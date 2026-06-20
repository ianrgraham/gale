//! GPU log-conformation (Fattal–Kupferman) viscoelastic transport operator —
//! reusable library component.
//!
//! Carries the `#[cuda_module]` device kernel plus a host launch wrapper
//! ([`logconf_psi_rhs`]) for the high-Wi-robust constitutive update. Per node it
//! eigendecomposes Ψ (`sqrt`) and forms the matrix `exp` (`exp`) — a full battery
//! of libdevice math, the last libdevice-blocked operator. Element-local
//! (nodal-collocation advection). Validated bit-for-bit against the CPU oracle
//! `gale::dg::LogConfOldroydB::psi_rhs`.
//!
//!   ∂ₜΨ = −(u·∇)Ψ + (ΩΨ − ΨΩ) + 2B + (1/λ)(e^{−Ψ} − I).

use cuda_core::{CudaContext, CudaStream, DeviceBuffer, LaunchConfig};
#[cfg(feature = "autodiff")]
use cuda_device::device;
use cuda_device::{kernel, thread, DisjointSlice, SharedArray};
use cuda_host::cuda_module;
use gale::dg::{LogConfOldroydB, Mesh2d};
use std::sync::Arc;

const NN_MAX: usize = 81;

#[cuda_module]
mod kernels {
    use super::*;
    #[cfg(feature = "autodiff")]
    use core::autodiff::autodiff_forward;

    /// `∂ₜΨ` for one element per block, one node per thread.
    #[kernel]
    #[allow(clippy::too_many_arguments)]
    pub fn psi_rhs(
        d: &[f64], ux: &[f64], uy: &[f64], pxx: &[f64], pxy: &[f64], pyy: &[f64],
        rx: &[f64], ry: &[f64], sx: &[f64], sy: &[f64],
        inv_lambda: f64, n1: u32,
        mut dxx: DisjointSlice<f64>, mut dxy: DisjointSlice<f64>, mut dyy: DisjointSlice<f64>,
    ) {
        static mut DS: SharedArray<f64, NN_MAX> = SharedArray::UNINIT;
        static mut UXS: SharedArray<f64, NN_MAX> = SharedArray::UNINIT;
        static mut UYS: SharedArray<f64, NN_MAX> = SharedArray::UNINIT;
        static mut AXX: SharedArray<f64, NN_MAX> = SharedArray::UNINIT;
        static mut AXY: SharedArray<f64, NN_MAX> = SharedArray::UNINIT;
        static mut AYY: SharedArray<f64, NN_MAX> = SharedArray::UNINIT;
        let n1 = n1 as usize;
        let nn = n1 * n1;
        let e = thread::blockIdx_x() as usize;
        let m = thread::threadIdx_x() as usize;
        let b = e * nn + m;
        unsafe {
            DS[m] = d[m];
            UXS[m] = ux[b];
            UYS[m] = uy[b];
            AXX[m] = pxx[b];
            AXY[m] = pxy[b];
            AYY[m] = pyy[b];
        }
        thread::sync_threads();

        let i = m % n1;
        let j = m / n1;
        // r/s derivatives of ux, uy, and the three Ψ components.
        let (mut ru, mut su) = (0.0f64, 0.0f64);
        let (mut rv, mut sv) = (0.0f64, 0.0f64);
        let (mut rxx, mut sxx) = (0.0f64, 0.0f64);
        let (mut rxy, mut sxy) = (0.0f64, 0.0f64);
        let (mut ryy, mut syy) = (0.0f64, 0.0f64);
        let mut k = 0usize;
        while k < n1 {
            let dik = unsafe { DS[i * n1 + k] };
            let djk = unsafe { DS[j * n1 + k] };
            let cr = k + j * n1;
            let cs = i + k * n1;
            unsafe {
                ru += dik * UXS[cr];
                su += djk * UXS[cs];
                rv += dik * UYS[cr];
                sv += djk * UYS[cs];
                rxx += dik * AXX[cr];
                sxx += djk * AXX[cs];
                rxy += dik * AXY[cr];
                sxy += djk * AXY[cs];
                ryy += dik * AYY[cr];
                syy += djk * AYY[cs];
            }
            k += 1;
        }
        let (rxm, rym, sxm, sym) = (rx[b], ry[b], sx[b], sy[b]);
        let lxx = rxm * ru + sxm * su;
        let lxy = rym * ru + sym * su;
        let lyx = rxm * rv + sxm * sv;
        let lyy = rym * rv + sym * sv;
        let pxx_x = rxm * rxx + sxm * sxx;
        let pxx_y = rym * rxx + sym * sxx;
        let pxy_x = rxm * rxy + sxm * sxy;
        let pxy_y = rym * rxy + sym * sxy;
        let pyy_x = rxm * ryy + sxm * syy;
        let pyy_y = rym * ryy + sym * syy;

        let (u, v) = (unsafe { UXS[m] }, unsafe { UYS[m] });
        let (p, q, r) = (unsafe { AXX[m] }, unsafe { AXY[m] }, unsafe { AYY[m] });

        // Eigendecomposition of Ψ. Eigenvector for the larger eigenvalue computed
        // DIRECTLY (avoids `atan2`, which the backend's device-intrinsic table does
        // not yet map): v=(μ₁−r, q); μ₁−r = ½(p−r)+rad ≥ 0 matches the CPU `c≥0`
        // branch, so the frames agree to round-off.
        let tr = 0.5 * (p + r);
        let diff = p - r;
        let rad = (0.25 * diff * diff + q * q).sqrt();
        let mu1 = tr + rad;
        let mu2 = tr - rad;
        let mu1r = mu1 - r;
        let nrm = (mu1r * mu1r + q * q).sqrt();
        let (c0, s0) = if nrm > 1e-300 { (mu1r / nrm, q / nrm) } else { (0.0, 1.0) };
        let l1 = mu1.exp();
        let l2 = mu2.exp();

        // Frame for B/M: align with the rate-of-strain near the isotropic point.
        let mut c = c0;
        let mut s = s0;
        if (mu1 - mu2).abs() < 1e-7 {
            let bb = 0.5 * (lxy + lyx);
            let dd = lyy;
            let radd = (0.25 * (lxx - dd) * (lxx - dd) + bb * bb).sqrt();
            let mu1dr = 0.5 * (lxx + dd) + radd - dd;
            let nrmd = (mu1dr * mu1dr + bb * bb).sqrt();
            if nrmd > 1e-300 {
                c = mu1dr / nrmd;
                s = bb / nrmd;
            } else {
                c = 0.0;
                s = 1.0;
            }
        }

        // M = RᵀLR, then B = R diag(m11,m22) Rᵀ.
        let a1 = c * lxx + s * lyx;
        let a2 = c * lxy + s * lyy;
        let bb1 = -s * lxx + c * lyx;
        let bb2 = -s * lxy + c * lyy;
        let m11 = a1 * c + a2 * s;
        let m12 = -a1 * s + a2 * c;
        let m21 = bb1 * c + bb2 * s;
        let m22 = -bb1 * s + bb2 * c;
        let bxx = c * c * m11 + s * s * m22;
        let bxy = c * s * (m11 - m22);
        let byy = s * s * m11 + c * c * m22;

        // Rotation rate ω (rotation-invariant in 2D).
        let denom = l2 - l1;
        let omega = if denom.abs() > 1e-12 {
            (m12 * l2 + m21 * l1) / denom
        } else {
            0.0
        };
        let rot_xx = 2.0 * omega * q;
        let rot_xy = omega * (r - p);
        let rot_yy = -2.0 * omega * q;

        // (1/λ)(e^{−Ψ} − I) using the original eigenframe.
        let e1 = (-mu1).exp();
        let e2 = (-mu2).exp();
        let em_xx = c0 * c0 * e1 + s0 * s0 * e2;
        let em_xy = c0 * s0 * (e1 - e2);
        let em_yy = s0 * s0 * e1 + c0 * c0 * e2;

        let adv_xx = u * pxx_x + v * pxx_y;
        let adv_xy = u * pxy_x + v * pxy_y;
        let adv_yy = u * pyy_x + v * pyy_y;

        if let Some(o) = dxx.get_mut(thread::index_1d()) {
            *o = -adv_xx + rot_xx + 2.0 * bxx + inv_lambda * (em_xx - 1.0);
        }
        if let Some(o) = dxy.get_mut(thread::index_1d()) {
            *o = -adv_xy + rot_xy + 2.0 * bxy + inv_lambda * em_xy;
        }
        if let Some(o) = dyy.get_mut(thread::index_1d()) {
            *o = -adv_yy + rot_yy + 2.0 * byy + inv_lambda * (em_yy - 1.0);
        }
    }

    /// Per-node IMEX implicit relaxation solve (Phase 3). Solves `Ψ − γ·S(Ψ) = B` with
    /// `S(Ψ) = (1/λ)(e^{−Ψ} − I)` by eigendecomposing the RHS `B` and, per eigenvalue `μ`,
    /// Newton-solving the scalar `ψ − (γ/λ)(e^{−ψ} − 1) = μ` (monotone ⇒ globally
    /// convergent), then recomposing in `B`'s eigenframe. Purely pointwise (no neighbour
    /// data); mirrors `gale::dg::LogConfOldroydB::implicit_relax_solve`. One node per thread.
    #[kernel]
    #[allow(clippy::too_many_arguments)]
    pub fn implicit_relax(
        bxx: &[f64], bxy: &[f64], byy: &[f64],
        gamma: f64, inv_lambda: f64, alpha: f64, ext: f64, n1: u32,
        mut oxx: DisjointSlice<f64>, mut oxy: DisjointSlice<f64>, mut oyy: DisjointSlice<f64>,
    ) {
        let nn = (n1 as usize) * (n1 as usize);
        let e = thread::blockIdx_x() as usize;
        let m = thread::threadIdx_x() as usize;
        let b = e * nn + m;
        let (p, q, r) = (bxx[b], bxy[b], byy[b]);
        // Eigendecompose B (atan2-free, identical to the psi_rhs frame).
        let tr = 0.5 * (p + r);
        let diff = p - r;
        let rad = (0.25 * diff * diff + q * q).sqrt();
        let mu1 = tr + rad;
        let mu2 = tr - rad;
        let mu1r = mu1 - r;
        let nrm = (mu1r * mu1r + q * q).sqrt();
        let (c0, s0) = if nrm > 1e-300 { (mu1r / nrm, q / nrm) } else { (0.0, 1.0) };
        let gl = gamma * inv_lambda;
        let mut psi1 = mu1;
        let mut psi2 = mu2;
        if ext < 1e300 {
            // FENE-P: bisection on the trace T ∈ (0, b). For a trial T, each eigenvalue solves
            // the monotone scalar ψ − (γ/λ)e^{−ψ} = μ − (γ/λ)f(T); consistency Σe^{ψ_i} = T is a
            // monotone 1-D root. The bracket keeps T < b ⇒ bound-preserving.
            let mut lo = 1e-12;
            let mut hi = ext * (1.0 - 1e-12);
            let mut it = 0usize;
            while it < 80 {
                let t = 0.5 * (lo + hi);
                let ff = (1.0 - 2.0 / ext) / (1.0 - t / ext);
                let beta1 = mu1 - gl * ff;
                let mut q1 = mu1;
                let mut j1 = 0usize;
                while j1 < 50 {
                    let e = (-q1).exp();
                    let st = (q1 - gl * e - beta1) / (1.0 + gl * e);
                    q1 -= st;
                    if st.abs() < 1e-14 {
                        break;
                    }
                    j1 += 1;
                }
                let beta2 = mu2 - gl * ff;
                let mut q2 = mu2;
                let mut j2 = 0usize;
                while j2 < 50 {
                    let e = (-q2).exp();
                    let st = (q2 - gl * e - beta2) / (1.0 + gl * e);
                    q2 -= st;
                    if st.abs() < 1e-14 {
                        break;
                    }
                    j2 += 1;
                }
                psi1 = q1;
                psi2 = q2;
                if q1.exp() + q2.exp() - t > 0.0 {
                    lo = t;
                } else {
                    hi = t;
                }
                it += 1;
            }
        } else {
            // Oldroyd-B (α=0) / Giesekus Newton per eigenvalue:
            //   g(ψ) = ψ + (γ/λ)(1−e^{−ψ}) + (γα/λ)(e^ψ−1)²e^{−ψ} − μ,  g′ > 0.
            let mut it = 0usize;
            while it < 60 {
                let em1 = (-psi1).exp();
                let ep1 = psi1.exp();
                let w1 = ep1 - 1.0;
                let g1 = psi1 + gl * (1.0 - em1) + gl * alpha * w1 * w1 * em1 - mu1;
                let gp1 = 1.0 + gl * em1 * (1.0 + alpha * (ep1 * ep1 - 1.0));
                let step1 = g1 / gp1;
                psi1 -= step1;
                let em2 = (-psi2).exp();
                let ep2 = psi2.exp();
                let w2 = ep2 - 1.0;
                let g2 = psi2 + gl * (1.0 - em2) + gl * alpha * w2 * w2 * em2 - mu2;
                let gp2 = 1.0 + gl * em2 * (1.0 + alpha * (ep2 * ep2 - 1.0));
                let step2 = g2 / gp2;
                psi2 -= step2;
                if step1.abs() < 1e-14 && step2.abs() < 1e-14 {
                    break;
                }
                it += 1;
            }
        }
        // Recompose Ψ = R diag(ψ1, ψ2) Rᵀ.
        if let Some(o) = oxx.get_mut(thread::index_1d()) {
            *o = c0 * c0 * psi1 + s0 * s0 * psi2;
        }
        if let Some(o) = oxy.get_mut(thread::index_1d()) {
            *o = c0 * s0 * (psi1 - psi2);
        }
        if let Some(o) = oyy.get_mut(thread::index_1d()) {
            *o = s0 * s0 * psi1 + c0 * c0 * psi2;
        }
    }

    /// Differentiable `#[device]` core of [`implicit_relax`] for std::autodiff →
    /// Enzyme (via `cargo oxide --features autodiff`). Identical math, but over
    /// **raw pointers** — the `&[f64]`/`DisjointSlice` ABI is mistranslated under
    /// differentiation (`<[T]>::get_mut` → discarded stack copy; `o[i]=` is an
    /// unsupported 2-level write). `inv_lambda` (1/λ) is the `Dual` differentiation
    /// target; the three outputs carry the tangent shadow. `index_1d()` =
    /// blockIdx·blockDim + threadIdx = the node `b` (block_dim = nn, as launched).
    /// The pipeline synthesizes `d_implicit_relax_core` + `d_implicit_relax_core_primal`.
    #[cfg(feature = "autodiff")]
    #[device]
    #[autodiff_forward(
        d_implicit_relax_core,
        Const, Const, Const, Const, Dual, Const, Const, Dual, Dual, Dual
    )]
    #[allow(clippy::too_many_arguments)]
    pub fn implicit_relax_core(
        bxx: *const f64, bxy: *const f64, byy: *const f64,
        gamma: f64, inv_lambda: f64, alpha: f64, ext: f64,
        oxx: *mut f64, oxy: *mut f64, oyy: *mut f64,
    ) {
        let off = thread::index_1d().get() * 8;
        let (p, q, r) = unsafe {
            (
                *((bxx as usize).wrapping_add(off) as *const f64),
                *((bxy as usize).wrapping_add(off) as *const f64),
                *((byy as usize).wrapping_add(off) as *const f64),
            )
        };
        // Eigendecompose B (atan2-free, identical to `implicit_relax`).
        let tr = 0.5 * (p + r);
        let diff = p - r;
        let rad = (0.25 * diff * diff + q * q).sqrt();
        let mu1 = tr + rad;
        let mu2 = tr - rad;
        let mu1r = mu1 - r;
        let nrm = (mu1r * mu1r + q * q).sqrt();
        let (c0, s0) = if nrm > 1e-300 { (mu1r / nrm, q / nrm) } else { (0.0, 1.0) };
        let gl = gamma * inv_lambda;
        let mut psi1 = mu1;
        let mut psi2 = mu2;
        if ext < 1e300 {
            let mut lo = 1e-12;
            let mut hi = ext * (1.0 - 1e-12);
            let mut it = 0usize;
            while it < 80 {
                let t = 0.5 * (lo + hi);
                let ff = (1.0 - 2.0 / ext) / (1.0 - t / ext);
                let beta1 = mu1 - gl * ff;
                let mut q1 = mu1;
                let mut j1 = 0usize;
                while j1 < 50 {
                    let e = (-q1).exp();
                    let st = (q1 - gl * e - beta1) / (1.0 + gl * e);
                    q1 -= st;
                    if st.abs() < 1e-14 {
                        break;
                    }
                    j1 += 1;
                }
                let beta2 = mu2 - gl * ff;
                let mut q2 = mu2;
                let mut j2 = 0usize;
                while j2 < 50 {
                    let e = (-q2).exp();
                    let st = (q2 - gl * e - beta2) / (1.0 + gl * e);
                    q2 -= st;
                    if st.abs() < 1e-14 {
                        break;
                    }
                    j2 += 1;
                }
                psi1 = q1;
                psi2 = q2;
                if q1.exp() + q2.exp() - t > 0.0 {
                    lo = t;
                } else {
                    hi = t;
                }
                it += 1;
            }
        } else {
            let mut it = 0usize;
            while it < 60 {
                let em1 = (-psi1).exp();
                let ep1 = psi1.exp();
                let w1 = ep1 - 1.0;
                let g1 = psi1 + gl * (1.0 - em1) + gl * alpha * w1 * w1 * em1 - mu1;
                let gp1 = 1.0 + gl * em1 * (1.0 + alpha * (ep1 * ep1 - 1.0));
                let step1 = g1 / gp1;
                psi1 -= step1;
                let em2 = (-psi2).exp();
                let ep2 = psi2.exp();
                let w2 = ep2 - 1.0;
                let g2 = psi2 + gl * (1.0 - em2) + gl * alpha * w2 * w2 * em2 - mu2;
                let gp2 = 1.0 + gl * em2 * (1.0 + alpha * (ep2 * ep2 - 1.0));
                let step2 = g2 / gp2;
                psi2 -= step2;
                if step1.abs() < 1e-14 && step2.abs() < 1e-14 {
                    break;
                }
                it += 1;
            }
        }
        unsafe {
            *((oxx as usize).wrapping_add(off) as *mut f64) = c0 * c0 * psi1 + s0 * s0 * psi2;
            *((oxy as usize).wrapping_add(off) as *mut f64) = c0 * s0 * (psi1 - psi2);
            *((oyy as usize).wrapping_add(off) as *mut f64) = s0 * s0 * psi1 + c0 * c0 * psi2;
        }
    }

    /// `tr exp(Ψ)` for a symmetric 2×2 `Ψ=[a,b,d]` (atan2-free eigenvalues `tr±rad`).
    fn tr_exp2(a: f64, b: f64, d: f64) -> f64 {
        let tr = 0.5 * (a + d);
        let diff = a - d;
        let rad = (0.25 * diff * diff + b * b).sqrt();
        (tr + rad).exp() + (tr - rad).exp()
    }

    /// Log-conformation **FENE-P trace-bound limiter** (Phase 4 GPU port). One block per
    /// element, one thread per node. Computes the quadrature-weighted cell mean `Ψ̄` (each
    /// thread reduces the shared arrays — `nn` small, no power-of-2 dependence), the per-node
    /// `θ` keeping `tr exp(Ψ̄+θΔ) ≤ b` (bisection on the convex `g(θ)`), the element-wide
    /// `θ = min`, then applies `Ψ ← Ψ̄ + θ(Ψ−Ψ̄)`. Mirrors `gale::dg::limit_logconf_trace_bound`.
    #[kernel]
    pub fn limit_logconf_trace(
        pxx: &[f64], pxy: &[f64], pyy: &[f64], jw: &[f64],
        b_max: f64, n1: u32,
        mut oxx: DisjointSlice<f64>, mut oxy: DisjointSlice<f64>, mut oyy: DisjointSlice<f64>,
    ) {
        static mut PXX: SharedArray<f64, NN_MAX> = SharedArray::UNINIT;
        static mut PXY: SharedArray<f64, NN_MAX> = SharedArray::UNINIT;
        static mut PYY: SharedArray<f64, NN_MAX> = SharedArray::UNINIT;
        static mut WS: SharedArray<f64, NN_MAX> = SharedArray::UNINIT;
        static mut TH: SharedArray<f64, NN_MAX> = SharedArray::UNINIT;
        let nn = (n1 as usize) * (n1 as usize);
        let e = thread::blockIdx_x() as usize;
        let m = thread::threadIdx_x() as usize;
        let b = e * nn + m;
        unsafe {
            PXX[m] = pxx[b];
            PXY[m] = pxy[b];
            PYY[m] = pyy[b];
            WS[m] = jw[b];
        }
        thread::sync_threads();
        // Quadrature-weighted cell mean (redundant per-thread reduction over shared).
        let (mut wsum, mut mxx, mut mxy, mut myy) = (0.0f64, 0.0f64, 0.0f64, 0.0f64);
        let mut j = 0usize;
        while j < nn {
            let w = unsafe { WS[j] };
            wsum += w;
            unsafe {
                mxx += w * PXX[j];
                mxy += w * PXY[j];
                myy += w * PYY[j];
            }
            j += 1;
        }
        mxx /= wsum;
        mxy /= wsum;
        myy /= wsum;
        // Per-node θ (this thread's node). Skip if the mean is itself over the bound.
        let (dxx, dxy, dyy) =
            (unsafe { PXX[m] } - mxx, unsafe { PXY[m] } - mxy, unsafe { PYY[m] } - myy);
        let mut th = 1.0f64;
        if tr_exp2(mxx, mxy, myy) <= b_max && tr_exp2(mxx + dxx, mxy + dxy, myy + dyy) > b_max {
            let (mut lo, mut hi) = (0.0f64, 1.0f64);
            let mut it = 0usize;
            while it < 60 {
                let mid = 0.5 * (lo + hi);
                if tr_exp2(mxx + mid * dxx, mxy + mid * dxy, myy + mid * dyy) <= b_max {
                    lo = mid;
                } else {
                    hi = mid;
                }
                it += 1;
            }
            th = lo;
        }
        unsafe { TH[m] = th; }
        thread::sync_threads();
        // Element θ = min over nodes (redundant per-thread reduction).
        let mut theta = 1.0f64;
        let mut j2 = 0usize;
        while j2 < nn {
            let t = unsafe { TH[j2] };
            if t < theta {
                theta = t;
            }
            j2 += 1;
        }
        if let Some(o) = oxx.get_mut(thread::index_1d()) {
            *o = mxx + theta * (unsafe { PXX[m] } - mxx);
        }
        if let Some(o) = oxy.get_mut(thread::index_1d()) {
            *o = mxy + theta * (unsafe { PXY[m] } - mxy);
        }
        if let Some(o) = oyy.get_mut(thread::index_1d()) {
            *o = myy + theta * (unsafe { PYY[m] } - myy);
        }
    }

    /// Recover the **conformation** `C = exp(Ψ)` from the log-conformation `Ψ=[a,b,d]` per node
    /// (symmetric matrix exp via eigendecomposition). One thread per node (grid-stride). Mirrors
    /// the host `gale::dg::LogConfOldroydB::conformation` (`sym_apply` with `f = exp`).
    #[kernel]
    pub fn conformation(
        pxx: &[f64], pxy: &[f64], pyy: &[f64], n1: u32,
        mut cxx: DisjointSlice<f64>, mut cxy: DisjointSlice<f64>, mut cyy: DisjointSlice<f64>,
    ) {
        let nn = (n1 as usize) * (n1 as usize);
        let e = thread::blockIdx_x() as usize;
        let m = thread::threadIdx_x() as usize;
        let b = e * nn + m;
        let (pa, pb, pd) = (pxx[b], pxy[b], pyy[b]);
        let tr = 0.5 * (pa + pd);
        let diff = pa - pd;
        let rad = (0.25 * diff * diff + pb * pb).sqrt();
        let (m1, m2) = (tr + rad, tr - rad);
        let theta = 0.5f64 * (2.0f64 * pb).atan2(diff);
        let (c, s) = (theta.cos(), theta.sin());
        let (e1, e2) = (m1.exp(), m2.exp());
        if let Some(o) = cxx.get_mut(thread::index_1d()) {
            *o = c * c * e1 + s * s * e2;
        }
        if let Some(o) = cxy.get_mut(thread::index_1d()) {
            *o = c * s * (e1 - e2);
        }
        if let Some(o) = cyy.get_mut(thread::index_1d()) {
            *o = s * s * e1 + c * c * e2;
        }
    }

    /// **Upwind advection surface lift** for the 3-component conformation field `Ψ` — the DG
    /// inter-element flux correction the volume `psi_rhs` omits. One block per element, **one thread
    /// per node, fully node-parallel**: thread `m` owns node `m`, visits the ≤2 faces it lies on
    /// (corner ⇒ 2, via the closed-form tensor-product membership the SIPG `operator` already uses —
    /// South t=0:a=i, East t=1:a=j, North t=2:a=i, West t=3:a=j), and accumulates the inflow
    /// correction `sw·uₙ·(Ψ_int − Ψ_ext)` (`uₙ = u·n`, only where `uₙ < 0`) into registers, then
    /// writes `out = rface / jw`. Mirrors the host `upwind_advection_lift`. Conforming meshes
    /// (interior + boundary faces); boundary faces pass `face_nbr = self` so they contribute zero
    /// (no-slip walls have `uₙ = 0`; no-inflow boundaries use the interior trace).
    ///
    /// Replaces the previous thread-0-serial implementation (one node-thread walked all 4×n1 faces
    /// while the other 15 idled — the measured 409 µs/call hot spot). No shared memory, no barrier;
    /// the per-node accumulation order (faces in t=0..4 order) is identical to the old thread-0 loop,
    /// so the result is bit-for-bit unchanged. The `un<0` inflow test is kept as a branchless select
    /// (`fac = (un<0) ? sw·uₙ : 0`) — `fac·0`-style adds when `uₙ≥0` are no-ops, preserving bit-equality
    /// while avoiding data-dependent warp divergence on the sign test.
    #[kernel]
    #[allow(clippy::too_many_arguments)]
    pub fn upwind_lift(
        ux: &[f64], uy: &[f64], pxx: &[f64], pxy: &[f64], pyy: &[f64], jw: &[f64], n1: u32,
        face_vl: &[u32], face_nx: &[f64], face_ny: &[f64], face_sw: &[f64], face_nbr: &[u32],
        mut oxx: DisjointSlice<f64>, mut oxy: DisjointSlice<f64>, mut oyy: DisjointSlice<f64>,
    ) {
        let _ = face_vl; // node index is `m` itself (vl == m by the closed-form face-node convention)
        let n1 = n1 as usize;
        let nn = n1 * n1;
        let e = thread::blockIdx_x() as usize;
        let m = thread::threadIdx_x() as usize;
        let b = e * nn + m;
        let i = m % n1;
        let j = m / n1;
        let (uxb, uyb) = (ux[b], uy[b]);
        let (pa, pb_, pd) = (pxx[b], pxy[b], pyy[b]); // this node's interior Ψ traces
        let mut rxx = 0.0f64;
        let mut rxy = 0.0f64;
        let mut ryy = 0.0f64;
        let mut t = 0usize;
        while t < 4 {
            // Closed-form face membership: which edge node `m` lies on and its along-edge index `a`.
            let (on, a) = if t == 0 {
                (j == 0, i) // South
            } else if t == 1 {
                (i == n1 - 1, j) // East
            } else if t == 2 {
                (j == n1 - 1, i) // North
            } else {
                (i == 0, j) // West
            };
            if on {
                let idx = (e * 4 + t) * n1 + a;
                let un = uxb * face_nx[idx] + uyb * face_ny[idx];
                let ext = face_nbr[idx] as usize;
                // Branchless inflow gate: zero contribution when uₙ ≥ 0 (outflow / no-slip).
                let fac = if un < 0.0 { face_sw[idx] * un } else { 0.0 };
                rxx += fac * (pa - pxx[ext]);
                rxy += fac * (pb_ - pxy[ext]);
                ryy += fac * (pd - pyy[ext]);
            }
            t += 1;
        }
        let inv = 1.0 / jw[b];
        if let Some(o) = oxx.get_mut(thread::index_1d()) {
            *o = rxx * inv;
        }
        if let Some(o) = oxy.get_mut(thread::index_1d()) {
            *o = rxy * inv;
        }
        if let Some(o) = oyy.get_mut(thread::index_1d()) {
            *o = ryy * inv;
        }
    }
}

/// Compute the log-conformation (Fattal–Kupferman) Ψ time-derivative `∂ₜΨ` on the
/// GPU for an Oldroyd-B fluid, returning the three independent components
/// `[∂ₜΨxx, ∂ₜΨxy, ∂ₜΨyy]` in the same `[Vec<f64>; 3]` layout as the CPU oracle.
/// Reusable host wrapper around the [`kernels::psi_rhs`] device kernel: flattens
/// the mesh metrics, uploads the velocity + Ψ state, launches one block per
/// element (one thread per node), and gathers the result. Bit-for-bit equal to
/// `gale::dg::LogConfOldroydB::psi_rhs`.
///
/// `lc` carries the relaxation time λ; only its reciprocal `1/λ` is needed by the
/// kernel. `psi`, `ux`, and `uy` must each be `n_elements·n_nodes` long.
pub fn logconf_psi_rhs(
    mesh: &Mesh2d,
    lc: &LogConfOldroydB,
    psi: &[Vec<f64>; 3],
    ux: &[f64],
    uy: &[f64],
) -> Result<[Vec<f64>; 3], Box<dyn std::error::Error>> {
    GpuLogConf::new()?.psi_rhs(mesh, lc, psi, ux, uy)
}

/// GPU per-node IMEX implicit relaxation solve (Phase 3): solve `Ψ − γ·S(Ψ) = B` for the
/// log-conformation `Ψ`, where `S(Ψ) = (1/λ)(e^{−Ψ} − I)` and `B = b` is the accumulated
/// explicit stage. Host wrapper around [`kernels::implicit_relax`]; the solve is local and
/// pointwise (one thread per node, no halo). Matches the CPU oracle
/// `gale::dg::LogConfOldroydB::implicit_relax_solve` to round-off. `gamma` is the stage
/// coefficient `dt·a^I_{ii}`; `b` is the `[Bxx, Bxy, Byy]` RHS, each `n_elements·n_nodes` long.
pub fn logconf_implicit_relax(
    mesh: &Mesh2d,
    lc: &LogConfOldroydB,
    b: &[Vec<f64>; 3],
    gamma: f64,
) -> Result<[Vec<f64>; 3], Box<dyn std::error::Error>> {
    GpuLogConf::new()?.implicit_relax(mesh, lc, b, gamma)
}

/// Persistent GPU handle for the log-conformation transport kernels. Holds the CUDA context +
/// loaded kernel module so the per-step constitutive evaluations REUSE them, instead of creating a
/// context and JIT-loading the whole kernel bundle on every call — which is ~0.3 s of host-side
/// setup each and, called 3× per SSP-RK3 step, leaves the GPU almost idle (the device does one tiny
/// kernel then waits while the host reloads the module). Mirrors the persistent
/// [`GpuPoisson`](crate::operators::poisson::GpuPoisson). The handle is mesh-independent (only the
/// data uploads depend on the mesh), so it never needs rebuilding on an AMR remesh: build once,
/// reuse for the whole run. The free functions [`logconf_psi_rhs`]/[`logconf_implicit_relax`] are
/// one-shot shims (build a fresh handle per call) kept for tests and non-hot callers.
pub struct GpuLogConf {
    stream: Arc<CudaStream>,
    module: kernels::LoadedModule,
}

impl GpuLogConf {
    /// Create the CUDA context and load the kernel module once.
    pub fn new() -> Result<Self, Box<dyn std::error::Error>> {
        let ctx = CudaContext::new(0)?;
        let stream = ctx.default_stream();
        let module = kernels::load(&ctx)?;
        Ok(Self { stream, module })
    }

    /// Build on a caller-provided stream (shares one non-legacy stream with the `GpuPoissonMg`
    /// handles in a device-resident step — see [`GpuPoissonMg::new_on_stream`]).
    pub fn new_on_stream(stream: std::sync::Arc<CudaStream>) -> Result<Self, Box<dyn std::error::Error>> {
        let module = kernels::load(stream.context())?;
        Ok(Self { stream, module })
    }

    /// `∂ₜΨ` (the log-conformation transport rhs) on the GPU, reusing this handle's context/module.
    /// Bit-for-bit equal to the one-shot [`logconf_psi_rhs`] (and the CPU oracle).
    pub fn psi_rhs(
        &self,
        mesh: &Mesh2d,
        lc: &LogConfOldroydB,
        psi: &[Vec<f64>; 3],
        ux: &[f64],
        uy: &[f64],
    ) -> Result<[Vec<f64>; 3], Box<dyn std::error::Error>> {
        let nn = mesh.refq.n_nodes();
        let ne = mesh.n_elements();
        let ndof = ne * nn;
        let n1 = mesh.order + 1;
        assert_eq!(ux.len(), ndof, "velocity length must be n_elements·n_nodes");
        assert_eq!(uy.len(), ndof, "velocity length must be n_elements·n_nodes");
        for c in psi {
            assert_eq!(c.len(), ndof, "Ψ component length must be n_elements·n_nodes");
        }
        assert!(nn <= NN_MAX, "p too large for NN_MAX={NN_MAX}");

        // Flatten per-node metrics.
        let (mut rx, mut ry, mut sx, mut sy) =
            (vec![0.0; ndof], vec![0.0; ndof], vec![0.0; ndof], vec![0.0; ndof]);
        for (e, el) in mesh.elements.iter().enumerate() {
            for k in 0..nn {
                rx[e * nn + k] = el.geom.rx[k];
                ry[e * nn + k] = el.geom.ry[k];
                sx[e * nn + k] = el.geom.sx[k];
                sy[e * nn + k] = el.geom.sy[k];
            }
        }

        let stream = &self.stream;
        let module = &self.module;
        let up = |v: &[f64]| DeviceBuffer::from_host(stream, v);
        let d_dev = up(&mesh.refq.line.diff)?;
        let ux_d = up(ux)?;
        let uy_d = up(uy)?;
        let pxx_d = up(&psi[0])?;
        let pxy_d = up(&psi[1])?;
        let pyy_d = up(&psi[2])?;
        let rx_d = up(&rx)?;
        let ry_d = up(&ry)?;
        let sx_d = up(&sx)?;
        let sy_d = up(&sy)?;
        let mut dxx = DeviceBuffer::<f64>::zeroed(stream, ndof)?;
        let mut dxy = DeviceBuffer::<f64>::zeroed(stream, ndof)?;
        let mut dyy = DeviceBuffer::<f64>::zeroed(stream, ndof)?;

        let cfg = LaunchConfig { grid_dim: (ne as u32, 1, 1), block_dim: (nn as u32, 1, 1), shared_mem_bytes: 0 };
        module.psi_rhs(
            stream, cfg, &d_dev, &ux_d, &uy_d, &pxx_d, &pxy_d, &pyy_d,
            &rx_d, &ry_d, &sx_d, &sy_d, lc.lambda.recip(), n1 as u32,
            &mut dxx, &mut dxy, &mut dyy,
        )?;
        Ok([dxx.to_host_vec(stream)?, dxy.to_host_vec(stream)?, dyy.to_host_vec(stream)?])
    }

    /// IMEX implicit relaxation solve on the GPU, reusing this handle's context/module. Bit-for-bit
    /// equal to the one-shot [`logconf_implicit_relax`].
    pub fn implicit_relax(
        &self,
        mesh: &Mesh2d,
        lc: &LogConfOldroydB,
        b: &[Vec<f64>; 3],
        gamma: f64,
    ) -> Result<[Vec<f64>; 3], Box<dyn std::error::Error>> {
        let nn = mesh.refq.n_nodes();
        let ne = mesh.n_elements();
        let ndof = ne * nn;
        let n1 = mesh.order + 1;
        for c in b {
            assert_eq!(c.len(), ndof, "B component length must be n_elements·n_nodes");
        }
        assert!(nn <= NN_MAX, "p too large for NN_MAX={NN_MAX}");

        let stream = &self.stream;
        let module = &self.module;
        let up = |v: &[f64]| DeviceBuffer::from_host(stream, v);
        let bxx_d = up(&b[0])?;
        let bxy_d = up(&b[1])?;
        let byy_d = up(&b[2])?;
        let mut oxx = DeviceBuffer::<f64>::zeroed(stream, ndof)?;
        let mut oxy = DeviceBuffer::<f64>::zeroed(stream, ndof)?;
        let mut oyy = DeviceBuffer::<f64>::zeroed(stream, ndof)?;

        let cfg = LaunchConfig { grid_dim: (ne as u32, 1, 1), block_dim: (nn as u32, 1, 1), shared_mem_bytes: 0 };
        module.implicit_relax(
            stream, cfg, &bxx_d, &bxy_d, &byy_d,
            gamma, lc.lambda.recip(), lc.mobility, lc.extensibility, n1 as u32,
            &mut oxx, &mut oxy, &mut oyy,
        )?;
        Ok([oxx.to_host_vec(stream)?, oxy.to_host_vec(stream)?, oyy.to_host_vec(stream)?])
    }

    // ===== Device-resident launch wrappers (DeviceBuffer in/out, no host transfer) =============
    // These take pre-uploaded device buffers (state + mesh metrics/face data, owned by the caller)
    // and just launch the kernel on this handle's stream — the building blocks for a device-resident
    // viscoelastic step. `n1 = order+1`, `ne = n_elements`.

    /// `C = exp(Ψ)` on device (block per element). See [`kernels::conformation`].
    #[allow(clippy::too_many_arguments)]
    pub fn conformation_dev(
        &self, ne: usize, n1: u32,
        pxx: &DeviceBuffer<f64>, pxy: &DeviceBuffer<f64>, pyy: &DeviceBuffer<f64>,
        cxx: &mut DeviceBuffer<f64>, cxy: &mut DeviceBuffer<f64>, cyy: &mut DeviceBuffer<f64>,
    ) -> Result<(), Box<dyn std::error::Error>> {
        let nn = n1 * n1;
        let cfg = LaunchConfig { grid_dim: (ne as u32, 1, 1), block_dim: (nn, 1, 1), shared_mem_bytes: 0 };
        self.module.conformation(&self.stream, cfg, pxx, pxy, pyy, n1, cxx, cxy, cyy)?;
        Ok(())
    }

    /// Log-conf volume rhs `∂ₜΨ` on device (block per element). See [`kernels::psi_rhs`].
    #[allow(clippy::too_many_arguments)]
    pub fn psi_rhs_dev(
        &self, ne: usize, n1: u32, d: &DeviceBuffer<f64>,
        ux: &DeviceBuffer<f64>, uy: &DeviceBuffer<f64>,
        pxx: &DeviceBuffer<f64>, pxy: &DeviceBuffer<f64>, pyy: &DeviceBuffer<f64>,
        rx: &DeviceBuffer<f64>, ry: &DeviceBuffer<f64>, sx: &DeviceBuffer<f64>, sy: &DeviceBuffer<f64>,
        inv_lambda: f64,
        dxx: &mut DeviceBuffer<f64>, dxy: &mut DeviceBuffer<f64>, dyy: &mut DeviceBuffer<f64>,
    ) -> Result<(), Box<dyn std::error::Error>> {
        let nn = n1 * n1;
        let cfg = LaunchConfig { grid_dim: (ne as u32, 1, 1), block_dim: (nn, 1, 1), shared_mem_bytes: 0 };
        self.module.psi_rhs(&self.stream, cfg, d, ux, uy, pxx, pxy, pyy, rx, ry, sx, sy, inv_lambda, n1, dxx, dxy, dyy)?;
        Ok(())
    }

    /// Upwind advection surface lift for Ψ on device (block per element). See [`kernels::upwind_lift`].
    #[allow(clippy::too_many_arguments)]
    pub fn upwind_lift_dev(
        &self, ne: usize, n1: u32,
        ux: &DeviceBuffer<f64>, uy: &DeviceBuffer<f64>,
        pxx: &DeviceBuffer<f64>, pxy: &DeviceBuffer<f64>, pyy: &DeviceBuffer<f64>, jw: &DeviceBuffer<f64>,
        face_vl: &DeviceBuffer<u32>, face_nx: &DeviceBuffer<f64>, face_ny: &DeviceBuffer<f64>,
        face_sw: &DeviceBuffer<f64>, face_nbr: &DeviceBuffer<u32>,
        oxx: &mut DeviceBuffer<f64>, oxy: &mut DeviceBuffer<f64>, oyy: &mut DeviceBuffer<f64>,
    ) -> Result<(), Box<dyn std::error::Error>> {
        let nn = n1 * n1;
        let cfg = LaunchConfig { grid_dim: (ne as u32, 1, 1), block_dim: (nn, 1, 1), shared_mem_bytes: 0 };
        self.module.upwind_lift(&self.stream, cfg, ux, uy, pxx, pxy, pyy, jw, n1, face_vl, face_nx, face_ny, face_sw, face_nbr, oxx, oxy, oyy)?;
        Ok(())
    }

    /// FENE-P trace-bound limiter on device (block per element). See [`kernels::limit_logconf_trace`].
    #[allow(clippy::too_many_arguments)]
    pub fn limit_trace_dev(
        &self, ne: usize, n1: u32,
        pxx: &DeviceBuffer<f64>, pxy: &DeviceBuffer<f64>, pyy: &DeviceBuffer<f64>, jw: &DeviceBuffer<f64>,
        b_max: f64,
        oxx: &mut DeviceBuffer<f64>, oxy: &mut DeviceBuffer<f64>, oyy: &mut DeviceBuffer<f64>,
    ) -> Result<(), Box<dyn std::error::Error>> {
        let nn = n1 * n1;
        let cfg = LaunchConfig { grid_dim: (ne as u32, 1, 1), block_dim: (nn, 1, 1), shared_mem_bytes: 0 };
        self.module.limit_logconf_trace(&self.stream, cfg, pxx, pxy, pyy, jw, b_max, n1, oxx, oxy, oyy)?;
        Ok(())
    }

    /// This handle's CUDA stream — so a device-resident caller can allocate buffers on the same
    /// stream/context the kernels run on (interoperating with a `GpuPoissonMg` on the same context).
    pub fn stream(&self) -> &CudaStream {
        &self.stream
    }
}

/// Differentiate the implicit relaxation w.r.t. `1/λ`: returns `[∂Ψxx, ∂Ψxy, ∂Ψyy]
/// / ∂(1/λ)` per node, computed by Enzyme forward-mode through the
/// pipeline-synthesized `d_implicit_relax_core` kernel, launched via the **normal
/// cuda-host bundle loader**. The headline differentiable-gale primitive (inverse
/// rheology / parameter inference), now produced by `cargo oxide --features
/// autodiff` with no manual scripts. Needs the `autodiff` feature.
#[cfg(feature = "autodiff")]
pub fn logconf_implicit_relax_grad(
    mesh: &Mesh2d,
    lc: &LogConfOldroydB,
    b: &[Vec<f64>; 3],
    gamma: f64,
) -> Result<[Vec<f64>; 3], Box<dyn std::error::Error>> {
    use std::ffi::c_void;
    let nn = mesh.refq.n_nodes();
    let ne = mesh.n_elements();
    let ndof = ne * nn;
    for c in b {
        assert_eq!(c.len(), ndof, "B component length must be n_elements·n_nodes");
    }
    assert!(nn <= NN_MAX, "p too large for NN_MAX={NN_MAX}");

    let ctx = CudaContext::new(0)?;
    let stream = ctx.default_stream();
    let bxx = DeviceBuffer::from_host(&stream, &b[0])?;
    let bxy = DeviceBuffer::from_host(&stream, &b[1])?;
    let byy = DeviceBuffer::from_host(&stream, &b[2])?;
    // Primal outputs (discarded) + tangent shadows (the gradient), zero-seeded.
    let oxx = DeviceBuffer::<f64>::zeroed(&stream, ndof)?;
    let oxy = DeviceBuffer::<f64>::zeroed(&stream, ndof)?;
    let oyy = DeviceBuffer::<f64>::zeroed(&stream, ndof)?;
    let doxx = DeviceBuffer::<f64>::zeroed(&stream, ndof)?;
    let doxy = DeviceBuffer::<f64>::zeroed(&stream, ndof)?;
    let doyy = DeviceBuffer::<f64>::zeroed(&stream, ndof)?;

    let module = kernels::load(&ctx)?;
    let func = module.as_cuda_module().load_function("d_implicit_relax_core")?;

    // d_implicit_relax_core(bxx, bxy, byy, gamma, inv_lambda[seed=1], alpha, ext,
    //                       oxx, doxx, oxy, doxy, oyy, doyy)
    let mut a_bxx = bxx.cu_deviceptr();
    let mut a_bxy = bxy.cu_deviceptr();
    let mut a_byy = byy.cu_deviceptr();
    let mut a_gamma = gamma;
    let mut a_invlam = lc.lambda.recip();
    let mut a_alpha = lc.mobility;
    let mut a_ext = lc.extensibility;
    let mut a_oxx = oxx.cu_deviceptr();
    let mut a_doxx = doxx.cu_deviceptr();
    let mut a_oxy = oxy.cu_deviceptr();
    let mut a_doxy = doxy.cu_deviceptr();
    let mut a_oyy = oyy.cu_deviceptr();
    let mut a_doyy = doyy.cu_deviceptr();
    let mut args: Vec<*mut c_void> = vec![
        &mut a_bxx as *mut _ as *mut c_void,
        &mut a_bxy as *mut _ as *mut c_void,
        &mut a_byy as *mut _ as *mut c_void,
        &mut a_gamma as *mut _ as *mut c_void,
        &mut a_invlam as *mut _ as *mut c_void,
        &mut a_alpha as *mut _ as *mut c_void,
        &mut a_ext as *mut _ as *mut c_void,
        &mut a_oxx as *mut _ as *mut c_void,
        &mut a_doxx as *mut _ as *mut c_void,
        &mut a_oxy as *mut _ as *mut c_void,
        &mut a_doxy as *mut _ as *mut c_void,
        &mut a_oyy as *mut _ as *mut c_void,
        &mut a_doyy as *mut _ as *mut c_void,
    ];
    unsafe {
        cuda_core::launch_kernel_on_stream(
            &func,
            (ne as u32, 1, 1),
            (nn as u32, 1, 1),
            0,
            &stream,
            &mut args,
        )?;
    }
    Ok([
        doxx.to_host_vec(&stream)?,
        doxy.to_host_vec(&stream)?,
        doyy.to_host_vec(&stream)?,
    ])
}

/// GPU log-conformation FENE-P trace-bound limiter (Phase 4): enforce `tr exp(Ψ) ≤ b_max` on the
/// log-conformation field, returning the limited `[Ψxx, Ψxy, Ψyy]`. Host wrapper around
/// [`kernels::limit_logconf_trace`] (one block per element). Matches the CPU oracle
/// `gale::dg::limit_logconf_trace_bound`. SPD is preserved for free (`C = exp(Ψ)`).
pub fn logconf_limit_trace(
    mesh: &Mesh2d,
    psi: &[Vec<f64>; 3],
    b_max: f64,
) -> Result<[Vec<f64>; 3], Box<dyn std::error::Error>> {
    let nn = mesh.refq.n_nodes();
    let ne = mesh.n_elements();
    let ndof = ne * nn;
    let n1 = mesh.order + 1;
    for c in psi {
        assert_eq!(c.len(), ndof, "Ψ component length must be n_elements·n_nodes");
    }
    assert!(nn <= NN_MAX, "p too large for NN_MAX={NN_MAX}");

    let mut jw = vec![0.0; ndof];
    for (e, el) in mesh.elements.iter().enumerate() {
        for k in 0..nn {
            jw[e * nn + k] = el.geom.jw[k];
        }
    }

    let ctx = CudaContext::new(0)?;
    let stream = ctx.default_stream();
    let up = |v: &[f64]| DeviceBuffer::from_host(&stream, v);
    let pxx_d = up(&psi[0])?;
    let pxy_d = up(&psi[1])?;
    let pyy_d = up(&psi[2])?;
    let jw_d = up(&jw)?;
    let mut oxx = DeviceBuffer::<f64>::zeroed(&stream, ndof)?;
    let mut oxy = DeviceBuffer::<f64>::zeroed(&stream, ndof)?;
    let mut oyy = DeviceBuffer::<f64>::zeroed(&stream, ndof)?;

    let module = kernels::load(&ctx)?;
    let cfg = LaunchConfig { grid_dim: (ne as u32, 1, 1), block_dim: (nn as u32, 1, 1), shared_mem_bytes: 0 };
    module.limit_logconf_trace(
        &stream, cfg, &pxx_d, &pxy_d, &pyy_d, &jw_d, b_max, n1 as u32,
        &mut oxx, &mut oxy, &mut oyy,
    )?;
    Ok([oxx.to_host_vec(&stream)?, oxy.to_host_vec(&stream)?, oyy.to_host_vec(&stream)?])
}
