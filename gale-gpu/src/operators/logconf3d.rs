//! GPU log-conformation (Fattal–Kupferman) viscoelastic transport on the **3D hex
//! mesh** — the 3D analogue of [`crate::operators::logconf`], and the GPU
//! counterpart of `gale::dg::LogConfOldroydB3d::psi_rhs`. This is the hardest
//! kernel in the crate: per node it eigendecomposes the symmetric 3×3 Ψ with a
//! 12-sweep cyclic **Jacobi** solver (no closed form in 3D), then assembles the
//! full Fattal–Kupferman rotation/stretch decomposition in the eigenframe.
//!
//!   ∂ₜΨ = −(u·∇)Ψ + (ΩΨ − ΨΩ) + 2B + (1/λ)(e^{−Ψ} − I).
//!
//! The Jacobi eigensolver is a **device `.func`** (`sym_eig3`) that takes the 6
//! upper entries of a symmetric 3×3 as scalars and RETURNS A 12-TUPLE
//! `(λ0,λ1,λ2, V00..V22)` — cuda-oxide mishandles nested-array writes, so the
//! whole port uses FLAT scalar locals (a00..a22, v00..v22). `theta.signum()` is
//! replaced by a manual sign branch (signum is unmapped in the device-intrinsic
//! table); `.abs()`, `.sqrt()`, `.exp()` are all mapped and used as-is.
//!
//! To keep the kernel signature narrow (the NVVM-text backend mis-parses very
//! wide signatures) the metrics are packed node-major (`met[b*9 + {rx,ry,rz,sx,
//! sy,sz,tx,ty,tz}]`, metric *columns* per the chain rule, exactly as in
//! [`crate::operators::oldroyd3d`]) and the 6 Ψ components are packed node-major
//! (`psi[b*6 + o]`); velocity stays 3 separate slices and the 6 outputs are 6
//! separate [`DisjointSlice`]s. Validated bit-for-bit against the CPU oracle
//! `gale::dg::LogConfOldroydB3d::psi_rhs` (~1e-9, libdevice `exp`).

use cuda_core::{CudaContext, DeviceBuffer, LaunchConfig};
use cuda_device::{kernel, thread, DisjointSlice, SharedArray};
use cuda_host::cuda_module;
use gale::dg::{LogConfOldroydB3d, Mesh3d};

const NN_MAX: usize = 125; // (p+1)³ up to p=4

#[cuda_module]
mod kernels {
    use super::*;

    /// Symmetric-3×3 eigendecomposition by cyclic Jacobi (12 sweeps). Input is the
    /// 6 upper entries `[xx, xy, xz, yy, yz, zz]`; returns the 3 eigenvalues plus
    /// the eigenvector matrix `V` row-major as `(λ0,λ1,λ2, v00,v01,v02, v10,v11,
    /// v12, v20,v21,v22)` where `V[i][k]` is the i-th component of the k-th
    /// eigenvector, so `A = V diag(λ) Vᵀ`. Exact flat-scalar port of the CPU
    /// `gale::dg::sym_eig3` — no nested arrays (cuda-oxide mishandles those), and
    /// `theta.signum()` replaced by an explicit sign branch (unmapped intrinsic).
    fn sym_eig3(
        mxx: f64, mxy: f64, mxz: f64, myy: f64, myz: f64, mzz: f64,
    ) -> (f64, f64, f64, f64, f64, f64, f64, f64, f64, f64, f64, f64) {
        // A (symmetric, full 3×3 flat scalars).
        let mut a00 = mxx;
        let mut a01 = mxy;
        let mut a02 = mxz;
        let mut a10 = mxy;
        let mut a11 = myy;
        let mut a12 = myz;
        let mut a20 = mxz;
        let mut a21 = myz;
        let mut a22 = mzz;
        // V = I.
        let mut v00 = 1.0f64;
        let mut v01 = 0.0f64;
        let mut v02 = 0.0f64;
        let mut v10 = 0.0f64;
        let mut v11 = 1.0f64;
        let mut v12 = 0.0f64;
        let mut v20 = 0.0f64;
        let mut v21 = 0.0f64;
        let mut v22 = 1.0f64;

        let mut sweep = 0usize;
        while sweep < 12 {
            // off = max(|a01|,|a02|,|a12|).
            let mut off = a01.abs();
            let o2 = a02.abs();
            if o2 > off {
                off = o2;
            }
            let o3 = a12.abs();
            if o3 > off {
                off = o3;
            }
            if off < 1e-300 {
                break;
            }

            // --- pair (p,q) = (0,1) ---
            let apq = a01;
            if apq.abs() >= 1e-300 {
                let theta = (a11 - a00) / (2.0 * apq);
                let sgn = if theta >= 0.0 { 1.0f64 } else { -1.0f64 };
                let t = sgn / (theta.abs() + (theta * theta + 1.0).sqrt());
                let c = 1.0 / (t * t + 1.0).sqrt();
                let s = t * c;
                // A ← Jᵀ A J: rotate columns p,q then rows p,q. (p=0, q=1)
                // columns: a[i][0], a[i][1] for i=0,1,2
                let a00n = c * a00 - s * a01;
                let a01n = s * a00 + c * a01;
                let a10n = c * a10 - s * a11;
                let a11n = s * a10 + c * a11;
                let a20n = c * a20 - s * a21;
                let a21n = s * a20 + c * a21;
                a00 = a00n;
                a01 = a01n;
                a10 = a10n;
                a11 = a11n;
                a20 = a20n;
                a21 = a21n;
                // rows: a[0][i], a[1][i] for i=0,1,2
                let r00 = c * a00 - s * a10;
                let r10 = s * a00 + c * a10;
                let r01 = c * a01 - s * a11;
                let r11 = s * a01 + c * a11;
                let r02 = c * a02 - s * a12;
                let r12 = s * a02 + c * a12;
                a00 = r00;
                a10 = r10;
                a01 = r01;
                a11 = r11;
                a02 = r02;
                a12 = r12;
                // V ← V J: rotate columns p,q of V.
                let nv00 = c * v00 - s * v01;
                let nv01 = s * v00 + c * v01;
                let nv10 = c * v10 - s * v11;
                let nv11 = s * v10 + c * v11;
                let nv20 = c * v20 - s * v21;
                let nv21 = s * v20 + c * v21;
                v00 = nv00;
                v01 = nv01;
                v10 = nv10;
                v11 = nv11;
                v20 = nv20;
                v21 = nv21;
            }

            // --- pair (p,q) = (0,2) ---
            let apq = a02;
            if apq.abs() >= 1e-300 {
                let theta = (a22 - a00) / (2.0 * apq);
                let sgn = if theta >= 0.0 { 1.0f64 } else { -1.0f64 };
                let t = sgn / (theta.abs() + (theta * theta + 1.0).sqrt());
                let c = 1.0 / (t * t + 1.0).sqrt();
                let s = t * c;
                // columns 0,2
                let a00n = c * a00 - s * a02;
                let a02n = s * a00 + c * a02;
                let a10n = c * a10 - s * a12;
                let a12n = s * a10 + c * a12;
                let a20n = c * a20 - s * a22;
                let a22n = s * a20 + c * a22;
                a00 = a00n;
                a02 = a02n;
                a10 = a10n;
                a12 = a12n;
                a20 = a20n;
                a22 = a22n;
                // rows 0,2
                let r00 = c * a00 - s * a20;
                let r20 = s * a00 + c * a20;
                let r01 = c * a01 - s * a21;
                let r21 = s * a01 + c * a21;
                let r02 = c * a02 - s * a22;
                let r22 = s * a02 + c * a22;
                a00 = r00;
                a20 = r20;
                a01 = r01;
                a21 = r21;
                a02 = r02;
                a22 = r22;
                // V columns 0,2
                let nv00 = c * v00 - s * v02;
                let nv02 = s * v00 + c * v02;
                let nv10 = c * v10 - s * v12;
                let nv12 = s * v10 + c * v12;
                let nv20 = c * v20 - s * v22;
                let nv22 = s * v20 + c * v22;
                v00 = nv00;
                v02 = nv02;
                v10 = nv10;
                v12 = nv12;
                v20 = nv20;
                v22 = nv22;
            }

            // --- pair (p,q) = (1,2) ---
            let apq = a12;
            if apq.abs() >= 1e-300 {
                let theta = (a22 - a11) / (2.0 * apq);
                let sgn = if theta >= 0.0 { 1.0f64 } else { -1.0f64 };
                let t = sgn / (theta.abs() + (theta * theta + 1.0).sqrt());
                let c = 1.0 / (t * t + 1.0).sqrt();
                let s = t * c;
                // columns 1,2
                let a01n = c * a01 - s * a02;
                let a02n = s * a01 + c * a02;
                let a11n = c * a11 - s * a12;
                let a12n = s * a11 + c * a12;
                let a21n = c * a21 - s * a22;
                let a22n = s * a21 + c * a22;
                a01 = a01n;
                a02 = a02n;
                a11 = a11n;
                a12 = a12n;
                a21 = a21n;
                a22 = a22n;
                // rows 1,2
                let r10 = c * a10 - s * a20;
                let r20 = s * a10 + c * a20;
                let r11 = c * a11 - s * a21;
                let r21 = s * a11 + c * a21;
                let r12 = c * a12 - s * a22;
                let r22 = s * a12 + c * a22;
                a10 = r10;
                a20 = r20;
                a11 = r11;
                a21 = r21;
                a12 = r12;
                a22 = r22;
                // V columns 1,2
                let nv01 = c * v01 - s * v02;
                let nv02 = s * v01 + c * v02;
                let nv11 = c * v11 - s * v12;
                let nv12 = s * v11 + c * v12;
                let nv21 = c * v21 - s * v22;
                let nv22 = s * v21 + c * v22;
                v01 = nv01;
                v02 = nv02;
                v11 = nv11;
                v12 = nv12;
                v21 = nv21;
                v22 = nv22;
            }

            sweep += 1;
        }

        (a00, a11, a22, v00, v01, v02, v10, v11, v12, v20, v21, v22)
    }

    /// `∂ₜΨ` for one element per block, one node per thread. Loads the diff matrix,
    /// the 3 velocity components, and the 6 Ψ components into shared memory; forms
    /// r/s/t reference derivatives by sum factorization; the physical velocity
    /// gradient `L` (chain rule, metric columns) and `∇Ψ`; then the per-node
    /// Fattal–Kupferman algebra (eigenframe rotation Ω, stretch B, and the
    /// matrix-exp relaxation `e^{−Ψ}`).
    ///
    /// Node index `m = i + j·n1 + k·n1²`. Reference derivatives:
    /// `ur = Σ_a DS[i·n1+a]·F[a + j·n1 + k·n2]`,
    /// `us = Σ_a DS[j·n1+a]·F[i + a·n1 + k·n2]`,
    /// `ut = Σ_a DS[k·n1+a]·F[i + j·n1 + a·n2]`.
    #[kernel]
    #[allow(clippy::too_many_arguments)]
    pub fn logconf3d_psi_rhs(
        d: &[f64], ux: &[f64], uy: &[f64], uz: &[f64], psi: &[f64], met: &[f64],
        inv_lambda: f64, n1: u32,
        mut dpxx: DisjointSlice<f64>, mut dpxy: DisjointSlice<f64>, mut dpxz: DisjointSlice<f64>,
        mut dpyy: DisjointSlice<f64>, mut dpyz: DisjointSlice<f64>, mut dpzz: DisjointSlice<f64>,
    ) {
        static mut DS: SharedArray<f64, NN_MAX> = SharedArray::UNINIT;
        static mut US: SharedArray<f64, NN_MAX> = SharedArray::UNINIT;
        static mut VS: SharedArray<f64, NN_MAX> = SharedArray::UNINIT;
        static mut WS: SharedArray<f64, NN_MAX> = SharedArray::UNINIT;
        static mut PXX: SharedArray<f64, NN_MAX> = SharedArray::UNINIT;
        static mut PXY: SharedArray<f64, NN_MAX> = SharedArray::UNINIT;
        static mut PXZ: SharedArray<f64, NN_MAX> = SharedArray::UNINIT;
        static mut PYY: SharedArray<f64, NN_MAX> = SharedArray::UNINIT;
        static mut PYZ: SharedArray<f64, NN_MAX> = SharedArray::UNINIT;
        static mut PZZ: SharedArray<f64, NN_MAX> = SharedArray::UNINIT;
        let n1 = n1 as usize;
        let n2 = n1 * n1;
        let nn = n1 * n2;
        let e = thread::blockIdx_x() as usize;
        let m = thread::threadIdx_x() as usize;
        let b = e * nn + m;
        let po = b * 6;
        unsafe {
            DS[m] = d[m];
            US[m] = ux[b];
            VS[m] = uy[b];
            WS[m] = uz[b];
            PXX[m] = psi[po];
            PXY[m] = psi[po + 1];
            PXZ[m] = psi[po + 2];
            PYY[m] = psi[po + 3];
            PYZ[m] = psi[po + 4];
            PZZ[m] = psi[po + 5];
        }
        thread::sync_threads();

        let i = m % n1;
        let j = (m / n1) % n1;
        let k = m / n2;
        // r/s/t derivatives of all nine fields (sum factorization).
        let (mut ur_u, mut us_u, mut ut_u) = (0.0f64, 0.0f64, 0.0f64);
        let (mut ur_v, mut us_v, mut ut_v) = (0.0f64, 0.0f64, 0.0f64);
        let (mut ur_w, mut us_w, mut ut_w) = (0.0f64, 0.0f64, 0.0f64);
        let (mut ur_xx, mut us_xx, mut ut_xx) = (0.0f64, 0.0f64, 0.0f64);
        let (mut ur_xy, mut us_xy, mut ut_xy) = (0.0f64, 0.0f64, 0.0f64);
        let (mut ur_xz, mut us_xz, mut ut_xz) = (0.0f64, 0.0f64, 0.0f64);
        let (mut ur_yy, mut us_yy, mut ut_yy) = (0.0f64, 0.0f64, 0.0f64);
        let (mut ur_yz, mut us_yz, mut ut_yz) = (0.0f64, 0.0f64, 0.0f64);
        let (mut ur_zz, mut us_zz, mut ut_zz) = (0.0f64, 0.0f64, 0.0f64);
        let mut a = 0usize;
        while a < n1 {
            let dia = unsafe { DS[i * n1 + a] };
            let dja = unsafe { DS[j * n1 + a] };
            let dka = unsafe { DS[k * n1 + a] };
            let cr = a + j * n1 + k * n2; // varies along r
            let cs = i + a * n1 + k * n2; // varies along s
            let ct = i + j * n1 + a * n2; // varies along t
            unsafe {
                ur_u += dia * US[cr]; us_u += dja * US[cs]; ut_u += dka * US[ct];
                ur_v += dia * VS[cr]; us_v += dja * VS[cs]; ut_v += dka * VS[ct];
                ur_w += dia * WS[cr]; us_w += dja * WS[cs]; ut_w += dka * WS[ct];
                ur_xx += dia * PXX[cr]; us_xx += dja * PXX[cs]; ut_xx += dka * PXX[ct];
                ur_xy += dia * PXY[cr]; us_xy += dja * PXY[cs]; ut_xy += dka * PXY[ct];
                ur_xz += dia * PXZ[cr]; us_xz += dja * PXZ[cs]; ut_xz += dka * PXZ[ct];
                ur_yy += dia * PYY[cr]; us_yy += dja * PYY[cs]; ut_yy += dka * PYY[ct];
                ur_yz += dia * PYZ[cr]; us_yz += dja * PYZ[cs]; ut_yz += dka * PYZ[ct];
                ur_zz += dia * PZZ[cr]; us_zz += dja * PZZ[cs]; ut_zz += dka * PZZ[ct];
            }
            a += 1;
        }

        // Metric columns: gx = rx·ur + sx·us + tx·ut, etc. (same layout as oldroyd3d).
        let mo = b * 9;
        let mrx = met[mo];
        let mry = met[mo + 1];
        let mrz = met[mo + 2];
        let msx = met[mo + 3];
        let msy = met[mo + 4];
        let msz = met[mo + 5];
        let mtx = met[mo + 6];
        let mty = met[mo + 7];
        let mtz = met[mo + 8];

        // Velocity gradient L (Lᵢⱼ = ∂uᵢ/∂xⱼ).
        let l00 = mrx * ur_u + msx * us_u + mtx * ut_u; // ∂ux/∂x
        let l01 = mry * ur_u + msy * us_u + mty * ut_u; // ∂ux/∂y
        let l02 = mrz * ur_u + msz * us_u + mtz * ut_u; // ∂ux/∂z
        let l10 = mrx * ur_v + msx * us_v + mtx * ut_v; // ∂uy/∂x
        let l11 = mry * ur_v + msy * us_v + mty * ut_v; // ∂uy/∂y
        let l12 = mrz * ur_v + msz * us_v + mtz * ut_v; // ∂uy/∂z
        let l20 = mrx * ur_w + msx * us_w + mtx * ut_w; // ∂uz/∂x
        let l21 = mry * ur_w + msy * us_w + mty * ut_w; // ∂uz/∂y
        let l22 = mrz * ur_w + msz * us_w + mtz * ut_w; // ∂uz/∂z

        // ∇Ψ for each of the 6 components (x,y,z partials) — advection only.
        let pxx_x = mrx * ur_xx + msx * us_xx + mtx * ut_xx;
        let pxx_y = mry * ur_xx + msy * us_xx + mty * ut_xx;
        let pxx_z = mrz * ur_xx + msz * us_xx + mtz * ut_xx;
        let pxy_x = mrx * ur_xy + msx * us_xy + mtx * ut_xy;
        let pxy_y = mry * ur_xy + msy * us_xy + mty * ut_xy;
        let pxy_z = mrz * ur_xy + msz * us_xy + mtz * ut_xy;
        let pxz_x = mrx * ur_xz + msx * us_xz + mtx * ut_xz;
        let pxz_y = mry * ur_xz + msy * us_xz + mty * ut_xz;
        let pxz_z = mrz * ur_xz + msz * us_xz + mtz * ut_xz;
        let pyy_x = mrx * ur_yy + msx * us_yy + mtx * ut_yy;
        let pyy_y = mry * ur_yy + msy * us_yy + mty * ut_yy;
        let pyy_z = mrz * ur_yy + msz * us_yy + mtz * ut_yy;
        let pyz_x = mrx * ur_yz + msx * us_yz + mtx * ut_yz;
        let pyz_y = mry * ur_yz + msy * us_yz + mty * ut_yz;
        let pyz_z = mrz * ur_yz + msz * us_yz + mtz * ut_yz;
        let pzz_x = mrx * ur_zz + msx * us_zz + mtx * ut_zz;
        let pzz_y = mry * ur_zz + msy * us_zz + mty * ut_zz;
        let pzz_z = mrz * ur_zz + msz * us_zz + mtz * ut_zz;

        // Local Ψ (6 scalars).
        let psixx = unsafe { PXX[m] };
        let psixy = unsafe { PXY[m] };
        let psixz = unsafe { PXZ[m] };
        let psiyy = unsafe { PYY[m] };
        let psiyz = unsafe { PYZ[m] };
        let psizz = unsafe { PZZ[m] };

        // 1. Eigendecomposition of Ψ.
        let (mu0, mu1, mu2, ev00, ev01, ev02, ev10, ev11, ev12, ev20, ev21, ev22) =
            sym_eig3(psixx, psixy, psixz, psiyy, psiyz, psizz);
        let mut v00 = ev00;
        let mut v01 = ev01;
        let mut v02 = ev02;
        let mut v10 = ev10;
        let mut v11 = ev11;
        let mut v12 = ev12;
        let mut v20 = ev20;
        let mut v21 = ev21;
        let mut v22 = ev22;

        // 2. Rate of strain D = ½(L + Lᵀ) (6 upper scalars).
        let d00 = l00;
        let d01 = 0.5 * (l01 + l10);
        let d02 = 0.5 * (l02 + l20);
        let d11 = l11;
        let d12 = 0.5 * (l12 + l21);
        let d22 = l22;

        // 3. Near-isotropic: recompute V from D's eigenframe (eigenvalues discarded).
        let s01 = (mu0 - mu1).abs();
        let s02 = (mu0 - mu2).abs();
        let s12 = (mu1 - mu2).abs();
        let mut spread = s01;
        if s02 > spread {
            spread = s02;
        }
        if s12 > spread {
            spread = s12;
        }
        if spread < 1e-7 {
            let (_, _, _, dv00, dv01, dv02, dv10, dv11, dv12, dv20, dv21, dv22) =
                sym_eig3(d00, d01, d02, d11, d12, d22);
            v00 = dv00;
            v01 = dv01;
            v02 = dv02;
            v10 = dv10;
            v11 = dv11;
            v12 = dv12;
            v20 = dv20;
            v21 = dv21;
            v22 = dv22;
        }

        // 4. lam_k = exp(mu_k).
        let lam0 = mu0.exp();
        let lam1 = mu1.exp();
        let lam2 = mu2.exp();

        // 5. M = Vᵀ L V (full 3×3). First T = L V, then M = Vᵀ T.
        // T[i][k] = Σ_p L[i][p] V[p][k].
        let t00 = l00 * v00 + l01 * v10 + l02 * v20;
        let t01 = l00 * v01 + l01 * v11 + l02 * v21;
        let t02 = l00 * v02 + l01 * v12 + l02 * v22;
        let t10 = l10 * v00 + l11 * v10 + l12 * v20;
        let t11 = l10 * v01 + l11 * v11 + l12 * v21;
        let t12 = l10 * v02 + l11 * v12 + l12 * v22;
        let t20 = l20 * v00 + l21 * v10 + l22 * v20;
        let t21 = l20 * v01 + l21 * v11 + l22 * v21;
        let t22 = l20 * v02 + l21 * v12 + l22 * v22;
        // M[a][k] = Σ_i V[i][a] T[i][k]  (Vᵀ has rows = V columns).
        let mm00 = v00 * t00 + v10 * t10 + v20 * t20;
        let mm01 = v00 * t01 + v10 * t11 + v20 * t21;
        let mm02 = v00 * t02 + v10 * t12 + v20 * t22;
        let mm10 = v01 * t00 + v11 * t10 + v21 * t20;
        let mm11 = v01 * t01 + v11 * t11 + v21 * t21;
        let mm12 = v01 * t02 + v11 * t12 + v21 * t22;
        let mm20 = v02 * t00 + v12 * t10 + v22 * t20;
        let mm21 = v02 * t01 + v12 * t11 + v22 * t21;
        let mm22 = v02 * t02 + v12 * t12 + v22 * t22;

        // 6. B = V diag(M00,M11,M22) Vᵀ (6 upper scalars).
        // B[i][j] = Σ_k V[i][k] M_kk V[j][k].
        let bxx = v00 * mm00 * v00 + v01 * mm11 * v01 + v02 * mm22 * v02;
        let bxy = v00 * mm00 * v10 + v01 * mm11 * v11 + v02 * mm22 * v12;
        let bxz = v00 * mm00 * v20 + v01 * mm11 * v21 + v02 * mm22 * v22;
        let byy = v10 * mm00 * v10 + v11 * mm11 * v11 + v12 * mm22 * v12;
        let byz = v10 * mm00 * v20 + v11 * mm11 * v21 + v12 * mm22 * v22;
        let bzz = v20 * mm00 * v20 + v21 * mm11 * v21 + v22 * mm22 * v22;

        // 7. Ω_eig antisymmetric: ω_ij = (M_ij λ_j + M_ji λ_i)/(λ_j − λ_i).
        let den01 = lam1 - lam0;
        let w01 = if den01.abs() > 1e-12 {
            (mm01 * lam1 + mm10 * lam0) / den01
        } else {
            0.0
        };
        let den02 = lam2 - lam0;
        let w02 = if den02.abs() > 1e-12 {
            (mm02 * lam2 + mm20 * lam0) / den02
        } else {
            0.0
        };
        let den12 = lam2 - lam1;
        let w12 = if den12.abs() > 1e-12 {
            (mm12 * lam2 + mm21 * lam1) / den12
        } else {
            0.0
        };
        // om (antisymmetric, diag 0): om01=w01, om10=-w01, om02=w02, om20=-w02,
        // om12=w12, om21=-w12.
        let om00 = 0.0f64;
        let om01 = w01;
        let om02 = w02;
        let om10 = -w01;
        let om11 = 0.0f64;
        let om12 = w12;
        let om20 = -w02;
        let om21 = -w12;
        let om22 = 0.0f64;
        // Ω = V om Vᵀ. First S = om Vᵀ: S[a][j] = Σ_k om[a][k] V[j][k].
        let s00 = om00 * v00 + om01 * v01 + om02 * v02;
        let s01b = om00 * v10 + om01 * v11 + om02 * v12;
        let s02b = om00 * v20 + om01 * v21 + om02 * v22;
        let s10 = om10 * v00 + om11 * v01 + om12 * v02;
        let s11 = om10 * v10 + om11 * v11 + om12 * v12;
        let s12b = om10 * v20 + om11 * v21 + om12 * v22;
        let s20 = om20 * v00 + om21 * v01 + om22 * v02;
        let s21 = om20 * v10 + om21 * v11 + om22 * v12;
        let s22 = om20 * v20 + om21 * v21 + om22 * v22;
        // Ω[i][j] = Σ_a V[i][a] S[a][j].
        let omg00 = v00 * s00 + v01 * s10 + v02 * s20;
        let omg01 = v00 * s01b + v01 * s11 + v02 * s21;
        let omg02 = v00 * s02b + v01 * s12b + v02 * s22;
        let omg10 = v10 * s00 + v11 * s10 + v12 * s20;
        let omg11 = v10 * s01b + v11 * s11 + v12 * s21;
        let omg12 = v10 * s02b + v11 * s12b + v12 * s22;
        let omg20 = v20 * s00 + v21 * s10 + v22 * s20;
        let omg21 = v20 * s01b + v21 * s11 + v22 * s21;
        let omg22 = v20 * s02b + v21 * s12b + v22 * s22;

        // 8. ΩΨ − ΨΩ (full 3×3, Ψ expanded). Symmetric Ψ ⇒ psi10=psi01, etc.
        // OP = Ω Ψ: OP[i][j] = Σ_p Ω[i][p] Ψ[p][j].
        let op00 = omg00 * psixx + omg01 * psixy + omg02 * psixz;
        let op01 = omg00 * psixy + omg01 * psiyy + omg02 * psiyz;
        let op02 = omg00 * psixz + omg01 * psiyz + omg02 * psizz;
        let op11 = omg10 * psixy + omg11 * psiyy + omg12 * psiyz;
        let op12 = omg10 * psixz + omg11 * psiyz + omg12 * psizz;
        let op22 = omg20 * psixz + omg21 * psiyz + omg22 * psizz;
        // PO = Ψ Ω: PO[i][j] = Σ_p Ψ[i][p] Ω[p][j].
        let po00 = psixx * omg00 + psixy * omg10 + psixz * omg20;
        let po01 = psixx * omg01 + psixy * omg11 + psixz * omg21;
        let po02 = psixx * omg02 + psixy * omg12 + psixz * omg22;
        let po11 = psixy * omg01 + psiyy * omg11 + psiyz * omg21;
        let po12 = psixy * omg02 + psiyy * omg12 + psiyz * omg22;
        let po22 = psixz * omg02 + psiyz * omg12 + psizz * omg22;
        let rot_xx = op00 - po00;
        let rot_xy = op01 - po01;
        let rot_xz = op02 - po02;
        let rot_yy = op11 - po11;
        let rot_yz = op12 - po12;
        let rot_zz = op22 - po22;

        // 9. em = V diag(exp(-mu_k)) Vᵀ (6 upper scalars).
        let e0 = (-mu0).exp();
        let e1 = (-mu1).exp();
        let e2 = (-mu2).exp();
        let em_xx = v00 * e0 * v00 + v01 * e1 * v01 + v02 * e2 * v02;
        let em_xy = v00 * e0 * v10 + v01 * e1 * v11 + v02 * e2 * v12;
        let em_xz = v00 * e0 * v20 + v01 * e1 * v21 + v02 * e2 * v22;
        let em_yy = v10 * e0 * v10 + v11 * e1 * v11 + v12 * e2 * v12;
        let em_yz = v10 * e0 * v20 + v11 * e1 * v21 + v12 * e2 * v22;
        let em_zz = v20 * e0 * v20 + v21 * e1 * v21 + v22 * e2 * v22;

        // 10. Assemble outputs. eye=[1,0,0,1,0,1].
        let u = unsafe { US[m] };
        let v = unsafe { VS[m] };
        let w = unsafe { WS[m] };
        let adv_xx = u * pxx_x + v * pxx_y + w * pxx_z;
        let adv_xy = u * pxy_x + v * pxy_y + w * pxy_z;
        let adv_xz = u * pxz_x + v * pxz_y + w * pxz_z;
        let adv_yy = u * pyy_x + v * pyy_y + w * pyy_z;
        let adv_yz = u * pyz_x + v * pyz_y + w * pyz_z;
        let adv_zz = u * pzz_x + v * pzz_y + w * pzz_z;

        let out_xx = -adv_xx + rot_xx + 2.0 * bxx + inv_lambda * (em_xx - 1.0);
        let out_xy = -adv_xy + rot_xy + 2.0 * bxy + inv_lambda * em_xy;
        let out_xz = -adv_xz + rot_xz + 2.0 * bxz + inv_lambda * em_xz;
        let out_yy = -adv_yy + rot_yy + 2.0 * byy + inv_lambda * (em_yy - 1.0);
        let out_yz = -adv_yz + rot_yz + 2.0 * byz + inv_lambda * em_yz;
        let out_zz = -adv_zz + rot_zz + 2.0 * bzz + inv_lambda * (em_zz - 1.0);

        if let Some(o) = dpxx.get_mut(thread::index_1d()) { *o = out_xx; }
        if let Some(o) = dpxy.get_mut(thread::index_1d()) { *o = out_xy; }
        if let Some(o) = dpxz.get_mut(thread::index_1d()) { *o = out_xz; }
        if let Some(o) = dpyy.get_mut(thread::index_1d()) { *o = out_yy; }
        if let Some(o) = dpyz.get_mut(thread::index_1d()) { *o = out_yz; }
        if let Some(o) = dpzz.get_mut(thread::index_1d()) { *o = out_zz; }
    }
}

/// Compute the 3D log-conformation (Fattal–Kupferman) Ψ time-derivative `∂ₜΨ` on
/// the GPU for an Oldroyd-B fluid, returning the six independent components
/// `[∂ₜΨxx, ∂ₜΨxy, ∂ₜΨxz, ∂ₜΨyy, ∂ₜΨyz, ∂ₜΨzz]` in the same `[Vec<f64>; 6]` layout
/// as the CPU oracle. Reusable host wrapper around the [`kernels::logconf3d_psi_rhs`]
/// device kernel: packs the 9 mesh metrics node-major (`met[b*9 + {rx,ry,rz,sx,sy,
/// sz,tx,ty,tz}]`), pads the `n1²` diff matrix to `n³`, packs the 6 Ψ components
/// node-major (`psi[b*6 + o]`), uploads the state, launches one block per element
/// (one thread per node), and gathers the result. Bit-for-bit equal to
/// `gale::dg::LogConfOldroydB3d::psi_rhs`.
///
/// `psi` holds the log-conformation field as `[Ψxx, Ψxy, Ψxz, Ψyy, Ψyz, Ψzz]`,
/// each of length `n_elements·n_nodes`; `ux`/`uy`/`uz` are the velocity components
/// in the same layout. `lc` carries the relaxation time λ; only its reciprocal
/// `1/λ` is needed by the kernel.
pub fn logconf3d_psi_rhs(
    mesh: &Mesh3d,
    lc: &LogConfOldroydB3d,
    psi: &[Vec<f64>; 6],
    ux: &[f64],
    uy: &[f64],
    uz: &[f64],
) -> Result<[Vec<f64>; 6], Box<dyn std::error::Error>> {
    let nn = mesh.refh.n_nodes();
    let ne = mesh.n_elements();
    let ndof = ne * nn;
    let n1 = mesh.order + 1;
    assert_eq!(ux.len(), ndof, "ux length must be n_elements·n_nodes");
    assert_eq!(uy.len(), ndof, "uy length must be n_elements·n_nodes");
    assert_eq!(uz.len(), ndof, "uz length must be n_elements·n_nodes");
    for (o, pi) in psi.iter().enumerate() {
        assert_eq!(pi.len(), ndof, "Ψ component {o} length must be n_elements·n_nodes");
    }
    assert!(nn <= NN_MAX, "p too large for NN_MAX={NN_MAX}");

    // Pack the 9 metric terms node-major (met[b*9 + idx]) and the 6 Ψ components
    // node-major (psi_pack[b*6 + o]).
    let mut met = vec![0.0; ndof * 9];
    let mut psi_pack = vec![0.0; ndof * 6];
    for (e, el) in mesh.elements.iter().enumerate() {
        let g = &el.geom;
        for k in 0..nn {
            let b = e * nn + k;
            met[b * 9] = g.rx[k];
            met[b * 9 + 1] = g.ry[k];
            met[b * 9 + 2] = g.rz[k];
            met[b * 9 + 3] = g.sx[k];
            met[b * 9 + 4] = g.sy[k];
            met[b * 9 + 5] = g.sz[k];
            met[b * 9 + 6] = g.tx[k];
            met[b * 9 + 7] = g.ty[k];
            met[b * 9 + 8] = g.tz[k];
            for o in 0..6 {
                psi_pack[b * 6 + o] = psi[o][b];
            }
        }
    }

    let ctx = CudaContext::new(0)?;
    let stream = ctx.default_stream();
    let up = |v: &[f64]| DeviceBuffer::from_host(&stream, v);
    // Pad the n1² diff matrix to n³ so the kernel loads DS[m] unconditionally.
    let mut d_pad = vec![0.0; nn];
    d_pad[..mesh.refh.line.diff.len()].copy_from_slice(&mesh.refh.line.diff);
    let d_dev = up(&d_pad)?;
    let ux_dev = up(ux)?;
    let uy_dev = up(uy)?;
    let uz_dev = up(uz)?;
    let psi_dev = up(&psi_pack)?;
    let met_dev = up(&met)?;
    let mut dxx = DeviceBuffer::<f64>::zeroed(&stream, ndof)?;
    let mut dxy = DeviceBuffer::<f64>::zeroed(&stream, ndof)?;
    let mut dxz = DeviceBuffer::<f64>::zeroed(&stream, ndof)?;
    let mut dyy = DeviceBuffer::<f64>::zeroed(&stream, ndof)?;
    let mut dyz = DeviceBuffer::<f64>::zeroed(&stream, ndof)?;
    let mut dzz = DeviceBuffer::<f64>::zeroed(&stream, ndof)?;

    let module = kernels::load(&ctx)?;
    let cfg = LaunchConfig { grid_dim: (ne as u32, 1, 1), block_dim: (nn as u32, 1, 1), shared_mem_bytes: 0 };
    module.logconf3d_psi_rhs(
        &stream, cfg, &d_dev, &ux_dev, &uy_dev, &uz_dev, &psi_dev, &met_dev,
        lc.lambda.recip(), n1 as u32,
        &mut dxx, &mut dxy, &mut dxz, &mut dyy, &mut dyz, &mut dzz,
    )?;
    Ok([
        dxx.to_host_vec(&stream)?,
        dxy.to_host_vec(&stream)?,
        dxz.to_host_vec(&stream)?,
        dyy.to_host_vec(&stream)?,
        dyz.to_host_vec(&stream)?,
        dzz.to_host_vec(&stream)?,
    ])
}
