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

use cuda_core::{CudaContext, DeviceBuffer, LaunchConfig};
use cuda_device::{kernel, thread, DisjointSlice, SharedArray};
use cuda_host::cuda_module;
use gale::dg::{LogConfOldroydB, Mesh2d};

const NN_MAX: usize = 81;

#[cuda_module]
mod kernels {
    use super::*;

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

    let ctx = CudaContext::new(0)?;
    let stream = ctx.default_stream();
    let up = |v: &[f64]| DeviceBuffer::from_host(&stream, v);
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
    let mut dxx = DeviceBuffer::<f64>::zeroed(&stream, ndof)?;
    let mut dxy = DeviceBuffer::<f64>::zeroed(&stream, ndof)?;
    let mut dyy = DeviceBuffer::<f64>::zeroed(&stream, ndof)?;

    let module = kernels::load(&ctx)?;
    let cfg = LaunchConfig { grid_dim: (ne as u32, 1, 1), block_dim: (nn as u32, 1, 1), shared_mem_bytes: 0 };
    module.psi_rhs(
        &stream, cfg, &d_dev, &ux_d, &uy_d, &pxx_d, &pxy_d, &pyy_d,
        &rx_d, &ry_d, &sx_d, &sy_d, lc.lambda.recip(), n1 as u32,
        &mut dxx, &mut dxy, &mut dyy,
    )?;
    Ok([dxx.to_host_vec(&stream)?, dxy.to_host_vec(&stream)?, dyy.to_host_vec(&stream)?])
}
