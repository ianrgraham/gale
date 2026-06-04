//! GPU Oldroyd-B conformation transport (direct form) on the **3D hex mesh** — the
//! 3D analogue of [`crate::operators::oldroyd`], and the GPU counterpart of
//! `gale::dg::OldroydB3d::conformation_rhs`.
//!
//! The conformation tensor is now symmetric **3×3 = 6 components**
//! `C = [Cxx, Cxy, Cxz, Cyy, Cyz, Czz]` and `L = ∇u` is a full 3×3. The kernel is
//! purely element-local (nodal-collocation advection — no faces, no neighbor gather)
//! and libdevice-free (only arithmetic + the sum-factorized gradient pattern already
//! proven on GPU by [`crate::operators::poisson3d`] / [`crate::operators::advection3d`]),
//! so it runs on the embedded `#[cuda_module]` path on sm_70. Validated bit-for-bit
//! against the CPU oracle `gale::dg::OldroydB3d::conformation_rhs`.
//!
//!   ∂ₜC = −(u·∇)C + (L·C + C·Lᵀ) − (1/λ)(C − I),   L = ∇u.
//!
//! To keep the kernel signature narrow (the NVVM-text backend mis-parses very wide
//! signatures) the metrics are packed node-major (`met[b*9 + {rx,ry,rz,sx,sy,sz,tx,
//! ty,tz}]`, metric *columns* per the chain rule) and the 6 conformation components
//! are packed node-major (`c[b*6 + o]`); velocity stays 3 separate slices and the 6
//! outputs are 6 separate [`DisjointSlice`]s.

use cuda_core::{CudaContext, DeviceBuffer, LaunchConfig};
use cuda_device::{kernel, thread, DisjointSlice, SharedArray};
use cuda_host::cuda_module;
use gale::dg::Mesh3d;

const NN_MAX: usize = 125; // (p+1)³ up to p=4

#[cuda_module]
mod kernels {
    use super::*;

    /// `∂ₜC` for one element per block, one node per thread. Loads the diff matrix,
    /// the 3 velocity components, and the 6 conformation components into shared
    /// memory; computes r/s/t reference derivatives of all 9 fields by sum
    /// factorization, then the physical velocity gradient `L` (chain rule, metric
    /// columns) and `∇C`; finally the pointwise advection + upper-convected
    /// stretching `L·C + C·Lᵀ` + relaxation.
    ///
    /// Node index `m = i + j·n1 + k·n1²`. Reference derivatives:
    /// `ur = Σ_a DS[i·n1+a]·F[a + j·n1 + k·n2]`,
    /// `us = Σ_a DS[j·n1+a]·F[i + a·n1 + k·n2]`,
    /// `ut = Σ_a DS[k·n1+a]·F[i + j·n1 + a·n2]`.
    #[kernel]
    #[allow(clippy::too_many_arguments)]
    pub fn oldroyd3d_conf_rhs(
        d: &[f64], ux: &[f64], uy: &[f64], uz: &[f64], c: &[f64], met: &[f64],
        inv_lambda: f64, n1: u32,
        mut dcxx: DisjointSlice<f64>, mut dcxy: DisjointSlice<f64>, mut dcxz: DisjointSlice<f64>,
        mut dcyy: DisjointSlice<f64>, mut dcyz: DisjointSlice<f64>, mut dczz: DisjointSlice<f64>,
    ) {
        static mut DS: SharedArray<f64, NN_MAX> = SharedArray::UNINIT;
        static mut US: SharedArray<f64, NN_MAX> = SharedArray::UNINIT;
        static mut VS: SharedArray<f64, NN_MAX> = SharedArray::UNINIT;
        static mut WS: SharedArray<f64, NN_MAX> = SharedArray::UNINIT;
        static mut CXX: SharedArray<f64, NN_MAX> = SharedArray::UNINIT;
        static mut CXY: SharedArray<f64, NN_MAX> = SharedArray::UNINIT;
        static mut CXZ: SharedArray<f64, NN_MAX> = SharedArray::UNINIT;
        static mut CYY: SharedArray<f64, NN_MAX> = SharedArray::UNINIT;
        static mut CYZ: SharedArray<f64, NN_MAX> = SharedArray::UNINIT;
        static mut CZZ: SharedArray<f64, NN_MAX> = SharedArray::UNINIT;
        let n1 = n1 as usize;
        let n2 = n1 * n1;
        let nn = n1 * n2;
        let e = thread::blockIdx_x() as usize;
        let m = thread::threadIdx_x() as usize;
        let b = e * nn + m;
        let co = b * 6;
        unsafe {
            DS[m] = d[m];
            US[m] = ux[b];
            VS[m] = uy[b];
            WS[m] = uz[b];
            CXX[m] = c[co];
            CXY[m] = c[co + 1];
            CXZ[m] = c[co + 2];
            CYY[m] = c[co + 3];
            CYZ[m] = c[co + 4];
            CZZ[m] = c[co + 5];
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
                ur_xx += dia * CXX[cr]; us_xx += dja * CXX[cs]; ut_xx += dka * CXX[ct];
                ur_xy += dia * CXY[cr]; us_xy += dja * CXY[cs]; ut_xy += dka * CXY[ct];
                ur_xz += dia * CXZ[cr]; us_xz += dja * CXZ[cs]; ut_xz += dka * CXZ[ct];
                ur_yy += dia * CYY[cr]; us_yy += dja * CYY[cs]; ut_yy += dka * CYY[ct];
                ur_yz += dia * CYZ[cr]; us_yz += dja * CYZ[cs]; ut_yz += dka * CYZ[ct];
                ur_zz += dia * CZZ[cr]; us_zz += dja * CZZ[cs]; ut_zz += dka * CZZ[ct];
            }
            a += 1;
        }

        // Metric columns: gx = rx·ur + sx·us + tx·ut, etc.
        let mo = b * 9;
        let (rx, ry, rz) = (met[mo], met[mo + 1], met[mo + 2]);
        let (sx, sy, sz) = (met[mo + 3], met[mo + 4], met[mo + 5]);
        let (tx, ty, tz) = (met[mo + 6], met[mo + 7], met[mo + 8]);

        // Velocity gradient L (Lᵢⱼ = ∂uᵢ/∂xⱼ). Row i comes from the i-th velocity
        // component, column j from the physical direction.
        let l00 = rx * ur_u + sx * us_u + tx * ut_u; // ∂ux/∂x
        let l01 = ry * ur_u + sy * us_u + ty * ut_u; // ∂ux/∂y
        let l02 = rz * ur_u + sz * us_u + tz * ut_u; // ∂ux/∂z
        let l10 = rx * ur_v + sx * us_v + tx * ut_v; // ∂uy/∂x
        let l11 = ry * ur_v + sy * us_v + ty * ut_v; // ∂uy/∂y
        let l12 = rz * ur_v + sz * us_v + tz * ut_v; // ∂uy/∂z
        let l20 = rx * ur_w + sx * us_w + tx * ut_w; // ∂uz/∂x
        let l21 = ry * ur_w + sy * us_w + ty * ut_w; // ∂uz/∂y
        let l22 = rz * ur_w + sz * us_w + tz * ut_w; // ∂uz/∂z

        // ∇C for each of the 6 components: (x, y, z) partials.
        let cxx_x = rx * ur_xx + sx * us_xx + tx * ut_xx;
        let cxx_y = ry * ur_xx + sy * us_xx + ty * ut_xx;
        let cxx_z = rz * ur_xx + sz * us_xx + tz * ut_xx;
        let cxy_x = rx * ur_xy + sx * us_xy + tx * ut_xy;
        let cxy_y = ry * ur_xy + sy * us_xy + ty * ut_xy;
        let cxy_z = rz * ur_xy + sz * us_xy + tz * ut_xy;
        let cxz_x = rx * ur_xz + sx * us_xz + tx * ut_xz;
        let cxz_y = ry * ur_xz + sy * us_xz + ty * ut_xz;
        let cxz_z = rz * ur_xz + sz * us_xz + tz * ut_xz;
        let cyy_x = rx * ur_yy + sx * us_yy + tx * ut_yy;
        let cyy_y = ry * ur_yy + sy * us_yy + ty * ut_yy;
        let cyy_z = rz * ur_yy + sz * us_yy + tz * ut_yy;
        let cyz_x = rx * ur_yz + sx * us_yz + tx * ut_yz;
        let cyz_y = ry * ur_yz + sy * us_yz + ty * ut_yz;
        let cyz_z = rz * ur_yz + sz * us_yz + tz * ut_yz;
        let czz_x = rx * ur_zz + sx * us_zz + tx * ut_zz;
        let czz_y = ry * ur_zz + sy * us_zz + ty * ut_zz;
        let czz_z = rz * ur_zz + sz * us_zz + tz * ut_zz;

        // Local conformation tensor (expand 6 → symmetric 3×3).
        let cm00 = unsafe { CXX[m] };
        let cm01 = unsafe { CXY[m] };
        let cm02 = unsafe { CXZ[m] };
        let cm11 = unsafe { CYY[m] };
        let cm12 = unsafe { CYZ[m] };
        let cm22 = unsafe { CZZ[m] };
        // Symmetric ⇒ cm10=cm01, cm20=cm02, cm21=cm12.

        // LC = L·C (only need the 6 upper-triangular entries below).
        // LC[i][j] = Σ_p L[i][p]·C[p][j].
        let lc00 = l00 * cm00 + l01 * cm01 + l02 * cm02;
        let lc01 = l00 * cm01 + l01 * cm11 + l02 * cm12;
        let lc02 = l00 * cm02 + l01 * cm12 + l02 * cm22;
        let lc11 = l10 * cm01 + l11 * cm11 + l12 * cm12;
        let lc12 = l10 * cm02 + l11 * cm12 + l12 * cm22;
        let lc22 = l20 * cm02 + l21 * cm12 + l22 * cm22;

        // CLt = C·Lᵀ. CLt[i][j] = Σ_p C[i][p]·L[j][p].
        let clt00 = cm00 * l00 + cm01 * l01 + cm02 * l02;
        let clt01 = cm00 * l10 + cm01 * l11 + cm02 * l12;
        let clt02 = cm00 * l20 + cm01 * l21 + cm02 * l22;
        let clt11 = cm01 * l10 + cm11 * l11 + cm12 * l12;
        let clt12 = cm01 * l20 + cm11 * l21 + cm12 * l22;
        let clt22 = cm02 * l20 + cm12 * l21 + cm22 * l22;

        let u = unsafe { US[m] };
        let v = unsafe { VS[m] };
        let w = unsafe { WS[m] };

        // Per-component: out = -adv + stretch + relax. eye = [1,0,0,1,0,1].
        let adv_xx = u * cxx_x + v * cxx_y + w * cxx_z;
        let adv_xy = u * cxy_x + v * cxy_y + w * cxy_z;
        let adv_xz = u * cxz_x + v * cxz_y + w * cxz_z;
        let adv_yy = u * cyy_x + v * cyy_y + w * cyy_z;
        let adv_yz = u * cyz_x + v * cyz_y + w * cyz_z;
        let adv_zz = u * czz_x + v * czz_y + w * czz_z;

        let out_xx = -adv_xx + (lc00 + clt00) - inv_lambda * (cm00 - 1.0);
        let out_xy = -adv_xy + (lc01 + clt01) - inv_lambda * cm01;
        let out_xz = -adv_xz + (lc02 + clt02) - inv_lambda * cm02;
        let out_yy = -adv_yy + (lc11 + clt11) - inv_lambda * (cm11 - 1.0);
        let out_yz = -adv_yz + (lc12 + clt12) - inv_lambda * cm12;
        let out_zz = -adv_zz + (lc22 + clt22) - inv_lambda * (cm22 - 1.0);

        if let Some(o) = dcxx.get_mut(thread::index_1d()) { *o = out_xx; }
        if let Some(o) = dcxy.get_mut(thread::index_1d()) { *o = out_xy; }
        if let Some(o) = dcxz.get_mut(thread::index_1d()) { *o = out_xz; }
        if let Some(o) = dcyy.get_mut(thread::index_1d()) { *o = out_yy; }
        if let Some(o) = dcyz.get_mut(thread::index_1d()) { *o = out_yz; }
        if let Some(o) = dczz.get_mut(thread::index_1d()) { *o = out_zz; }
    }
}

/// Compute the 3D Oldroyd-B conformation-transport RHS `∂ₜC` on the GPU for a hex
/// mesh, returning the nodal time-derivative of the six independent conformation
/// components `[Cxx, Cxy, Cxz, Cyy, Cyz, Czz]`. Reusable host wrapper around the
/// [`kernels::oldroyd3d_conf_rhs`] device kernel: packs the 9 mesh metrics
/// node-major (`met[b*9 + {rx,ry,rz,sx,sy,sz,tx,ty,tz}]`), pads the `n1²` diff
/// matrix to `n³`, packs the 6 conformation components node-major (`c[b*6 + o]`),
/// uploads the state, launches one block per element (one thread per node), and
/// gathers the result. Bit-for-bit equal to
/// `gale::dg::OldroydB3d::conformation_rhs`.
///
/// `c` holds the conformation field as `[Cxx, Cxy, Cxz, Cyy, Cyz, Czz]`, each of
/// length `n_elements·n_nodes`; `ux`/`uy`/`uz` are the velocity components in the
/// same layout; `lambda` is the relaxation time. The returned `[Vec<f64>; 6]`
/// follows the same `[dCxx, dCxy, dCxz, dCyy, dCyz, dCzz]` layout as the CPU oracle.
pub fn oldroyd3d_conf_rhs(
    mesh: &Mesh3d,
    c: &[Vec<f64>; 6],
    ux: &[f64],
    uy: &[f64],
    uz: &[f64],
    lambda: f64,
) -> Result<[Vec<f64>; 6], Box<dyn std::error::Error>> {
    let nn = mesh.refh.n_nodes();
    let ne = mesh.n_elements();
    let ndof = ne * nn;
    let n1 = mesh.order + 1;
    assert_eq!(ux.len(), ndof, "ux length must be n_elements·n_nodes");
    assert_eq!(uy.len(), ndof, "uy length must be n_elements·n_nodes");
    assert_eq!(uz.len(), ndof, "uz length must be n_elements·n_nodes");
    for (o, ci) in c.iter().enumerate() {
        assert_eq!(ci.len(), ndof, "C component {o} length must be n_elements·n_nodes");
    }
    assert!(nn <= NN_MAX, "p too large for NN_MAX={NN_MAX}");

    // Pack the 9 metric terms node-major (met[b*9 + idx]) and the 6 conformation
    // components node-major (c_pack[b*6 + o]).
    let mut met = vec![0.0; ndof * 9];
    let mut c_pack = vec![0.0; ndof * 6];
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
                c_pack[b * 6 + o] = c[o][b];
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
    let c_dev = up(&c_pack)?;
    let met_dev = up(&met)?;
    let mut dxx = DeviceBuffer::<f64>::zeroed(&stream, ndof)?;
    let mut dxy = DeviceBuffer::<f64>::zeroed(&stream, ndof)?;
    let mut dxz = DeviceBuffer::<f64>::zeroed(&stream, ndof)?;
    let mut dyy = DeviceBuffer::<f64>::zeroed(&stream, ndof)?;
    let mut dyz = DeviceBuffer::<f64>::zeroed(&stream, ndof)?;
    let mut dzz = DeviceBuffer::<f64>::zeroed(&stream, ndof)?;

    let module = kernels::load(&ctx)?;
    let cfg = LaunchConfig { grid_dim: (ne as u32, 1, 1), block_dim: (nn as u32, 1, 1), shared_mem_bytes: 0 };
    module.oldroyd3d_conf_rhs(
        &stream, cfg, &d_dev, &ux_dev, &uy_dev, &uz_dev, &c_dev, &met_dev,
        lambda.recip(), n1 as u32,
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
