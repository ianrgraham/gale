//! GPU port of the **Oldroyd-B conformation transport** rhs (direct form), the
//! viscoelastic constitutive update on the headline path. Purely element-local
//! (nodal-collocation advection — no faces, no neighbor gather) and libdevice-free
//! (only arithmetic + the sum-factorized gradient pattern already proven on GPU),
//! so it runs on the embedded `#[cuda_module]` path on sm_70. Validated bit-for-bit
//! against the CPU oracle `OldroydB::conformation_rhs`.
//!
//!   ∂ₜC = −(u·∇)C + (L·C + C·Lᵀ) − (1/λ)(C − I),   L = ∇u.
//!
//! Run: cargo oxide run --bin gpu-oldroyd

use cuda_core::{CudaContext, DeviceBuffer, LaunchConfig};
use cuda_device::{kernel, thread, DisjointSlice, SharedArray};
use cuda_host::cuda_module;
use gale::dg::{Mesh2d, OldroydB};

const NN_MAX: usize = 81;

#[cuda_module]
mod kernels {
    use super::*;

    /// `∂ₜC` for one element per block, one node per thread. Computes ∇u and ∇C by
    /// sum factorization, then the pointwise advection + upper-convected stretching
    /// + relaxation.
    #[kernel]
    #[allow(clippy::too_many_arguments)]
    pub fn conf_rhs(
        d: &[f64], ux: &[f64], uy: &[f64], cxx: &[f64], cxy: &[f64], cyy: &[f64],
        rx: &[f64], ry: &[f64], sx: &[f64], sy: &[f64],
        inv_lambda: f64, n1: u32,
        mut dcxx: DisjointSlice<f64>, mut dcxy: DisjointSlice<f64>, mut dcyy: DisjointSlice<f64>,
    ) {
        static mut DS: SharedArray<f64, NN_MAX> = SharedArray::UNINIT;
        static mut US: SharedArray<f64, NN_MAX> = SharedArray::UNINIT;
        static mut VS: SharedArray<f64, NN_MAX> = SharedArray::UNINIT;
        static mut XX: SharedArray<f64, NN_MAX> = SharedArray::UNINIT;
        static mut XY: SharedArray<f64, NN_MAX> = SharedArray::UNINIT;
        static mut YY: SharedArray<f64, NN_MAX> = SharedArray::UNINIT;
        let n1 = n1 as usize;
        let nn = n1 * n1;
        let e = thread::blockIdx_x() as usize;
        let m = thread::threadIdx_x() as usize;
        let b = e * nn + m;
        unsafe {
            DS[m] = d[m];
            US[m] = ux[b];
            VS[m] = uy[b];
            XX[m] = cxx[b];
            XY[m] = cxy[b];
            YY[m] = cyy[b];
        }
        thread::sync_threads();

        let i = m % n1;
        let j = m / n1;
        // r- and s-derivatives of all five fields (sum factorization).
        let (mut ur_u, mut us_u) = (0.0f64, 0.0f64);
        let (mut ur_v, mut us_v) = (0.0f64, 0.0f64);
        let (mut ur_xx, mut us_xx) = (0.0f64, 0.0f64);
        let (mut ur_xy, mut us_xy) = (0.0f64, 0.0f64);
        let (mut ur_yy, mut us_yy) = (0.0f64, 0.0f64);
        let mut k = 0usize;
        while k < n1 {
            let dik = unsafe { DS[i * n1 + k] };
            let djk = unsafe { DS[j * n1 + k] };
            let cr = k + j * n1; // (k, j) — varies along r
            let cs = i + k * n1; // (i, k) — varies along s
            unsafe {
                ur_u += dik * US[cr];
                us_u += djk * US[cs];
                ur_v += dik * VS[cr];
                us_v += djk * VS[cs];
                ur_xx += dik * XX[cr];
                us_xx += djk * XX[cs];
                ur_xy += dik * XY[cr];
                us_xy += djk * XY[cs];
                ur_yy += dik * YY[cr];
                us_yy += djk * YY[cs];
            }
            k += 1;
        }
        let (rxm, rym, sxm, sym) = (rx[b], ry[b], sx[b], sy[b]);
        // Velocity gradient L (Lᵢⱼ = ∂uᵢ/∂xⱼ).
        let lxx = rxm * ur_u + sxm * us_u;
        let lxy = rym * ur_u + sym * us_u;
        let lyx = rxm * ur_v + sxm * us_v;
        let lyy = rym * ur_v + sym * us_v;
        // ∇C components.
        let cxx_x = rxm * ur_xx + sxm * us_xx;
        let cxx_y = rym * ur_xx + sym * us_xx;
        let cxy_x = rxm * ur_xy + sxm * us_xy;
        let cxy_y = rym * ur_xy + sym * us_xy;
        let cyy_x = rxm * ur_yy + sxm * us_yy;
        let cyy_y = rym * ur_yy + sym * us_yy;

        let (u, v) = (unsafe { US[m] }, unsafe { VS[m] });
        let (cxxm, cxym, cyym) = (unsafe { XX[m] }, unsafe { XY[m] }, unsafe { YY[m] });

        let adv_xx = u * cxx_x + v * cxx_y;
        let adv_xy = u * cxy_x + v * cxy_y;
        let adv_yy = u * cyy_x + v * cyy_y;
        let s_xx = 2.0 * (lxx * cxxm + lxy * cxym);
        let s_xy = lxx * cxym + lxy * cyym + lyx * cxxm + lyy * cxym;
        let s_yy = 2.0 * (lyx * cxym + lyy * cyym);
        let r_xx = -inv_lambda * (cxxm - 1.0);
        let r_xy = -inv_lambda * cxym;
        let r_yy = -inv_lambda * (cyym - 1.0);

        if let Some(o) = dcxx.get_mut(thread::index_1d()) {
            *o = -adv_xx + s_xx + r_xx;
        }
        if let Some(o) = dcxy.get_mut(thread::index_1d()) {
            *o = -adv_xy + s_xy + r_xy;
        }
        if let Some(o) = dcyy.get_mut(thread::index_1d()) {
            *o = -adv_yy + s_yy + r_yy;
        }
    }
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let p = 4;
    println!("=== GPU Oldroyd-B conformation rhs vs CPU (p={p}) ===\n");

    let mesh = Mesh2d::rectangular(p, 4, 4, [0.0, 1.0], [0.0, 1.0]);
    let nn = mesh.refq.n_nodes();
    let ne = mesh.n_elements();
    let ndof = ne * nn;
    let lambda = 0.7;
    let op = OldroydB::new(&mesh, lambda, 1.0);

    // Smooth velocity + SPD conformation fields.
    let (mut ux, mut uy) = (vec![0.0; ndof], vec![0.0; ndof]);
    let mut c = [vec![0.0; ndof], vec![0.0; ndof], vec![0.0; ndof]];
    for (e, el) in mesh.elements.iter().enumerate() {
        for k in 0..nn {
            let (x, y) = (el.geom.x[k], el.geom.y[k]);
            ux[e * nn + k] = (2.0 * x).sin() * y + 0.3 * x;
            uy[e * nn + k] = -0.4 * (3.0 * y).cos() * x;
            c[0][e * nn + k] = 1.5 + 0.4 * (x + y).sin();
            c[1][e * nn + k] = 0.2 * x - 0.1 * y;
            c[2][e * nn + k] = 1.3 + 0.3 * (x * y).cos();
        }
    }
    let cpu = op.conformation_rhs(&c, &ux, &uy);

    // Flatten per-node metrics.
    let (mut rx, mut ry, mut sx, mut sy) = (vec![0.0; ndof], vec![0.0; ndof], vec![0.0; ndof], vec![0.0; ndof]);
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
    let ux_dev = up(&ux)?;
    let uy_dev = up(&uy)?;
    let cxx_dev = up(&c[0])?;
    let cxy_dev = up(&c[1])?;
    let cyy_dev = up(&c[2])?;
    let rx_dev = up(&rx)?;
    let ry_dev = up(&ry)?;
    let sx_dev = up(&sx)?;
    let sy_dev = up(&sy)?;
    let mut dxx = DeviceBuffer::<f64>::zeroed(&stream, ndof)?;
    let mut dxy = DeviceBuffer::<f64>::zeroed(&stream, ndof)?;
    let mut dyy = DeviceBuffer::<f64>::zeroed(&stream, ndof)?;

    let module = kernels::load(&ctx)?;
    let cfg = LaunchConfig { grid_dim: (ne as u32, 1, 1), block_dim: (nn as u32, 1, 1), shared_mem_bytes: 0 };
    module.conf_rhs(
        &stream, cfg, &d_dev, &ux_dev, &uy_dev, &cxx_dev, &cxy_dev, &cyy_dev,
        &rx_dev, &ry_dev, &sx_dev, &sy_dev, lambda.recip(), (p + 1) as u32,
        &mut dxx, &mut dxy, &mut dyy,
    )?;
    let gpu = [dxx.to_host_vec(&stream)?, dxy.to_host_vec(&stream)?, dyy.to_host_vec(&stream)?];

    let mut max_abs = 0.0f64;
    let mut scale = 1e-300f64;
    for v in 0..3 {
        for i in 0..ndof {
            max_abs = max_abs.max((gpu[v][i] - cpu[v][i]).abs());
            scale = scale.max(cpu[v][i].abs());
        }
    }
    println!("dofs={ndof}  max|gpu − cpu| / |rhs| = {:.3e}", max_abs / scale);
    if max_abs / scale < 1e-12 {
        println!("\nPASS: GPU Oldroyd-B conformation rhs matches the CPU oracle.");
        Ok(())
    } else {
        eprintln!("\nFAIL: GPU/CPU mismatch.");
        std::process::exit(1);
    }
}
