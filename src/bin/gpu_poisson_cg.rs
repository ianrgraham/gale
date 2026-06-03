//! GPU port (step 7d): **conjugate gradient entirely on the GPU**, solving the
//! SIPG Poisson system with the device operator pipeline (step 7c) plus a
//! dot-product reduction and axpy/xpby vector kernels. Vectors stay resident on
//! the device; only the CG scalars (`α`, `β`, residual) transfer host-side.
//! Validated against the CPU oracle `Poisson::cg`.
//!
//! Run: cargo oxide run --bin gpu-poisson-cg

use cuda_core::{CudaContext, DeviceBuffer, LaunchConfig};
use cuda_device::{kernel, thread, DisjointSlice, SharedArray};
use cuda_host::cuda_module;
use gale::dg::{Edge, Mesh2d, Neighbor, Poisson};

const P: usize = 4;
const N1: usize = P + 1;
const NN: usize = N1 * N1;
const BND: u32 = u32::MAX;
const RED: usize = 256; // reduction block size

#[cuda_module]
mod kernels {
    use super::*;

    #[kernel]
    #[allow(clippy::too_many_arguments)]
    pub fn gradient(
        d: &[f64], u: &[f64], rx: &[f64], ry: &[f64], sx: &[f64], sy: &[f64],
        mut gx: DisjointSlice<f64>, mut gy: DisjointSlice<f64>,
    ) {
        static mut DS: SharedArray<f64, NN> = SharedArray::UNINIT;
        static mut US: SharedArray<f64, NN> = SharedArray::UNINIT;
        let e = thread::blockIdx_x() as usize;
        let m = thread::threadIdx_x() as usize;
        unsafe {
            DS[m] = d[m];
            US[m] = u[e * NN + m];
        }
        thread::sync_threads();
        let i = m % N1;
        let j = m / N1;
        let mut ur = 0.0f64;
        let mut us = 0.0f64;
        let mut k = 0usize;
        while k < N1 {
            unsafe {
                ur += DS[i * N1 + k] * US[k + j * N1];
                us += DS[j * N1 + k] * US[i + k * N1];
            }
            k += 1;
        }
        let b = e * NN + m;
        let gxv = rx[b] * ur + sx[b] * us;
        let gyv = ry[b] * ur + sy[b] * us;
        if let Some(o) = gx.get_mut(thread::index_1d()) {
            *o = gxv;
        }
        if let Some(o) = gy.get_mut(thread::index_1d()) {
            *o = gyv;
        }
    }

    #[kernel]
    #[allow(clippy::too_many_arguments)]
    pub fn operator(
        d: &[f64], u: &[f64], gx: &[f64], gy: &[f64],
        rx: &[f64], ry: &[f64], sx: &[f64], sy: &[f64], jw: &[f64],
        face_vl: &[u32], face_nx: &[f64], face_ny: &[f64], face_sw: &[f64],
        face_nbr: &[u32], face_tau: &[f64], mut out: DisjointSlice<f64>,
    ) {
        static mut DS: SharedArray<f64, NN> = SharedArray::UNINIT;
        static mut PR: SharedArray<f64, NN> = SharedArray::UNINIT;
        static mut PS: SharedArray<f64, NN> = SharedArray::UNINIT;
        static mut RF: SharedArray<f64, NN> = SharedArray::UNINIT;
        static mut HX: SharedArray<f64, NN> = SharedArray::UNINIT;
        static mut HY: SharedArray<f64, NN> = SharedArray::UNINIT;
        let e = thread::blockIdx_x() as usize;
        let m = thread::threadIdx_x() as usize;
        let b = e * NN + m;
        unsafe {
            DS[m] = d[m];
            RF[m] = 0.0;
            HX[m] = 0.0;
            HY[m] = 0.0;
            let wx = jw[b] * gx[b];
            let wy = jw[b] * gy[b];
            PR[m] = rx[b] * wx + ry[b] * wy;
            PS[m] = sx[b] * wx + sy[b] * wy;
        }
        thread::sync_threads();
        if m == 0 {
            let mut t = 0usize;
            while t < 4 {
                let tau = face_tau[e * 4 + t];
                let mut a = 0usize;
                while a < N1 {
                    let idx = (e * 4 + t) * N1 + a;
                    let vl = face_vl[idx] as usize;
                    let nx = face_nx[idx];
                    let ny = face_ny[idx];
                    let sw = face_sw[idx];
                    let nbr = face_nbr[idx];
                    let dun_e = nx * gx[e * NN + vl] + ny * gy[e * NN + vl];
                    let ug = u[e * NN + vl];
                    let (avg, jump, gfac) = if nbr == BND {
                        (dun_e, ug, 1.0)
                    } else {
                        let ng = nbr as usize;
                        (0.5 * (dun_e + nx * gx[ng] + ny * gy[ng]), ug - u[ng], 0.5)
                    };
                    let g = gfac * sw * jump;
                    unsafe {
                        RF[vl] += -sw * avg + tau * sw * jump;
                        HX[vl] += g * nx;
                        HY[vl] += g * ny;
                    }
                    a += 1;
                }
                t += 1;
            }
        }
        thread::sync_threads();
        let hxm = unsafe { HX[m] };
        let hym = unsafe { HY[m] };
        unsafe {
            PR[m] -= rx[b] * hxm + ry[b] * hym;
            PS[m] -= sx[b] * hxm + sy[b] * hym;
        }
        thread::sync_threads();
        let i = m % N1;
        let j = m / N1;
        let mut acc = 0.0f64;
        let mut k = 0usize;
        while k < N1 {
            unsafe {
                acc += DS[k * N1 + i] * PR[k + j * N1] + DS[k * N1 + j] * PS[i + k * N1];
            }
            k += 1;
        }
        let rfm = unsafe { RF[m] };
        if let Some(o) = out.get_mut(thread::index_1d()) {
            *o = acc + rfm;
        }
    }

    /// Single-block dot product: grid-stride accumulate + tree reduce → partial[0].
    #[kernel]
    pub fn dot_partial(a: &[f64], b: &[f64], n: u64, mut partial: DisjointSlice<f64>) {
        static mut SH: SharedArray<f64, RED> = SharedArray::UNINIT;
        let tid = thread::threadIdx_x() as usize;
        let stride = thread::blockDim_x() as usize;
        let mut acc = 0.0f64;
        let mut i = tid;
        while i < n as usize {
            acc += a[i] * b[i];
            i += stride;
        }
        unsafe { SH[tid] = acc; }
        thread::sync_threads();
        let mut s = stride / 2;
        while s > 0 {
            if tid < s {
                unsafe { SH[tid] = SH[tid] + SH[tid + s]; }
            }
            thread::sync_threads();
            s /= 2;
        }
        if tid == 0 {
            if let Some(o) = partial.get_mut(thread::index_1d()) {
                *o = unsafe { SH[0] };
            }
        }
    }

    /// y ← y + a·x
    #[kernel]
    pub fn axpy(mut y: DisjointSlice<f64>, x: &[f64], a: f64) {
        let idx = thread::index_1d();
        let i = idx.get();
        if let Some(o) = y.get_mut(idx) {
            *o += a * x[i];
        }
    }

    /// y ← x + b·y
    #[kernel]
    pub fn xpby(mut y: DisjointSlice<f64>, x: &[f64], b: f64) {
        let idx = thread::index_1d();
        let i = idx.get();
        if let Some(o) = y.get_mut(idx) {
            *o = x[i] + b * *o;
        }
    }
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    use std::f64::consts::PI;
    println!("=== GPU conjugate gradient vs CPU oracle (p={P}) ===\n");

    let mesh = Mesh2d::rectangular(P, 3, 3, [0.0, 1.0], [0.0, 1.0]);
    let refq = &mesh.refq;
    let nn = refq.n_nodes();
    let ne = mesh.n_elements();
    let ndof = ne * nn;
    assert_eq!(nn, NN);
    let poisson = Poisson::new(&mesh, 5.0);

    // MMS: u = sin(πx)sin(πy), −Δu = 2π²u, homogeneous Dirichlet.
    let exact = |x: f64, y: f64| (PI * x).sin() * (PI * y).sin();
    let mut f = vec![0.0; ndof];
    for (e, el) in mesh.elements.iter().enumerate() {
        for k in 0..nn {
            f[e * nn + k] = 2.0 * PI * PI * exact(el.geom.x[k], el.geom.y[k]);
        }
    }
    let b = poisson.rhs(&f, |_, _| 0.0);

    // CPU reference solve.
    let (u_cpu, it_cpu, _res) = poisson.cg(&b, 1e-10, 20000);

    // Flatten metrics + face metadata (as in step 7c).
    let (mut rx, mut ry, mut sx, mut sy, mut jw) =
        (vec![0.0; ndof], vec![0.0; ndof], vec![0.0; ndof], vec![0.0; ndof], vec![0.0; ndof]);
    for (e, el) in mesh.elements.iter().enumerate() {
        for k in 0..nn {
            rx[e * nn + k] = el.geom.rx[k];
            ry[e * nn + k] = el.geom.ry[k];
            sx[e * nn + k] = el.geom.sx[k];
            sy[e * nn + k] = el.geom.sy[k];
            jw[e * nn + k] = el.geom.jw[k];
        }
    }
    let alpha_p = 5.0;
    let p1 = (P + 1) as f64;
    let h: Vec<f64> = mesh.elements.iter().map(|el| el.geom.jw.iter().sum::<f64>().sqrt()).collect();
    let nfc = ne * 4 * N1;
    let (mut fvl, mut fnx, mut fny, mut fsw, mut fnbr, mut ftau) =
        (vec![0u32; nfc], vec![0.0; nfc], vec![0.0; nfc], vec![0.0; nfc], vec![BND; nfc], vec![0.0; ne * 4]);
    for (e, el) in mesh.elements.iter().enumerate() {
        for (t, edge) in Edge::ALL.iter().enumerate() {
            let face = &el.faces[*edge as usize];
            let nb = &el.neighbors[*edge as usize];
            ftau[e * 4 + t] = match nb {
                Neighbor::Interior { elem: re, .. } => alpha_p * p1 * p1 / h[e].min(h[*re]),
                Neighbor::Boundary { .. } => alpha_p * p1 * p1 / h[e],
            };
            for a in 0..N1 {
                let idx = (e * 4 + t) * N1 + a;
                fvl[idx] = face.nodes[a] as u32;
                fnx[idx] = face.nx[a];
                fny[idx] = face.ny[a];
                fsw[idx] = face.sw[a];
                if let Neighbor::Interior { elem: re, edge: redge, perm } = nb {
                    let rf = &mesh.elements[*re].faces[*redge as usize];
                    fnbr[idx] = (*re * NN + rf.nodes[perm[a]]) as u32;
                }
            }
        }
    }

    // Device setup.
    let ctx = CudaContext::new(0)?;
    let stream = ctx.default_stream();
    let up = |v: &[f64]| DeviceBuffer::from_host(&stream, v);
    let d_dev = up(&refq.line.diff)?;
    let rx_dev = up(&rx)?;
    let ry_dev = up(&ry)?;
    let sx_dev = up(&sx)?;
    let sy_dev = up(&sy)?;
    let jw_dev = up(&jw)?;
    let fvl_dev = DeviceBuffer::from_host(&stream, &fvl)?;
    let fnx_dev = up(&fnx)?;
    let fny_dev = up(&fny)?;
    let fsw_dev = up(&fsw)?;
    let fnbr_dev = DeviceBuffer::from_host(&stream, &fnbr)?;
    let ftau_dev = up(&ftau)?;

    // CG vectors (resident on device).
    let mut x = DeviceBuffer::<f64>::zeroed(&stream, ndof)?;
    let mut r = up(&b)?; // r = b − A·0 = b
    let mut p = up(&b)?; // p = r
    let mut ap = DeviceBuffer::<f64>::zeroed(&stream, ndof)?;
    let mut gx = DeviceBuffer::<f64>::zeroed(&stream, ndof)?;
    let mut gy = DeviceBuffer::<f64>::zeroed(&stream, ndof)?;
    let mut partial = DeviceBuffer::<f64>::zeroed(&stream, 1)?;

    let module = kernels::load(&ctx)?;
    let cfg = LaunchConfig { grid_dim: (ne as u32, 1, 1), block_dim: (nn as u32, 1, 1), shared_mem_bytes: 0 };
    let red = LaunchConfig { grid_dim: (1, 1, 1), block_dim: (RED as u32, 1, 1), shared_mem_bytes: 0 };
    let vec_cfg = LaunchConfig::for_num_elems(ndof as u32);

    // dot(a,b) helper.
    let n64 = ndof as u64;
    macro_rules! dot {
        ($a:expr, $b:expr) => {{
            module.dot_partial(&stream, red, $a, $b, n64, &mut partial)?;
            partial.to_host_vec(&stream)?[0]
        }};
    }
    // A·p helper (gradient then operator into ap).
    macro_rules! apply {
        ($field:expr, $dst:expr) => {{
            module.gradient(&stream, cfg, &d_dev, $field, &rx_dev, &ry_dev, &sx_dev, &sy_dev, &mut gx, &mut gy)?;
            module.operator(
                &stream, cfg, &d_dev, $field, &gx, &gy, &rx_dev, &ry_dev, &sx_dev, &sy_dev, &jw_dev,
                &fvl_dev, &fnx_dev, &fny_dev, &fsw_dev, &fnbr_dev, &ftau_dev, $dst,
            )?;
        }};
    }

    let bnorm = dot!(&r, &r).sqrt().max(1e-300);
    let mut rs = dot!(&r, &r);
    let mut iters = 0;
    for it in 0..20000 {
        apply!(&p, &mut ap);
        let pap = dot!(&p, &ap);
        let alpha = rs / pap;
        module.axpy(&stream, vec_cfg, &mut x, &p, alpha)?; // x += α p
        module.axpy(&stream, vec_cfg, &mut r, &ap, -alpha)?; // r −= α ap
        let rs_new = dot!(&r, &r);
        iters = it + 1;
        if rs_new.sqrt() / bnorm < 1e-10 {
            break;
        }
        let beta = rs_new / rs;
        module.xpby(&stream, vec_cfg, &mut p, &r, beta)?; // p = r + β p
        rs = rs_new;
    }

    let u_gpu = x.to_host_vec(&stream)?;

    // Compare GPU vs CPU solve.
    let mut diff = 0.0f64;
    let mut cpun = 0.0f64;
    let mut err_gpu = 0.0f64;
    for (e, el) in mesh.elements.iter().enumerate() {
        for k in 0..nn {
            let i = e * nn + k;
            diff += el.geom.jw[k] * (u_gpu[i] - u_cpu[i]).powi(2);
            cpun += el.geom.jw[k] * u_cpu[i].powi(2);
            err_gpu += el.geom.jw[k] * (u_gpu[i] - exact(el.geom.x[k], el.geom.y[k])).powi(2);
        }
    }
    let rel = (diff / cpun.max(1e-300)).sqrt();
    println!("dofs={ndof}   CG iters: cpu={it_cpu}  gpu={iters}");
    println!("‖u_gpu − u_cpu‖ / ‖u_cpu‖ = {rel:.3e}");
    println!("‖u_gpu − u_exact‖ (MMS)   = {:.3e}", err_gpu.sqrt());
    if rel < 1e-8 {
        println!("\nPASS: GPU CG matches the CPU solve.");
        Ok(())
    } else {
        eprintln!("\nFAIL: GPU/CPU solve mismatch.");
        std::process::exit(1);
    }
}
