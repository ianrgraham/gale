//! GPU port (step 7f): the **p-multigrid V-cycle preconditioner + PCG, fully
//! on-device**, validated against the CPU oracle `PMultigrid::pcg`.
//!
//! Setup (meshes, transfer matrices, diagonals, smoother weights) is reused from
//! the validated CPU `PMultigrid`; only the iteration runs on the GPU. Kernels are
//! **order-agnostic** (runtime `n1`, shared sized to `NN_MAX`), so one set serves
//! every level. The V-cycle is written iteratively (down → coarse CG → up) and the
//! orchestration uses macros so each kernel launch is its own statement
//! (borrow-clean). Libdevice-free ⇒ embedded path works on sm_70.
//!
//! Run: cargo oxide run --bin gpu-poisson-pcg

use cuda_core::{CudaContext, DeviceBuffer, LaunchConfig};
use cuda_device::{kernel, thread, DisjointSlice, SharedArray};
use cuda_host::cuda_module;
use gale::dg::{Edge, Neighbor, PMultigrid, Poisson};

const NN_MAX: usize = 81; // (order 8 + 1)²
const RED: usize = 256;
const BND: u32 = u32::MAX;

#[cuda_module]
mod kernels {
    use super::*;

    #[kernel]
    #[allow(clippy::too_many_arguments)]
    pub fn gradient(
        d: &[f64], u: &[f64], rx: &[f64], ry: &[f64], sx: &[f64], sy: &[f64], n1: u32,
        mut gx: DisjointSlice<f64>, mut gy: DisjointSlice<f64>,
    ) {
        static mut DS: SharedArray<f64, NN_MAX> = SharedArray::UNINIT;
        static mut US: SharedArray<f64, NN_MAX> = SharedArray::UNINIT;
        let n1 = n1 as usize;
        let nn = n1 * n1;
        let e = thread::blockIdx_x() as usize;
        let m = thread::threadIdx_x() as usize;
        let base = e * nn;
        unsafe {
            DS[m] = d[m];
            US[m] = u[base + m];
        }
        thread::sync_threads();
        let i = m % n1;
        let j = m / n1;
        let mut ur = 0.0f64;
        let mut us = 0.0f64;
        let mut k = 0usize;
        while k < n1 {
            unsafe {
                ur += DS[i * n1 + k] * US[k + j * n1];
                us += DS[j * n1 + k] * US[i + k * n1];
            }
            k += 1;
        }
        let gxv = rx[base + m] * ur + sx[base + m] * us;
        let gyv = ry[base + m] * ur + sy[base + m] * us;
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
        d: &[f64], u: &[f64], gx: &[f64], gy: &[f64], rx: &[f64], ry: &[f64], sx: &[f64],
        sy: &[f64], jw: &[f64], n1: u32, face_vl: &[u32], face_nx: &[f64], face_ny: &[f64],
        face_sw: &[f64], face_nbr: &[u32], face_tau: &[f64], mut out: DisjointSlice<f64>,
    ) {
        static mut DS: SharedArray<f64, NN_MAX> = SharedArray::UNINIT;
        static mut PR: SharedArray<f64, NN_MAX> = SharedArray::UNINIT;
        static mut PS: SharedArray<f64, NN_MAX> = SharedArray::UNINIT;
        static mut RF: SharedArray<f64, NN_MAX> = SharedArray::UNINIT;
        static mut HX: SharedArray<f64, NN_MAX> = SharedArray::UNINIT;
        static mut HY: SharedArray<f64, NN_MAX> = SharedArray::UNINIT;
        let n1 = n1 as usize;
        let nn = n1 * n1;
        let e = thread::blockIdx_x() as usize;
        let m = thread::threadIdx_x() as usize;
        let b = e * nn + m;
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
                while a < n1 {
                    let idx = (e * 4 + t) * n1 + a;
                    let vl = face_vl[idx] as usize;
                    let nx = face_nx[idx];
                    let ny = face_ny[idx];
                    let sw = face_sw[idx];
                    let nbr = face_nbr[idx];
                    let dun_e = nx * gx[e * nn + vl] + ny * gy[e * nn + vl];
                    let ug = u[e * nn + vl];
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
        let i = m % n1;
        let j = m / n1;
        let mut acc = 0.0f64;
        let mut k = 0usize;
        while k < n1 {
            unsafe {
                acc += DS[k * n1 + i] * PR[k + j * n1] + DS[k * n1 + j] * PS[i + k * n1];
            }
            k += 1;
        }
        let rfm = unsafe { RF[m] };
        if let Some(o) = out.get_mut(thread::index_1d()) {
            *o = acc + rfm;
        }
    }

    /// y ← y + ω·invd·(b − ap)  (damped-Jacobi update)
    #[kernel]
    pub fn jacobi(mut y: DisjointSlice<f64>, b: &[f64], ap: &[f64], invd: &[f64], omega: f64) {
        let idx = thread::index_1d();
        let i = idx.get();
        if let Some(o) = y.get_mut(idx) {
            *o += omega * invd[i] * (b[i] - ap[i]);
        }
    }

    /// out ← a − b
    #[kernel]
    pub fn sub(mut out: DisjointSlice<f64>, a: &[f64], b: &[f64]) {
        let idx = thread::index_1d();
        let i = idx.get();
        if let Some(o) = out.get_mut(idx) {
            *o = a[i] - b[i];
        }
    }

    /// y ← c·y
    #[kernel]
    pub fn scal(mut y: DisjointSlice<f64>, c: f64) {
        let idx = thread::index_1d();
        if let Some(o) = y.get_mut(idx) {
            *o *= c;
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

    /// Prolong coarse→fine, per element (block = fine nodes). `out[e·nf + m]`.
    #[kernel]
    pub fn prolong(interp: &[f64], coarse: &[f64], n1f: u32, n1c: u32, mut out: DisjointSlice<f64>) {
        static mut CS: SharedArray<f64, NN_MAX> = SharedArray::UNINIT;
        static mut IM: SharedArray<f64, NN_MAX> = SharedArray::UNINIT;
        let n1f = n1f as usize;
        let n1c = n1c as usize;
        let nf = n1f * n1f;
        let nc = n1c * n1c;
        let e = thread::blockIdx_x() as usize;
        let m = thread::threadIdx_x() as usize;
        unsafe {
            if m < nc {
                CS[m] = coarse[e * nc + m];
            }
            if m < n1f * n1c {
                IM[m] = interp[m];
            }
        }
        thread::sync_threads();
        let iff = m % n1f;
        let jf = m / n1f;
        let mut sacc = 0.0f64;
        let mut jc = 0usize;
        while jc < n1c {
            let mut ic = 0usize;
            while ic < n1c {
                sacc += unsafe { IM[iff * n1c + ic] * IM[jf * n1c + jc] * CS[ic + jc * n1c] };
                ic += 1;
            }
            jc += 1;
        }
        if let Some(o) = out.get_mut(thread::index_1d()) {
            *o = sacc;
        }
    }

    /// Restrict fine→coarse, per element (block = fine nodes, write coarse for m<nc).
    #[kernel]
    pub fn restrict(interp: &[f64], fine: &[f64], n1f: u32, n1c: u32, mut out: DisjointSlice<f64>) {
        static mut FS: SharedArray<f64, NN_MAX> = SharedArray::UNINIT;
        static mut IM: SharedArray<f64, NN_MAX> = SharedArray::UNINIT;
        let n1f = n1f as usize;
        let n1c = n1c as usize;
        let nf = n1f * n1f;
        let nc = n1c * n1c;
        let e = thread::blockIdx_x() as usize;
        let m = thread::threadIdx_x() as usize;
        unsafe {
            FS[m] = fine[e * nf + m];
            if m < n1f * n1c {
                IM[m] = interp[m];
            }
        }
        thread::sync_threads();
        if m < nc {
            let ic = m % n1c;
            let jc = m / n1c;
            let mut sacc = 0.0f64;
            let mut jf = 0usize;
            while jf < n1f {
                let mut iff = 0usize;
                while iff < n1f {
                    sacc += unsafe { IM[iff * n1c + ic] * IM[jf * n1c + jc] * FS[iff + jf * n1f] };
                    iff += 1;
                }
                jf += 1;
            }
            unsafe {
                *out.get_unchecked_mut(e * nc + m) = sacc;
            }
        }
    }
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    use std::f64::consts::PI;
    let order = 4;
    println!("=== GPU p-multigrid PCG vs CPU oracle (p={order}) ===\n");

    // CPU setup (validated) + reference solve.
    let mg = PMultigrid::new(order, 3, 3, [0.0, 1.0], [0.0, 1.0], 5.0);
    let nlev = mg.n_levels();
    let (n_pre, n_post) = mg.smoothing();
    let fine = mg.mesh(0);
    let nn0 = fine.refq.n_nodes();
    let n0 = fine.n_elements() * nn0;
    let exact = |x: f64, y: f64| (PI * x).sin() * (PI * y).sin();
    let mut frc = vec![0.0; n0];
    for (e, el) in fine.elements.iter().enumerate() {
        for k in 0..nn0 {
            frc[e * nn0 + k] = 2.0 * PI * PI * exact(el.geom.x[k], el.geom.y[k]);
        }
    }
    let rhs = Poisson::new(fine, mg.alpha).rhs(&frc, |_, _| 0.0);
    let (u_cpu, it_cpu) = mg.pcg(&rhs, 1e-10, 2000);

    // ---- device buffers (struct-of-arrays per level) ----
    let ctx = CudaContext::new(0)?;
    let stream = ctx.default_stream();
    let up = |v: &[f64]| DeviceBuffer::from_host(&stream, v);
    let upu = |v: &[u32]| DeviceBuffer::from_host(&stream, v);

    let mut n1v = Vec::new();
    let mut nev = Vec::new();
    let mut ndofv = Vec::new();
    let (mut dl, mut rxl, mut ryl, mut sxl, mut syl, mut jwl, mut invd) =
        (vec![], vec![], vec![], vec![], vec![], vec![], vec![]);
    let (mut fvl, mut fnbr) = (vec![], vec![]);
    let (mut fnx, mut fny, mut fsw, mut ftau) = (vec![], vec![], vec![], vec![]);
    let mut omega = Vec::new();
    let (mut xb, mut bb, mut rb, mut apb, mut gxb, mut gyb, mut tmpb) =
        (vec![], vec![], vec![], vec![], vec![], vec![], vec![]);

    for l in 0..nlev {
        let m = mg.mesh(l);
        let nn = m.refq.n_nodes();
        let ne = m.n_elements();
        let ndof = ne * nn;
        let n1 = (mg.level_order(l) + 1) as u32;
        // metrics + face metadata
        let (mut rx, mut ry, mut sx, mut sy, mut jw) =
            (vec![0.0; ndof], vec![0.0; ndof], vec![0.0; ndof], vec![0.0; ndof], vec![0.0; ndof]);
        for (e, el) in m.elements.iter().enumerate() {
            for k in 0..nn {
                rx[e * nn + k] = el.geom.rx[k];
                ry[e * nn + k] = el.geom.ry[k];
                sx[e * nn + k] = el.geom.sx[k];
                sy[e * nn + k] = el.geom.sy[k];
                jw[e * nn + k] = el.geom.jw[k];
            }
        }
        let p1 = (mg.level_order(l) + 1) as f64;
        let h: Vec<f64> = m.elements.iter().map(|el| el.geom.jw.iter().sum::<f64>().sqrt()).collect();
        let n1u = n1 as usize;
        let nfc = ne * 4 * n1u;
        let (mut vl, mut nx, mut ny, mut sw, mut nbr, mut tau) =
            (vec![0u32; nfc], vec![0.0; nfc], vec![0.0; nfc], vec![0.0; nfc], vec![BND; nfc], vec![0.0; ne * 4]);
        for (e, el) in m.elements.iter().enumerate() {
            for (t, edge) in Edge::ALL.iter().enumerate() {
                let face = &el.faces[*edge as usize];
                let nb = &el.neighbors[*edge as usize];
                tau[e * 4 + t] = match nb {
                    Neighbor::Interior { elem: re, .. } => mg.alpha * p1 * p1 / h[e].min(h[*re]),
                    Neighbor::Boundary { .. } => mg.alpha * p1 * p1 / h[e],
                };
                for a in 0..n1u {
                    let idx = (e * 4 + t) * n1u + a;
                    vl[idx] = face.nodes[a] as u32;
                    nx[idx] = face.nx[a];
                    ny[idx] = face.ny[a];
                    sw[idx] = face.sw[a];
                    if let Neighbor::Interior { elem: re, edge: redge, perm } = nb {
                        let rf = &m.elements[*re].faces[*redge as usize];
                        nbr[idx] = (*re * nn + rf.nodes[perm[a]]) as u32;
                    }
                }
            }
        }
        n1v.push(n1);
        nev.push(ne as u32);
        ndofv.push(ndof);
        dl.push(up(&m.refq.line.diff)?);
        rxl.push(up(&rx)?);
        ryl.push(up(&ry)?);
        sxl.push(up(&sx)?);
        syl.push(up(&sy)?);
        jwl.push(up(&jw)?);
        invd.push(up(mg.inv_diagonal(l))?);
        fvl.push(upu(&vl)?);
        fnbr.push(upu(&nbr)?);
        fnx.push(up(&nx)?);
        fny.push(up(&ny)?);
        fsw.push(up(&sw)?);
        ftau.push(up(&tau)?);
        omega.push(mg.jacobi_omega(l));
        xb.push(DeviceBuffer::<f64>::zeroed(&stream, ndof)?);
        bb.push(DeviceBuffer::<f64>::zeroed(&stream, ndof)?);
        rb.push(DeviceBuffer::<f64>::zeroed(&stream, ndof)?);
        apb.push(DeviceBuffer::<f64>::zeroed(&stream, ndof)?);
        gxb.push(DeviceBuffer::<f64>::zeroed(&stream, ndof)?);
        gyb.push(DeviceBuffer::<f64>::zeroed(&stream, ndof)?);
        tmpb.push(DeviceBuffer::<f64>::zeroed(&stream, ndof)?);
    }
    // transfer matrices (coarse l+1 → fine l)
    let mut interp = Vec::new();
    for l in 0..nlev - 1 {
        interp.push(up(mg.interp_matrix(l))?);
    }

    // per-level launch configs
    let cfg: Vec<LaunchConfig> = (0..nlev)
        .map(|l| LaunchConfig { grid_dim: (nev[l], 1, 1), block_dim: (n1v[l] * n1v[l], 1, 1), shared_mem_bytes: 0 })
        .collect();
    let vcfg: Vec<LaunchConfig> = ndofv.iter().map(|&n| LaunchConfig::for_num_elems(n as u32)).collect();
    let redcfg = LaunchConfig { grid_dim: (1, 1, 1), block_dim: (RED as u32, 1, 1), shared_mem_bytes: 0 };

    // PCG vectors (finest level, separate from V-cycle scratch).
    let mut psol = DeviceBuffer::<f64>::zeroed(&stream, n0)?;
    let mut pres = up(&rhs)?;
    let mut pp = DeviceBuffer::<f64>::zeroed(&stream, n0)?;
    let mut pz = DeviceBuffer::<f64>::zeroed(&stream, n0)?;
    let mut pap = DeviceBuffer::<f64>::zeroed(&stream, n0)?;
    let rhs_dev = up(&rhs)?;
    let mut partial = DeviceBuffer::<f64>::zeroed(&stream, 1)?;

    let module = kernels::load(&ctx)?;

    macro_rules! dot {
        ($a:expr, $b:expr, $n:expr) => {{
            module.dot_partial(&stream, redcfg, $a, $b, $n as u64, &mut partial)?;
            partial.to_host_vec(&stream)?[0]
        }};
    }
    macro_rules! dcopy {
        ($dst:expr, $src:expr, $n:expr) => {{
            unsafe {
                cuda_core::memory::memcpy_dtod_async($dst.cu_deviceptr(), $src.cu_deviceptr(), $n * 8, stream.cu_stream())?;
            }
        }};
    }
    // matvec: dst = A·src at level $l (src/dst external; gx/gy scratch from Vecs).
    macro_rules! matvec {
        ($l:expr, $src:expr, $dst:expr) => {{
            let l = $l;
            module.gradient(&stream, cfg[l], &dl[l], $src, &rxl[l], &ryl[l], &sxl[l], &syl[l], n1v[l], &mut gxb[l], &mut gyb[l])?;
            module.operator(
                &stream, cfg[l], &dl[l], $src, &gxb[l], &gyb[l], &rxl[l], &ryl[l], &sxl[l], &syl[l], &jwl[l], n1v[l],
                &fvl[l], &fnx[l], &fny[l], &fsw[l], &fnbr[l], &ftau[l], $dst,
            )?;
        }};
    }
    macro_rules! coarse_cg {
        ($l:expr) => {{
            let l = $l;
            let n = ndofv[l];
            module.scal(&stream, vcfg[l], &mut xb[l], 0.0)?;
            dcopy!(&rb[l], &bb[l], n);
            dcopy!(&tmpb[l], &bb[l], n);
            let bn = dot!(&bb[l], &bb[l], n).sqrt().max(1e-300);
            let mut rs = dot!(&rb[l], &rb[l], n);
            for _ in 0..300 {
                matvec!(l, &tmpb[l], &mut apb[l]);
                let pap_d = dot!(&tmpb[l], &apb[l], n);
                let alpha = rs / pap_d;
                module.axpy(&stream, vcfg[l], &mut xb[l], &tmpb[l], alpha)?;
                module.axpy(&stream, vcfg[l], &mut rb[l], &apb[l], -alpha)?;
                let rsn = dot!(&rb[l], &rb[l], n);
                if rsn.sqrt() / bn < 1e-10 {
                    break;
                }
                let beta = rsn / rs;
                module.xpby(&stream, vcfg[l], &mut tmpb[l], &rb[l], beta)?;
                rs = rsn;
            }
        }};
    }
    // One V-cycle: input in bb[0], solution in xb[0].
    macro_rules! vcycle {
        () => {{
            let last = nlev - 1;
            for l in 0..last {
                module.scal(&stream, vcfg[l], &mut xb[l], 0.0)?;
                for _ in 0..n_pre {
                    matvec!(l, &xb[l], &mut apb[l]);
                    module.jacobi(&stream, vcfg[l], &mut xb[l], &bb[l], &apb[l], &invd[l], omega[l])?;
                }
                matvec!(l, &xb[l], &mut apb[l]);
                module.sub(&stream, vcfg[l], &mut rb[l], &bb[l], &apb[l])?;
                module.restrict(&stream, cfg[l], &interp[l], &rb[l], n1v[l], n1v[l + 1], &mut bb[l + 1])?;
            }
            coarse_cg!(last);
            for l in (0..last).rev() {
                module.prolong(&stream, cfg[l], &interp[l], &xb[l + 1], n1v[l], n1v[l + 1], &mut tmpb[l])?;
                module.axpy(&stream, vcfg[l], &mut xb[l], &tmpb[l], 1.0)?;
                for _ in 0..n_post {
                    matvec!(l, &xb[l], &mut apb[l]);
                    module.jacobi(&stream, vcfg[l], &mut xb[l], &bb[l], &apb[l], &invd[l], omega[l])?;
                }
            }
        }};
    }

    // ---- preconditioned CG on the device ----
    module.scal(&stream, vcfg[0], &mut psol, 0.0)?;
    dcopy!(&bb[0], &pres, n0);
    vcycle!();
    dcopy!(&pz, &xb[0], n0);
    dcopy!(&pp, &pz, n0);
    let bn = dot!(&rhs_dev, &rhs_dev, n0).sqrt().max(1e-300);
    let mut rz = dot!(&pres, &pz, n0);
    let mut iters = 0;
    for it in 0..2000 {
        matvec!(0, &pp, &mut pap);
        let pap_d = dot!(&pp, &pap, n0);
        let alpha = rz / pap_d;
        module.axpy(&stream, vcfg[0], &mut psol, &pp, alpha)?;
        module.axpy(&stream, vcfg[0], &mut pres, &pap, -alpha)?;
        iters = it + 1;
        if dot!(&pres, &pres, n0).sqrt() / bn < 1e-10 {
            break;
        }
        dcopy!(&bb[0], &pres, n0);
        vcycle!();
        dcopy!(&pz, &xb[0], n0);
        let rz_new = dot!(&pres, &pz, n0);
        let beta = rz_new / rz;
        module.xpby(&stream, vcfg[0], &mut pp, &pz, beta)?;
        rz = rz_new;
    }
    let u_gpu = psol.to_host_vec(&stream)?;

    // compare
    let mut diff = 0.0f64;
    let mut nrm = 0.0f64;
    for i in 0..n0 {
        diff += (u_gpu[i] - u_cpu[i]).powi(2);
        nrm += u_cpu[i].powi(2);
    }
    let rel = (diff / nrm.max(1e-300)).sqrt();
    println!("dofs={n0}  levels={nlev}  PCG iters: cpu={it_cpu}  gpu={iters}");
    println!("‖u_gpu − u_cpu‖ / ‖u_cpu‖ = {rel:.3e}");
    if rel < 1e-7 {
        println!("\nPASS: GPU p-multigrid PCG matches the CPU oracle.");
        Ok(())
    } else {
        eprintln!("\nFAIL: GPU/CPU mismatch.");
        std::process::exit(1);
    }
}
