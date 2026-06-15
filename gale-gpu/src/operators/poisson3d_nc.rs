//! GPU **3D non-conforming (2:1 octree) SIPG Poisson** operator — the 3D analogue of
//! [`poisson_nc`](super::poisson_nc) and the AMR companion to the conforming
//! [`poisson3d`](super::poisson3d). Matrix-free `(λM + A)·u` on a hex mesh whose 2:1 interfaces
//! couple a coarse hex face to four fine quarter-faces through the hex-face mortar `P`
//! (`RefineHex::mortar_to_fine_face`, the tensor product `P_ha ⊗ P_hb` of the two 1D matrices)
//! and its transpose `Pᵀ` (`mortar_gather_face`). Validated bit-for-bit against the CPU oracle
//! `gale::dg::Poisson3d::apply` (which carries the same mortar assembly).
//!
//! The kernel is gather-form (thread 0 runs the face loop into shared `RF/HX/HY/HZ`; all threads do
//! the volume sum-factorisation + symmetry lift), per-element metrics so coarse and fine sizes mix.

use cuda_core::{CudaContext, DeviceBuffer, LaunchConfig};
use cuda_device::{kernel, thread, DisjointSlice, SharedArray};
use cuda_host::cuda_module;
use gale::dg::{Face, Mesh3d, Neighbor3, RefineQuad};

const NN_MAX: usize = 125; // (order 4 + 1)³ — raise for higher p (watch static shared budget)
const RED: usize = 256; // reduction block size for the CG dot product

const K_CONF: u32 = 0; // conforming Interior / Dirichlet(BND) / Neumann(NEU)
const K_FINE_TO_COARSE: u32 = 1; // this hex is FINE; one coarse neighbour (coarse trace)
const K_COARSE_TO_FINE: u32 = 2; // this hex is COARSE; four fine neighbours (quarters 0..3)

const BND: u32 = u32::MAX; // Dirichlet boundary face
const NEU: u32 = u32::MAX - 1; // Neumann boundary face (natural BC)

#[cuda_module]
mod kernels {
    use super::*;

    /// Per-element physical gradient `(gx,gy,gz)` of `u` (per-node metrics `met[b*9]`). One block
    /// per element, `nn` threads. Renamed `_3dnc` for crate-wide kernel-name uniqueness.
    #[kernel]
    #[allow(clippy::too_many_arguments)]
    pub fn gradient_3dnc(
        d: &[f64], u: &[f64], met: &[f64], n1: u32,
        mut gx: DisjointSlice<f64>, mut gy: DisjointSlice<f64>, mut gz: DisjointSlice<f64>,
    ) {
        static mut DS: SharedArray<f64, NN_MAX> = SharedArray::UNINIT;
        static mut US: SharedArray<f64, NN_MAX> = SharedArray::UNINIT;
        let n1 = n1 as usize;
        let n2 = n1 * n1;
        let nn = n1 * n2;
        let e = thread::blockIdx_x() as usize;
        let m = thread::threadIdx_x() as usize;
        let b = e * nn + m;
        unsafe {
            DS[m] = d[m];
            US[m] = u[b];
        }
        thread::sync_threads();
        let i = m % n1;
        let j = (m / n1) % n1;
        let k = m / n2;
        let (mut ur, mut us, mut ut) = (0.0f64, 0.0f64, 0.0f64);
        let mut a = 0usize;
        while a < n1 {
            unsafe {
                ur += DS[i * n1 + a] * US[a + j * n1 + k * n2];
                us += DS[j * n1 + a] * US[i + a * n1 + k * n2];
                ut += DS[k * n1 + a] * US[i + j * n1 + a * n2];
            }
            a += 1;
        }
        let mo = b * 9;
        // physical grad = Jᵀ·(ur,us,ut): gx = rx·ur + sx·us + tx·ut, etc. met = [rx,ry,rz,sx,sy,sz,tx,ty,tz]
        let gxv = met[mo] * ur + met[mo + 3] * us + met[mo + 6] * ut;
        let gyv = met[mo + 1] * ur + met[mo + 4] * us + met[mo + 7] * ut;
        let gzv = met[mo + 2] * ur + met[mo + 5] * us + met[mo + 8] * ut;
        if let Some(o) = gx.get_mut(thread::index_1d()) {
            *o = gxv;
        }
        if let Some(o) = gy.get_mut(thread::index_1d()) {
            *o = gyv;
        }
        if let Some(o) = gz.get_mut(thread::index_1d()) {
            *o = gzv;
        }
    }

    /// 3D non-conforming SIPG operator. Conforming faces use `face_nbr` (per face node); FTC faces
    /// project the coarse trace onto this fine hex's nodes via `P_ha⊗P_hb`; CTF faces `Pᵀ`-gather
    /// the four fine quarter-faces back to the coarse test nodes. The coarse-edge normal (`enx/…`)
    /// is used on both sides (mirrors the CPU). `out` must be distinct from `u`.
    #[kernel]
    #[allow(clippy::too_many_arguments)]
    pub fn operator_3dnc(
        d: &[f64], u: &[f64], gx: &[f64], gy: &[f64], gz: &[f64], met: &[f64], jw: &[f64], n1: u32,
        // conforming-face data, `(e*6+t)*n2 + a`:
        face_vl: &[u32], face_nx: &[f64], face_ny: &[f64], face_nz: &[f64], face_sw: &[f64], face_nbr: &[u32],
        // per-face metadata, `e*6 + t`:
        fkind: &[u32], enx: &[f64], eny: &[f64], enz: &[f64], etau: &[f64], quad0: &[u32],
        // NC sorted data: self/coarse traces `(e*6+t)*n2 + k`, fine quarters `((e*6+t)*4 + q)*n2 + i`:
        self_sorted: &[u32], self_sw: &[f64], coarse_sorted: &[u32], finef_sorted: &[u32], finef_sw: &[f64],
        // 1D mortar matrices (n1*n1), `P[i*n1+j]`:
        p0: &[f64], p1: &[f64],
        lambda: f64, mut out: DisjointSlice<f64>,
    ) {
        static mut DS: SharedArray<f64, NN_MAX> = SharedArray::UNINIT;
        static mut PR: SharedArray<f64, NN_MAX> = SharedArray::UNINIT;
        static mut PS: SharedArray<f64, NN_MAX> = SharedArray::UNINIT;
        static mut PT: SharedArray<f64, NN_MAX> = SharedArray::UNINIT;
        static mut RF: SharedArray<f64, NN_MAX> = SharedArray::UNINIT;
        static mut HX: SharedArray<f64, NN_MAX> = SharedArray::UNINIT;
        static mut HY: SharedArray<f64, NN_MAX> = SharedArray::UNINIT;
        static mut HZ: SharedArray<f64, NN_MAX> = SharedArray::UNINIT;
        let n1 = n1 as usize;
        let n2 = n1 * n1;
        let nn = n1 * n2;
        let e = thread::blockIdx_x() as usize;
        let m = thread::threadIdx_x() as usize;
        let b = e * nn + m;
        let mo = b * 9;
        unsafe {
            DS[m] = d[m];
            RF[m] = 0.0;
            HX[m] = 0.0;
            HY[m] = 0.0;
            HZ[m] = 0.0;
            let wx = jw[b] * gx[b];
            let wy = jw[b] * gy[b];
            let wz = jw[b] * gz[b];
            PR[m] = met[mo] * wx + met[mo + 1] * wy + met[mo + 2] * wz;
            PS[m] = met[mo + 3] * wx + met[mo + 4] * wy + met[mo + 5] * wz;
            PT[m] = met[mo + 6] * wx + met[mo + 7] * wy + met[mo + 8] * wz;
        }
        thread::sync_threads();
        if m == 0 {
            let mut t = 0usize;
            while t < 6 {
                let ft = e * 6 + t;
                let kind = fkind[ft];
                let tau = etau[ft];
                if kind == K_CONF {
                    let mut a = 0usize;
                    while a < n2 {
                        let idx = ft * n2 + a;
                        let nbr = face_nbr[idx];
                        if nbr != NEU {
                            let vl = face_vl[idx] as usize;
                            let nx = face_nx[idx];
                            let ny = face_ny[idx];
                            let nz = face_nz[idx];
                            let sw = face_sw[idx];
                            let dun_e = nx * gx[e * nn + vl] + ny * gy[e * nn + vl] + nz * gz[e * nn + vl];
                            let ug = u[e * nn + vl];
                            let (avg, jump, gfac) = if nbr == BND {
                                (dun_e, ug, 1.0)
                            } else {
                                let ng = nbr as usize;
                                (0.5 * (dun_e + nx * gx[ng] + ny * gy[ng] + nz * gz[ng]), ug - u[ng], 0.5)
                            };
                            let g = gfac * sw * jump;
                            unsafe {
                                RF[vl] += -sw * avg + tau * sw * jump;
                                HX[vl] += g * nx;
                                HY[vl] += g * ny;
                                HZ[vl] += g * nz;
                            }
                        }
                        a += 1;
                    }
                } else if kind == K_FINE_TO_COARSE {
                    // This hex is FINE; project the coarse trace onto each fine node i=(ia,ib).
                    let fnx = enx[ft];
                    let fny = eny[ft];
                    let fnz = enz[ft];
                    let quad = quad0[ft] as usize;
                    let (ha, hb) = (quad % 2, quad / 2);
                    let mut i = 0usize;
                    while i < n2 {
                        let ia = i % n1;
                        let ib = i / n1;
                        let si = self_sorted[ft * n2 + i] as usize;
                        let sw = self_sw[ft * n2 + i];
                        let mut u_nbr = 0.0f64;
                        let mut dun_nbr = 0.0f64;
                        let mut jb = 0usize;
                        while jb < n1 {
                            let pb = if hb == 0 { p0[ib * n1 + jb] } else { p1[ib * n1 + jb] };
                            let mut ja = 0usize;
                            while ja < n1 {
                                let pa = if ha == 0 { p0[ia * n1 + ja] } else { p1[ia * n1 + ja] };
                                let ck = coarse_sorted[ft * n2 + (ja + jb * n1)] as usize;
                                let w = pa * pb;
                                u_nbr += w * u[ck];
                                dun_nbr += w * (fnx * gx[ck] + fny * gy[ck] + fnz * gz[ck]);
                                ja += 1;
                            }
                            jb += 1;
                        }
                        let u_self = u[e * nn + si];
                        let dun_self = fnx * gx[e * nn + si] + fny * gy[e * nn + si] + fnz * gz[e * nn + si];
                        let avg = 0.5 * (dun_self + dun_nbr);
                        let jump = u_self - u_nbr;
                        let g = 0.5 * sw * jump;
                        unsafe {
                            RF[si] += -sw * avg + tau * sw * jump;
                            HX[si] += g * fnx;
                            HY[si] += g * fny;
                            HZ[si] += g * fnz;
                        }
                        i += 1;
                    }
                } else {
                    // This hex is COARSE; integrate on each of the 4 fine quarter mortars and
                    // Pᵀ-gather to the coarse test nodes j=(ja,jb).
                    let fnx = enx[ft];
                    let fny = eny[ft];
                    let fnz = enz[ft];
                    let mut h = 0usize;
                    while h < 4 {
                        let (ha, hb) = (h % 2, h / 2);
                        let mut j = 0usize;
                        while j < n2 {
                            let ja = j % n1;
                            let jb = j / n1;
                            let cj = self_sorted[ft * n2 + j] as usize;
                            let mut rf_acc = 0.0f64;
                            let mut hl_acc = 0.0f64;
                            let mut i = 0usize;
                            while i < n2 {
                                let ia = i % n1;
                                let ib = i / n1;
                                // project coarse trace onto fine node i
                                let mut ucp = 0.0f64;
                                let mut dncp = 0.0f64;
                                let mut kb = 0usize;
                                while kb < n1 {
                                    let pb = if hb == 0 { p0[ib * n1 + kb] } else { p1[ib * n1 + kb] };
                                    let mut ka = 0usize;
                                    while ka < n1 {
                                        let pa = if ha == 0 { p0[ia * n1 + ka] } else { p1[ia * n1 + ka] };
                                        let ck = self_sorted[ft * n2 + (ka + kb * n1)] as usize;
                                        let w = pa * pb;
                                        ucp += w * u[e * nn + ck];
                                        dncp += w
                                            * (fnx * gx[e * nn + ck] + fny * gy[e * nn + ck] + fnz * gz[e * nn + ck]);
                                        ka += 1;
                                    }
                                    kb += 1;
                                }
                                let fk = finef_sorted[(ft * 4 + h) * n2 + i] as usize;
                                let swf = finef_sw[(ft * 4 + h) * n2 + i];
                                let uf = u[fk];
                                let dunf = fnx * gx[fk] + fny * gy[fk] + fnz * gz[fk];
                                let jump = ucp - uf;
                                let avg = 0.5 * (dncp + dunf);
                                let gc = -swf * avg + tau * swf * jump;
                                let gl = 0.5 * swf * jump;
                                let pa = if ha == 0 { p0[ia * n1 + ja] } else { p1[ia * n1 + ja] };
                                let pb = if hb == 0 { p0[ib * n1 + jb] } else { p1[ib * n1 + jb] };
                                let w = pa * pb;
                                rf_acc += w * gc;
                                hl_acc += w * gl;
                                i += 1;
                            }
                            unsafe {
                                RF[cj] += rf_acc;
                                HX[cj] += hl_acc * fnx;
                                HY[cj] += hl_acc * fny;
                                HZ[cj] += hl_acc * fnz;
                            }
                            j += 1;
                        }
                        h += 1;
                    }
                }
                t += 1;
            }
        }
        thread::sync_threads();
        let hxm = unsafe { HX[m] };
        let hym = unsafe { HY[m] };
        let hzm = unsafe { HZ[m] };
        unsafe {
            PR[m] -= met[mo] * hxm + met[mo + 1] * hym + met[mo + 2] * hzm;
            PS[m] -= met[mo + 3] * hxm + met[mo + 4] * hym + met[mo + 5] * hzm;
            PT[m] -= met[mo + 6] * hxm + met[mo + 7] * hym + met[mo + 8] * hzm;
        }
        thread::sync_threads();
        let i = m % n1;
        let j = (m / n1) % n1;
        let k = m / n2;
        let mut acc = 0.0f64;
        let mut a = 0usize;
        while a < n1 {
            unsafe {
                acc += DS[a * n1 + i] * PR[a + j * n1 + k * n2];
                acc += DS[a * n1 + j] * PS[i + a * n1 + k * n2];
                acc += DS[a * n1 + k] * PT[i + j * n1 + a * n2];
            }
            a += 1;
        }
        let rfm = unsafe { RF[m] };
        if let Some(o) = out.get_mut(thread::index_1d()) {
            *o = acc + rfm + lambda * jw[b] * u[b];
        }
    }

    /// y ← y + a·x  (CG vector op; `_3dnc` suffix for crate-wide kernel-name uniqueness)
    #[kernel]
    pub fn axpy_3dnc(mut y: DisjointSlice<f64>, x: &[f64], a: f64) {
        let idx = thread::index_1d();
        let i = idx.get();
        if let Some(o) = y.get_mut(idx) {
            *o += a * x[i];
        }
    }

    /// y ← x + b·y
    #[kernel]
    pub fn xpby_3dnc(mut y: DisjointSlice<f64>, x: &[f64], b: f64) {
        let idx = thread::index_1d();
        let i = idx.get();
        if let Some(o) = y.get_mut(idx) {
            *o = x[i] + b * *o;
        }
    }

    /// Multi-block grid-stride dot product (one partial per block; host sums them).
    #[kernel]
    pub fn dot_3dnc_partial(a: &[f64], b: &[f64], n: u64, mut partial: DisjointSlice<f64>) {
        static mut SH: SharedArray<f64, RED> = SharedArray::UNINIT;
        let tid = thread::threadIdx_x() as usize;
        let gstride = (thread::gridDim_x() * thread::blockDim_x()) as usize;
        let mut acc = 0.0f64;
        let mut i = (thread::blockIdx_x() * thread::blockDim_x()) as usize + tid;
        while i < n as usize {
            acc += a[i] * b[i];
            i += gstride;
        }
        unsafe { SH[tid] = acc; }
        thread::sync_threads();
        let mut s = thread::blockDim_x() as usize / 2;
        while s > 0 {
            if tid < s {
                unsafe { SH[tid] = SH[tid] + SH[tid + s]; }
            }
            thread::sync_threads();
            s /= 2;
        }
        if tid == 0 {
            unsafe { *partial.get_unchecked_mut(thread::blockIdx_x() as usize) = SH[0]; }
        }
    }
}

fn dot_blocks(ndof: usize) -> usize {
    ndof.div_ceil(RED).clamp(1, 1024)
}

/// Hex-face trace nodes in (a,b) tensor order — sorted by the two tangential coords (b major, a
/// minor), with a TOLERANT primary compare (the trilinear geometry gives within-row coords that
/// differ by FP rounding; an exact compare would scramble each row). Mirrors the CPU `sorted_face`.
fn sorted_face(mesh: &Mesh3d, e: usize, fc: Face) -> Vec<usize> {
    let fd = &mesh.elements[e].faces[fc as usize];
    let g = &mesh.elements[e].geom;
    let (a, b) = match fc.normal_axis() {
        0 => (1usize, 2usize),
        1 => (0, 2),
        _ => (0, 1),
    };
    let coord = |k: usize, ax: usize| {
        let nd = fd.nodes[k];
        [g.x[nd], g.y[nd], g.z[nd]][ax]
    };
    let tol = 1e-9;
    let mut idx: Vec<usize> = (0..fd.nodes.len()).collect();
    idx.sort_by(|&p, &q| {
        let (bp, bq) = (coord(p, b), coord(q, b));
        if (bp - bq).abs() > tol {
            bp.partial_cmp(&bq).unwrap()
        } else {
            coord(p, a).partial_cmp(&coord(q, a)).unwrap()
        }
    });
    idx
}

#[allow(clippy::type_complexity)]
struct NcArrays3 {
    nn: usize,
    ne: usize,
    ndof: usize,
    n1: u32,
    diff: Vec<f64>,
    met: Vec<f64>,
    jw: Vec<f64>,
    face_vl: Vec<u32>,
    face_nx: Vec<f64>,
    face_ny: Vec<f64>,
    face_nz: Vec<f64>,
    face_sw: Vec<f64>,
    face_nbr: Vec<u32>,
    fkind: Vec<u32>,
    enx: Vec<f64>,
    eny: Vec<f64>,
    enz: Vec<f64>,
    etau: Vec<f64>,
    quad0: Vec<u32>,
    self_sorted: Vec<u32>,
    self_sw: Vec<f64>,
    coarse_sorted: Vec<u32>,
    finef_sorted: Vec<u32>,
    finef_sw: Vec<f64>,
    p0: Vec<f64>,
    p1: Vec<f64>,
}

fn flatten3d_nc(mesh: &Mesh3d, alpha: f64, reaction_unused: f64, neumann_tags: &[u32]) -> NcArrays3 {
    let _ = reaction_unused;
    let nn = mesh.refh.n_nodes();
    let ne = mesh.n_elements();
    let ndof = ne * nn;
    let order = mesh.order;
    let n1 = (order + 1) as usize;
    let n2 = n1 * n1;
    assert!(nn <= NN_MAX, "p too large for NN_MAX={NN_MAX}");

    let mut met = vec![0.0; ndof * 9];
    let mut jw = vec![0.0; ndof];
    for (e, el) in mesh.elements.iter().enumerate() {
        for k in 0..nn {
            let g = &el.geom;
            let o = (e * nn + k) * 9;
            met[o] = g.rx[k];
            met[o + 1] = g.ry[k];
            met[o + 2] = g.rz[k];
            met[o + 3] = g.sx[k];
            met[o + 4] = g.sy[k];
            met[o + 5] = g.sz[k];
            met[o + 6] = g.tx[k];
            met[o + 7] = g.ty[k];
            met[o + 8] = g.tz[k];
            jw[e * nn + k] = g.jw[k];
        }
    }

    let p1f = (order + 1) as f64;
    let h: Vec<f64> = mesh.elements.iter().map(|el| el.geom.jw.iter().sum::<f64>().cbrt()).collect();

    // The two 1D mortar matrices P0=p_left, P1=p_right (RefineHex's hex-face mortar is their tensor
    // product). RefineQuad shares the identical 1D matrices, so reconstruct them as the columns
    // P[i*n1+j] = mortar_to_fine(e_j, half)[i] — the same construction the 2D GPU path uses.
    let mortar = RefineQuad::new(order);
    let (mut p0, mut p1) = (vec![0.0; n1 * n1], vec![0.0; n1 * n1]);
    for j in 0..n1 {
        let mut ej = vec![0.0; n1];
        ej[j] = 1.0;
        let col0 = mortar.mortar_to_fine(&ej, 0);
        let col1 = mortar.mortar_to_fine(&ej, 1);
        for i in 0..n1 {
            p0[i * n1 + j] = col0[i];
            p1[i * n1 + j] = col1[i];
        }
    }

    let nfc = ne * 6 * n2;
    let (mut face_vl, mut face_nx, mut face_ny, mut face_nz, mut face_sw, mut face_nbr) = (
        vec![0u32; nfc], vec![0.0; nfc], vec![0.0; nfc], vec![0.0; nfc], vec![0.0; nfc], vec![BND; nfc],
    );
    let (mut fkind, mut enx, mut eny, mut enz, mut etau, mut quad0) = (
        vec![K_CONF; ne * 6], vec![0.0; ne * 6], vec![0.0; ne * 6], vec![0.0; ne * 6], vec![0.0; ne * 6],
        vec![0u32; ne * 6],
    );
    let (mut self_sorted, mut self_sw, mut coarse_sorted) =
        (vec![0u32; nfc], vec![0.0; nfc], vec![0u32; nfc]);
    let (mut finef_sorted, mut finef_sw) = (vec![0u32; ne * 6 * 4 * n2], vec![0.0; ne * 6 * 4 * n2]);

    for (e, el) in mesh.elements.iter().enumerate() {
        for (t, face) in Face::ALL.iter().enumerate() {
            let ft = e * 6 + t;
            let fd = &el.faces[*face as usize];
            match &el.neighbors[*face as usize] {
                Neighbor3::Interior { elem: re, face: rface, perm } => {
                    fkind[ft] = K_CONF;
                    etau[ft] = alpha * p1f * p1f / h[e].min(h[*re]);
                    let rf = &mesh.elements[*re].faces[*rface as usize];
                    for a in 0..n2 {
                        let idx = ft * n2 + a;
                        face_vl[idx] = fd.nodes[a] as u32;
                        face_nx[idx] = fd.nx[a];
                        face_ny[idx] = fd.ny[a];
                        face_nz[idx] = fd.nz[a];
                        face_sw[idx] = fd.sw[a];
                        face_nbr[idx] = (*re * nn + rf.nodes[perm[a]]) as u32;
                    }
                }
                Neighbor3::Boundary { tag } => {
                    fkind[ft] = K_CONF;
                    etau[ft] = alpha * p1f * p1f / h[e];
                    let neu = neumann_tags.contains(tag);
                    for a in 0..n2 {
                        let idx = ft * n2 + a;
                        face_vl[idx] = fd.nodes[a] as u32;
                        face_nx[idx] = fd.nx[a];
                        face_ny[idx] = fd.ny[a];
                        face_nz[idx] = fd.nz[a];
                        face_sw[idx] = fd.sw[a];
                        face_nbr[idx] = if neu { NEU } else { BND };
                    }
                }
                Neighbor3::FineToCoarse { coarse, face: cface, quad } => {
                    fkind[ft] = K_FINE_TO_COARSE;
                    etau[ft] = alpha * p1f * p1f / h[e].min(h[*coarse]);
                    quad0[ft] = *quad as u32;
                    let es = sorted_face(mesh, e, *face);
                    enx[ft] = fd.nx[es[0]];
                    eny[ft] = fd.ny[es[0]];
                    enz[ft] = fd.nz[es[0]];
                    for k in 0..n2 {
                        self_sorted[ft * n2 + k] = fd.nodes[es[k]] as u32; // local node
                        self_sw[ft * n2 + k] = fd.sw[es[k]];
                    }
                    let cs = sorted_face(mesh, *coarse, *cface);
                    let cf = &mesh.elements[*coarse].faces[*cface as usize];
                    for k in 0..n2 {
                        coarse_sorted[ft * n2 + k] = (*coarse * nn + cf.nodes[cs[k]]) as u32; // global
                    }
                }
                Neighbor3::CoarseToFine { fine } => {
                    fkind[ft] = K_COARSE_TO_FINE;
                    let mut hmin = h[e];
                    for &(re, _) in fine.iter() {
                        hmin = hmin.min(h[re]);
                    }
                    etau[ft] = alpha * p1f * p1f / hmin;
                    let cs = sorted_face(mesh, e, *face);
                    enx[ft] = fd.nx[cs[0]];
                    eny[ft] = fd.ny[cs[0]];
                    enz[ft] = fd.nz[cs[0]];
                    for k in 0..n2 {
                        self_sorted[ft * n2 + k] = fd.nodes[cs[k]] as u32; // local node
                    }
                    for (q, &(re, rface)) in fine.iter().enumerate() {
                        let fs = sorted_face(mesh, re, rface);
                        let ff = &mesh.elements[re].faces[rface as usize];
                        for i in 0..n2 {
                            finef_sorted[(ft * 4 + q) * n2 + i] = (re * nn + ff.nodes[fs[i]]) as u32;
                            finef_sw[(ft * 4 + q) * n2 + i] = ff.sw[fs[i]];
                        }
                    }
                }
            }
        }
    }

    // The 1D diff matrix (n1²) padded to nn so the kernel's `DS[m]=d[m]` (m<nn) never OOBs; the
    // sum-fac only reads the first n1² entries (the real matrix).
    let mut diff = vec![0.0; nn];
    diff[..mesh.refh.line.diff.len()].copy_from_slice(&mesh.refh.line.diff);

    NcArrays3 {
        nn, ne, ndof, n1: n1 as u32, diff, met, jw, face_vl, face_nx,
        face_ny, face_nz, face_sw, face_nbr, fkind, enx, eny, enz, etau, quad0, self_sorted, self_sw,
        coarse_sorted, finef_sorted, finef_sw, p0, p1,
    }
}

/// Apply the matrix-free SIPG `(λM + A)·u` once on the GPU for a 3D 2:1 non-conforming hex mesh.
/// Bit-for-bit equal to `gale::dg::Poisson3d::with_bc(mesh, alpha, reaction, neumann).apply(u)`.
pub fn poisson3d_nc_apply(
    mesh: &Mesh3d,
    u: &[f64],
    alpha: f64,
    reaction: f64,
    neumann_tags: &[u32],
) -> Result<Vec<f64>, Box<dyn std::error::Error>> {
    let ma = flatten3d_nc(mesh, alpha, reaction, neumann_tags);
    assert_eq!(u.len(), ma.ndof, "state length must be n_elements·n_nodes");
    let ctx = CudaContext::new(0)?;
    let stream = ctx.default_stream();
    let up = |v: &[f64]| DeviceBuffer::from_host(&stream, v);
    let upu = |v: &[u32]| DeviceBuffer::from_host(&stream, v);

    let d_dev = up(&ma.diff)?;
    let u_dev = up(u)?;
    let met_dev = up(&ma.met)?;
    let jw_dev = up(&ma.jw)?;
    let face_vl = upu(&ma.face_vl)?;
    let face_nx = up(&ma.face_nx)?;
    let face_ny = up(&ma.face_ny)?;
    let face_nz = up(&ma.face_nz)?;
    let face_sw = up(&ma.face_sw)?;
    let face_nbr = upu(&ma.face_nbr)?;
    let fkind = upu(&ma.fkind)?;
    let enx = up(&ma.enx)?;
    let eny = up(&ma.eny)?;
    let enz = up(&ma.enz)?;
    let etau = up(&ma.etau)?;
    let quad0 = upu(&ma.quad0)?;
    let self_sorted = upu(&ma.self_sorted)?;
    let self_sw = up(&ma.self_sw)?;
    let coarse_sorted = upu(&ma.coarse_sorted)?;
    let finef_sorted = upu(&ma.finef_sorted)?;
    let finef_sw = up(&ma.finef_sw)?;
    let p0 = up(&ma.p0)?;
    let p1 = up(&ma.p1)?;
    let mut gx = DeviceBuffer::<f64>::zeroed(&stream, ma.ndof)?;
    let mut gy = DeviceBuffer::<f64>::zeroed(&stream, ma.ndof)?;
    let mut gz = DeviceBuffer::<f64>::zeroed(&stream, ma.ndof)?;
    let mut out = DeviceBuffer::<f64>::zeroed(&stream, ma.ndof)?;

    let module = kernels::load(&ctx)?;
    let nev = ma.ne as u32;
    let nnu = ma.nn as u32;
    let gcfg = LaunchConfig { grid_dim: (nev, 1, 1), block_dim: (nnu, 1, 1), shared_mem_bytes: 0 };
    let ocfg =
        LaunchConfig { grid_dim: (nev, 1, 1), block_dim: (nnu, 1, 1), shared_mem_bytes: 0 };
    module.gradient_3dnc(&stream, gcfg, &d_dev, &u_dev, &met_dev, ma.n1, &mut gx, &mut gy, &mut gz)?;
    module.operator_3dnc(
        &stream, ocfg, &d_dev, &u_dev, &gx, &gy, &gz, &met_dev, &jw_dev, ma.n1, &face_vl, &face_nx,
        &face_ny, &face_nz, &face_sw, &face_nbr, &fkind, &enx, &eny, &enz, &etau, &quad0, &self_sorted,
        &self_sw, &coarse_sorted, &finef_sorted, &finef_sw, &p0, &p1, reaction, &mut out,
    )?;
    Ok(out.to_host_vec(&stream)?)
}

/// Device-resident conjugate gradient for `(reaction·M + A)·x = b` on a 3D **2:1 non-conforming**
/// hex mesh, using the NC operator (`gradient_3dnc` → `operator_3dnc`) with the mortar coupling.
/// `deflate` removes the constant nullspace each iteration (singular pure-Neumann pressure). The
/// mesh is uploaded once; only CG vectors move per iteration (dot scalars read back per the
/// multi-block reduction). Mirrors `cg3d_impl` (conforming) and the 2D `poisson_nc` CG path.
#[allow(clippy::too_many_arguments)]
fn cg3d_nc_impl(
    mesh: &Mesh3d,
    b: &[f64],
    alpha: f64,
    reaction: f64,
    neumann_tags: &[u32],
    deflate: bool,
    tol: f64,
    maxit: usize,
) -> Result<(Vec<f64>, usize), Box<dyn std::error::Error>> {
    let ma = flatten3d_nc(mesh, alpha, reaction, neumann_tags);
    let ndof = ma.ndof;
    assert_eq!(b.len(), ndof, "rhs length must be n_elements·n_nodes");
    let ctx = CudaContext::new(0)?;
    let stream = ctx.default_stream();
    let up = |v: &[f64]| DeviceBuffer::from_host(&stream, v);
    let upu = |v: &[u32]| DeviceBuffer::from_host(&stream, v);

    // Constant NC mesh arrays (uploaded once).
    let d_dev = up(&ma.diff)?;
    let met_dev = up(&ma.met)?;
    let jw_dev = up(&ma.jw)?;
    let face_vl = upu(&ma.face_vl)?;
    let face_nx = up(&ma.face_nx)?;
    let face_ny = up(&ma.face_ny)?;
    let face_nz = up(&ma.face_nz)?;
    let face_sw = up(&ma.face_sw)?;
    let face_nbr = upu(&ma.face_nbr)?;
    let fkind = upu(&ma.fkind)?;
    let enx = up(&ma.enx)?;
    let eny = up(&ma.eny)?;
    let enz = up(&ma.enz)?;
    let etau = up(&ma.etau)?;
    let quad0 = upu(&ma.quad0)?;
    let self_sorted = upu(&ma.self_sorted)?;
    let self_sw = up(&ma.self_sw)?;
    let coarse_sorted = upu(&ma.coarse_sorted)?;
    let finef_sorted = upu(&ma.finef_sorted)?;
    let finef_sw = up(&ma.finef_sw)?;
    let p0 = up(&ma.p0)?;
    let p1 = up(&ma.p1)?;

    // CG vectors.
    let mut x = DeviceBuffer::<f64>::zeroed(&stream, ndof)?;
    let mut r = up(b)?;
    let mut p = DeviceBuffer::<f64>::zeroed(&stream, ndof)?;
    let mut ap = DeviceBuffer::<f64>::zeroed(&stream, ndof)?;
    let mut gx = DeviceBuffer::<f64>::zeroed(&stream, ndof)?;
    let mut gy = DeviceBuffer::<f64>::zeroed(&stream, ndof)?;
    let mut gz = DeviceBuffer::<f64>::zeroed(&stream, ndof)?;
    let nb = dot_blocks(ndof);
    let mut partial = DeviceBuffer::<f64>::zeroed(&stream, nb)?;
    let ones = up(&vec![1.0f64; ndof])?;

    let module = kernels::load(&ctx)?;
    let nev = ma.ne as u32;
    let nnu = ma.nn as u32;
    let gcfg = LaunchConfig { grid_dim: (nev, 1, 1), block_dim: (nnu, 1, 1), shared_mem_bytes: 0 };
    let ocfg = LaunchConfig { grid_dim: (nev, 1, 1), block_dim: (nnu, 1, 1), shared_mem_bytes: 0 };
    let red = LaunchConfig { grid_dim: (nb as u32, 1, 1), block_dim: (RED as u32, 1, 1), shared_mem_bytes: 0 };
    let vec_cfg = LaunchConfig::for_num_elems(ndof as u32);
    let n64 = ndof as u64;
    let ninv = 1.0 / ndof as f64;

    macro_rules! dot {
        ($a:expr, $b:expr) => {{
            module.dot_3dnc_partial(&stream, red, $a, $b, n64, &mut partial)?;
            partial.to_host_vec(&stream)?.iter().sum::<f64>()
        }};
    }
    macro_rules! deflate {
        ($v:expr) => {{
            if deflate {
                let mean = dot!($v, &ones) * ninv;
                module.axpy_3dnc(&stream, vec_cfg, $v, &ones, -mean)?;
            }
        }};
    }
    macro_rules! apply {
        ($field:expr, $dst:expr) => {{
            module.gradient_3dnc(&stream, gcfg, &d_dev, $field, &met_dev, ma.n1, &mut gx, &mut gy, &mut gz)?;
            module.operator_3dnc(
                &stream, ocfg, &d_dev, $field, &gx, &gy, &gz, &met_dev, &jw_dev, ma.n1, &face_vl,
                &face_nx, &face_ny, &face_nz, &face_sw, &face_nbr, &fkind, &enx, &eny, &enz, &etau,
                &quad0, &self_sorted, &self_sw, &coarse_sorted, &finef_sorted, &finef_sw, &p0, &p1,
                reaction, $dst,
            )?;
        }};
    }

    deflate!(&mut r);
    module.xpby_3dnc(&stream, vec_cfg, &mut p, &r, 0.0)?; // p = r
    let bn = dot!(&r, &r).sqrt().max(1e-300);
    let mut rs = dot!(&r, &r);
    let mut iters = 0;
    for it in 0..maxit {
        apply!(&p, &mut ap);
        let pap = dot!(&p, &ap);
        let alpha_cg = rs / pap;
        module.axpy_3dnc(&stream, vec_cfg, &mut x, &p, alpha_cg)?;
        module.axpy_3dnc(&stream, vec_cfg, &mut r, &ap, -alpha_cg)?;
        deflate!(&mut r);
        let rs_new = dot!(&r, &r);
        iters = it + 1;
        if rs_new.sqrt() / bn < tol {
            break;
        }
        let beta = rs_new / rs;
        module.xpby_3dnc(&stream, vec_cfg, &mut p, &r, beta)?;
        rs = rs_new;
    }
    Ok((x.to_host_vec(&stream)?, iters))
}

/// Tag-aware 3D NC CG for `(reaction·M + A)·x = b` (Dirichlet except `neumann_tags`; non-deflated) —
/// the velocity-Helmholtz / outflow-pinned-pressure path on a non-conforming hex mesh.
pub fn helmholtz3d_nc_cg_solve_tags(
    mesh: &Mesh3d,
    b: &[f64],
    alpha: f64,
    reaction: f64,
    neumann_tags: &[u32],
    tol: f64,
    maxit: usize,
) -> Result<(Vec<f64>, usize), Box<dyn std::error::Error>> {
    cg3d_nc_impl(mesh, b, alpha, reaction, neumann_tags, false, tol, maxit)
}

/// Deflated 3D NC CG for the singular pure-Neumann pressure-Poisson `A·x = b` on a non-conforming
/// hex mesh (constant nullspace removed each iteration).
pub fn pressure3d_nc_cg_solve(
    mesh: &Mesh3d,
    b: &[f64],
    alpha: f64,
    tol: f64,
    maxit: usize,
) -> Result<(Vec<f64>, usize), Box<dyn std::error::Error>> {
    cg3d_nc_impl(mesh, b, alpha, 0.0, &mesh.boundary_tags(), true, tol, maxit)
}
