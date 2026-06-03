//! `gale-gpu` — GPU device kernels + host launch wrappers for gale's DG operators,
//! as **reusable library components** (not one-off binaries).
//!
//! This crate carries the `#[cuda_module]` device code, so it must be built with
//! the cuda-oxide codegen backend (`cargo oxide`). It depends on the pure-host
//! `gale` lib for mesh/operator types; `gale` does not depend on it, so `gale`'s
//! `cargo test` remains a pure-host CPU oracle. Each GPU operator here is validated
//! bit-for-bit against that oracle.
//!
//! Reusing a `#[cuda_module]` from a *dependency library crate* relies on the fork
//! fix that anchors the embedded artifact object so the linker retains it across
//! the rlib boundary (see `docs/cuda-oxide-codegen-notes.md` §3 / fork commit
//! `02129b8`).

use cuda_core::{CudaContext, DeviceBuffer, LaunchConfig};
use cuda_device::{kernel, thread, DisjointSlice, SharedArray};
use cuda_host::cuda_module;
use gale::dg::{Edge, Mesh2d, Neighbor};

const NN_MAX: usize = 81; // (p+1)² up to p=8

#[cuda_module]
mod kernels {
    use super::*;

    /// Weak-form linear-advection RHS, Rusanov flux. `u` is the global state;
    /// `face_nbr` indexes into it. `e = blockIdx.x`, node `m = threadIdx.x`.
    #[kernel]
    #[allow(clippy::too_many_arguments)]
    pub fn advect_rhs(
        d: &[f64], u: &[f64], rx: &[f64], ry: &[f64], sx: &[f64], sy: &[f64], jw: &[f64],
        ax: f64, ay: f64, n1: u32,
        face_vl: &[u32], face_nx: &[f64], face_ny: &[f64], face_sw: &[f64], face_nbr: &[u32],
        mut out: DisjointSlice<f64>,
    ) {
        static mut DS: SharedArray<f64, NN_MAX> = SharedArray::UNINIT;
        static mut PR: SharedArray<f64, NN_MAX> = SharedArray::UNINIT;
        static mut PS: SharedArray<f64, NN_MAX> = SharedArray::UNINIT;
        static mut RFACE: SharedArray<f64, NN_MAX> = SharedArray::UNINIT;
        let n1 = n1 as usize;
        let nn = n1 * n1;
        let e = thread::blockIdx_x() as usize;
        let m = thread::threadIdx_x() as usize;
        let b = e * nn + m;
        unsafe {
            DS[m] = d[m];
            RFACE[m] = 0.0;
            let um = u[b];
            let wfx = jw[b] * ax * um;
            let wfy = jw[b] * ay * um;
            PR[m] = rx[b] * wfx + ry[b] * wfy;
            PS[m] = sx[b] * wfx + sy[b] * wfy;
        }
        thread::sync_threads();
        if m == 0 {
            let mut t = 0usize;
            while t < 4 {
                let mut a = 0usize;
                while a < n1 {
                    let idx = (e * 4 + t) * n1 + a;
                    let vl = face_vl[idx] as usize;
                    let nx = face_nx[idx];
                    let ny = face_ny[idx];
                    let sw = face_sw[idx];
                    let um = u[e * nn + vl];
                    let up = u[face_nbr[idx] as usize];
                    let an = ax * nx + ay * ny;
                    let fstar = 0.5 * an * (um + up) - 0.5 * an.abs() * (up - um);
                    unsafe {
                        RFACE[vl] += sw * fstar;
                    }
                    a += 1;
                }
                t += 1;
            }
        }
        thread::sync_threads();
        let i = m % n1;
        let j = m / n1;
        let mut vol = 0.0f64;
        let mut k = 0usize;
        while k < n1 {
            unsafe {
                vol += DS[k * n1 + i] * PR[k + j * n1] + DS[k * n1 + j] * PS[i + k * n1];
            }
            k += 1;
        }
        let rf = unsafe { RFACE[m] };
        if let Some(o) = out.get_mut(thread::index_1d()) {
            *o = (vol - rf) / jw[b];
        }
    }
}

/// Compute the linear-advection weak-form RHS `∂ₜu = M⁻¹L(u)` on the GPU for a
/// **periodic** quad mesh, returning the nodal time-derivative. Reusable host
/// wrapper around the [`kernels::advect_rhs`] device kernel: flattens the mesh
/// metrics + face connectivity, uploads, launches one block per element, and
/// gathers the result. Bit-for-bit equal to `gale::dg::Hyperbolic` advection.
///
/// Panics if the mesh has boundary faces (this PoC kernel handles interior faces
/// only — use a periodic mesh).
pub fn advection_rhs(
    mesh: &Mesh2d,
    u: &[f64],
    ax: f64,
    ay: f64,
) -> Result<Vec<f64>, Box<dyn std::error::Error>> {
    let nn = mesh.refq.n_nodes();
    let ne = mesh.n_elements();
    let ndof = ne * nn;
    let n1 = mesh.order + 1;
    assert_eq!(u.len(), ndof, "state length must be n_elements·n_nodes");
    assert!(nn <= NN_MAX, "p too large for NN_MAX={NN_MAX}");

    // Flatten per-node metrics.
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

    // Flatten face connectivity (interior faces only).
    let nfc = ne * 4 * n1;
    let (mut fvl, mut fnx, mut fny, mut fsw, mut fnbr) =
        (vec![0u32; nfc], vec![0.0; nfc], vec![0.0; nfc], vec![0.0; nfc], vec![0u32; nfc]);
    for (e, el) in mesh.elements.iter().enumerate() {
        for (t, edge) in Edge::ALL.iter().enumerate() {
            let face = &el.faces[*edge as usize];
            let Neighbor::Interior { elem: re, edge: redge, perm } = &el.neighbors[*edge as usize]
            else {
                panic!("advection_rhs: mesh has a boundary face (use a periodic mesh)");
            };
            let rf = &mesh.elements[*re].faces[*redge as usize];
            for a in 0..n1 {
                let idx = (e * 4 + t) * n1 + a;
                fvl[idx] = face.nodes[a] as u32;
                fnx[idx] = face.nx[a];
                fny[idx] = face.ny[a];
                fsw[idx] = face.sw[a];
                fnbr[idx] = (*re * nn + rf.nodes[perm[a]]) as u32;
            }
        }
    }

    let ctx = CudaContext::new(0)?;
    let stream = ctx.default_stream();
    let up = |v: &[f64]| DeviceBuffer::from_host(&stream, v);
    let upu = |v: &[u32]| DeviceBuffer::from_host(&stream, v);
    let d_dev = up(&mesh.refq.line.diff)?;
    let u_dev = up(u)?;
    let rx_dev = up(&rx)?;
    let ry_dev = up(&ry)?;
    let sx_dev = up(&sx)?;
    let sy_dev = up(&sy)?;
    let jw_dev = up(&jw)?;
    let fvl_dev = upu(&fvl)?;
    let fnx_dev = up(&fnx)?;
    let fny_dev = up(&fny)?;
    let fsw_dev = up(&fsw)?;
    let fnbr_dev = upu(&fnbr)?;
    let mut out_dev = DeviceBuffer::<f64>::zeroed(&stream, ndof)?;

    let module = kernels::load(&ctx)?;
    let cfg = LaunchConfig { grid_dim: (ne as u32, 1, 1), block_dim: (nn as u32, 1, 1), shared_mem_bytes: 0 };
    module.advect_rhs(
        &stream, cfg, &d_dev, &u_dev, &rx_dev, &ry_dev, &sx_dev, &sy_dev, &jw_dev, ax, ay,
        n1 as u32, &fvl_dev, &fnx_dev, &fny_dev, &fsw_dev, &fnbr_dev, &mut out_dev,
    )?;
    Ok(out_dev.to_host_vec(&stream)?)
}
