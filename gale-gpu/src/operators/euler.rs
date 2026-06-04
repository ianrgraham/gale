//! GPU compressible-Euler (weak-form DG, nv=4) operator — reusable library component.
//!
//! Carries the `#[cuda_module]` device kernel ([`kernels::euler_rhs`]) plus a host
//! launch wrapper ([`euler_rhs`]). The wave speed needs in-kernel `sqrt(γp/ρ)` and
//! `abs` (the libdevice / NVVM-text path), so this operator exercises the exporter
//! fix that anchors libdevice math. Validated bit-for-bit against
//! `gale::dg::Hyperbolic` (Euler, weak form, Rusanov / LLF flux).
//!
//! State (SoA): `u0=ρ`, `u1=ρu`, `u2=ρv`, `u3=E`. Periodic mesh ⇒ no boundary flux.

use cuda_core::{CudaContext, DeviceBuffer, LaunchConfig};
use cuda_device::{kernel, thread, DisjointSlice, SharedArray};
use cuda_host::cuda_module;
use gale::dg::{Edge, Mesh2d, Neighbor};

const NN_MAX: usize = 81; // (p+1)² up to p=8

#[cuda_module]
mod kernels {
    use super::*;

    /// `∂ₜu = M⁻¹[Dxᵀ(W Fx)+Dyᵀ(W Fy) − ∮ F*·n]`, Euler + Rusanov, one element/block.
    ///
    /// Named `euler_rhs` and unique crate-wide: kernel export names share a single
    /// device bundle in cuda-oxide, so they must not clash with other operators'
    /// kernels (cf. `advect2d_rhs`). The wave-speed `sqrt`/`abs` go through libdevice.
    #[kernel]
    #[allow(clippy::too_many_arguments)]
    pub fn euler_rhs(
        d: &[f64],
        u0: &[f64], u1: &[f64], u2: &[f64], u3: &[f64],
        rx: &[f64], ry: &[f64], sx: &[f64], sy: &[f64], jw: &[f64],
        gamma: f64, n1: u32,
        face_vl: &[u32], face_nx: &[f64], face_ny: &[f64], face_sw: &[f64], face_nbr: &[u32],
        mut o0: DisjointSlice<f64>, mut o1: DisjointSlice<f64>,
        mut o2: DisjointSlice<f64>, mut o3: DisjointSlice<f64>,
    ) {
        static mut DS: SharedArray<f64, NN_MAX> = SharedArray::UNINIT;
        static mut PR0: SharedArray<f64, NN_MAX> = SharedArray::UNINIT;
        static mut PR1: SharedArray<f64, NN_MAX> = SharedArray::UNINIT;
        static mut PR2: SharedArray<f64, NN_MAX> = SharedArray::UNINIT;
        static mut PR3: SharedArray<f64, NN_MAX> = SharedArray::UNINIT;
        static mut PS0: SharedArray<f64, NN_MAX> = SharedArray::UNINIT;
        static mut PS1: SharedArray<f64, NN_MAX> = SharedArray::UNINIT;
        static mut PS2: SharedArray<f64, NN_MAX> = SharedArray::UNINIT;
        static mut PS3: SharedArray<f64, NN_MAX> = SharedArray::UNINIT;
        static mut RF0: SharedArray<f64, NN_MAX> = SharedArray::UNINIT;
        static mut RF1: SharedArray<f64, NN_MAX> = SharedArray::UNINIT;
        static mut RF2: SharedArray<f64, NN_MAX> = SharedArray::UNINIT;
        static mut RF3: SharedArray<f64, NN_MAX> = SharedArray::UNINIT;
        let n1 = n1 as usize;
        let nn = n1 * n1;
        let e = thread::blockIdx_x() as usize;
        let m = thread::threadIdx_x() as usize;
        let b = e * nn + m;
        let g1 = gamma - 1.0;

        // Flux of the local state → contravariant volume sources.
        unsafe {
            DS[m] = d[m];
            RF0[m] = 0.0;
            RF1[m] = 0.0;
            RF2[m] = 0.0;
            RF3[m] = 0.0;
            let (r, mx, my, en) = (u0[b], u1[b], u2[b], u3[b]);
            let (vx, vy) = (mx / r, my / r);
            let p = g1 * (en - 0.5 * r * (vx * vx + vy * vy));
            // Fx, Fy.
            let fx0 = mx;
            let fx1 = mx * vx + p;
            let fx2 = mx * vy;
            let fx3 = vx * (en + p);
            let fy0 = my;
            let fy1 = my * vx;
            let fy2 = my * vy + p;
            let fy3 = vy * (en + p);
            let w = jw[b];
            let (rxw, ryw, sxw, syw) = (rx[b] * w, ry[b] * w, sx[b] * w, sy[b] * w);
            PR0[m] = rxw * fx0 + ryw * fy0;
            PR1[m] = rxw * fx1 + ryw * fy1;
            PR2[m] = rxw * fx2 + ryw * fy2;
            PR3[m] = rxw * fx3 + ryw * fy3;
            PS0[m] = sxw * fx0 + syw * fy0;
            PS1[m] = sxw * fx1 + syw * fy1;
            PS2[m] = sxw * fx2 + syw * fy2;
            PS3[m] = sxw * fx3 + syw * fy3;
        }
        thread::sync_threads();

        // Rusanov surface flux (serial on thread 0): RFACE[vl] += sw·F*·n.
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
                    let gi = e * nn + vl;
                    let ng = face_nbr[idx] as usize;
                    // Both states.
                    let (rm, mxm, mym, enm) = (u0[gi], u1[gi], u2[gi], u3[gi]);
                    let (rp, mxp, myp, enp) = (u0[ng], u1[ng], u2[ng], u3[ng]);
                    let (vxm, vym) = (mxm / rm, mym / rm);
                    let (vxp, vyp) = (mxp / rp, myp / rp);
                    let pm = g1 * (enm - 0.5 * rm * (vxm * vxm + vym * vym));
                    let pp = g1 * (enp - 0.5 * rp * (vxp * vxp + vyp * vyp));
                    // Normal fluxes F·n for both states.
                    let fnm0 = mxm * nx + mym * ny;
                    let fnm1 = (mxm * vxm + pm) * nx + (mxm * vym) * ny;
                    let fnm2 = (mym * vxm) * nx + (mym * vym + pm) * ny;
                    let fnm3 = (vxm * (enm + pm)) * nx + (vym * (enm + pm)) * ny;
                    let fnp0 = mxp * nx + myp * ny;
                    let fnp1 = (mxp * vxp + pp) * nx + (mxp * vyp) * ny;
                    let fnp2 = (myp * vxp) * nx + (myp * vyp + pp) * ny;
                    let fnp3 = (vxp * (enp + pp)) * nx + (vyp * (enp + pp)) * ny;
                    // Max wave speed: |V·n| + c, c = sqrt(γ p/ρ)  (libdevice sqrt + abs).
                    let vnm = vxm * nx + vym * ny;
                    let vnp = vxp * nx + vyp * ny;
                    let cm = (gamma * pm / rm).sqrt();
                    let cp = (gamma * pp / rp).sqrt();
                    let lam_m = vnm.abs() + cm;
                    let lam_p = vnp.abs() + cp;
                    let lam = if lam_m > lam_p { lam_m } else { lam_p };
                    let inv_jw = sw / jw[gi];
                    unsafe {
                        RF0[vl] += inv_jw * (0.5 * (fnm0 + fnp0) - 0.5 * lam * (rp - rm));
                        RF1[vl] += inv_jw * (0.5 * (fnm1 + fnp1) - 0.5 * lam * (mxp - mxm));
                        RF2[vl] += inv_jw * (0.5 * (fnm2 + fnp2) - 0.5 * lam * (myp - mym));
                        RF3[vl] += inv_jw * (0.5 * (fnm3 + fnp3) - 0.5 * lam * (enp - enm));
                    }
                    a += 1;
                }
                t += 1;
            }
        }
        thread::sync_threads();

        // Volume transpose + surface, divided by the mass.
        let i = m % n1;
        let j = m / n1;
        let (mut v0, mut v1, mut v2, mut v3) = (0.0f64, 0.0f64, 0.0f64, 0.0f64);
        let mut k = 0usize;
        while k < n1 {
            let dri = unsafe { DS[k * n1 + i] };
            let dsj = unsafe { DS[k * n1 + j] };
            let cr = k + j * n1;
            let cs = i + k * n1;
            unsafe {
                v0 += dri * PR0[cr] + dsj * PS0[cs];
                v1 += dri * PR1[cr] + dsj * PS1[cs];
                v2 += dri * PR2[cr] + dsj * PS2[cs];
                v3 += dri * PR3[cr] + dsj * PS3[cs];
            }
            k += 1;
        }
        let w = jw[b];
        if let Some(o) = o0.get_mut(thread::index_1d()) {
            *o = v0 / w - unsafe { RF0[m] };
        }
        if let Some(o) = o1.get_mut(thread::index_1d()) {
            *o = v1 / w - unsafe { RF1[m] };
        }
        if let Some(o) = o2.get_mut(thread::index_1d()) {
            *o = v2 / w - unsafe { RF2[m] };
        }
        if let Some(o) = o3.get_mut(thread::index_1d()) {
            *o = v3 / w - unsafe { RF3[m] };
        }
    }
}

/// Compute the compressible-Euler weak-form RHS `∂ₜu = M⁻¹L(u)` on the GPU for a
/// **periodic** quad mesh, returning the nodal time-derivative of each conserved
/// component. Reusable host wrapper around the [`kernels::euler_rhs`] device kernel:
/// flattens the mesh metrics + face connectivity, uploads, launches one block per
/// element, and gathers the four component results. Bit-for-bit equal to
/// `gale::dg::Hyperbolic` (Euler, weak form, Rusanov).
///
/// `state` is the SoA conserved state `[ρ, ρu, ρv, E]`, each of length
/// `n_elements·n_nodes`; the returned `Vec<Vec<f64>>` preserves that 4-component
/// layout.
///
/// Panics if the mesh has boundary faces (this PoC kernel handles interior faces
/// only — use a periodic mesh).
pub fn euler_rhs(
    mesh: &Mesh2d,
    state: &[Vec<f64>],
    gamma: f64,
) -> Result<Vec<Vec<f64>>, Box<dyn std::error::Error>> {
    let nn = mesh.refq.n_nodes();
    let ne = mesh.n_elements();
    let ndof = ne * nn;
    let n1 = mesh.order + 1;
    assert_eq!(state.len(), 4, "Euler state must have 4 components [ρ, ρu, ρv, E]");
    for c in state {
        assert_eq!(c.len(), ndof, "state length must be n_elements·n_nodes");
    }
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
                panic!("euler_rhs: mesh has a boundary face (use a periodic mesh)");
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
    let u0 = up(&state[0])?;
    let u1 = up(&state[1])?;
    let u2 = up(&state[2])?;
    let u3 = up(&state[3])?;
    let rxd = up(&rx)?;
    let ryd = up(&ry)?;
    let sxd = up(&sx)?;
    let syd = up(&sy)?;
    let jwd = up(&jw)?;
    let fvld = upu(&fvl)?;
    let fnxd = up(&fnx)?;
    let fnyd = up(&fny)?;
    let fswd = up(&fsw)?;
    let fnbrd = upu(&fnbr)?;
    let mut d0 = DeviceBuffer::<f64>::zeroed(&stream, ndof)?;
    let mut d1 = DeviceBuffer::<f64>::zeroed(&stream, ndof)?;
    let mut d2 = DeviceBuffer::<f64>::zeroed(&stream, ndof)?;
    let mut d3 = DeviceBuffer::<f64>::zeroed(&stream, ndof)?;

    let module = kernels::load(&ctx)?;
    let cfg = LaunchConfig { grid_dim: (ne as u32, 1, 1), block_dim: (nn as u32, 1, 1), shared_mem_bytes: 0 };
    module.euler_rhs(
        &stream, cfg, &d_dev, &u0, &u1, &u2, &u3, &rxd, &ryd, &sxd, &syd, &jwd, gamma,
        n1 as u32, &fvld, &fnxd, &fnyd, &fswd, &fnbrd, &mut d0, &mut d1, &mut d2, &mut d3,
    )?;
    Ok(vec![
        d0.to_host_vec(&stream)?,
        d1.to_host_vec(&stream)?,
        d2.to_host_vec(&stream)?,
        d3.to_host_vec(&stream)?,
    ])
}
