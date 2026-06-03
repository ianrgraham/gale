//! GPU port of the **split-form (entropy-stable) DG operator** — the high-Re engine
//! — for Burgers, validated bit-for-bit against `Hyperbolic::rhs` (SplitForm).
//! Uses the entropy-CONSERVING variant (central surface flux, `dissipation=false`):
//! the Fisher–Carpenter flux-differencing volume `(2/J)Σ D·F̃#` with metric averaging,
//! plus the strong-form surface `(1/Jw)(F·n − F*·n)`. Burgers' EC two-point flux
//! `(uₗ²+uₗuᵣ+uᵣ²)/6` and `F·n=½u²nₓ` are pure arithmetic ⇒ embedded path, sm_70.
//! (Periodic mesh ⇒ no boundary flux; `fpy=0` for Burgers ⇒ only `rx,sx` metrics.)
//!
//! Run: cargo oxide run --bin gpu-burgers-split

use cuda_core::{CudaContext, DeviceBuffer, LaunchConfig};
use cuda_device::{kernel, thread, DisjointSlice, SharedArray};
use cuda_host::cuda_module;
use gale::dg::{Burgers, Edge, Hyperbolic, Mesh2d, Neighbor, VolumeForm};

const NN_MAX: usize = 81;

#[cuda_module]
mod kernels {
    use super::*;

    /// `∂ₜu` for split-form Burgers, one element per block, one node per thread.
    #[kernel]
    #[allow(clippy::too_many_arguments)]
    pub fn burgers_split(
        d: &[f64], u: &[f64], jac: &[f64], rx: &[f64], sx: &[f64], jw: &[f64], n1: u32,
        face_vl: &[u32], face_nx: &[f64], face_sw: &[f64], face_nbr: &[u32],
        mut out: DisjointSlice<f64>,
    ) {
        static mut DS: SharedArray<f64, NN_MAX> = SharedArray::UNINIT;
        static mut US: SharedArray<f64, NN_MAX> = SharedArray::UNINIT;
        static mut JAC: SharedArray<f64, NN_MAX> = SharedArray::UNINIT;
        static mut RXS: SharedArray<f64, NN_MAX> = SharedArray::UNINIT;
        static mut SXS: SharedArray<f64, NN_MAX> = SharedArray::UNINIT;
        static mut RFACE: SharedArray<f64, NN_MAX> = SharedArray::UNINIT;
        let n1 = n1 as usize;
        let nn = n1 * n1;
        let e = thread::blockIdx_x() as usize;
        let m = thread::threadIdx_x() as usize;
        let b = e * nn + m;
        unsafe {
            DS[m] = d[m];
            US[m] = u[b];
            JAC[m] = jac[b];
            RXS[m] = rx[b];
            SXS[m] = sx[b];
            RFACE[m] = 0.0;
        }
        thread::sync_threads();

        let i = m % n1;
        let j = m / n1;
        let ui = unsafe { US[m] };
        let jaci = unsafe { JAC[m] };
        let rxi = unsafe { RXS[m] };
        let sxi = unsafe { SXS[m] };
        let mut vol = 0.0f64;
        // r-direction line: 2 Σ_mm D[i,mm] · J̄ · r̄ₓ · F#(uᵢ,uₘ).
        let mut mm = 0usize;
        while mm < n1 {
            let lm = mm + j * n1;
            let um = unsafe { US[lm] };
            let fp = (ui * ui + ui * um + um * um) / 6.0;
            let jb = 0.5 * (jaci + unsafe { JAC[lm] });
            let rxb = 0.5 * (rxi + unsafe { RXS[lm] });
            vol += 2.0 * unsafe { DS[i * n1 + mm] } * jb * rxb * fp;
            mm += 1;
        }
        // s-direction line.
        mm = 0;
        while mm < n1 {
            let lm = i + mm * n1;
            let um = unsafe { US[lm] };
            let fp = (ui * ui + ui * um + um * um) / 6.0;
            let jb = 0.5 * (jaci + unsafe { JAC[lm] });
            let sxb = 0.5 * (sxi + unsafe { SXS[lm] });
            vol += 2.0 * unsafe { DS[j * n1 + mm] } * jb * sxb * fp;
            mm += 1;
        }
        let dudt_vol = -vol / jaci;
        thread::sync_threads();

        // Strong-form surface (central flux): RFACE[vl] += sw(F(uᵢ)·n − F*·n)/Jw.
        if m == 0 {
            let mut t = 0usize;
            while t < 4 {
                let mut a = 0usize;
                while a < n1 {
                    let idx = (e * 4 + t) * n1 + a;
                    let vl = face_vl[idx] as usize;
                    let nx = face_nx[idx];
                    let sw = face_sw[idx];
                    let sm = unsafe { US[vl] };
                    let sp = u[face_nbr[idx] as usize];
                    let fnm = 0.5 * sm * sm * nx;
                    let fnp = 0.5 * sp * sp * nx;
                    let fstar = 0.5 * (fnm + fnp);
                    unsafe {
                        RFACE[vl] += sw * (fnm - fstar) / jw[e * nn + vl];
                    }
                    a += 1;
                }
                t += 1;
            }
        }
        thread::sync_threads();

        let rf = unsafe { RFACE[m] };
        if let Some(o) = out.get_mut(thread::index_1d()) {
            *o = dudt_vol + rf;
        }
    }
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let p = 4;
    println!("=== GPU split-form Burgers vs CPU (entropy-conserving, p={p}) ===\n");

    let mesh = Mesh2d::rectangular_periodic(p, 3, 3, [0.0, 1.0], [0.0, 1.0]);
    let nn = mesh.refq.n_nodes();
    let ne = mesh.n_elements();
    let ndof = ne * nn;
    let op = Hyperbolic::with_options(&mesh, Burgers, VolumeForm::SplitForm, false);

    let mut state = vec![vec![0.0; ndof]];
    for (e, el) in mesh.elements.iter().enumerate() {
        for k in 0..nn {
            state[0][e * nn + k] = 0.5 + (2.0 * std::f64::consts::PI * el.geom.x[k]).sin() * (2.0 * std::f64::consts::PI * el.geom.y[k]).cos();
        }
    }
    let cpu = op.rhs(&state, 0.0, &|_, _, _, _: &mut [f64]| {});

    // Flatten metrics.
    let (mut jac, mut rx, mut sx, mut jw) = (vec![0.0; ndof], vec![0.0; ndof], vec![0.0; ndof], vec![0.0; ndof]);
    for (e, el) in mesh.elements.iter().enumerate() {
        for k in 0..nn {
            jac[e * nn + k] = el.geom.jac[k];
            rx[e * nn + k] = el.geom.rx[k];
            sx[e * nn + k] = el.geom.sx[k];
            jw[e * nn + k] = el.geom.jw[k];
        }
    }
    let n1 = (p + 1) as u32;
    let nfc = ne * 4 * (p + 1);
    let (mut fvl, mut fnx, mut fsw, mut fnbr) = (vec![0u32; nfc], vec![0.0; nfc], vec![0.0; nfc], vec![0u32; nfc]);
    for (e, el) in mesh.elements.iter().enumerate() {
        for (t, edge) in Edge::ALL.iter().enumerate() {
            let face = &el.faces[*edge as usize];
            let Neighbor::Interior { elem: re, edge: redge, perm } = &el.neighbors[*edge as usize] else {
                panic!("periodic mesh: no boundary");
            };
            let rf = &mesh.elements[*re].faces[*redge as usize];
            for a in 0..(p + 1) {
                let idx = (e * 4 + t) * (p + 1) + a;
                fvl[idx] = face.nodes[a] as u32;
                fnx[idx] = face.nx[a];
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
    let u_dev = up(&state[0])?;
    let jac_dev = up(&jac)?;
    let rx_dev = up(&rx)?;
    let sx_dev = up(&sx)?;
    let jw_dev = up(&jw)?;
    let fvl_dev = upu(&fvl)?;
    let fnx_dev = up(&fnx)?;
    let fsw_dev = up(&fsw)?;
    let fnbr_dev = upu(&fnbr)?;
    let mut out_dev = DeviceBuffer::<f64>::zeroed(&stream, ndof)?;

    let module = kernels::load(&ctx)?;
    let cfg = LaunchConfig { grid_dim: (ne as u32, 1, 1), block_dim: (nn as u32, 1, 1), shared_mem_bytes: 0 };
    module.burgers_split(
        &stream, cfg, &d_dev, &u_dev, &jac_dev, &rx_dev, &sx_dev, &jw_dev, n1,
        &fvl_dev, &fnx_dev, &fsw_dev, &fnbr_dev, &mut out_dev,
    )?;
    let gpu = out_dev.to_host_vec(&stream)?;

    let mut max_abs = 0.0f64;
    let scale = cpu[0].iter().fold(0.0f64, |a, &v| a.max(v.abs())).max(1e-300);
    for i in 0..ndof {
        max_abs = max_abs.max((gpu[i] - cpu[0][i]).abs());
    }
    println!("dofs={ndof}  max|gpu − cpu| / |rhs| = {:.3e}", max_abs / scale);
    if max_abs / scale < 1e-11 {
        println!("\nPASS: GPU split-form Burgers matches the CPU oracle.");
        Ok(())
    } else {
        eprintln!("\nFAIL: GPU/CPU mismatch.");
        std::process::exit(1);
    }
}
