//! Validates the two NEW device kernels for viscoelastic residency against their host references:
//!   * `conformation` (Ψ → C = exp Ψ) vs `gale::dg::LogConfOldroydB::conformation`
//!   * `upwind_lift` (3-component DG upwind advection surface flux) vs
//!     `gale::dg::upwind_advection_lift`
//! on a walled (Kolmogorov-style) mesh, so boundary faces (face_nbr = self ⇒ zero contribution)
//! are exercised. These are the crux kernels of the device-resident VE step; both must match the
//! host to round-off. Run: cargo oxide run --bin resident-ve-kernels-check

use gale::dg::{upwind_advection_lift, Edge, LogConfOldroydB, Mesh2d, Neighbor};
use gale_gpu::operators::logconf::GpuLogConf;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let p = 3usize;
    let (nx, ny) = (8usize, 8usize);
    let xr = [0.0, 1.0];
    let yr = [0.0, 1.0];
    let mesh = Mesh2d::rectangular(p, nx, ny, xr, yr);
    let nn = mesh.refq.n_nodes();
    let ne = mesh.n_elements();
    let ndof = ne * nn;
    let n1 = (p + 1) as u32;
    let two_pi = std::f64::consts::TAU;
    println!("=== device VE kernels vs host — conformation & upwind lift, {nx}×{ny} p={p} ===");

    // Non-trivial smooth test state: Ψ (symmetric) + velocity.
    let (mut pxx, mut pxy, mut pyy) = (vec![0.0; ndof], vec![0.0; ndof], vec![0.0; ndof]);
    let (mut ux, mut uy) = (vec![0.0; ndof], vec![0.0; ndof]);
    for (e, el) in mesh.elements.iter().enumerate() {
        for k in 0..nn {
            let (x, y) = (el.geom.x[k], el.geom.y[k]);
            let g = e * nn + k;
            pxx[g] = 0.4 * (two_pi * x).sin() + 0.1;
            pxy[g] = 0.2 * (two_pi * x).cos() * (two_pi * y).sin();
            pyy[g] = 0.3 * (two_pi * y).cos() - 0.1;
            ux[g] = (two_pi * 2.0 * y).sin();
            uy[g] = 0.2 * (two_pi * x).sin() * (two_pi * y).sin();
        }
    }

    // Flatten ALL faces (interior + boundary). Boundary faces ⇒ face_nbr = self node (zero lift).
    let nfc = ne * 4 * (n1 as usize);
    let (mut fvl, mut fnx, mut fny, mut fsw, mut fnbr) =
        (vec![0u32; nfc], vec![0.0; nfc], vec![0.0; nfc], vec![0.0; nfc], vec![0u32; nfc]);
    for (e, el) in mesh.elements.iter().enumerate() {
        for (t, edge) in Edge::ALL.iter().enumerate() {
            let face = &el.faces[*edge as usize];
            for a in 0..n1 as usize {
                let idx = (e * 4 + t) * n1 as usize + a;
                let vl = face.nodes[a];
                fvl[idx] = vl as u32;
                fnx[idx] = face.nx[a];
                fny[idx] = face.ny[a];
                fsw[idx] = face.sw[a];
                fnbr[idx] = match &el.neighbors[*edge as usize] {
                    Neighbor::Interior { elem: re, edge: redge, perm } => {
                        let rf = &mesh.elements[*re].faces[*redge as usize];
                        (*re * nn + rf.nodes[perm[a]]) as u32
                    }
                    _ => (e * nn + vl) as u32, // boundary / NC ⇒ self ⇒ no correction
                };
            }
        }
    }

    let jw: Vec<f64> = mesh.elements.iter().flat_map(|el| el.geom.jw.iter().copied()).collect();
    let rx: Vec<f64> = mesh.elements.iter().flat_map(|el| el.geom.rx.iter().copied()).collect();
    let _ = &rx; // (metrics not needed for these two kernels)

    // Host references.
    let lc = LogConfOldroydB::new(&mesh, 1.0, 1.0);
    let psi = [pxx.clone(), pxy.clone(), pyy.clone()];
    let cc_host = lc.conformation(&psi);
    let lift_host = upwind_advection_lift(&mesh, &psi, &ux, &uy, |_| None);

    // Device kernels.
    let h = GpuLogConf::new()?;
    let pxx_d = up(&h, &pxx)?;
    let pxy_d = up(&h, &pxy)?;
    let pyy_d = up(&h, &pyy)?;
    let ux_d = up(&h, &ux)?;
    let uy_d = up(&h, &uy)?;
    let jw_d = up(&h, &jw)?;
    let fvl_d = upu(&h, &fvl)?;
    let fnx_d = up(&h, &fnx)?;
    let fny_d = up(&h, &fny)?;
    let fsw_d = up(&h, &fsw)?;
    let fnbr_d = upu(&h, &fnbr)?;
    let (mut cxx, mut cxy, mut cyy) = (zeros(&h, ndof)?, zeros(&h, ndof)?, zeros(&h, ndof)?);
    let (mut lxx, mut lxy, mut lyy) = (zeros(&h, ndof)?, zeros(&h, ndof)?, zeros(&h, ndof)?);

    h.conformation_dev(ne, n1, &pxx_d, &pxy_d, &pyy_d, &mut cxx, &mut cxy, &mut cyy)?;
    h.upwind_lift_dev(
        ne, n1, &ux_d, &uy_d, &pxx_d, &pxy_d, &pyy_d, &jw_d,
        &fvl_d, &fnx_d, &fny_d, &fsw_d, &fnbr_d, &mut lxx, &mut lxy, &mut lyy,
    )?;
    let cc_dev = [dn(&h, &cxx)?, dn(&h, &cxy)?, dn(&h, &cyy)?];
    let lift_dev = [dn(&h, &lxx)?, dn(&h, &lxy)?, dn(&h, &lyy)?];

    let rel = |a: &[f64], b: &[f64]| -> f64 {
        let d: f64 = a.iter().zip(b).map(|(x, y)| (x - y).powi(2)).sum();
        let n: f64 = b.iter().map(|x| x * x).sum::<f64>().max(1e-300);
        (d / n).sqrt()
    };
    let rc = (0..3).map(|i| rel(&cc_dev[i], &cc_host[i])).fold(0.0f64, f64::max);
    let rl = (0..3).map(|i| rel(&lift_dev[i], &lift_host[i])).fold(0.0f64, f64::max);
    println!("  conformation rel err = {rc:.3e}   upwind-lift rel err = {rl:.3e}");
    if rc < 1e-12 && rl < 1e-12 {
        println!("OK: device conformation & upwind-lift match the host to round-off.");
        Ok(())
    } else {
        eprintln!("FAIL: device VE kernels diverge from host.");
        std::process::exit(1);
    }
}

type B = cuda_core::DeviceBuffer<f64>;
type Bu = cuda_core::DeviceBuffer<u32>;
fn up(h: &GpuLogConf, v: &[f64]) -> Result<B, Box<dyn std::error::Error>> {
    Ok(cuda_core::DeviceBuffer::from_host(h.stream(), v)?)
}
fn upu(h: &GpuLogConf, v: &[u32]) -> Result<Bu, Box<dyn std::error::Error>> {
    Ok(cuda_core::DeviceBuffer::from_host(h.stream(), v)?)
}
fn zeros(h: &GpuLogConf, n: usize) -> Result<B, Box<dyn std::error::Error>> {
    Ok(cuda_core::DeviceBuffer::<f64>::zeroed(h.stream(), n)?)
}
fn dn(h: &GpuLogConf, b: &B) -> Result<Vec<f64>, Box<dyn std::error::Error>> {
    Ok(b.to_host_vec(h.stream())?)
}
