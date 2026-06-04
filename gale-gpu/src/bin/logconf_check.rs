//! Validation harness for the reusable [`gale_gpu::logconf_psi_rhs`] library
//! component: computes the GPU log-conformation Ψ rhs and checks it against the
//! CPU oracle `gale::dg::LogConfOldroydB::psi_rhs`. Exercises the last
//! libdevice-blocked operator (full `sqrt`/`exp` battery) coexisting with the
//! other `#[cuda_module]`s in the same crate bundle.
//!
//! Run: cargo oxide run --bin gpu-logconf

use gale::dg::{LogConfOldroydB, Mesh2d};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let p = 4;
    let lambda = 0.7;
    println!("=== gale_gpu::logconf_psi_rhs (library) vs CPU oracle (p={p}) ===\n");

    let mesh = Mesh2d::rectangular(p, 4, 4, [0.0, 1.0], [0.0, 1.0]);
    let nn = mesh.refq.n_nodes();
    let ne = mesh.n_elements();
    let ndof = ne * nn;
    let lc = LogConfOldroydB::new(&mesh, lambda, 1.0);

    // Smooth velocity + a moderately anisotropic Ψ field (SPD C = exp Ψ).
    let (mut uxv, mut uyv) = (vec![0.0; ndof], vec![0.0; ndof]);
    let mut psi = [vec![0.0; ndof], vec![0.0; ndof], vec![0.0; ndof]];
    for (e, el) in mesh.elements.iter().enumerate() {
        for k in 0..nn {
            let (x, y) = (el.geom.x[k], el.geom.y[k]);
            uxv[e * nn + k] = (2.0 * x).sin() * y + 0.3 * x;
            uyv[e * nn + k] = -0.4 * (3.0 * y).cos() * x;
            psi[0][e * nn + k] = 0.5 + 0.3 * (x + y).sin();
            psi[1][e * nn + k] = 0.2 * x - 0.1 * y;
            psi[2][e * nn + k] = -0.3 + 0.2 * (x * y).cos();
        }
    }
    let cpu = lc.psi_rhs(&psi, &uxv, &uyv);

    // GPU path through the reusable library wrapper.
    let gpu = gale_gpu::logconf_psi_rhs(&mesh, &lc, &psi, &uxv, &uyv)?;

    let mut max_abs = 0.0f64;
    let mut scale = 1e-300f64;
    for vv in 0..3 {
        for i in 0..ndof {
            max_abs = max_abs.max((gpu[vv][i] - cpu[vv][i]).abs());
            scale = scale.max(cpu[vv][i].abs());
        }
    }
    println!("dofs={ndof}  max|gpu − cpu| / |rhs| = {:.3e}", max_abs / scale);
    if max_abs / scale < 1e-9 {
        println!("\nPASS: GPU log-conformation rhs (library) matches the CPU oracle.");
        Ok(())
    } else {
        eprintln!("\nFAIL: GPU/CPU mismatch.");
        std::process::exit(1);
    }
}
