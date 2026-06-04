//! Validation harness for the GPU 3D log-conformation rhs [`gale_gpu::logconf3d_psi_rhs`]
//! (Fattal–Kupferman, with an on-device 3×3 Jacobi eigensolver) vs the CPU oracle
//! `gale::dg::LogConfOldroydB3d::psi_rhs`. Checks two states: an anisotropic Ψ (general
//! eigenframe) and a near-equilibrium Ψ≈0 (exercises the isotropic-branch second eig).
//!
//! Run: cargo oxide run --bin logconf3d-check

use gale::dg::{LogConfOldroydB3d, Mesh3d};

fn nodal(mesh: &Mesh3d, f: impl Fn(f64, f64, f64) -> f64) -> Vec<f64> {
    let nn = mesh.refh.n_nodes();
    let mut v = vec![0.0; mesh.n_elements() * nn];
    for (e, el) in mesh.elements.iter().enumerate() {
        for k in 0..nn {
            v[e * nn + k] = f(el.geom.x[k], el.geom.y[k], el.geom.z[k]);
        }
    }
    v
}

fn rel_l2(a: &[f64], b: &[f64]) -> f64 {
    let num: f64 = a.iter().zip(b).map(|(x, y)| (x - y).powi(2)).sum();
    let den: f64 = b.iter().map(|y| y * y).sum::<f64>().max(1e-300);
    (num / den).sqrt()
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let p = 3;
    let mesh = Mesh3d::rectangular(p, 2, 2, 2, [0.0, 1.0], [0.0, 1.0], [0.0, 1.0]);
    let lc = LogConfOldroydB3d::new(&mesh, 0.5, 0.5);
    println!("=== GPU 3D log-conformation rhs (3×3 Jacobi eig) vs CPU oracle (p={p}) ===\n");

    let ux = nodal(&mesh, |x, y, z| 0.3 * (x + 0.5 * y - 0.2 * z));
    let uy = nodal(&mesh, |x, y, z| -0.2 * (y - 0.3 * x + 0.1 * z));
    let uz = nodal(&mesh, |x, y, z| 0.15 * (z + 0.2 * x - 0.4 * y));

    let mut ok = true;
    for (label, amp) in [("anisotropic Ψ", 0.4), ("near-equilibrium Ψ≈0", 1e-4)] {
        // Symmetric Ψ with distinct diagonal (anisotropic) scaled by amp.
        let psi = [
            nodal(&mesh, |x, _, _| amp * (0.7 + 0.3 * x)),
            nodal(&mesh, |x, y, _| amp * 0.2 * (x - y)),
            nodal(&mesh, |x, _, z| amp * 0.15 * (x + z)),
            nodal(&mesh, |_, y, _| amp * (0.4 + 0.2 * y)),
            nodal(&mesh, |_, y, z| amp * 0.1 * (y - z)),
            nodal(&mesh, |_, _, z| amp * (0.5 - 0.2 * z)),
        ];
        let r_cpu = lc.psi_rhs(&psi, &ux, &uy, &uz);
        let r_gpu = gale_gpu::logconf3d_psi_rhs(&mesh, &lc, &psi, &ux, &uy, &uz)?;
        let err = (0..6).fold(0.0f64, |a, o| a.max(rel_l2(&r_gpu[o], &r_cpu[o])));
        let pass = err < 1e-9;
        ok &= pass;
        println!("{label}: max rel = {err:.3e}   {}", if pass { "OK" } else { "FAIL" });
    }

    if ok {
        println!("\nPASS: GPU 3D log-conformation rhs (on-device 3×3 eigensolver) matches the CPU oracle.");
        Ok(())
    } else {
        eprintln!("\nFAIL: GPU 3D log-conformation mismatch.");
        std::process::exit(1);
    }
}
