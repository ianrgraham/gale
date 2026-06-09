//! Validates the GPU per-node IMEX implicit relaxation solve
//! (`gale_gpu::logconf_implicit_relax`, Phase 3) against the CPU oracle
//! `gale::dg::LogConfOldroydB::implicit_relax_solve`, and confirms the solved field
//! satisfies the implicit stage equation `Ψ − γ·S(Ψ) = B`.

use gale::dg::{LogConfOldroydB, Mesh2d};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let p = 4;
    let mesh = Mesh2d::rectangular(p, 4, 4, [0.0, 1.0], [0.0, 1.0]);
    let nn = mesh.refq.n_nodes();
    let ne = mesh.n_elements();
    let ndof = ne * nn;
    let lambda = 0.5;
    let lc = LogConfOldroydB::new(&mesh, lambda, 1.0);
    // Stage coefficient γ = dt·a^I_{ii} for ARS(2,2,2) at dt = 0.02.
    let gamma = (1.0 - 0.5_f64.sqrt()) * 0.02;

    // Smooth synthetic RHS B (the accumulated explicit stage) — a symmetric tensor field.
    let mut b = [vec![0.0; ndof], vec![0.0; ndof], vec![0.0; ndof]];
    for (e, el) in mesh.elements.iter().enumerate() {
        for k in 0..nn {
            let (x, y) = (el.geom.x[k], el.geom.y[k]);
            let g = e * nn + k;
            b[0][g] = 0.6 + 0.4 * (x + y).sin();
            b[1][g] = 0.2 * x - 0.1 * y;
            b[2][g] = 0.3 + 0.3 * (x * y).cos();
        }
    }

    let cpu = lc.implicit_relax_solve(&b, gamma);
    let gpu = gale_gpu::logconf_implicit_relax(&mesh, &lc, &b, gamma)?;

    let mut max_abs = 0.0f64;
    let mut scale = 1e-300f64;
    for v in 0..3 {
        for i in 0..ndof {
            max_abs = max_abs.max((gpu[v][i] - cpu[v][i]).abs());
            scale = scale.max(cpu[v][i].abs());
        }
    }
    println!("dofs={ndof}  γ={gamma:.5}  max|gpu − cpu| / |Ψ| = {:.3e}", max_abs / scale);

    // Independent correctness: the GPU result must satisfy Ψ − γ·S(Ψ) − B ≈ 0.
    let s = lc.relax_source(&gpu);
    let mut res = 0.0f64;
    for v in 0..3 {
        for i in 0..ndof {
            res = res.max((gpu[v][i] - gamma * s[v][i] - b[v][i]).abs());
        }
    }
    println!("max residual |Ψ − γ·S(Ψ) − B| = {res:.3e}");

    if max_abs / scale < 1e-9 && res < 1e-9 {
        println!("\nPASS: GPU implicit relaxation solve matches the CPU oracle and satisfies the stage equation.");
        Ok(())
    } else {
        eprintln!("\nFAIL: GPU/CPU mismatch or non-zero stage residual.");
        std::process::exit(1);
    }
}
