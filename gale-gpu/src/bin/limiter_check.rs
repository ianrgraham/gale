//! Validates the GPU log-conformation FENE-P trace-bound limiter
//! (`gale_gpu::logconf_limit_trace`, Phase 4) against the CPU oracle
//! `gale::dg::limit_logconf_trace_bound`, and confirms `tr exp(Ψ) ≤ b`.

use gale::dg::{limit_logconf_trace_bound, LogConfOldroydB, Mesh2d};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let p = 4;
    let mesh = Mesh2d::rectangular(p, 4, 4, [0.0, 1.0], [0.0, 1.0]);
    let nn = mesh.refq.n_nodes();
    let ne = mesh.n_elements();
    let ndof = ne * nn;
    let lc = LogConfOldroydB::new(&mesh, 1.0, 1.0);
    let b = 12.0;

    // A log-conf field with sharp features so some nodes exceed tr exp(Ψ) = b while the
    // element means stay admissible.
    let mut psi = [vec![0.0; ndof], vec![0.0; ndof], vec![0.0; ndof]];
    for (e, el) in mesh.elements.iter().enumerate() {
        for k in 0..nn {
            let (x, y) = (el.geom.x[k], el.geom.y[k]);
            let g = e * nn + k;
            psi[0][g] = 1.6 * (5.0 * (x - 0.5)).tanh() + 0.6; // Ψxx up to ~2.2 ⇒ tr C up to ~10+
            psi[1][g] = 0.3 * (x + y);
            psi[2][g] = 0.4 * (3.0 * y).sin();
        }
    }

    let mut cpu = psi.clone();
    limit_logconf_trace_bound(&mesh, &mut cpu, b);
    let gpu = gale_gpu::logconf_limit_trace(&mesh, &psi, b)?;

    let mut max_abs = 0.0f64;
    let mut scale = 1e-300f64;
    for v in 0..3 {
        for i in 0..ndof {
            max_abs = max_abs.max((gpu[v][i] - cpu[v][i]).abs());
            scale = scale.max(cpu[v][i].abs());
        }
    }
    // tr exp(Ψ) ≤ b for the GPU result (SPD is automatic via exp).
    let c = lc.conformation(&gpu);
    let mut max_tr = 0.0f64;
    for i in 0..ndof {
        max_tr = max_tr.max(c[0][i] + c[2][i]);
    }
    println!(
        "dofs={ndof}  b={b}  max|gpu − cpu| / |Ψ| = {:.3e}  max tr C = {max_tr:.4}",
        max_abs / scale
    );

    if max_abs / scale < 1e-9 && max_tr <= b * (1.0 + 1e-9) {
        println!("\nPASS: GPU log-conf trace limiter matches the CPU oracle and enforces tr C ≤ b.");
        Ok(())
    } else {
        eprintln!("\nFAIL: GPU/CPU mismatch or trace bound violated.");
        std::process::exit(1);
    }
}
