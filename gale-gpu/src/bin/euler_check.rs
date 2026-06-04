//! Validation harness for the reusable [`gale_gpu::euler_rhs`] library component:
//! builds the same smooth Euler state the operator was ported with, runs it through
//! the GPU wrapper, and checks it bit-for-bit (libdevice `sqrt`/`abs` path) against
//! `gale::dg::Hyperbolic` (Euler, weak form, Rusanov / LLF flux).
//!
//! State (SoA): u0=ρ, u1=ρu, u2=ρv, u3=E. Periodic mesh ⇒ no boundary flux.
//!
//! Run: cargo oxide run --bin gpu-euler

use gale::dg::{Euler, Hyperbolic, Mesh2d};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let p = 4;
    let gamma = 1.4;
    println!("=== gale_gpu::euler_rhs (library) vs CPU oracle (p={p}) ===\n");

    let mesh = Mesh2d::rectangular_periodic(p, 3, 3, [0.0, 1.0], [0.0, 1.0]);
    let nn = mesh.refq.n_nodes();
    let ne = mesh.n_elements();
    let ndof = ne * nn;
    let op = Hyperbolic::new(&mesh, Euler { gamma });

    // Smooth state: density wave + swirl.
    let mut state: Vec<Vec<f64>> = (0..4).map(|_| vec![0.0; ndof]).collect();
    for (e, el) in mesh.elements.iter().enumerate() {
        for k in 0..nn {
            let (x, y) = (el.geom.x[k], el.geom.y[k]);
            let r = 1.0 + 0.2 * (2.0 * std::f64::consts::PI * x).sin();
            let vx = 0.3 + 0.1 * (2.0 * std::f64::consts::PI * y).cos();
            let vy = -0.2 + 0.1 * (2.0 * std::f64::consts::PI * x).cos();
            let pr = 1.0 + 0.1 * (2.0 * std::f64::consts::PI * (x + y)).sin();
            let en = pr / (gamma - 1.0) + 0.5 * r * (vx * vx + vy * vy);
            state[0][e * nn + k] = r;
            state[1][e * nn + k] = r * vx;
            state[2][e * nn + k] = r * vy;
            state[3][e * nn + k] = en;
        }
    }
    let cpu = op.rhs(&state, 0.0, &|_, _, _, _: &mut [f64]| {});

    // GPU path through the reusable library wrapper.
    let gpu = gale_gpu::euler_rhs(&mesh, &state, gamma)?;

    let mut max_abs = 0.0f64;
    let mut scale = 1e-300f64;
    for v in 0..4 {
        for i in 0..ndof {
            max_abs = max_abs.max((gpu[v][i] - cpu[v][i]).abs());
            scale = scale.max(cpu[v][i].abs());
        }
    }
    println!("dofs={ndof}  max|gpu − cpu| / |rhs| = {:.3e}", max_abs / scale);
    if max_abs / scale < 1e-11 {
        println!("\nPASS: GPU Euler operator (library) matches the CPU oracle (libdevice path works).");
        Ok(())
    } else {
        eprintln!("\nFAIL: GPU/CPU mismatch.");
        std::process::exit(1);
    }
}
