//! Where do the CG iterations actually go in a dual-splitting flow step? Each step does
//! one **pressure** solve (Neumann, deflated, reaction 0 — the ill-conditioned one) and
//! two **velocity** Helmholtz solves (reaction λ = 1/(νΔt) — better-conditioned for small
//! Δt). This probe reports the GPU CG iteration counts for each, across mesh refinement
//! (where multigrid's value shows up — pressure iters grow with 1/h, Helmholtz shouldn't)
//! and a λ sweep (Δt at ν=1). Decides which solve, if any, is worth preconditioning.
//!
//! Run: cargo oxide run --bin cg-iter-probe

use gale::dg::{Mesh2d, Poisson};
use gale_gpu::{helmholtz_cg_solve, pressure_cg_solve};

/// Smooth nodal field f(x,y) over the mesh (mean-removed for the Neumann-compatible RHS).
fn nodal(mesh: &Mesh2d, f: impl Fn(f64, f64) -> f64) -> Vec<f64> {
    let nn = mesh.refq.n_nodes();
    let mut v = vec![0.0; mesh.n_elements() * nn];
    for (e, el) in mesh.elements.iter().enumerate() {
        for k in 0..nn {
            v[e * nn + k] = f(el.geom.x[k], el.geom.y[k]);
        }
    }
    let mean = v.iter().sum::<f64>() / v.len() as f64;
    v.iter_mut().for_each(|x| *x -= mean);
    v
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let (p, alpha, tol, maxit) = (4usize, 5.0, 1e-10, 50_000);
    let nu = 1.0;
    let dts = [0.05, 0.01, 0.001]; // λ = 1/(νΔt) = 20, 100, 1000
    println!("=== CG iterations per dual-splitting solve (p={p}, ν={nu}, tol={tol:.0e}) ===");
    println!("pressure = Neumann/deflated (reaction 0);  velocity = Helmholtz (reaction λ=1/(νΔt))\n");
    println!(
        "{:>7} {:>8} {:>10} | velocity Helmholtz iters @ λ =",
        "grid", "ndof", "pressure"
    );
    println!("{:>7} {:>8} {:>10} | {:>8} {:>8} {:>8}", "", "", "(λ=0)", "20", "100", "1000");

    for &g in &[8usize, 16, 32, 64] {
        let mesh = Mesh2d::rectangular(p, g, g, [0.0, 1.0], [0.0, 1.0]);
        let ndof = mesh.n_elements() * mesh.refq.n_nodes();
        let src = nodal(&mesh, |x, y| (std::f64::consts::PI * x).sin() * (std::f64::consts::PI * y).sin());

        // Pressure: pure-Neumann, deflated.
        let pp = Poisson::with_bc(&mesh, alpha, 0.0, mesh.boundary_tags());
        let bp = pp.rhs_mixed(&src, |_, _| 0.0, |_, _| 0.0);
        let (_x, pit) = pressure_cg_solve(&mesh, &bp, alpha, tol, maxit)?;

        let mut vits = Vec::new();
        for &dt in &dts {
            let lambda = 1.0 / (nu * dt);
            let hop = Poisson::with_reaction(&mesh, alpha, lambda);
            let fxv: Vec<f64> = src.iter().map(|v| lambda * v).collect();
            let bv = hop.rhs(&fxv, |_, _| 0.0);
            let (_u, vit) = helmholtz_cg_solve(&mesh, &bv, alpha, lambda, tol, maxit)?;
            vits.push(vit);
        }
        println!(
            "{:>5}² {:>8} {:>10} | {:>8} {:>8} {:>8}",
            g, ndof, pit, vits[0], vits[1], vits[2]
        );
    }
    println!("\n(If pressure iters grow with grid and dwarf velocity, MG should target the pressure solve.)");
    Ok(())
}
