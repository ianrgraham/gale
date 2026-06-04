//! Validation harness for the reusable [`gale_gpu::helmholtz_cg_solve`] library
//! component — the **viscous-velocity solve** of the dual-splitting Stokes/NS
//! scheme. Solves the SIPG Helmholtz system `(λM + A)·u = b` by conjugate gradient
//! entirely on the GPU and checks it against the CPU oracle
//! `gale::dg::Poisson::with_reaction(mesh, alpha, λ).cg`.
//!
//! Run: cargo oxide run --bin helmholtz-check

use gale::dg::{Mesh2d, Poisson};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    use std::f64::consts::PI;
    const P: usize = 4;
    let alpha = 5.0;
    let lambda = 50.0; // ~ 1/(νΔt) scale of the viscous Helmholtz solve
    println!("=== gale_gpu::helmholtz_cg_solve (λ={lambda}) vs CPU oracle (p={P}) ===\n");

    let mesh = Mesh2d::rectangular(P, 3, 3, [0.0, 1.0], [0.0, 1.0]);
    let nn = mesh.refq.n_nodes();
    let ne = mesh.n_elements();
    let ndof = ne * nn;
    let op = Poisson::with_reaction(&mesh, alpha, lambda);

    // MMS: u = sin(πx)sin(πy) (zero on the boundary ⇒ homogeneous Dirichlet).
    // (λM + A)u ↔ (λ − Δ)u = (λ + 2π²)u in strong form.
    let exact = |x: f64, y: f64| (PI * x).sin() * (PI * y).sin();
    let mut f = vec![0.0; ndof];
    for (e, el) in mesh.elements.iter().enumerate() {
        for k in 0..nn {
            f[e * nn + k] = (lambda + 2.0 * PI * PI) * exact(el.geom.x[k], el.geom.y[k]);
        }
    }
    let b = op.rhs(&f, |_, _| 0.0);

    // CPU reference solve.
    let (u_cpu, it_cpu, _res) = op.cg(&b, 1e-10, 20000);

    // GPU path through the reusable library wrapper (same tol / max-iters).
    let (u_gpu, iters) = gale_gpu::helmholtz_cg_solve(&mesh, &b, alpha, lambda, 1e-10, 20000)?;

    let mut diff = 0.0f64;
    let mut cpun = 0.0f64;
    let mut err_gpu = 0.0f64;
    for (e, el) in mesh.elements.iter().enumerate() {
        for k in 0..nn {
            let i = e * nn + k;
            diff += el.geom.jw[k] * (u_gpu[i] - u_cpu[i]).powi(2);
            cpun += el.geom.jw[k] * u_cpu[i].powi(2);
            err_gpu += el.geom.jw[k] * (u_gpu[i] - exact(el.geom.x[k], el.geom.y[k])).powi(2);
        }
    }
    let rel = (diff / cpun.max(1e-300)).sqrt();
    println!("dofs={ndof}   Helmholtz CG iters: cpu={it_cpu}  gpu={iters}");
    println!("‖u_gpu − u_cpu‖ / ‖u_cpu‖ = {rel:.3e}");
    println!("‖u_gpu − u_exact‖ (MMS)   = {:.3e}", err_gpu.sqrt());
    if rel < 1e-8 {
        println!("\nPASS: GPU Helmholtz CG (library) matches the CPU viscous solve.");
        Ok(())
    } else {
        eprintln!("\nFAIL: GPU/CPU Helmholtz mismatch.");
        std::process::exit(1);
    }
}
