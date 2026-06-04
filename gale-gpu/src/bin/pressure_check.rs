//! Validation harness for the reusable [`gale_gpu::pressure_cg_solve`] library
//! component — the **pure-Neumann pressure-projection solve** of the dual-splitting
//! Stokes/NS scheme. Solves the singular SIPG pressure-Poisson by deflated CG on the
//! GPU and checks it against the CPU oracle
//! `gale::dg::Poisson::with_bc(mesh, alpha, 0, all-tags).cg_deflated`.
//!
//! Run: cargo oxide run --bin pressure-check

use gale::dg::{Mesh2d, Poisson};

/// Remove the arithmetic mean (the undetermined constant of a pure-Neumann solve).
fn demean(v: &mut [f64]) {
    let m = v.iter().sum::<f64>() / v.len() as f64;
    v.iter_mut().for_each(|x| *x -= m);
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    use std::f64::consts::PI;
    const P: usize = 4;
    let alpha = 5.0;
    println!("=== gale_gpu::pressure_cg_solve (deflated, Neumann) vs CPU oracle (p={P}) ===\n");

    let mesh = Mesh2d::rectangular(P, 3, 3, [0.0, 1.0], [0.0, 1.0]);
    let nn = mesh.refq.n_nodes();
    let ne = mesh.n_elements();
    let ndof = ne * nn;
    // Pure-Neumann pressure operator (all four boundary tags Neumann), as Stokes uses.
    let pressure = Poisson::with_bc(&mesh, alpha, 0.0, vec![0, 1, 2, 3]);

    // Neumann MMS: u = cos(πx)cos(πy) has ∂u/∂n = 0 on ∂[0,1]²; −Δu = 2π²u = f;
    // ∫f = 0 ⇒ compatible. Solution is determined up to an additive constant.
    let exact = |x: f64, y: f64| (PI * x).cos() * (PI * y).cos();
    let mut f = vec![0.0; ndof];
    for (e, el) in mesh.elements.iter().enumerate() {
        for k in 0..nn {
            f[e * nn + k] = 2.0 * PI * PI * exact(el.geom.x[k], el.geom.y[k]);
        }
    }
    let b = pressure.rhs_mixed(&f, |_, _| 0.0, |_, _| 0.0);

    // CPU reference (deflated CG) and GPU path (deflated CG on device).
    let (mut u_cpu, it_cpu) = pressure.cg_deflated(&b, 1e-10, 20000);
    let (mut u_gpu, iters) = gale_gpu::pressure_cg_solve(&mesh, &b, alpha, 1e-10, 20000)?;

    // Both are defined up to a constant — compare after removing the mean.
    demean(&mut u_cpu);
    demean(&mut u_gpu);
    let mut diff = 0.0f64;
    let mut cpun = 0.0f64;
    let mut err_gpu = 0.0f64;
    let mut exn = 0.0f64;
    // Mean-removed exact for the MMS check.
    let mut ex = vec![0.0; ndof];
    for (e, el) in mesh.elements.iter().enumerate() {
        for k in 0..nn {
            ex[e * nn + k] = exact(el.geom.x[k], el.geom.y[k]);
        }
    }
    demean(&mut ex);
    for (e, el) in mesh.elements.iter().enumerate() {
        for k in 0..nn {
            let i = e * nn + k;
            diff += el.geom.jw[k] * (u_gpu[i] - u_cpu[i]).powi(2);
            cpun += el.geom.jw[k] * u_cpu[i].powi(2);
            err_gpu += el.geom.jw[k] * (u_gpu[i] - ex[i]).powi(2);
            exn += el.geom.jw[k] * ex[i].powi(2);
        }
    }
    let rel = (diff / cpun.max(1e-300)).sqrt();
    println!("dofs={ndof}   deflated-CG iters: cpu={it_cpu}  gpu={iters}");
    println!("‖u_gpu − u_cpu‖ / ‖u_cpu‖ = {rel:.3e}");
    println!("‖u_gpu − u_exact‖ / ‖u_exact‖ (MMS) = {:.3e}", (err_gpu / exn.max(1e-300)).sqrt());
    if rel < 1e-8 {
        println!("\nPASS: GPU deflated pressure CG (library) matches the CPU solve.");
        Ok(())
    } else {
        eprintln!("\nFAIL: GPU/CPU pressure mismatch.");
        std::process::exit(1);
    }
}
