//! "Are we over-solving?" — does the per-step elliptic-solve tolerance need to be 1e-10, or is
//! the time-discretization error the real floor? Sweeps the solve tol on a time-accurate run
//! (decaying Taylor–Green/Stokes vortex, exact solution) and reports the physical error
//! ‖u − exact‖ + the drift from the tight-tol reference, alongside the per-tol MG-PCG iteration
//! count on a representative pressure RHS (the cost we'd save). If the physical error is flat from
//! 1e-10 up to some looser tol, every iteration below that tol is wasted.
//!
//! Run: cargo oxide run --bin solve-tol-sweep

use gale::dg::{Mesh2d, PMultigrid, Poisson};
use gale_gpu::{GpuPoissonMg, GpuStokes};
use std::f64::consts::PI;

fn nodal(mesh: &Mesh2d, f: impl Fn(f64, f64) -> f64) -> Vec<f64> {
    let nn = mesh.refq.n_nodes();
    let mut v = vec![0.0; mesh.n_elements() * nn];
    for (e, el) in mesh.elements.iter().enumerate() {
        for k in 0..nn {
            v[e * nn + k] = f(el.geom.x[k], el.geom.y[k]);
        }
    }
    v
}

fn broadband(fine: &Mesh2d) -> Vec<f64> {
    let n0 = fine.n_elements() * fine.refq.n_nodes();
    let mut s: Vec<f64> = (0..n0)
        .map(|i| {
            let h = (i as u64).wrapping_mul(2862933555777941757).wrapping_add(3037000493);
            ((h >> 11) as f64 / (1u64 << 53) as f64) * 2.0 - 1.0
        })
        .collect();
    let m = s.iter().sum::<f64>() / n0 as f64;
    s.iter_mut().for_each(|v| *v -= m);
    s
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let (nu, p, alpha, xr) = (1.0, 4usize, 5.0, [0.0, 1.0]);
    let (g, dt, nsteps) = (16usize, 0.005, 20usize);
    let lambda = 1.0 / (nu * dt);
    let mesh = Mesh2d::rectangular(p, g, g, xr, xr);
    let tags = mesh.boundary_tags();
    let decay = move |t: f64| (-2.0 * PI * PI * nu * t).exp();
    let eu = move |x: f64, y: f64, t: f64| -(PI * x).cos() * (PI * y).sin() * decay(t);
    let ev = move |x: f64, y: f64, t: f64| (PI * x).sin() * (PI * y).cos() * decay(t);
    let zero = |_: f64, _: f64, _: f64| 0.0;
    let t_end = dt * nsteps as f64;

    // Persistent MG handles (pressure: deflated Neumann; velocity: Helmholtz λ), so per-step cost
    // is the solve and the tol is the only thing we vary.
    let mgp = GpuPoissonMg::new(PMultigrid::with_bc(p, g, g, xr, xr, alpha, 0.0, tags.clone()))?;
    let mgv = GpuPoissonMg::new(PMultigrid::with_reaction(p, g, g, xr, xr, alpha, lambda))?;

    let run = |tol: f64| -> Result<Vec<f64>, Box<dyn std::error::Error>> {
        let gst = GpuStokes::new(&mesh, alpha, nu, dt)
            .with_mg_pressure(&mgp)
            .with_mg_velocity(&mgv, &mgv)
            .with_tol(tol);
        let mut ux = nodal(&mesh, |x, y| eu(x, y, 0.0));
        let mut uy = nodal(&mesh, |x, y| ev(x, y, 0.0));
        let mut t = 0.0;
        for _ in 0..nsteps {
            t += dt;
            let (nx, ny) = gst.step(&ux, &uy, t, eu, ev, zero, zero)?;
            ux = nx;
            uy = ny;
        }
        let mut out = ux;
        out.extend(uy);
        Ok(out)
    };
    let l2 = |a: &[f64]| -> f64 {
        let gst = GpuStokes::new(&mesh, alpha, nu, dt);
        let n = a.len() / 2;
        (gst.l2_norm(&a[..n]).powi(2) + gst.l2_norm(&a[n..]).powi(2)).sqrt()
    };

    let exu = nodal(&mesh, |x, y| eu(x, y, t_end));
    let exv = nodal(&mesh, |x, y| ev(x, y, t_end));
    let mut exact = exu;
    exact.extend(exv);

    let reference = run(1e-12)?; // tightest-tol trajectory

    println!("=== Over-solving check: decaying vortex (p={p}, {g}², dt={dt}, {nsteps} steps, ν={nu}) ===\n");
    println!("{:>8}  {:>14}  {:>16}", "tol", "‖u−exact‖", "‖u−u(1e-12)‖");
    for &tol in &[1e-10, 1e-8, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2] {
        let u = run(tol)?;
        let err_exact = {
            let d: Vec<f64> = u.iter().zip(&exact).map(|(a, b)| a - b).collect();
            l2(&d)
        };
        let drift = {
            let d: Vec<f64> = u.iter().zip(&reference).map(|(a, b)| a - b).collect();
            l2(&d)
        };
        println!("{:>8.0e}  {:>14.3e}  {:>16.3e}", tol, err_exact, drift);
    }

    // Iteration cost vs tol: a representative pressure solve at a production-ish size.
    let gc = 64usize;
    let cmesh = Mesh2d::rectangular(p, gc, gc, xr, xr);
    let ctags = cmesh.boundary_tags();
    let crhs = Poisson::with_bc(&cmesh, alpha, 0.0, ctags.clone()).rhs_mixed(&broadband(&cmesh), |_, _| 0.0, |_, _| 0.0);
    let cmg = GpuPoissonMg::new(PMultigrid::with_bc(p, gc, gc, xr, xr, alpha, 0.0, ctags))?;
    println!("\nMG-PCG iters vs tol (pressure, {gc}² p={p}):");
    println!("{:>8}  {:>6}", "tol", "iters");
    for &tol in &[1e-10, 1e-8, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2] {
        let (_x, it) = cmg.solve(&crhs, tol, 100_000)?;
        println!("{:>8.0e}  {:>6}", tol, it);
    }
    Ok(())
}
