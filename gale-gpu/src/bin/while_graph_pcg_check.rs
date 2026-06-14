//! Validates the **WHILE-graph PCG** (the whole outer CG loop captured into a CUDA graph WHILE
//! conditional node, convergence decided device-side by `pcg_cond` — NO per-iteration host residual
//! readback) against the standard readback PCG. Same operator, same RHS ⇒ same solution and the
//! same iteration count (the device convergence test mirrors the host `rel < tol`). Covers both the
//! non-singular velocity Helmholtz and the singular (deflated) pure-Neumann pressure. This is the
//! capstone of the device-resident flow work: combined with `solve_dev`, an entire elliptic solve
//! is one `cuGraphLaunch` with zero host synchronization. Run: cargo oxide run --bin while-graph-pcg-check

use gale::dg::{Mesh2d, PMultigrid, Poisson};
use gale_gpu::operators::poisson::GpuPoissonMg;
use std::f64::consts::PI;

fn rel(a: &[f64], b: &[f64]) -> f64 {
    let d: f64 = a.iter().zip(b).map(|(x, y)| (x - y).powi(2)).sum();
    let n: f64 = b.iter().map(|x| x * x).sum::<f64>().max(1e-300);
    (d / n).sqrt()
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let (p, alpha, tol, maxit) = (4usize, 5.0, 1e-8, 4000);
    let g = 24usize;
    let xr = [0.0, 1.0];
    let yr = [0.0, 1.0];
    let mesh = Mesh2d::rectangular(p, g, g, xr, yr);
    let nn = mesh.refq.n_nodes();
    let ndof = mesh.n_elements() * nn;
    println!("=== WHILE-graph PCG vs readback PCG ({g}² p={p}, tol={tol}) ===\n");

    let mut worst = 0.0f64;
    let mut ok = true;

    // --- Case 1: non-singular velocity Helmholtz (λM + A) -------------------------------
    {
        let lambda = 100.0;
        let mut src = vec![0.0; ndof];
        for (e, el) in mesh.elements.iter().enumerate() {
            for k in 0..nn {
                src[e * nn + k] = (PI * el.geom.x[k]).sin() * (PI * el.geom.y[k]).sin();
            }
        }
        let op = Poisson::with_reaction(&mesh, alpha, lambda);
        let rhs = op.rhs(&src.iter().map(|v| lambda * v).collect::<Vec<_>>(), |_, _| 0.0);

        let h_ref = GpuPoissonMg::new(PMultigrid::with_reaction(p, g, g, xr, yr, alpha, lambda))?;
        let (x_ref, it_ref) = h_ref.solve(&rhs, tol, maxit)?;
        let h_while = GpuPoissonMg::new(PMultigrid::with_reaction(p, g, g, xr, yr, alpha, lambda))?
            .with_while_graph(true)?;
        let (x_while, it_while) = h_while.solve(&rhs, tol, maxit)?;

        let r = rel(&x_while, &x_ref);
        worst = worst.max(r);
        let pass = r < 1e-6;
        ok &= pass;
        println!("Helmholtz (λ={lambda}): readback {it_ref} iters, WHILE-graph {it_while} iters, rel = {r:.3e}  [{}]", if pass { "OK" } else { "FAIL" });
    }

    // --- Case 2: singular pure-Neumann pressure (deflated) ------------------------------
    {
        let tags = mesh.boundary_tags();
        let op = Poisson::with_bc(&mesh, alpha, 0.0, tags.clone());
        // Compatible (zero-mean) RHS for the singular system.
        let mut f = vec![0.0; ndof];
        for (e, el) in mesh.elements.iter().enumerate() {
            for k in 0..nn {
                f[e * nn + k] = (2.0 * PI * el.geom.x[k]).cos() * (2.0 * PI * el.geom.y[k]).cos();
            }
        }
        let rhs = op.rhs(&f, |_, _| 0.0);

        let h_ref = GpuPoissonMg::new(PMultigrid::with_bc(p, g, g, xr, yr, alpha, 0.0, tags.clone()))?;
        let (x_ref, it_ref) = h_ref.solve(&rhs, tol, maxit)?;
        let h_while = GpuPoissonMg::new(PMultigrid::with_bc(p, g, g, xr, yr, alpha, 0.0, tags.clone()))?
            .with_while_graph(true)?;
        let (x_while, it_while) = h_while.solve(&rhs, tol, maxit)?;

        // Singular ⇒ solution defined up to a constant; compare mean-removed.
        let mr = |v: &[f64]| -> Vec<f64> {
            let m = v.iter().sum::<f64>() / v.len() as f64;
            v.iter().map(|x| x - m).collect()
        };
        let r = rel(&mr(&x_while), &mr(&x_ref));
        worst = worst.max(r);
        let pass = r < 1e-6;
        ok &= pass;
        println!("singular pressure: readback {it_ref} iters, WHILE-graph {it_while} iters, rel (mean-removed) = {r:.3e}  [{}]", if pass { "OK" } else { "FAIL" });
    }

    // --- Timing: per-solve cost, readback vs WHILE-graph (Helmholtz, cold solves) -------
    {
        let lambda = 100.0;
        let mut src = vec![0.0; ndof];
        for (e, el) in mesh.elements.iter().enumerate() {
            for k in 0..nn {
                src[e * nn + k] = (PI * el.geom.x[k]).sin() * (PI * el.geom.y[k]).sin();
            }
        }
        let op = Poisson::with_reaction(&mesh, alpha, lambda);
        let rhs = op.rhs(&src.iter().map(|v| lambda * v).collect::<Vec<_>>(), |_, _| 0.0);
        let reps = 40usize;
        let time_solves = |h: &GpuPoissonMg| -> Result<f64, Box<dyn std::error::Error>> {
            h.solve(&rhs, tol, maxit)?; // warm up (JIT/alloc)
            let t = std::time::Instant::now();
            for _ in 0..reps {
                h.solve(&rhs, tol, maxit)?;
            }
            Ok(1e3 * t.elapsed().as_secs_f64() / reps as f64)
        };
        let h_ref = GpuPoissonMg::new(PMultigrid::with_reaction(p, g, g, xr, yr, alpha, lambda))?;
        let ms_ref = time_solves(&h_ref)?;
        let h_while = GpuPoissonMg::new(PMultigrid::with_reaction(p, g, g, xr, yr, alpha, lambda))?.with_while_graph(true)?;
        let ms_while = time_solves(&h_while)?;
        println!(
            "\nTIMING (cold Helmholtz solve, {reps} reps): readback {ms_ref:.3} ms/solve  vs  WHILE-graph {ms_while:.3} ms/solve  ({:.2}×)",
            ms_ref / ms_while
        );
        println!("(per-solve capture re-instantiates the graph each call; amortized capture-once is a follow-on.)");
    }

    println!("\nworst rel = {worst:.3e}");
    if ok {
        println!("OK: WHILE-graph PCG matches the readback PCG — convergence loop runs device-side, no per-iteration readback.");
        Ok(())
    } else {
        eprintln!("FAIL: WHILE-graph PCG mismatch.");
        std::process::exit(1);
    }
}
