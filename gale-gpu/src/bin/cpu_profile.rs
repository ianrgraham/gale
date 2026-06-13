//! **CPU-path profile** (data for the CPU performance work). Measures, on representative
//! flow-sized elliptic problems, the two levers: (1) ALGORITHM — unpreconditioned `cg`
//! (what the CPU flow solvers currently use) vs the existing-but-unused `PMultigrid::pcg`;
//! (2) PARALLELISM — `Poisson::apply` throughput, and (run the whole bin under
//! `/usr/bin/time -v`) the "Percent of CPU" of the pcg solve, which exposes how much the
//! serial CG vector ops (`dot`/`axpy`) drag down the parallel `apply`.
//!
//! Run: cargo oxide run --bin cpu-profile   (CPU_N sets the grid; default 64)

use gale::dg::{Mesh2d, PMultigrid, Poisson};
use std::time::Instant;

fn broadband(n: usize) -> Vec<f64> {
    let mut s: Vec<f64> = (0..n)
        .map(|i| {
            let h = (i as u64).wrapping_mul(2862933555777941757).wrapping_add(3037000493);
            ((h >> 11) as f64 / (1u64 << 53) as f64) * 2.0 - 1.0
        })
        .collect();
    let m = s.iter().sum::<f64>() / n as f64;
    s.iter_mut().for_each(|v| *v -= m);
    s
}

fn time<T>(f: impl FnOnce() -> T) -> (T, f64) {
    let t0 = Instant::now();
    let r = f();
    (r, t0.elapsed().as_secs_f64() * 1e3)
}

fn main() {
    let p = 4;
    let n = std::env::var("CPU_N").ok().and_then(|v| v.parse().ok()).unwrap_or(64usize);
    let xr = [0.0, 1.0];
    let mesh = Mesh2d::rectangular(p, n, n, xr, xr);
    let ndof = mesh.n_elements() * mesh.refq.n_nodes();
    let alpha = 5.0;
    let (tol, maxit) = (1e-8, 100000);
    println!("=== CPU profile: {n}×{n} p={p}, ndof={ndof} ===\n");

    // --- ALGORITHM: the REAL flow pressure operator — SINGULAR pure-Neumann (reaction 0,
    // every boundary tag Neumann), solved with DEFLATED CG. (The all-Dirichlet variant the
    // first cut measured is NOT what the splitting solves; the closed-box/no-outflow pressure
    // is singular.) Compare deflated CG vs the deflated MG-PCG across sizes to find the win and
    // the crossover below which the V-cycle overhead makes MG-PCG a net loss on small meshes.
    let ntags = mesh.boundary_tags();
    let pois = Poisson::with_bc(&mesh, alpha, 0.0, ntags.clone());
    let mg = PMultigrid::with_bc(p, n, n, xr, xr, alpha, 0.0, ntags.clone());
    let f = broadband(ndof);
    let b = pois.rhs_mixed(&f, |_, _| 0.0, |_, _| 0.0);
    let ((_pc, cg_it), cg_ms) = time(|| pois.cg_deflated(&b, tol, maxit));
    let ((_pm, mg_it), mg_ms) = time(|| mg.pcg_deflated(&b, tol, maxit));
    println!("PRESSURE Poisson (SINGULAR pure-Neumann, deflated — the real flow operator):");
    println!("  deflated cg         : {cg_it:6} iters  {cg_ms:9.1} ms   <- what CPU flow used before");
    println!("  deflated p-MG-PCG   : {mg_it:6} iters  {mg_ms:9.1} ms   <- {:.1}× faster, {:.0}× fewer iters",
             cg_ms / mg_ms.max(1e-9), cg_it as f64 / mg_it.max(1) as f64);

    // --- ALGORITHM: velocity-Helmholtz (large reaction ⇒ mass-dominated, well-conditioned) ---
    let lambda = 1.0e5;
    let hpois = Poisson::with_reaction(&mesh, alpha, lambda);
    let hmg = PMultigrid::with_reaction(p, n, n, xr, xr, alpha, lambda);
    let bh = hpois.rhs(&f, |_, _| 0.0);
    let ((_, hcg_it, _), hcg_ms) = time(|| hpois.cg(&bh, tol, maxit));
    let ((_, hmg_it), hmg_ms) = time(|| hmg.pcg(&bh, tol, maxit));
    println!("\nVELOCITY Helmholtz (λ={lambda:.0e}, mass-dominated):");
    println!("  unpreconditioned cg : {hcg_it:6} iters  {hcg_ms:9.1} ms");
    println!("  p-MG-PCG            : {hmg_it:6} iters  {hmg_ms:9.1} ms");

    // --- PARALLELISM: apply throughput + the per-iter serial overhead ---
    let u = broadband(ndof);
    let reps = 50;
    let (_, ap_ms) = time(|| {
        let mut acc = 0.0;
        for _ in 0..reps {
            acc += pois.apply(&u)[0];
        }
        acc
    });
    let per_apply = ap_ms / reps as f64;
    // Isolate the V-cycle cost: PMultigrid::apply REBUILDS a Poisson operator each call
    // (multigrid.rs apply_level), vs the persistent Poisson::apply. And one precondition()
    // (a full V-cycle) — the per-iteration cost that capped the MG-PCG win at 4.6×.
    let (_, mgap_ms) = time(|| {
        let mut acc = 0.0;
        for _ in 0..reps {
            acc += mg.apply(&u)[0];
        }
        acc
    });
    let (_, pre_ms) = time(|| {
        let mut acc = 0.0;
        for _ in 0..5 {
            acc += mg.precondition(&u)[0];
        }
        acc
    });
    let levels: Vec<usize> = mg.meshes.iter().map(|m| m.n_elements()).collect();
    // Per-apply FLOOR check: time apply on coarse levels. If a p=1 / small-element apply
    // isn't proportionally cheaper than the p=4 finest, a fixed per-apply overhead
    // (RefineQuad::new + allocations per call) dominates the V-cycle's ~56 applies.
    for &lvl in &[0usize, 2, 4, 6] {
        if lvl < mg.meshes.len() {
            let m = &mg.meshes[lvl];
            let op = Poisson::with_reaction(m, alpha, 0.0);
            let un = vec![0.5; m.n_elements() * m.refq.n_nodes()];
            let (_, t) = time(|| {
                let mut a = 0.0;
                for _ in 0..reps {
                    a += op.apply(&un)[0];
                }
                a
            });
            println!("  apply level {lvl} ({:5} elem, p={}): {:.3} ms", m.n_elements(), m.order, t / reps as f64);
        }
    }
    println!("\nV-CYCLE BREAKDOWN:");
    println!("  PMultigrid::apply    : {:.3} ms/apply  (rebuilds Poisson each call; vs persistent {per_apply:.3})", mgap_ms / reps as f64);
    println!("  precondition (1 V-cyc): {:.1} ms      (this × {mg_it} = the MG-PCG cost)", pre_ms / 5.0);
    println!("  levels (elements)    : {levels:?}  (coarse_solve does ≤500 CG iters on the last)");

    println!("\nPARALLELISM:");
    println!("  Poisson::apply       : {per_apply:.3} ms/apply (parallel)");
    // The unpreconditioned cg did cg_it applies + cg_it serial dot/axpy passes; comparing its
    // ms/iter to a bare apply shows the serial vector-op overhead per iteration.
    println!("  cg ms/iter           : {:.3} ms/iter  (apply {:.3} + serial dot/axpy {:.3})",
             cg_ms / cg_it.max(1) as f64, per_apply, (cg_ms / cg_it.max(1) as f64 - per_apply).max(0.0));
    println!("\n(Run under `/usr/bin/time -v` and read 'Percent of CPU': 6400% = all 64 physical\n cores; the gap to that is the serial CG vector-op / assembly fraction.)");
}
