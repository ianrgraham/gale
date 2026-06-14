//! Validation for the GPU **SBM velocity** operator — Dirichlet surrogate with the high-order
//! Taylor correction (`S_h u = u + ∇u·d`) — against the CPU oracle
//! `gale::dg::ShiftedPoisson::…taylor(true)`. Increment 3 of the GPU SBM port: the genuinely new
//! surrogate-Nitsche kernel path (SURR faces + uploaded shift vectors). Checks both the single
//! `sbm_poisson_apply` (bit-for-bit) and the full `sbm_pcg_solve` (vs CPU MG-PCG).
//!
//! Run: cargo oxide run --bin sbm-velocity-check

use gale::dg::{CircleLevelSet, Mesh2d, ShiftedBoundary, ShiftedMultigrid, ShiftedPoisson};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    const P: usize = 4;
    let alpha = 5.0;
    let lambda = 100.0; // velocity Helmholtz reaction (mass term)
    let (nx, ny) = (16usize, 12usize);
    println!("=== GPU SBM velocity (Dirichlet+Taylor surrogate, λ={lambda}) vs CPU (p={P}, {nx}×{ny}) ===\n");

    let mesh = Mesh2d::rectangular(P, nx, ny, [0.0, 2.0], [0.0, 1.5]);
    let nn = mesh.refq.n_nodes();
    let ndof = mesh.n_elements() * nn;
    let ls = CircleLevelSet::new(0.7, 0.7, 0.25);
    let sb = ShiftedBoundary::new(&mesh, &ls);
    let neumann = vec![1u32]; // outflow Neumann; the other walls Dirichlet (cylinder velocity BCs)
    println!("active {}/{} elements, {} surrogate faces\n", sb.n_active(), mesh.n_elements(), sb.faces.len());

    let cpu_op = ShiftedPoisson::with_bc(&mesh, alpha, lambda, neumann.clone(), sb.clone()).taylor(true);

    let rand: Vec<f64> = (0..ndof)
        .map(|i| {
            let h = (i as u64).wrapping_mul(2862933555777941757).wrapping_add(3037000493);
            ((h >> 11) as f64 / (1u64 << 53) as f64) * 2.0 - 1.0
        })
        .collect();

    // (1) Operator apply: GPU must match CPU bit-for-bit (the surrogate Nitsche + Taylor terms).
    let cpu_a = cpu_op.apply(&rand);
    let gpu_a = gale_gpu::operators::poisson::sbm_poisson_apply(&mesh, &sb, &rand, alpha, lambda, &neumann, true, true)?;
    let active = cpu_op.active();
    let (mut amax, mut anorm) = (0.0f64, 0.0f64);
    for i in 0..ndof {
        amax = amax.max((gpu_a[i] - cpu_a[i]).abs());
        anorm = anorm.max(cpu_a[i].abs());
    }
    let arel = amax / anorm.max(1e-300);
    println!("(1) apply: max|GPU−CPU| = {amax:.3e}   rel = {arel:.3e}");

    // (2) Full MG-PCG: GPU sbm_pcg_solve vs CPU ShiftedMultigrid::pcg (range-consistent RHS).
    let smg = ShiftedMultigrid::new(P, nx, ny, [0.0, 2.0], [0.0, 1.5], alpha, lambda, neumann.clone(), &ls, true, true);
    let b = cpu_op.apply(&rand);
    let (tol, maxit) = (1e-9, 2000);
    let (cpu, cpu_it) = smg.pcg(&b, tol, maxit);
    let (gpu, gpu_it) = gale_gpu::operators::poisson::sbm_pcg_solve(&smg, &b, tol, maxit)?;
    let (mut diff, mut den) = (0.0f64, 0.0f64);
    for e in 0..mesh.n_elements() {
        if !active[e] {
            continue;
        }
        for k in 0..nn {
            let d = gpu[e * nn + k] - cpu[e * nn + k];
            diff += d * d;
            den += cpu[e * nn + k] * cpu[e * nn + k];
        }
    }
    let srel = (diff / den.max(1e-300)).sqrt();
    println!("(2) MG-PCG: CPU {cpu_it} iters, GPU {gpu_it} iters, rel ‖GPU−CPU‖ = {srel:.3e}");

    println!();
    if arel < 1e-12 && srel < 1e-6 && gpu_it <= cpu_it + 3 {
        println!("OK: GPU SBM velocity operator (Dirichlet+Taylor) matches the CPU oracle");
        Ok(())
    } else {
        eprintln!("FAIL: GPU SBM velocity deviates (apply rel {arel:.1e}, solve rel {srel:.1e}, {gpu_it} vs {cpu_it} iters)");
        std::process::exit(1);
    }
}
