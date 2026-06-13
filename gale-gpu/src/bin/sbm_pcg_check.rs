//! Validation harness for [`gale_gpu::operators::poisson::sbm_pcg_solve`] — the GPU SBM-aware
//! p-multigrid-preconditioned CG (natural-Neumann surrogate) — against the CPU oracle
//! `gale::dg::ShiftedMultigrid::pcg`. Increment 2 of the GPU SBM port (docs/sbm-status.md):
//! the full device V-cycle/PCG for the SBM operator.
//!
//! Run: cargo oxide run --bin sbm-pcg-check

use gale::dg::{CircleLevelSet, Mesh2d, ShiftedBoundary, ShiftedMultigrid, ShiftedPoisson};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    const P: usize = 4;
    let alpha = 5.0;
    let (nx, ny) = (16usize, 12usize);
    println!("=== gale_gpu::sbm_pcg_solve (GPU) vs CPU ShiftedMultigrid::pcg (p={P}, {nx}×{ny}) ===\n");

    let mesh = Mesh2d::rectangular(P, nx, ny, [0.0, 2.0], [0.0, 1.5]);
    let nn = mesh.refq.n_nodes();
    let ndof = mesh.n_elements() * nn;
    let ls = CircleLevelSet::new(0.7, 0.7, 0.25);
    let sb = ShiftedBoundary::new(&mesh, &ls);
    // Non-singular pressure: outer tags 0/2/3 Neumann, tag 1 Dirichlet (pins the constant) — the
    // cylinder pressure configuration. Natural-Neumann surrogate, reaction 0.
    let neumann = vec![0u32, 2, 3];
    println!("active {}/{} elements\n", sb.n_active(), mesh.n_elements());

    let smg = ShiftedMultigrid::new(P, nx, ny, [0.0, 2.0], [0.0, 1.5], alpha, 0.0, neumann.clone(), &ls, false, false);
    let cpu_op = ShiftedPoisson::with_bc(&mesh, alpha, 0.0, neumann.clone(), sb.clone()).surrogate_neumann();

    // A range-consistent RHS: b = A·x_true.
    let x_true: Vec<f64> = (0..ndof)
        .map(|i| {
            let h = (i as u64).wrapping_mul(2862933555777941757).wrapping_add(3037000493);
            ((h >> 11) as f64 / (1u64 << 53) as f64) * 2.0 - 1.0
        })
        .collect();
    let b = cpu_op.apply(&x_true);

    let (tol, maxit) = (1e-9, 2000);
    let (cpu, cpu_it) = smg.pcg(&b, tol, maxit);
    let (gpu, gpu_it) = gale_gpu::operators::poisson::sbm_pcg_solve(&smg, &b, tol, maxit)?;

    // Compare GPU vs CPU solution (active dofs; both are PCG to the same tol so they agree to ~tol,
    // not bit-for-bit — the on-device reductions differ in summation order).
    let active = cpu_op.active();
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
    let rel = (diff / den.max(1e-300)).sqrt();
    println!("CPU MG-PCG : {cpu_it} iters");
    println!("GPU MG-PCG : {gpu_it} iters");
    println!("rel ‖GPU−CPU‖ (active) = {rel:.3e}");

    if rel < 1e-6 && gpu_it <= cpu_it + 3 {
        println!("\nOK: GPU SBM-MG-PCG matches the CPU oracle and converges in a comparable iteration count");
        Ok(())
    } else {
        eprintln!("\nFAIL: GPU SBM-MG-PCG disagrees with CPU (rel {rel:.1e}, GPU {gpu_it} vs CPU {cpu_it} iters)");
        std::process::exit(1);
    }
}
