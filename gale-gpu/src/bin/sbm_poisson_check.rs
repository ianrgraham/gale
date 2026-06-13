//! Validation harness for [`gale_gpu::operators::poisson::sbm_poisson_apply`] — the GPU
//! Shifted Boundary Method operator (natural-Neumann surrogate) — against the CPU oracle
//! `gale::dg::ShiftedPoisson::…surrogate_neumann().apply`. This is increment 1 of the GPU SBM
//! port (docs/sbm-status.md): the matrix-free SBM matvec on device, bit-for-bit vs the CPU.
//!
//! Run: cargo oxide run --bin sbm-poisson-check

use gale::dg::{CircleLevelSet, Mesh2d, ShiftedBoundary, ShiftedPoisson};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    const P: usize = 4;
    let alpha = 5.0;
    println!("=== gale_gpu::sbm_poisson_apply (GPU) vs CPU ShiftedPoisson oracle (p={P}) ===\n");

    // Uniform axis-aligned rectangular mesh (the GPU affine-collapse requirement) with a circular
    // hole — the cylinder-pressure geometry. Tag the four box sides; outflow (tag 1) stays
    // Dirichlet, the rest Neumann — exactly the cylinder pressure operator's BCs.
    let mesh = Mesh2d::rectangular(P, 8, 6, [0.0, 2.0], [0.0, 1.5]);
    let nn = mesh.refq.n_nodes();
    let ne = mesh.n_elements();
    let ndof = ne * nn;
    let ls = CircleLevelSet::new(0.7, 0.7, 0.25);
    let sb = ShiftedBoundary::new(&mesh, &ls);
    let active = sb.active.clone();
    println!("mesh {}×{} p={P}, active {}/{} elements\n", 8, 6, sb.n_active(), ne);

    // A broadband state to exercise every term.
    let u: Vec<f64> = (0..ndof)
        .map(|i| {
            let h = (i as u64).wrapping_mul(2862933555777941757).wrapping_add(3037000493);
            ((h >> 11) as f64 / (1u64 << 53) as f64) * 2.0 - 1.0
        })
        .collect();

    let mut worst = 0.0f64;
    for (label, reaction) in [("pressure-Poisson (reaction 0)", 0.0), ("Helmholtz (reaction 5)", 5.0)] {
        let neumann = vec![0u32, 2, 3]; // outflow = tag 1 stays Dirichlet
        let cpu_op =
            ShiftedPoisson::with_bc(&mesh, alpha, reaction, neumann.clone(), sb.clone()).surrogate_neumann();
        let cpu = cpu_op.apply(&u);
        let gpu = gale_gpu::operators::poisson::sbm_poisson_apply(&mesh, &active, &u, alpha, reaction, &neumann)?;

        let mut max_abs = 0.0f64;
        let mut cpu_norm = 0.0f64;
        for i in 0..ndof {
            let d = (gpu[i] - cpu[i]).abs();
            max_abs = max_abs.max(d);
            cpu_norm = cpu_norm.max(cpu[i].abs());
        }
        let max_rel = max_abs / cpu_norm.max(1e-300);
        worst = worst.max(max_rel);
        println!("{label}: max|GPU−CPU| = {max_abs:.3e}   rel = {max_rel:.3e}");
    }

    println!();
    if worst < 1e-12 {
        println!("OK: GPU SBM operator matches the CPU oracle bit-for-bit (rel {worst:.1e})");
        Ok(())
    } else {
        eprintln!("FAIL: GPU SBM operator deviates from CPU oracle (rel {worst:.1e})");
        std::process::exit(1);
    }
}
