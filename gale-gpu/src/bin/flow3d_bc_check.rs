//! Validation harness for the GPU **3D per-region boundary conditions** —
//! [`gale_gpu::GpuStokes3d::with_bcs`] / `step_ns_forced_bc`. The 3D analogue of
//! `flow-bc-check`: a box with a west inlet `u=(1,0,0)` (tag 4), an east traction-free
//! outflow (tag 5), and free-slip (symmetry) on the four lateral faces (tags 0–3).
//! The uniform flow `u=(1,0,0)`, `p=0` is the exact steady state (harmonic,
//! divergence-free) and must be preserved — exercising inflow + outflow + symmetry and
//! the per-component velocity routing. Checked against the exact field and the validated
//! CPU `Stokes3d::with_bcs`.
//!
//! Run: cargo oxide run --bin flow3d-bc-check

use gale::dg::{BoundaryConditions3d, FlowBc3d, Mesh3d, Stokes3d};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("=== gale_gpu::GpuStokes3d::with_bcs per-region BCs vs CPU (uniform flow preserved) ===\n");

    let (nu, alpha, dt, nsteps) = (1.0, 5.0, 0.05, 15usize);
    let p = 3;
    let mesh = Mesh3d::rectangular(p, 3, 2, 2, [0.0, 2.0], [0.0, 1.0], [0.0, 1.0]);
    let ndof = mesh.n_elements() * mesh.refh.n_nodes();

    // West inlet (tag 4) u=(1,0,0), east outflow (tag 5), free-slip on the 4 lateral faces.
    let bcs = || {
        BoundaryConditions3d::no_slip()
            .set(4, FlowBc3d::velocity(|_, _, _, _| (1.0, 0.0, 0.0)))
            .set(5, FlowBc3d::Outflow)
            .set(0, FlowBc3d::Symmetry)
            .set(1, FlowBc3d::Symmetry)
            .set(2, FlowBc3d::Symmetry)
            .set(3, FlowBc3d::Symmetry)
    };

    let z = vec![0.0; ndof];

    let gbcs = bcs();
    let gst = gale_gpu::GpuStokes3d::with_bcs(&mesh, alpha, nu, dt, &gbcs);
    let (mut gux, mut guy, mut guz) = (vec![1.0; ndof], vec![0.0; ndof], vec![0.0; ndof]);

    let cbcs = bcs();
    let cst = Stokes3d::with_bcs(&mesh, alpha, nu, dt, &cbcs);
    let (mut cux, mut cuy, mut cuz) = (vec![1.0; ndof], vec![0.0; ndof], vec![0.0; ndof]);

    let mut t = 0.0;
    for _ in 0..nsteps {
        t += dt;
        let (nx, ny, nz) = gst.step_ns_forced_bc(&gux, &guy, &guz, t, &gbcs, &z, &z, &z)?;
        gux = nx;
        guy = ny;
        guz = nz;
        let (mx, my, mz) = cst.step_ns_forced_bc(&cux, &cuy, &cuz, t, &cbcs, &z, &z, &z);
        cux = mx;
        cuy = my;
        cuz = mz;
    }

    // Error vs the exact uniform flow.
    let err_u = gux.iter().fold(0.0f64, |a, &v| a.max((v - 1.0).abs()));
    let err_t = guy.iter().chain(guz.iter()).fold(0.0f64, |a, &v| a.max(v.abs()));
    // Error vs the CPU oracle.
    let dmax = |a: &[f64], b: &[f64]| a.iter().zip(b).fold(0.0f64, |m, (x, y)| m.max((x - y).abs()));
    let rel_cpu = dmax(&gux, &cux).max(dmax(&guy, &cuy)).max(dmax(&guz, &cuz));

    let pass = err_u < 1e-6 && err_t < 1e-6 && rel_cpu < 1e-7;
    println!(
        "inflow+outflow+symmetry: max|u_gpu−1|={err_u:.3e}  max|v,w|_gpu={err_t:.3e}  max|gpu−cpu|={rel_cpu:.3e}  {}",
        if pass { "OK" } else { "FAIL" }
    );

    if pass {
        println!("\nPASS: GPU 3D per-region BCs (inflow + outflow + symmetry) match the CPU oracle.");
        Ok(())
    } else {
        eprintln!("\nFAIL: GPU 3D per-region BC mismatch.");
        std::process::exit(1);
    }
}
