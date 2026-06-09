//! Validation harness for **split-form convection + per-region BCs** on the GPU —
//! [`gale_gpu::GpuStokes`] with `convection_scheme = SplitFormDg` driven through
//! `step_ns_forced_bc`. Box flow with a west inlet `u=(1,0)` (tag 3), east traction-free
//! outflow (tag 1), and free-slip (symmetry) top/bottom walls (tags 0,2). The uniform
//! flow `u=(1,0)`, `p=0` is the exact steady state — preserving it exercises every branch
//! of the entropy-stable convection ghost (Dirichlet inflow, transparent outflow,
//! reflected slip wall). Checked against the exact field and the CPU `Stokes` oracle.
//!
//! Run: cargo oxide run --bin flow-split-bc-check

use gale::dg::{BoundaryConditions, ConvectionScheme, FlowBc, Mesh2d, Stokes};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("=== GPU split-form convection + per-region BCs vs CPU (uniform flow preserved) ===\n");

    let (nu, alpha, dt, nsteps) = (1.0, 5.0, 0.05, 20usize);
    let p = 4;
    let mesh = Mesh2d::rectangular(p, 5, 3, [0.0, 2.0], [0.0, 1.0]);
    let ndof = mesh.n_elements() * mesh.refq.n_nodes();

    let bcs = || {
        BoundaryConditions::no_slip()
            .set(3, FlowBc::velocity(|_, _, _| (1.0, 0.0)))
            .set(1, FlowBc::Outflow)
            .set(0, FlowBc::Symmetry)
            .set(2, FlowBc::Symmetry)
    };
    let z = vec![0.0; ndof];

    let gbcs = bcs();
    let mut gst = gale_gpu::GpuStokes::with_bcs(&mesh, alpha, nu, dt, &gbcs);
    gst.convection_scheme = ConvectionScheme::SplitFormDg;
    let (mut gux, mut guy) = (vec![1.0; ndof], vec![0.0; ndof]);

    let cbcs = bcs();
    let mut cst = Stokes::with_bcs(&mesh, alpha, nu, dt, &cbcs);
    cst.convection_scheme = ConvectionScheme::SplitFormDg;
    let (mut cux, mut cuy) = (vec![1.0; ndof], vec![0.0; ndof]);

    let mut t = 0.0;
    for _ in 0..nsteps {
        t += dt;
        let (nx, ny) = gst.step_ns_forced_bc(&gux, &guy, t, &gbcs, &z, &z)?;
        gux = nx;
        guy = ny;
        let (mx, my) = cst.step_ns_forced_bc(&cux, &cuy, t, &cbcs, &z, &z);
        cux = mx;
        cuy = my;
    }

    let one = vec![1.0; ndof];
    let eu: Vec<f64> = gux.iter().map(|v| v - 1.0).collect();
    let err_exact = (gst.l2_norm(&eu).powi(2) + gst.l2_norm(&guy).powi(2)).sqrt() / gst.l2_norm(&one);
    let dx: Vec<f64> = gux.iter().zip(&cux).map(|(a, b)| a - b).collect();
    let dy: Vec<f64> = guy.iter().zip(&cuy).map(|(a, b)| a - b).collect();
    let cnorm = (cst.l2_norm(&cux).powi(2) + cst.l2_norm(&cuy).powi(2)).sqrt().max(1e-300);
    let rel_cpu = (gst.l2_norm(&dx).powi(2) + gst.l2_norm(&dy).powi(2)).sqrt() / cnorm;

    let pass = err_exact < 1e-6 && rel_cpu < 1e-7;
    println!(
        "split-form inflow+outflow+symmetry: ‖u_gpu−(1,0)‖/‖u‖ = {err_exact:.3e}   ‖u_gpu−u_cpu‖/‖u‖ = {rel_cpu:.3e}   {}",
        if pass { "OK" } else { "FAIL" }
    );

    if pass {
        println!("\nPASS: GPU split-form convection + per-region BCs match the CPU oracle.");
        Ok(())
    } else {
        eprintln!("\nFAIL: GPU split-form per-region BC mismatch.");
        std::process::exit(1);
    }
}
