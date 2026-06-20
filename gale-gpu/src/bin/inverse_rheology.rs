//! Inverse rheology by differentiating through gale's REAL GPU conformation solve.
//!
//! Fits the relaxation parameters (1/λ, Giesekus α) to "measured" conformation data
//! by gradient descent, with the loss gradient computed in a SINGLE reverse-mode
//! Enzyme pass through the pipeline-synthesized `dr_implicit_relax_core` kernel
//! (`logconf_implicit_relax_vjp`). Forward solve = `logconf_implicit_relax`.
//!
//! Run: cargo oxide run --features autodiff --bin inverse-rheology

use gale::dg::{LogConfOldroydB, Mesh2d};

fn solve(mesh: &Mesh2d, il: f64, al: f64, b: &[Vec<f64>; 3], gamma: f64) -> [Vec<f64>; 3] {
    let lc = LogConfOldroydB::new(mesh, 1.0 / il, 1.0).with_mobility(al);
    gale_gpu::logconf_implicit_relax(mesh, &lc, b, gamma).expect("forward solve")
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let p = 4;
    let mesh = Mesh2d::rectangular(p, 4, 4, [0.0, 1.0], [0.0, 1.0]);
    let nn = mesh.refq.n_nodes();
    let ne = mesh.n_elements();
    let ndof = ne * nn;
    let gamma = (1.0 - 0.5_f64.sqrt()) * 0.05;

    let mut b = [vec![0.0; ndof], vec![0.0; ndof], vec![0.0; ndof]];
    for (e, el) in mesh.elements.iter().enumerate() {
        for k in 0..nn {
            let (x, y) = (el.geom.x[k], el.geom.y[k]);
            let g = e * nn + k;
            // Large-deformation B so the Giesekus α term carries signal (α is
            // weakly identifiable at small stretch).
            b[0][g] = 1.6 + 1.1 * (x + y).sin();
            b[1][g] = 0.7 * x - 0.35 * y;
            b[2][g] = 1.1 + 0.9 * (x * y).cos();
        }
    }

    // Synthetic "measured" data at the true parameters.
    let (il_true, al_true) = (2.0, 0.3); // 1/λ = 2.0 (λ=0.5), Giesekus α = 0.3
    let meas = solve(&mesh, il_true, al_true, &b, gamma);
    println!("true params: 1/λ = {il_true:.4}, α = {al_true:.4}");

    // --- gradient-consistency sanity check: reverse VJP vs forward grad ---
    let (il0, al0) = (1.2, 0.1);
    let lc0 = LogConfOldroydB::new(&mesh, 1.0 / il0, 1.0).with_mobility(al0);
    let psi0 = solve(&mesh, il0, al0, &b, gamma);
    let mut r = [vec![0.0; ndof], vec![0.0; ndof], vec![0.0; ndof]];
    for c in 0..3 {
        for i in 0..ndof {
            r[c][i] = psi0[c][i] - meas[c][i]; // ∂L/∂Ψ for L = ½‖Ψ−Ψ_meas‖²
        }
    }
    let (dil_rev, _dal_rev) = gale_gpu::logconf_implicit_relax_vjp(&mesh, &lc0, &b, gamma, &r)?;
    let fwd = gale_gpu::logconf_implicit_relax_grad(&mesh, &lc0, &b, gamma)?;
    let dil_fwd: f64 = (0..3).map(|c| (0..ndof).map(|i| r[c][i] * fwd[c][i]).sum::<f64>()).sum();
    let rel = (dil_rev - dil_fwd).abs() / (dil_fwd.abs() + 1e-30);
    println!("consistency ∂L/∂(1/λ): reverse={dil_rev:.6} forward·r={dil_fwd:.6} rel={rel:.2e}");
    assert!(rel < 1e-6, "reverse VJP disagrees with forward-mode gradient");

    // --- gradient-descent fit: one reverse pass per step gives both params'
    // gradients; the whole loop runs under a single CUDA context. ---
    let n_iter: usize = std::env::var("FIT_ITERS").ok().and_then(|s| s.parse().ok()).unwrap_or(6000);
    // Adam step sizes (per-parameter adaptive); the (1/λ, α) valley is elongated.
    let (il, al, loss0) =
        gale_gpu::logconf_fit_relax_params(&mesh, &b, gamma, f64::INFINITY, &meas, il0, al0, n_iter, 1.5e-2, 1.5e-2)?;

    println!("recovered: 1/λ = {il:.4} (true {il_true:.4}), α = {al:.4} (true {al_true:.4})  final loss = {loss0:.3e}");
    let ok = (il - il_true).abs() < 2e-2 && (al - al_true).abs() < 2e-2;
    if ok {
        println!(
            "PASS: inverse rheology — params recovered by reverse-mode Enzyme through gale's \
             real GPU conformation solve, built by `cargo oxide --features autodiff`."
        );
        Ok(())
    } else {
        Err("parameter fit did not converge".into())
    }
}
