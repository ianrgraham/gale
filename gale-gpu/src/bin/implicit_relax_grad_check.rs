//! Validates gale's REAL `implicit_relax` differentiated through the automated
//! std::autodiff → cargo-oxide → Enzyme path: `logconf_implicit_relax_grad`
//! computes ∂Ψ/∂(1/λ) on the GPU via the pipeline-synthesized
//! `d_implicit_relax_core` kernel, checked against a central finite difference of
//! the GPU primal `logconf_implicit_relax`.
//!
//! Run: cargo oxide run --features autodiff --bin implicit-relax-grad-check

use gale::dg::{LogConfOldroydB, Mesh2d};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let p = 4;
    let mesh = Mesh2d::rectangular(p, 4, 4, [0.0, 1.0], [0.0, 1.0]);
    let nn = mesh.refq.n_nodes();
    let ne = mesh.n_elements();
    let ndof = ne * nn;
    let lambda = 0.5; // Oldroyd-B (α=0, b=∞ defaults) — the validated branch.
    let lc = LogConfOldroydB::new(&mesh, lambda, 1.0);
    let gamma = (1.0 - 0.5_f64.sqrt()) * 0.02;

    let mut b = [vec![0.0; ndof], vec![0.0; ndof], vec![0.0; ndof]];
    for (e, el) in mesh.elements.iter().enumerate() {
        for k in 0..nn {
            let (x, y) = (el.geom.x[k], el.geom.y[k]);
            let g = e * nn + k;
            b[0][g] = 0.6 + 0.4 * (x + y).sin();
            b[1][g] = 0.2 * x - 0.1 * y;
            b[2][g] = 0.3 + 0.3 * (x * y).cos();
        }
    }

    // Enzyme forward-mode ∂Ψ/∂(1/λ), built by `cargo oxide` and launched via cuda-host.
    let grad = gale_gpu::logconf_implicit_relax_grad(&mesh, &lc, &b, gamma)?;

    // Central finite difference of the GPU primal w.r.t. (1/λ).
    let il = 1.0 / lambda;
    let h = 1e-6;
    let lc_p = LogConfOldroydB::new(&mesh, 1.0 / (il + h), 1.0);
    let lc_m = LogConfOldroydB::new(&mesh, 1.0 / (il - h), 1.0);
    let psi_p = gale_gpu::logconf_implicit_relax(&mesh, &lc_p, &b, gamma)?;
    let psi_m = gale_gpu::logconf_implicit_relax(&mesh, &lc_m, &b, gamma)?;

    let mut max_rel = 0.0f64;
    let mut scale = 1e-30f64;
    for v in 0..3 {
        for i in 0..ndof {
            let fd = (psi_p[v][i] - psi_m[v][i]) / (2.0 * h);
            scale = scale.max(fd.abs());
            max_rel = max_rel.max((grad[v][i] - fd).abs());
        }
    }
    max_rel /= scale;
    println!("dofs={ndof}  ∂Ψ/∂(1/λ): max rel |enzyme − FD| = {max_rel:.3e}");
    println!(
        "  sample node 0: enzyme=[{:.6},{:.6},{:.6}]",
        grad[0][0], grad[1][0], grad[2][0]
    );
    if max_rel < 1e-4 {
        println!(
            "PASS: gale's REAL implicit_relax, differentiated by `cargo oxide --features autodiff`, \
             correct ∂/∂(1/λ) on the Titan V."
        );
        Ok(())
    } else {
        Err(format!("gradient mismatch: {max_rel:.3e}").into())
    }
}
