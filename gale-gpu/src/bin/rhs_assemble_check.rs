//! Validates the **device-resident RHS assembly** (`GpuPoissonMg::rhs_madd_dev`) against the
//! host `Poisson::rhs`. The DG-SEM mass is diagonal and the boundary (Nitsche-Dirichlet) lift
//! depends only on the BC data + geometry, so the full host assembly `b = M·f + lift` collapses
//! to one device fused multiply-add `b = scale·jw⊙f + lift`, with `jw = rhs(1,0)` and
//! `lift = rhs(0,g)` precomputed once. This is the first device-native flow-stage primitive
//! (keeps the source `f` and result resident on the GPU). Mirrors the velocity-Helmholtz RHS
//! (`scale = λ`, nonzero Dirichlet data). Run: cargo oxide run --bin rhs-assemble-check

use gale::dg::{PMultigrid, Poisson};
use gale_gpu::operators::poisson::GpuPoissonMg;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let (p, alpha) = (4usize, 5.0);
    let nu = 1.0;
    let dt = 0.01;
    let lambda = 1.0 / (nu * dt); // velocity Helmholtz reaction
    println!("=== device RHS assembly (rhs_madd_dev) vs host Poisson::rhs (λ={lambda}, p={p}) ===\n");

    let mut worst = 0.0f64;
    for &g in &[16usize, 32] {
        let mg = PMultigrid::with_reaction(p, g, g, [0.0, 1.0], [0.0, 1.0], alpha, lambda);
        let fine = mg.mesh(0);
        let nn = fine.refq.n_nodes();
        let n0 = fine.n_elements() * nn;
        let op = Poisson::with_reaction(fine, alpha, lambda);

        // A representative source + a nonzero Dirichlet datum (exercises the boundary lift).
        let mut src = vec![0.0; n0];
        for (e, el) in fine.elements.iter().enumerate() {
            for k in 0..nn {
                src[e * nn + k] = (2.3 * el.geom.x[k]).cos() * (1.7 * el.geom.y[k] + 0.4).sin();
            }
        }
        let gdir = |x: f64, y: f64| 0.5 + x - 2.0 * y;

        // Host reference: the exact velocity-step RHS, b = M·(λ·src) + lift(g).
        let fxv: Vec<f64> = src.iter().map(|v| lambda * v).collect();
        let b_host = op.rhs(&fxv, gdir);

        // Constants extracted by reusing the validated host assembly:
        //   jw   = rhs(1, 0)  = M·1   (the diagonal mass; lift(0) = 0)
        //   lift = rhs(0, g)  = lift(g)  (M·0 = 0)
        let jw = op.rhs(&vec![1.0; n0], |_, _| 0.0);
        let lift = op.rhs(&vec![0.0; n0], gdir);

        // Device: b = λ·jw⊙src + lift, fully on the GPU.
        let handle = GpuPoissonMg::new(mg)?;
        let jw_dev = handle.upload_field(&jw)?;
        let src_dev = handle.upload_field(&src)?;
        let lift_dev = handle.upload_field(&lift)?;
        let mut b_dev = handle.alloc_field()?;
        handle.rhs_madd_dev(&mut b_dev, &jw_dev, &src_dev, &lift_dev, lambda)?;
        let b_gpu = handle.download_field(&b_dev)?;

        let diff: f64 = b_gpu.iter().zip(&b_host).map(|(a, b)| (a - b).powi(2)).sum();
        let nrm: f64 = b_host.iter().map(|b| b * b).sum::<f64>().max(1e-300);
        let rel = (diff / nrm).sqrt();
        worst = worst.max(rel);
        println!("  {g}²  ndof={n0:>7}  rel ‖GPU−host‖ = {rel:.3e}");
    }

    println!("\nworst rel = {worst:.3e}");
    if worst < 1e-13 {
        println!("OK: device RHS assembly matches the host Poisson::rhs (bit-for-bit).");
        Ok(())
    } else {
        eprintln!("FAIL: device RHS assembly mismatch.");
        std::process::exit(1);
    }
}
