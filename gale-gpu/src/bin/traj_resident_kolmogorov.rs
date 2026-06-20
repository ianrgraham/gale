//! **Device-resident** Kolmogorov elastic-turbulence trajectory.
//!
//! Drives `GpuResidentVe` — the 100%-device-resident VE dual-splitting step (all handles on one
//! shared stream; conformation → stress-divergence → momentum → transport → implicit κ stress
//! diffusion → trace limiter all on device; fields stay resident across steps, the host only reads
//! back u + tr C every `EVERY` steps for the trajectory dump). This is GPU-compute-bound (~100% SM)
//! and ~4× the host-orchestrated `traj-amr-kolmogorov` integrator at 256², which idles the GPU ~50%
//! doing per-step host assembly. Uniform mesh (no AMR): at ≥128² the base already resolves the
//! strands (AMR added ~2% elements), and the on-device stress diffusion keeps it stable where the
//! un-stabilised host AMR run blew up.
//!
//! Writes the same `u` + `trC` trajectory `.h5` as `traj-amr-kolmogorov`, so `spectrum.py` consumes
//! it identically. Run:
//!   RK_N=256 RK_STEPS=6000 RK_EVERY=50 RK_LAMBDA=14 RK_KAPPA=0.01 \
//!     cargo oxide run --features traj --bin traj-resident-kolmogorov
//! (bare binary needs CUDA_OXIDE_TARGET=sm_70).

use gale::dg::{LogConfOldroydB, Mesh2d};
use gale_gpu::resident::GpuResidentVe;
use gale_traj::TrajectoryWriter;

fn env_usize(k: &str, d: usize) -> usize {
    std::env::var(k).ok().and_then(|v| v.parse().ok()).unwrap_or(d)
}
fn env_f64(k: &str, d: f64) -> f64 {
    std::env::var(k).ok().and_then(|v| v.parse().ok()).unwrap_or(d)
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let p = 3usize;
    let n = env_usize("RK_N", 128);
    let (xr, yr) = ([0.0, 1.0], [0.0, 1.0]);
    let nu0 = env_f64("RK_NU0", 0.1);
    let beta = env_f64("RK_BETA", 0.5);
    let lambda = env_f64("RK_LAMBDA", 14.0);
    let force = env_f64("RK_F", 2.0);
    let nk = env_f64("RK_NK", 2.0);
    let alpha = 5.0;
    let dt = env_f64("RK_DT", 2e-3);
    let steps = env_usize("RK_STEPS", 6000);
    let every = env_usize("RK_EVERY", 50);
    let bmax = env_f64("RK_BMAX", 500.0); // FENE-P-like trace cap (≈L²)
    let kappa = env_f64("RK_KAPPA", 0.01); // on-device implicit stress diffusion (Sc = ν₀/κ)
    // Solve tolerance: 1e-6 is plenty for turbulence (time-discretisation error dominates) and is
    // ~1.7× faster than 1e-9 at 256² (the step is elliptic-solve-bound — profiled 471→280 ms/step).
    let tol = env_f64("RK_TOL", 1e-6);
    let out = std::env::args()
        .nth(1)
        .or_else(|| std::env::var("RK_OUT").ok())
        .unwrap_or_else(|| format!("/tmp/rkolmo_{n}.h5"));

    let eta_s = beta * nu0;
    let eta_p = (1.0 - beta) * nu0;
    let two_pi = std::f64::consts::TAU;
    let fx = move |_x: f64, y: f64| force * (two_pi * nk * y).sin();
    let fy = |_x: f64, _y: f64| 0.0;
    let u_est = force / (nu0 * (two_pi * nk).powi(2));
    let (wi, re) = (lambda * u_est * two_pi * nk, u_est / nu0); // L = 1
    println!(
        "=== device-resident VE Kolmogorov {n}×{n} p={p}, {steps} steps (β={beta}, λ={lambda}, \
         F={force}, n={nk}) ⇒ U≈{u_est:.3}, Wi≈{wi:.2}, Re≈{re:.2}, El≈{:.1}, κ={kappa:.2e} \
         (Sc={:.1}) ===",
        wi / re.max(1e-9),
        nu0 / kappa.max(1e-30)
    );

    let mesh = Mesh2d::rectangular(p, n, n, xr, yr);
    let nn = mesh.refq.n_nodes();
    let ne = mesh.n_elements();
    // Kolmogorov IC: laminar shear + a small symmetry-breaking perturbation to seed the instability.
    let (mut ux0, mut uy0) = (vec![0.0; ne * nn], vec![0.0; ne * nn]);
    for (e, el) in mesh.elements.iter().enumerate() {
        for k in 0..nn {
            ux0[e * nn + k] = u_est * (two_pi * nk * el.geom.y[k]).sin();
            uy0[e * nn + k] =
                0.02 * u_est * (two_pi * el.geom.x[k]).sin() * (two_pi * el.geom.y[k]).sin();
        }
    }
    let mut ve = GpuResidentVe::new(&mesh, eta_s, eta_p, lambda, dt, alpha, kappa, fx, fy)?
        .with_trace_bound(bmax)
        .with_tol(tol, 4000);
    ve.set_velocity(&ux0, &uy0)?;

    // Ψ = log C ⇒ convert to C = exp(Ψ) before taking tr C (host, for the diagnostic only).
    let lc = LogConfOldroydB::new(&mesh, lambda, eta_p);
    let mut tw = TrajectoryWriter::create(&out, p, 2)?;
    let topo = tw.topology_for(&mesh)?; // uniform mesh ⇒ ONE fixed topology

    let pack = |ve: &GpuResidentVe| -> Result<(Vec<f32>, Vec<f32>, f64, f64, bool), Box<dyn std::error::Error>> {
        let (ux, uy) = ve.velocity()?;
        let psi = ve.psi()?;
        let cc = lc.conformation(&psi);
        let (mut uf, mut tf) = (vec![0f32; ne * nn * 2], vec![0f32; ne * nn]);
        let (mut trmax, mut vke, mut finite) = (0.0f64, 0.0f64, true);
        for g in 0..ne * nn {
            uf[g * 2] = ux[g] as f32;
            uf[g * 2 + 1] = uy[g] as f32;
            let tr = cc[0][g] + cc[2][g];
            tf[g] = tr as f32;
            trmax = trmax.max(tr);
            vke += uy[g] * uy[g];
            finite &= ux[g].is_finite() && uy[g].is_finite() && tr.is_finite();
        }
        vke /= (ne * nn) as f64;
        Ok((uf, tf, trmax, vke, finite))
    };

    let (u0, t0, tr0, vke0, _) = pack(&ve)?;
    tw.write_frame(0.0, 0, topo, ne, nn, &[("u", u0, 2), ("trC", t0, 1)], None)?;
    println!("  frame 0: {ne} elems, max tr C = {tr0:.3}, ⟨v²⟩ = {vke0:.3e}");

    let nchunks = steps / every;
    for c in 0..nchunks {
        ve.run(every)?;
        let (uf, tf, trmax, vke, finite) = pack(&ve)?;
        let step = ((c + 1) * every) as u64;
        if !finite {
            eprintln!("  BLEW UP at step {step} (non-finite) — raise κ or lower Wi");
            break;
        }
        tw.write_frame(step as f64 * dt, step, topo, ne, nn, &[("u", uf, 2), ("trC", tf, 1)], None)?;
        if c % 5 == 0 {
            println!("  frame {}: {ne} elems, max tr C = {trmax:.3}, ⟨v²⟩ = {vke:.3e}", c + 1);
        }
    }
    println!("wrote {} frames → {out}", tw.n_frames());
    Ok(())
}
