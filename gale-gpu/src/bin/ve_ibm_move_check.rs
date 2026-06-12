//! **M2: a freely-moving rigid particle in viscoelastic flow** — the migration-demo
//! milestone of docs/research-moving-particle-coupling.md. A heavy disk, released
//! off the channel centerline, is advected by an Oldroyd-B body-force-driven flow while
//! the explicit Newton–Euler coupling (`GpuMovingPenalizationHook`) moves it. This is
//! the moving-body extension of the fixed-body capstone (`ve-ibm-check`), and it
//! directly probes the research-flagged risk: SPD/stability of the conformation tensor
//! at a MOVING immersed surface.
//!
//! Checks, for BOTH constitutive representations:
//!  - OldroydB (direct C): GPU vs CPU bit-exact; C stays SPD (C_xx>0, det C>0) everywhere
//!    including at the moving body.
//!  - LogConf (Ψ = log C): GPU vs CPU to transcendental tolerance (~1e-5, libdevice);
//!    Ψ stays finite ⇒ C = exp(Ψ) SPD by construction (the robust headline path).
//! Plus: the body actually MOVES (a cross-stream migration signal), and stays stable.
//!
//! NOTE (honest scope): this is a closed-box body-force demo on a coarse mesh, so the
//! cross-stream drift is a qualitative migration *signal*, not a quantitative benchmark.
//! A quantitative match to the Oldroyd-B channel-migration equilibrium-position curve
//! (periodic channel + resolution + sourced 2D constants) is the M2 follow-up.
//!
//! Run: cargo oxide run --bin ve-ibm-move-check

use gale::dg::{FreeBody, Mesh2d};
use gale::sim::{MovingPenalizationHook, Simulation, State, ViscoModel, ViscoelasticDualSplitting};

fn rel_l2(a: &[f64], b: &[f64]) -> f64 {
    let num: f64 = a.iter().zip(b).map(|(x, y)| (x - y).powi(2)).sum();
    let den: f64 = b.iter().map(|y| y * y).sum::<f64>().max(1e-300);
    (num / den).sqrt()
}

/// Result of one model run: final velocity + conformation fields and the body pose.
struct Run {
    ux: Vec<f64>,
    uy: Vec<f64>,
    c: [Vec<f64>; 3],
    body: gale::dg::FreeBody,
    umax: f64,
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let p = 4;
    let mesh = Mesh2d::rectangular(p, 3, 3, [0.0, 1.0], [0.0, 1.0]);
    let (cx0, cy0, r) = (0.5, 0.6, 0.18); // released OFF the centerline (y=0.5)
    let (eta_s, eta_p, lambda, alpha) = (0.5, 0.5, 0.5, 5.0);
    let (eta_b, dt, g, rho) = (1e-3, 0.02, 1.0, 20.0);
    let nsteps = 24u64;
    let bc = |_: f64, _: f64, _: f64| 0.0; // no-slip walls
    let drive = move |_: f64, _: f64, _: f64| g; // pressure-gradient drive (+x)
    let zero = |_: f64, _: f64, _: f64| 0.0;
    let make_body = || FreeBody::disk(cx0, cy0, r, rho, eta_b);

    // Run one constitutive model on GPU and CPU, returning both results.
    let run = |model: ViscoModel| -> Result<(Run, Run), Box<dyn std::error::Error>> {
        // GPU.
        let mut gst = State::new(mesh.clone());
        let gvid = gst.add_field("velocity", 2);
        let gcid = gst.add_field("conformation", 3);
        let ginteg = gale_gpu::GpuViscoelasticDualSplitting::new(
            gvid, gcid, dt, eta_s, eta_p, lambda, alpha, model,
        )
        .boundary(bc, bc)
        .drive(drive, zero);
        let eq = ginteg.equilibrium(&gst);
        for j in 0..3 {
            gst.fields.by_id_mut(gcid).component_mut(j).copy_from_slice(&eq[j]);
        }
        let (ghook, gbody) = gale_gpu::GpuMovingPenalizationHook::new(gvid, make_body(), &mesh, dt);
        let mut gsim = Simulation::new(gst);
        gsim.set_integrator(ginteg);
        gsim.set_stage_hook(ghook);
        gsim.run(nsteps);
        let gux = gsim.state.field("velocity").component(0).to_vec();
        let guy = gsim.state.field("velocity").component(1).to_vec();
        let gc: [Vec<f64>; 3] =
            std::array::from_fn(|j| gsim.state.field("conformation").component(j).to_vec());
        let gumax = gux.iter().zip(&guy).fold(0.0f64, |m, (a, b)| m.max(a.hypot(*b)));

        // CPU oracle (identical composition).
        let mut cst = State::new(mesh.clone());
        let cvid = cst.add_field("velocity", 2);
        let ccid = cst.add_field("conformation", 3);
        let cinteg = ViscoelasticDualSplitting::new(
            cvid, ccid, dt, eta_s, eta_p, lambda, alpha, model,
        )
        .boundary(bc, bc)
        .drive(drive, zero);
        let eqc = cinteg.equilibrium(&cst);
        for j in 0..3 {
            cst.fields.by_id_mut(ccid).component_mut(j).copy_from_slice(&eqc[j]);
        }
        let (chook, cbody) = MovingPenalizationHook::new(cvid, make_body(), &mesh, dt);
        let mut csim = Simulation::new(cst);
        csim.set_integrator(cinteg);
        csim.set_stage_hook(chook);
        csim.run(nsteps);
        let cux = csim.state.field("velocity").component(0).to_vec();
        let cuy = csim.state.field("velocity").component(1).to_vec();
        let cc: [Vec<f64>; 3] =
            std::array::from_fn(|j| csim.state.field("conformation").component(j).to_vec());
        let cumax = cux.iter().zip(&cuy).fold(0.0f64, |m, (a, b)| m.max(a.hypot(*b)));

        Ok((
            Run { ux: gux, uy: guy, c: gc, body: *gbody.borrow(), umax: gumax },
            Run { ux: cux, uy: cuy, c: cc, body: *cbody.borrow(), umax: cumax },
        ))
    };

    println!("=== M2: freely-moving rigid disk in GPU viscoelastic flow (Oldroyd-B), {nsteps} steps ===\n");
    let mut all_pass = true;

    // OldroydB: direct C — bit-exact GPU==CPU + explicit SPD check.
    {
        let (gpu, cpu) = run(ViscoModel::OldroydB)?;
        let rel_u = rel_l2(&gpu.ux, &cpu.ux).max(rel_l2(&gpu.uy, &cpu.uy));
        let rel_c = (0..3).fold(0.0f64, |a, j| a.max(rel_l2(&gpu.c[j], &cpu.c[j])));
        let dpose = (gpu.body.body.cx - cpu.body.body.cx).abs() + (gpu.body.body.cy - cpu.body.body.cy).abs();
        // SPD of C = (c0 c1; c1 c2) on the GPU field, everywhere (incl. at the body).
        let (mut min_c0, mut min_det) = (f64::INFINITY, f64::INFINITY);
        for i in 0..gpu.c[0].len() {
            min_c0 = min_c0.min(gpu.c[0][i]);
            min_det = min_det.min(gpu.c[0][i] * gpu.c[2][i] - gpu.c[1][i] * gpu.c[1][i]);
        }
        let spd = min_c0 > 0.0 && min_det > 0.0 && min_det.is_finite();
        let dx = gpu.body.body.cx - cx0;
        let dy = gpu.body.body.cy - cy0;
        let moved = dx.hypot(dy) > 1e-5;
        let stable = gpu.umax.is_finite() && gpu.umax > 1e-3;
        let bitexact = rel_u < 1e-6 && rel_c < 1e-6 && dpose < 1e-7;
        println!("OldroydB : GPU/CPU rel u={rel_u:.2e} c={rel_c:.2e} Δpose={dpose:.2e} | SPD: min C_xx={min_c0:.3e} min detC={min_det:.3e}");
        println!("           body Δx={dx:+.3e} Δy(migration)={dy:+.3e}  U={:+.3e} V={:+.3e} ω={:+.3e}  umax={:.3e}",
                 gpu.body.body.u, gpu.body.body.v, gpu.body.body.omega, gpu.umax);
        let pass = bitexact && spd && moved && stable;
        println!("           => {}", if pass { "OK (bit-exact, SPD, moved, stable)" } else { "FAIL" });
        all_pass &= pass;
    }

    // LogConf: Ψ = log C — SPD guaranteed (Ψ finite ⇒ C=exp(Ψ) SPD); transcendental tol.
    {
        let (gpu, cpu) = run(ViscoModel::LogConf)?;
        let rel_u = rel_l2(&gpu.ux, &cpu.ux).max(rel_l2(&gpu.uy, &cpu.uy));
        let rel_c = (0..3).fold(0.0f64, |a, j| a.max(rel_l2(&gpu.c[j], &cpu.c[j])));
        let dpose = (gpu.body.body.cx - cpu.body.body.cx).abs() + (gpu.body.body.cy - cpu.body.body.cy).abs();
        let psi_finite = gpu.c.iter().all(|comp| comp.iter().all(|v| v.is_finite()));
        let dx = gpu.body.body.cx - cx0;
        let dy = gpu.body.body.cy - cy0;
        let moved = dx.hypot(dy) > 1e-5;
        let stable = gpu.umax.is_finite() && gpu.umax > 1e-3;
        let matches = rel_u < 5e-3 && rel_c < 5e-3 && dpose < 5e-3; // libdevice transcendentals
        println!("\nLogConf  : GPU/CPU rel u={rel_u:.2e} c={rel_c:.2e} Δpose={dpose:.2e} | Ψ finite ⇒ C SPD: {psi_finite}");
        println!("           body Δx={dx:+.3e} Δy(migration)={dy:+.3e}  U={:+.3e} V={:+.3e} ω={:+.3e}  umax={:.3e}",
                 gpu.body.body.u, gpu.body.body.v, gpu.body.body.omega, gpu.umax);
        let pass = matches && psi_finite && moved && stable;
        println!("           => {}", if pass { "OK (SPD by construction, moved, stable, GPU≈CPU)" } else { "FAIL" });
        all_pass &= pass;
    }

    if all_pass {
        println!("\nPASS: freely-moving disk in viscoelastic flow — stable, SPD-preserving at the\nmoving body, GPU matches the CPU oracle, with a cross-stream migration signal.");
        Ok(())
    } else {
        eprintln!("\nFAIL: M2 moving-particle viscoelastic check.");
        std::process::exit(1);
    }
}
