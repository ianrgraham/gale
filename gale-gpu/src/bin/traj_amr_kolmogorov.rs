//! 2D **viscoelastic Kolmogorov flow** — the canonical elastic-turbulence model — with AMR
//! resolving the thin birefringent stress strands. A sinusoidal body force `fx = F·sin(2πn y)`
//! (zero at the no-slip top/bottom walls) drives `n` counter-flowing shear bands; at high
//! Weissenberg number the elastic instability of the shear layers throws off thin polymer-stress
//! strands. The flow is advanced by the GPU log-conformation viscoelastic integrator and an
//! `AmrUpdater` refines where the conformation is under-resolved (the strands) and coarsens the
//! smooth bulk. The dump re-emits topology on each remesh; view with `gale-view --grid`.
//!
//! Inertialess elastic turbulence wants Re≪1, Wi≳O(1) (El = Wi/Re large). This is a research-grade
//! regime — start at moderate Wi (it should be stable with log-conformation) and push up; if it
//! blows up we add a bound-preserving / stress-diffusion stabilizer.
//!
//! Run:  cargo oxide run --features traj --bin traj-amr-kolmogorov
//! View: gale-view /tmp/kolmo.h5 --field trC --comp 0 --grid --all --out /tmp/kolmo
//!
//! Env: KO_N (base res), KO_NU0 (total visc), KO_BETA (η_s/η₀), KO_LAMBDA (relax → Wi), KO_F
//! (forcing), KO_NK (force wavenumber), KO_DT, TRAJ_STEPS/EVERY, KO_REFINE/KO_COARSEN, KO_AMR (0/1).

use gale::dg::Mesh2d;
use gale::sim::{AmrUpdater, Periodic, Simulation, State, ViscoModel};
use gale_traj::TrajectoryWriter;

fn env_usize(k: &str, d: usize) -> usize {
    std::env::var(k).ok().and_then(|v| v.parse().ok()).unwrap_or(d)
}
fn env_f64(k: &str, d: f64) -> f64 {
    std::env::var(k).ok().and_then(|v| v.parse().ok()).unwrap_or(d)
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let p = 3;
    let (nx, ny) = (env_usize("KO_N", 24), env_usize("KO_N", 24));
    let xr = [0.0, 1.0];
    let yr = [0.0, 1.0];
    let nu0 = env_f64("KO_NU0", 0.1); // total zero-shear kinematic viscosity
    let beta = env_f64("KO_BETA", 0.5); // solvent ratio η_s/η₀
    let lambda = env_f64("KO_LAMBDA", 1.0); // polymer relaxation time (↑ ⇒ ↑ Wi)
    let force = env_f64("KO_F", 2.0); // body-force amplitude
    let nk = env_f64("KO_NK", 2.0); // forcing wavenumber (shear bands)
    let alpha = 5.0;
    let dt = env_f64("KO_DT", 5e-4);
    let steps = env_usize("TRAJ_STEPS", 600);
    let every = env_usize("TRAJ_EVERY", 10);
    let refine_thr = env_f64("KO_REFINE", 3e-2);
    let coarsen_thr = env_f64("KO_COARSEN", 3e-3);
    let amr_on = env_usize("KO_AMR", 1) != 0;
    let bmax = env_f64("KO_BMAX", 0.0); // FENE-P-like trace cap (≈L²); 0 ⇒ unbounded Oldroyd-B
    let kappa = env_f64("KO_KAPPA", 0.0); // polymer stress diffusion κ (Sc = ν₀/κ); 0 ⇒ off
    let kimpl = env_usize("KO_KIMPL", 1) != 0; // 1 ⇒ implicit diffusion (no CFL); 0 ⇒ explicit
    // Output path: an explicit CLI arg / `KO_OUT` wins; otherwise auto-name from the swept
    // parameters so a parameter scan never overwrites itself. `Sc0` means diffusion off.
    let sc_tag = if kappa > 0.0 { format!("Sc{:.0}", nu0 / kappa) } else { "Sc0".to_string() };
    let out = std::env::args().nth(1).or_else(|| std::env::var("KO_OUT").ok()).unwrap_or_else(|| {
        format!(
            "/tmp/kolmo_lam{lambda}_F{force}_nu{nu0}_nk{nk}_{sc_tag}_amr{}.h5",
            amr_on as u8
        )
    });

    let eta_s = beta * nu0;
    let eta_p = (1.0 - beta) * nu0;
    let two_pi = std::f64::consts::TAU;
    let fx = move |_x: f64, y: f64, _t: f64| force * (two_pi * nk * y).sin();
    let fy = |_x: f64, _y: f64, _t: f64| 0.0;
    let zero = |_: f64, _: f64, _: f64| 0.0;
    // Rough Wi/Re from the laminar balance U ≈ F/(ν₀(2πn)²), γ̇ ≈ U·2πn.
    let u_est = force / (nu0 * (two_pi * nk).powi(2));
    let (wi, re) = (lambda * u_est * two_pi * nk, u_est / nu0); // L = 1
    println!(
        "=== VE Kolmogorov {nx}×{ny} p={p}, {steps} steps (β={beta}, λ={lambda}, F={force}, n={nk}) \
         ⇒ U≈{u_est:.3}, Wi≈{wi:.2}, Re≈{re:.2}, El≈{:.1}, AMR={amr_on} ===",
        wi / re.max(1e-9)
    );

    // Seed the laminar shear profile + a small symmetry-breaking perturbation so the elastic
    // instability has something to amplify.
    let u_ic = move |_x: f64, y: f64| u_est * (two_pi * nk * y).sin();
    let v_ic = move |x: f64, y: f64| 0.02 * u_est * (two_pi * x).sin() * (two_pi * y).sin();

    let mut st = State::new(Mesh2d::rectangular(p, nx, ny, xr, yr));
    let vid = st.add_field_from(
        "velocity",
        &[Box::new(u_ic) as Box<dyn Fn(f64, f64) -> f64>, Box::new(v_ic)],
    );
    // Conformation field. The LogConf model stores Ψ = log C, so equilibrium C = I is Ψ = 0.
    let cid = st.add_field_from(
        "conformation",
        &[
            Box::new(|_: f64, _: f64| 0.0) as Box<dyn Fn(f64, f64) -> f64>,
            Box::new(|_: f64, _: f64| 0.0),
            Box::new(|_: f64, _: f64| 0.0),
        ],
    );

    let mut integ = gale_gpu::GpuViscoelasticDualSplitting::new(
        vid, cid, dt, eta_s, eta_p, lambda, alpha, ViscoModel::LogConf,
    )
    .boundary(zero, zero)
    .drive(fx, fy);
    if bmax > 0.0 {
        integ = integ.with_trace_bound(bmax);
    }
    if kappa > 0.0 {
        integ = if kimpl {
            integ.with_stress_diffusion_implicit(kappa)
        } else {
            integ.with_stress_diffusion(kappa)
        };
        let mode = if kimpl { "implicit" } else { "explicit" };
        println!("  stress diffusion κ = {kappa:.2e}  (Sc = ν₀/κ = {:.2}, {mode})", nu0 / kappa);
    }
    let mut sim = Simulation::new(st);
    sim.set_integrator(integ);
    if amr_on {
        // Refine where the conformation (the stress strands) is under-resolved.
        sim.add_updater(
            AmrUpdater::new(p, nx, ny, xr, yr, "conformation", refine_thr).with_coarsening(coarsen_thr),
            Periodic::new(every as u64),
        );
    }

    let nn = sim.state.mesh.refq.n_nodes();
    let mut tw = TrajectoryWriter::create(&out, p, 2)?;
    // `topology_for` re-emits a topology whenever the mesh GEOMETRY changes (not just when the
    // element count changes) — an AMR remesh can swap which cells are refined at constant element
    // count, and pairing the new field with the old topology renders it at the wrong cells.
    let mut topo = tw.topology_for(&sim.state.mesh)?;

    // Pack velocity (2-comp) and the polymer-stretch diagnostic tr C (1-comp). The field stores
    // Ψ = log C, so convert Ψ → C = exp(Ψ) before taking the trace.
    let pack = |sim: &Simulation| -> (Vec<f32>, Vec<f32>, usize, f64, f64, bool) {
        let v = sim.state.field("velocity");
        let cfld = sim.state.field("conformation");
        let (ux, uy) = (v.component(0), v.component(1));
        let psi: [Vec<f64>; 3] =
            [cfld.component(0).to_vec(), cfld.component(1).to_vec(), cfld.component(2).to_vec()];
        let lc = gale::dg::LogConfOldroydB::new(&sim.state.mesh, lambda, eta_p);
        let cc = lc.conformation(&psi); // [Cxx, Cxy, Cyy] = exp(Ψ)
        let ne = sim.state.mesh.n_elements();
        let mut uf = vec![0f32; ne * nn * 2];
        let mut tf = vec![0f32; ne * nn];
        let mut trmax = 0.0f64;
        let mut vke = 0.0f64; // cross-stream kinetic energy ⟨v²⟩: 0 for laminar Kolmogorov,
        let mut finite = true; // grows/fluctuates at the elastic(-inertial) instability onset.
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
        (uf, tf, ne, trmax, vke, finite)
    };

    let (u0, t0, ne0, tr0, vke0, _) = pack(&sim);
    tw.write_frame(0.0, 0, topo, ne0, nn, &[("u", u0, 2), ("trC", t0, 1)], None)?;
    println!("  frame 0: {ne0} elems, max tr C = {tr0:.3}, ⟨v²⟩ = {vke0:.3e}");

    let nchunks = steps / every;
    for c in 0..nchunks {
        sim.run(every as u64);
        topo = tw.topology_for(&sim.state.mesh)?;
        let (uf, tf, ne, trmax, vke, finite) = pack(&sim);
        let step = ((c + 1) * every) as u64;
        // Stop BEFORE writing a non-finite frame: the trajectory then contains only valid frames,
        // and (with the per-frame flush in TrajectoryWriter) stays openable even on a blow-up.
        if !finite {
            eprintln!("  BLEW UP at step {step} (velocity/tr C non-finite) — lower Wi or add a stabilizer");
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
