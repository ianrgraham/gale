//! Validation harness for [`gale_gpu::GpuViscoelasticDualSplitting`] — the coupled GPU
//! viscoelastic flow integrator (build-order step 4). Runs a pressure-driven Oldroyd-B
//! **Poiseuille channel** through `gale::sim::Simulation` (velocity + conformation, GPU
//! velocity solve + GPU conformation transport) and checks it against the validated
//! direct CPU `gale::dg::ViscoelasticFlow` loop with identical setup, for both the
//! direct Oldroyd-B and the log-conformation models.
//!
//! Run: cargo oxide run --bin ve-check

use gale::dg::{ConstitutiveModel, Mesh2d, ViscoelasticFlow};
use gale::sim::{Simulation, State, ViscoModel};

fn rel_l2(a: &[f64], b: &[f64]) -> f64 {
    let num: f64 = a.iter().zip(b).map(|(x, y)| (x - y).powi(2)).sum();
    let den: f64 = b.iter().map(|y| y * y).sum::<f64>().max(1e-300);
    (num / den).sqrt()
}

/// `tol` is the GPU-vs-CPU agreement bound over the 40-step coupled run. It is not
/// bit-for-bit: the GPU/CPU CG solves converge to the same residual but different
/// round-off, and log-conformation adds libdevice eig/exp/log divergence — so the
/// bound is engineering-precision, looser for the transcendental-heavy log model.
fn run_model(model: ViscoModel, name: &str, tol: f64) -> Result<bool, Box<dyn std::error::Error>> {
    let p = 4;
    let mesh = Mesh2d::rectangular(p, 3, 3, [0.0, 1.0], [0.0, 1.0]);
    let (eta_s, eta_p, lambda, g) = (0.5, 0.5, 0.5, 1.0);
    let eta0 = eta_s + eta_p;
    let nn = mesh.refq.n_nodes();
    let dt = 0.02;
    let nsteps = 40u64;
    let u_exact = move |y: f64| (g / (2.0 * eta0)) * y * (1.0 - y);
    let bc_u = move |_x: f64, y: f64, _t: f64| u_exact(y);
    let bc_v = |_: f64, _: f64, _: f64| 0.0;
    let drive_x = move |_: f64, _: f64, _: f64| g;
    let zero_f = |_: f64, _: f64, _: f64| 0.0;

    // GPU framework run.
    let mut st = State::new(mesh.clone());
    let vid = st.add_field("velocity", 2);
    let cid = st.add_field("conformation", 3);
    let integ = gale_gpu::GpuViscoelasticDualSplitting::new(
        vid, cid, dt, eta_s, eta_p, lambda, 5.0, model,
    )
    .boundary(bc_u, bc_v)
    .drive(drive_x, zero_f);
    let eq = integ.equilibrium(&st);
    {
        let cf = st.fields.by_id_mut(cid);
        for j in 0..3 {
            cf.component_mut(j).copy_from_slice(&eq[j]);
        }
    }
    let mut sim = Simulation::new(st);
    sim.set_integrator(integ);
    sim.run(nsteps);

    // CPU reference: direct ViscoelasticFlow loop, identical setup. (with_model picks
    // the same constitutive model; ViscoelasticFlow::new uses Oldroyd-B.)
    let mut rux = vec![0.0; mesh.n_elements() * nn];
    let mut ruy = rux.clone();
    let (gux, guy, gc);
    let (rc_final, rux_final, ruy_final);
    match model {
        ViscoModel::OldroydB => {
            let ve = ViscoelasticFlow::new(&mesh, eta_s, eta_p, lambda, dt, 5.0);
            let mut rc = ve.model.equilibrium();
            let mut t = 0.0;
            for _ in 0..nsteps {
                t += dt;
                let (nx, ny, nc) = ve.step(&rux, &ruy, &rc, t, bc_u, bc_v, drive_x, zero_f);
                rux = nx;
                ruy = ny;
                rc = nc;
            }
            rc_final = rc;
            rux_final = rux;
            ruy_final = ruy;
        }
        ViscoModel::LogConf => {
            let m = gale::dg::LogConfOldroydB::new(&mesh, lambda, eta_p);
            let ve = ViscoelasticFlow::with_model(&mesh, eta_s, dt, 5.0, m);
            let mut rc = ve.model.equilibrium();
            let mut t = 0.0;
            for _ in 0..nsteps {
                t += dt;
                let (nx, ny, nc) = ve.step(&rux, &ruy, &rc, t, bc_u, bc_v, drive_x, zero_f);
                rux = nx;
                ruy = ny;
                rc = nc;
            }
            rc_final = rc;
            rux_final = rux;
            ruy_final = ruy;
        }
    }

    gux = sim.state.field("velocity").component(0).to_vec();
    guy = sim.state.field("velocity").component(1).to_vec();
    gc = [
        sim.state.field("conformation").component(0).to_vec(),
        sim.state.field("conformation").component(1).to_vec(),
        sim.state.field("conformation").component(2).to_vec(),
    ];

    let ru = rel_l2(&gux, &rux_final).max(rel_l2(&guy, &ruy_final));
    let rc = (0..3).fold(0.0f64, |a, j| a.max(rel_l2(&gc[j], &rc_final[j])));
    let umax = gux.iter().fold(0.0f64, |a, &v| a.max(v.abs()));
    let pass = ru < tol && rc < tol && umax.is_finite() && umax > 1e-3;
    println!(
        "{name}: vel rel={ru:.3e}  conf rel={rc:.3e}  umax={umax:.4}  {}",
        if pass { "OK" } else { "FAIL" }
    );
    Ok(pass)
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("=== GPU viscoelastic Poiseuille channel (40 steps, p=4) vs CPU ViscoelasticFlow ===\n");
    let mut ok = true;
    // Direct Oldroyd-B is pure arithmetic ⇒ tight (vel ~1e-7 vs CPU); it shares the exact
    // same GPU velocity solve as log-conf, so it is the real sentinel that the velocity
    // path (now the p-MG-PCG default) is correct. Log-conformation uses libdevice
    // transcendentals (eig/exp/log) whose GPU results differ from the host at ~1e-5 (the
    // conformation agreement); over 40 coupled steps that, plus the matrix-free reduction
    // ORDER (multi-block dot + gather-form SIPG face term), amplify through the stiff
    // exp/log transport. Switching the velocity solve from CG to the p-MG-PCG default adds
    // another round-off-level (~1e-7, per Oldroyd-B) perturbation that the log-conf
    // transport amplifies to ~3e-3 — physically the same trajectory (umax unchanged), so
    // the bound here is engineering-precision, not bit-level. Correctness is pinned by
    // Oldroyd-B (1e-6) and the conformation match; this gate only guards gross divergence.
    ok &= run_model(ViscoModel::OldroydB, "Oldroyd-B", 1e-6)?;
    ok &= run_model(ViscoModel::LogConf, "log-conformation", 5e-3)?;
    if ok {
        println!("\nPASS: coupled GPU viscoelastic flow matches the CPU oracle (both models).");
        Ok(())
    } else {
        eprintln!("\nFAIL: GPU viscoelastic mismatch.");
        std::process::exit(1);
    }
}
