//! Validation harness for the GPU 3D viscoelastic coupling
//! [`gale_gpu::GpuViscoelasticDualSplitting3d`] for **both** the direct Oldroyd-B and
//! the log-conformation models. Each drives a 3D channel through `Simulation<Mesh3d>`
//! and is checked against a CPU reference dual-split loop (host stress-divergence →
//! `Stokes3d::step_ns_forced` → host `step_ssp_rk3`). (The standalone conformation
//! rhs kernels are validated by `ve3d-check`'s Oldroyd path / `logconf3d-check`.)
//!
//! Run: cargo oxide run --bin ve3d-check

use gale::dg::{LogConfOldroydB3d, Mesh3d, OldroydB3d, Stokes3d};
use gale::sim::{Simulation, State, ViscoModel};

fn rel_l2(a: &[f64], b: &[f64]) -> f64 {
    let num: f64 = a.iter().zip(b).map(|(x, y)| (x - y).powi(2)).sum();
    let den: f64 = b.iter().map(|y| y * y).sum::<f64>().max(1e-300);
    (num / den).sqrt()
}

fn run(model: ViscoModel, name: &str, tol: f64) -> Result<bool, Box<dyn std::error::Error>> {
    let p = 3;
    let mesh = Mesh3d::rectangular(p, 2, 2, 2, [0.0, 1.0], [0.0, 1.0], [0.0, 1.0]);
    let (eta_s, eta_p, lambda, alpha) = (0.5, 0.5, 0.5, 5.0);
    let (g, dt, nsteps) = (1.0, 0.02, 16u64);
    let bc = |_: f64, _: f64, _: f64, _: f64| 0.0;
    let drive = move |_: f64, _: f64, _: f64, _: f64| g;
    let zero = |_: f64, _: f64, _: f64, _: f64| 0.0;

    // GPU framework run.
    let mut st: State<Mesh3d> = State::new(mesh.clone());
    let vid = st.add_field("velocity", 3);
    let cid = st.add_field("conformation", 6);
    let integ = gale_gpu::GpuViscoelasticDualSplitting3d::new(vid, cid, dt, eta_s, eta_p, lambda, alpha, model)
        .boundary(bc, bc, bc)
        .drive(drive, zero, zero);
    let eq = integ.equilibrium(&st);
    for o in 0..6 {
        st.fields.by_id_mut(cid).component_mut(o).copy_from_slice(&eq[o]);
    }
    let mut sim = Simulation::new(st);
    sim.set_integrator(integ);
    sim.run(nsteps);
    let gux = sim.state.field("velocity").component(0).to_vec();
    let gc: Vec<Vec<f64>> = (0..6).map(|o| sim.state.field("conformation").component(o).to_vec()).collect();

    // CPU reference dual-split loop (no CPU 3D-VE-flow integrator exists).
    let stokes = Stokes3d::new(&mesh, alpha, eta_s, dt);
    let nn = mesh.refh.n_nodes();
    let ndof = mesh.n_elements() * nn;
    let (mut rux, mut ruy, mut ruz) = (vec![0.0; ndof], vec![0.0; ndof], vec![0.0; ndof]);
    let ob = OldroydB3d::new(&mesh, lambda, eta_p);
    let lcm = LogConfOldroydB3d::new(&mesh, lambda, eta_p);
    let mut rc = match model {
        ViscoModel::OldroydB => ob.identity(),
        ViscoModel::LogConf => lcm.identity(),
    };
    let mut t = 0.0;
    for _ in 0..nsteps {
        t += dt;
        let (mut bx, by, bz) = match model {
            ViscoModel::OldroydB => ob.stress_divergence(&rc),
            ViscoModel::LogConf => lcm.stress_divergence(&rc),
        };
        for v in bx.iter_mut() {
            *v += g;
        }
        let (nx, ny, nz) = stokes.step_ns_forced(&rux, &ruy, &ruz, t, bc, bc, bc, &bx, &by, &bz);
        rc = match model {
            ViscoModel::OldroydB => ob.step_ssp_rk3(&rc, &nx, &ny, &nz, dt),
            ViscoModel::LogConf => lcm.step_ssp_rk3(&rc, &nx, &ny, &nz, dt),
        };
        rux = nx;
        ruy = ny;
        ruz = nz;
    }

    let vel_err = rel_l2(&gux, &rux);
    let conf_err = (0..6).fold(0.0f64, |a, o| a.max(rel_l2(&gc[o], &rc[o])));
    let umax = gux.iter().fold(0.0f64, |a, &v| a.max(v.abs()));
    let pass = vel_err < tol && conf_err < tol && umax > 1e-3;
    println!("{name}: vel rel = {vel_err:.3e}  conf rel = {conf_err:.3e}  umax = {umax:.4}  {}", if pass { "OK" } else { "FAIL" });
    Ok(pass)
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("=== GPU 3D viscoelastic coupled flow (both models) vs CPU oracle ===\n");
    let mut ok = true;
    // Oldroyd-B is pure arithmetic ⇒ tight; log-conformation uses libdevice eig/exp.
    ok &= run(ViscoModel::OldroydB, "Oldroyd-B", 1e-6)?;
    ok &= run(ViscoModel::LogConf, "log-conformation", 1e-4)?;
    if ok {
        println!("\nPASS: GPU 3D viscoelastic coupled flow matches the CPU oracle (both models).");
        Ok(())
    } else {
        eprintln!("\nFAIL: GPU 3D viscoelastic mismatch.");
        std::process::exit(1);
    }
}
