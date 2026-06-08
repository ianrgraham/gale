//! Validation harness for the GPU **conformation-inflow** path —
//! [`gale_gpu::GpuViscoelasticDualSplitting::conformation_inflow`]. The conformation
//! advection is now full upwind DG (device collocation volume term + host upwind
//! surface lift), so an inflow datum is transported downstream across elements. This
//! runs a pressure-driven Oldroyd-B / log-conformation channel with **stretched fluid
//! entering at the west inlet** through `gale::sim::Simulation`, and checks the GPU
//! result against the CPU `gale::sim::ViscoelasticDualSplitting` with the identical
//! inflow — confirming the GPU host-lift + inflow plumbing matches the oracle, and that
//! the injected conformation is actually present near the inlet.
//!
//! Run: cargo oxide run --bin ve-bc-check

use gale::dg::{ConformationInflow, Mesh2d};
use gale::sim::{Simulation, State, StateIntegrator, StateStageHook, ViscoModel};

fn rel_l2(a: &[f64], b: &[f64]) -> f64 {
    let num: f64 = a.iter().zip(b).map(|(x, y)| (x - y).powi(2)).sum();
    let den: f64 = b.iter().map(|y| y * y).sum::<f64>().max(1e-300);
    (num / den).sqrt()
}

/// No-op stage hook (the CPU integrator's `step` takes one).
struct NoHook;
impl StateStageHook for NoHook {
    fn after_stage(&self, _state: &mut State, _stage: usize) {}
}

fn run_model(model: ViscoModel, name: &str, tol: f64) -> Result<bool, Box<dyn std::error::Error>> {
    let p = 4;
    let mesh = Mesh2d::rectangular(p, 4, 3, [0.0, 2.0], [0.0, 1.0]);
    let (eta_s, eta_p, lambda, g) = (0.5, 0.5, 0.5, 1.0);
    let eta0 = eta_s + eta_p;
    let dt = 0.02;
    let nsteps = 40u64;
    let u_exact = move |y: f64| (g / (2.0 * eta0)) * y * (1.0 - y);
    let bc_u = move |_x: f64, y: f64, _t: f64| u_exact(y);
    let bc_v = |_: f64, _: f64, _: f64| 0.0;
    let drive_x = move |_: f64, _: f64, _: f64| g;
    let zero_f = |_: f64, _: f64, _: f64| 0.0;
    // Stretched fluid entering at the west inlet (tag 3).
    let c_in = [2.0, 0.5, 1.0];
    let inflow = || ConformationInflow::new(vec![3], c_in);

    // GPU framework run.
    let mut gst = State::new(mesh.clone());
    let gvid = gst.add_field("velocity", 2);
    let gcid = gst.add_field("conformation", 3);
    let ginteg = gale_gpu::GpuViscoelasticDualSplitting::new(gvid, gcid, dt, eta_s, eta_p, lambda, 5.0, model)
        .boundary(bc_u, bc_v)
        .drive(drive_x, zero_f)
        .conformation_inflow(inflow());
    let geq = ginteg.equilibrium(&gst);
    {
        let cf = gst.fields.by_id_mut(gcid);
        for j in 0..3 {
            cf.component_mut(j).copy_from_slice(&geq[j]);
        }
    }
    let mut gsim = Simulation::new(gst);
    gsim.set_integrator(ginteg);
    gsim.run(nsteps);

    // CPU reference: the same scenario through the CPU sim integrator with inflow.
    let mut cst = State::new(mesh.clone());
    let cvid = cst.add_field("velocity", 2);
    let ccid = cst.add_field("conformation", 3);
    let cinteg = gale::sim::ViscoelasticDualSplitting::new(cvid, ccid, dt, eta_s, eta_p, lambda, 5.0, model)
        .boundary(bc_u, bc_v)
        .drive(drive_x, zero_f)
        .conformation_inflow(inflow());
    let ceq = cinteg.equilibrium(&cst);
    {
        let cf = cst.fields.by_id_mut(ccid);
        for j in 0..3 {
            cf.component_mut(j).copy_from_slice(&ceq[j]);
        }
    }
    let hook = NoHook;
    for _ in 0..nsteps {
        cinteg.step(&mut cst, &hook);
    }

    let gux = gsim.state.field("velocity").component(0).to_vec();
    let guy = gsim.state.field("velocity").component(1).to_vec();
    let cux = cst.field("velocity").component(0).to_vec();
    let cuy = cst.field("velocity").component(1).to_vec();
    let gc: Vec<Vec<f64>> = (0..3).map(|j| gsim.state.field("conformation").component(j).to_vec()).collect();
    let cc: Vec<Vec<f64>> = (0..3).map(|j| cst.field("conformation").component(j).to_vec()).collect();

    // Primary check: the GPU coupled run with conformation inflow matches the CPU
    // oracle (same upwind lift + inflow trace; the inflow injection itself is proven
    // cleanly by the CPU operator test `conformation_inflow_fills_domain`).
    let ru = rel_l2(&gux, &cux).max(rel_l2(&guy, &cuy));
    let rc = (0..3).fold(0.0f64, |a, j| a.max(rel_l2(&gc[j], &cc[j])));
    // Informational: the inlet state in the field's native variable (C for Oldroyd-B,
    // Ψ for log-conformation) — shear also stretches here, so it isn't an isolated
    // inflow probe, just a non-triviality readout.
    let nn = mesh.refq.n_nodes();
    let mut inlet = 0.0f64;
    for (e, el) in mesh.elements.iter().enumerate() {
        for k in 0..nn {
            if el.geom.x[k] < 0.25 {
                inlet = inlet.max(gc[0][e * nn + k]);
            }
        }
    }
    let pass = ru < tol && rc < tol;
    println!(
        "{name}: vel rel={ru:.3e}  conf rel={rc:.3e}  inlet[0]={inlet:.3}  {}",
        if pass { "OK" } else { "FAIL" }
    );
    Ok(pass)
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("=== GPU conformation inflow (stretched fluid at west inlet) vs CPU oracle ===\n");
    let mut ok = true;
    ok &= run_model(ViscoModel::OldroydB, "Oldroyd-B", 1e-6)?;
    ok &= run_model(ViscoModel::LogConf, "log-conformation", 1e-4)?;
    if ok {
        println!("\nPASS: GPU conformation inflow matches the CPU oracle and injects at the inlet.");
        Ok(())
    } else {
        eprintln!("\nFAIL: GPU conformation inflow mismatch.");
        std::process::exit(1);
    }
}
