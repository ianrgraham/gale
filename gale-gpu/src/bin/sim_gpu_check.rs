//! Framework GPU execution across crates: assemble a `gale::sim::Simulation` with a
//! GPU-backed semidiscretization from `gale-gpu` and run it via the same HOOMD-style
//! API as the CPU path. Validates the GPU-run trajectory against a CPU-run one
//! (identical `Mol` + SSP-RK3 scheme; only the rhs backend differs).
//!
//! Run: cargo oxide run --bin sim-gpu-check

use gale::dg::{Hyperbolic, LinearAdvection, Mesh2d};
use gale::sim::{ClosureSemi, Integrator, Mol, Simulation, SspRk3, SspRk3State, State};
use std::f64::consts::PI;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let p = 4;
    let (ax, ay) = (0.8, 0.4);
    let dt = 1e-3;
    let nsteps = 200u64;
    println!("=== GPU-backed Simulation (gale-gpu) vs CPU-backed, {nsteps} steps (p={p}) ===\n");

    let mesh = Mesh2d::rectangular_periodic(p, 4, 4, [0.0, 1.0], [0.0, 1.0]);
    let init = |x: f64, y: f64| (2.0 * PI * x).sin() * (2.0 * PI * y).cos();

    // GPU-backed Simulation: Mol + SspRk3State driving gale_gpu::GpuAdvection, run
    // through the standard Simulation API — each rhs evaluation executes on the GPU.
    let mut gst = State::new(mesh.clone());
    let uid = gst.add_field_from("u", &[init]);
    let mut sim = Simulation::new(gst);
    sim.set_integrator(Mol::new(gale_gpu::GpuAdvection::new(uid, ax, ay), SspRk3State::new(dt)));
    sim.run(nsteps);
    let gpu = sim.state.field("u").component(0).to_vec();

    // CPU reference: same SSP-RK3 scheme, rhs from gale's CPU Hyperbolic operator.
    let hyp = Hyperbolic::new(&mesh, LinearAdvection { ax, ay });
    let bc = |_x: f64, _y: f64, _t: f64, _o: &mut [f64]| {};
    let csemi = ClosureSemi::new(1, hyp.ndof(), |s: &[Vec<f64>], t: f64| hyp.rhs(s, t, &bc));
    let scheme = SspRk3::new(dt);
    let mut cstate: Vec<Vec<f64>> = {
        let mut st = State::new(mesh.clone());
        st.add_field_from("u", &[init]);
        vec![st.field("u").component(0).to_vec()]
    };
    let mut t = 0.0;
    for _ in 0..nsteps {
        cstate = scheme.step(&csemi, &cstate, t);
        t += dt;
    }
    let cpu = &cstate[0];

    let scale = cpu.iter().fold(0.0f64, |a, &v| a.max(v.abs())).max(1e-300);
    let max_abs = gpu.iter().zip(cpu).fold(0.0f64, |a, (g, c)| a.max((g - c).abs()));
    println!("dofs={}  steps={nsteps}  max|gpu_sim − cpu_sim| / |u| = {:.3e}", gpu.len(), max_abs / scale);
    if max_abs / scale < 1e-9 {
        println!("\nPASS: GPU-backed Simulation matches the CPU-backed run through the framework.");
        Ok(())
    } else {
        eprintln!("\nFAIL: GPU/CPU simulation divergence {:.3e}", max_abs / scale);
        std::process::exit(1);
    }
}
