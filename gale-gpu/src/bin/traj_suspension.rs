//! Trajectory dump of a many-body suspension (M4) → one HDF5 file. Two freely-moving rigid disks
//! ride a uniform inflow stream `u=(1,0)`; the faster upstream disk overtakes the slower one and
//! short-range repulsion bounces them apart without overlap (same physics as `ibm_suspension_check`,
//! GPU-validated bit-for-bit vs the CPU oracle). The framework sim is advanced in chunks of
//! `TRAJ_EVERY` steps; after each chunk the velocity field + body poses are written as a frame.
//!
//! Run: cargo oxide run --features traj --bin traj-suspension
//! View: python gale-traj/python/view_traj.py /tmp/suspension.h5 --field u --comp mag --gif

use gale::dg::{FreeBody, Mesh2d, RigidBody, Suspension};
use gale::sim::{Simulation, State};
use gale_traj::TrajectoryWriter;

fn env_usize(k: &str, d: usize) -> usize {
    std::env::var(k).ok().and_then(|v| v.parse().ok()).unwrap_or(d)
}
fn env_f64(k: &str, d: f64) -> f64 {
    std::env::var(k).ok().and_then(|v| v.parse().ok()).unwrap_or(d)
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let p = 4;
    let n = env_usize("TRAJ_N", 16); // square mesh n×n (TRAJ_N to override)
    let (nx, ny) = (n, n);
    let mesh = Mesh2d::rectangular(p, nx, ny, [0.0, 1.0], [0.0, 1.0]);
    let nn = mesh.refq.n_nodes();
    let ne = mesh.n_elements();
    let ndof = ne * nn;
    let r = 0.10;
    // Finer mesh ⇒ smaller dt for the explicit convection CFL.
    let (rho_s, eta_b, nu) = (20.0, 1e-3, 0.1);
    let dt = env_f64("TRAJ_DT", 8e-4);
    let (rep_k, rep_range) = (40.0, 0.06);
    let steps = env_usize("TRAJ_STEPS", 400);
    let every = env_usize("TRAJ_EVERY", 4);
    let out = std::env::args().nth(1).unwrap_or_else(|| "/tmp/suspension.h5".to_string());
    let inflow = |_: f64, _: f64, _: f64| 1.0;
    let zero = |_: f64, _: f64, _: f64| 0.0;
    println!("=== traj suspension {nx}×{ny} p={p}, 2 disks r={r} in a stream, {steps} steps, dump every {every} ===");

    let mass = rho_s * std::f64::consts::PI * r * r;
    let inertia = 0.5 * mass * r * r;
    let mut a = FreeBody::new(RigidBody::disk(0.30, 0.5, r), mass, inertia, eta_b);
    let mut b = FreeBody::new(RigidBody::disk(0.55, 0.5, r), mass, inertia, eta_b);
    a.body.u = 2.0;
    b.body.u = 1.0;
    let susp = Suspension::new(vec![a, b], eta_b, rep_k, rep_range).with_walls([0.0, 1.0], [0.0, 1.0]);

    let mut st = State::new(mesh.clone());
    let vid = st.add_field("velocity", 2);
    for u in st.field_mut("velocity").component_mut(0).iter_mut() {
        *u = 1.0; // seed the uniform stream
    }
    let mut sim = Simulation::new(st);
    sim.set_integrator(gale_gpu::GpuDualSplitting::new(vid, dt, nu, 5.0).boundary(inflow, zero));
    let (hook, bodies) = gale_gpu::GpuMultiMovingPenalizationHook::new(vid, susp, &mesh, dt);
    sim.set_stage_hook(hook);

    let mut tw = TrajectoryWriter::create(&out, p, 2)?;
    let topo = tw.write_mesh2d(&mesh)?;
    tw.write_body_radii(&[r, r])?; // disk radii (static) → viewer draws the bodies at each pose

    let dump = |tw: &mut TrajectoryWriter,
                sim: &Simulation,
                step: u64|
     -> Result<(), Box<dyn std::error::Error>> {
        let v = sim.state.field("velocity");
        let (ux, uy) = (v.component(0), v.component(1));
        let mut field = vec![0f32; ndof * 2];
        for g in 0..ndof {
            field[g * 2] = ux[g] as f32;
            field[g * 2 + 1] = uy[g] as f32;
        }
        let poses: Vec<[f64; 3]> =
            bodies.borrow().bodies.iter().map(|b| [b.body.cx, b.body.cy, b.body.phi]).collect();
        tw.write_frame(step as f64 * dt, step, topo, ne, nn, &[("u", field, 2)], Some(&poses))?;
        Ok(())
    };

    dump(&mut tw, &sim, 0)?;
    let nchunks = steps / every;
    for c in 0..nchunks {
        sim.run(every as u64);
        dump(&mut tw, &sim, ((c + 1) * every) as u64)?;
    }
    println!("wrote {} frames → {out}", tw.n_frames());
    Ok(())
}
