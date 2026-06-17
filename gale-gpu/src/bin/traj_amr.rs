//! Trajectory dump of a **dynamically adaptive** 2D flow → one HDF5 file with a CHANGING mesh. A
//! localized divergence-free vortex on a Cartesian base grid is advanced by the GPU dual-splitting
//! integrator while an `AmrUpdater` refines the under-resolved vortex region and coarsens it back as
//! the vortex diffuses. Whenever the mesh changes (an AMR remesh) a NEW topology is written and the
//! subsequent frames reference it — so the trajectory carries a per-frame mesh, and `gale-view`
//! (which reads each frame's topology) renders the refining/coarsening grid.
//!
//! Run:  cargo oxide run --features traj --bin traj-amr
//! View: gale-view /tmp/amr.h5 --field u --comp mag --all --out /tmp/amr   (montage of frames)

use gale::dg::Mesh2d;
use gale::sim::{AmrUpdater, Periodic, Simulation, State};
use gale_traj::TrajectoryWriter;

fn env_usize(k: &str, d: usize) -> usize {
    std::env::var(k).ok().and_then(|v| v.parse().ok()).unwrap_or(d)
}
fn env_f64(k: &str, d: f64) -> f64 {
    std::env::var(k).ok().and_then(|v| v.parse().ok()).unwrap_or(d)
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let p = 3;
    let (nx, ny) = (env_usize("AMR_N", 8), env_usize("AMR_N", 8));
    let xr = [0.0, 1.0];
    let yr = [0.0, 1.0];
    let nu = env_f64("AMR_NU", 0.04);
    let alpha = 5.0;
    let dt = env_f64("AMR_DT", 5e-4);
    let steps = env_usize("TRAJ_STEPS", 160);
    let every = env_usize("TRAJ_EVERY", 8); // dump + adapt cadence
    let refine_thr = env_f64("AMR_REFINE", 3e-2);
    let coarsen_thr = env_f64("AMR_COARSEN", 3e-3);
    let out = std::env::args().nth(1).unwrap_or_else(|| "/tmp/amr.h5".to_string());

    // Divergence-free vortex (survives the projection): wide enough to be resolved (stable) yet
    // sharp enough that its region exceeds the refine threshold on the coarse base grid.
    let (cx, cy, s2) = (0.4, 0.55, env_f64("AMR_S2", 0.015));
    let g = move |x: f64, y: f64| (-((x - cx).powi(2) + (y - cy).powi(2)) / s2).exp();
    let vu = move |x: f64, y: f64| -(y - cy) * g(x, y);
    let vv = move |x: f64, y: f64| (x - cx) * g(x, y);
    let zero = |_: f64, _: f64, _: f64| 0.0;
    println!("=== traj AMR flow {nx}×{ny} p={p}, {steps} steps, adapt+dump every {every} (refine>{refine_thr}, coarsen<{coarsen_thr}) ===");

    let mut st = State::new(Mesh2d::rectangular(p, nx, ny, xr, yr));
    let vid = st.add_field_from(
        "velocity",
        &[Box::new(vu) as Box<dyn Fn(f64, f64) -> f64>, Box::new(vv)],
    );
    let mut sim = Simulation::new(st);
    sim.set_integrator(gale_gpu::GpuDualSplitting::new(vid, dt, nu, alpha).boundary(zero, zero));
    sim.add_updater(
        AmrUpdater::new(p, nx, ny, xr, yr, "velocity", refine_thr).with_coarsening(coarsen_thr),
        Periodic::new(every as u64),
    );

    let nn = sim.state.mesh.refq.n_nodes();
    let mut tw = TrajectoryWriter::create(&out, p, 2)?;
    // `topology_for` re-emits topology on any GEOMETRY change (not just element-count change), so a
    // constant-ndof AMR remesh doesn't pair the new field with a stale mesh (which mis-draws cells).
    let mut topo = tw.topology_for(&sim.state.mesh)?;

    // Pack the velocity field of the CURRENT mesh as interleaved [ne, nn, 2] f32.
    let pack = |sim: &Simulation| -> (Vec<f32>, usize) {
        let v = sim.state.field("velocity");
        let (ux, uy) = (v.component(0), v.component(1));
        let ne = sim.state.mesh.n_elements();
        let mut f = vec![0f32; ne * nn * 2];
        for g in 0..ne * nn {
            f[g * 2] = ux[g] as f32;
            f[g * 2 + 1] = uy[g] as f32;
        }
        (f, ne)
    };

    let (f0, ne0) = pack(&sim);
    tw.write_frame(0.0, 0, topo, ne0, nn, &[("u", f0, 2)], None)?;

    let nchunks = steps / every;
    for c in 0..nchunks {
        sim.run(every as u64);
        topo = tw.topology_for(&sim.state.mesh)?;
        let (f, ne) = pack(&sim);
        tw.write_frame(((c + 1) * every) as f64 * dt, ((c + 1) * every) as u64, topo, ne, nn, &[("u", f, 2)], None)?;
        if c % 4 == 0 || ne != 64 {
            println!("  chunk {c}: {ne} elems");
        }
    }
    println!("wrote {} frames → {out} (final mesh {} elems)", tw.n_frames(), sim.state.mesh.n_elements());
    Ok(())
}
