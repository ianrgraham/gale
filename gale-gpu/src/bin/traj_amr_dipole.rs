//! **Advecting vortex-pair** AMR showcase → one HDF5 file with a moving refined region. A
//! counter-rotating, divergence-free Gaussian dipole self-propagates across a 2D box (the jet
//! between the two vortices drives it), and the `AmrUpdater` refines the moving vortex cores while
//! coarsening the wake behind them — the signature use of dynamic AMR (refinement that TRACKS a
//! moving sharp feature). The dump re-emits the topology whenever the mesh changes, so `gale-view
//! --grid` shows the refined band traveling with the dipole.
//!
//! Run:  cargo oxide run --features traj --bin traj-amr-dipole
//! View: gale-view /tmp/dipole.h5 --field u --comp mag --grid --all --out /tmp/dipole
//!
//! Tunables (env): AMR_N base res, AMR_NU viscosity, AMR_DT step, AMR_AMP dipole strength,
//! AMR_S2 core size², AMR_SEP half-separation, AMR_REFINE/AMR_COARSEN thresholds, TRAJ_STEPS/EVERY.

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
    let (nx, ny) = (env_usize("AMR_N", 16), env_usize("AMR_N", 16));
    let xr = [0.0, 1.0];
    let yr = [0.0, 1.0];
    let nu = env_f64("AMR_NU", 2e-3);
    let alpha = 5.0;
    let dt = env_f64("AMR_DT", 1e-3);
    let steps = env_usize("TRAJ_STEPS", 700);
    let every = env_usize("TRAJ_EVERY", 10); // dump + adapt cadence
    let refine_thr = env_f64("AMR_REFINE", 3e-2);
    let coarsen_thr = env_f64("AMR_COARSEN", 3e-3);
    let out = std::env::args().nth(1).unwrap_or_else(|| "/tmp/dipole.h5".to_string());

    // Counter-rotating dipole via a streamfunction ψ = A·(g_top − g_bot) of two Gaussian cores at
    // (cx, 0.5 ± sep); u = ∂ψ/∂y, v = −∂ψ/∂x ⇒ exactly divergence-free (survives the projection).
    // The jet between the cores points +x, so the dipole self-advects rightward across the box.
    let cx = env_f64("AMR_CX", 0.25);
    let sep = env_f64("AMR_SEP", 0.09);
    let s2 = env_f64("AMR_S2", 0.004);
    let amp = env_f64("AMR_AMP", 0.08);
    let (yt, yb) = (0.5 + sep, 0.5 - sep);
    let gt = move |x: f64, y: f64| (-((x - cx).powi(2) + (y - yt).powi(2)) / s2).exp();
    let gb = move |x: f64, y: f64| (-((x - cx).powi(2) + (y - yb).powi(2)) / s2).exp();
    // ∂g/∂y = g·(−2(y−yc)/s2), ∂g/∂x = g·(−2(x−cx)/s2).
    let vu = move |x: f64, y: f64| {
        amp * (gt(x, y) * (-2.0 * (y - yt) / s2) - gb(x, y) * (-2.0 * (y - yb) / s2))
    };
    let vv = move |x: f64, y: f64| {
        -amp * (gt(x, y) * (-2.0 * (x - cx) / s2) - gb(x, y) * (-2.0 * (x - cx) / s2))
    };
    let zero = |_: f64, _: f64, _: f64| 0.0;
    println!("=== traj AMR vortex-pair {nx}×{ny} p={p}, {steps} steps, adapt+dump every {every} (ν={nu}, amp={amp}) ===");

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
    let mut topo = tw.write_mesh2d(&sim.state.mesh)?;
    let mut last_ne = sim.state.mesh.n_elements();

    let pack = |sim: &Simulation| -> (Vec<f32>, usize, f64) {
        let v = sim.state.field("velocity");
        let (ux, uy) = (v.component(0), v.component(1));
        let ne = sim.state.mesh.n_elements();
        let mut f = vec![0f32; ne * nn * 2];
        let mut umax = 0.0f64;
        for g in 0..ne * nn {
            f[g * 2] = ux[g] as f32;
            f[g * 2 + 1] = uy[g] as f32;
            umax = umax.max((ux[g] * ux[g] + uy[g] * uy[g]).sqrt());
        }
        (f, ne, umax)
    };

    let (f0, ne0, umax0) = pack(&sim);
    tw.write_frame(0.0, 0, topo, ne0, nn, &[("u", f0, 2)], None)?;
    println!("  frame 0: {ne0} elems, |u|max={umax0:.3}");

    let nchunks = steps / every;
    for c in 0..nchunks {
        sim.run(every as u64);
        let ne = sim.state.mesh.n_elements();
        if ne != last_ne {
            topo = tw.write_mesh2d(&sim.state.mesh)?;
            last_ne = ne;
        }
        let (f, ne, umax) = pack(&sim);
        let step = ((c + 1) * every) as u64;
        tw.write_frame(step as f64 * dt, step, topo, ne, nn, &[("u", f, 2)], None)?;
        if c % 5 == 0 {
            println!("  frame {}: {ne} elems, |u|max={umax:.3}", c + 1);
        }
    }
    println!("wrote {} frames → {out}", tw.n_frames());
    Ok(())
}
