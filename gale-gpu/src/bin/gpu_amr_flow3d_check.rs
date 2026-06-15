//! End-to-end **GPU 3D adaptive flow** check: the GPU 3D dual-splitting integrator
//! (`GpuDualSplitting3d`) running on a 2:1 non-conforming (octree AMR) hex mesh, with the elliptic
//! solves routed through the non-conforming path (`poisson3d_nc`).
//!
//! Checks:
//!   (A) GPU free-stream preservation through a coarsening remesh: a uniform `(1,0,0)` flow on a
//!       pre-refined hex mesh stays `(1,0,0)` while `AmrUpdater3d` coarsens it mid-run (the NC
//!       dual-splitting step and the restrict-remap both preserve a constant), and the mesh shrinks.
//!   (B) GPU vs CPU on a STATIC non-conforming mesh (one NS step): the GPU NC flow matches the CPU
//!       `DualSplitting3d` (which already handles NC via Poisson3d::apply) to solver tolerance.
//!
//! Run: cargo oxide run --bin gpu-amr-flow3d-check

use gale::dg::Mesh3d;
use gale::sim::{AmrUpdater3d, DualSplitting3d, OnStep, Simulation, State};

fn maxd(a: &[f64], b: &[f64]) -> f64 {
    a.iter().zip(b).map(|(x, y)| (x - y).abs()).fold(0.0, f64::max)
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let (p, nx, ny, nz) = (2usize, 2usize, 2usize, 2usize);
    let (xr, yr, zr) = ([0.0, 1.0], [0.0, 1.0], [0.0, 1.0]);
    let (nu, alpha, dt) = (0.05, 5.0, 1e-2);
    let mut ok = true;

    // (A) GPU free stream through a coarsening remesh.
    {
        let pre = vec![(0usize, 0usize, 0usize), (1, 1, 1)];
        let mut st = State::new(Mesh3d::cartesian_refined(p, nx, ny, nz, xr, yr, zr, &pre));
        let vid = st.add_field_from(
            "velocity",
            &[
                Box::new(|_: f64, _: f64, _: f64| 1.0) as Box<dyn Fn(f64, f64, f64) -> f64>,
                Box::new(|_: f64, _: f64, _: f64| 0.0),
                Box::new(|_: f64, _: f64, _: f64| 0.0),
            ],
        );
        let pre_ne = st.mesh.n_elements();
        let mut sim = Simulation::new(st);
        sim.set_integrator(gale_gpu::GpuDualSplitting3d::new(vid, dt, nu, alpha).boundary(
            |_, _, _, _| 1.0,
            |_, _, _, _| 0.0,
            |_, _, _, _| 0.0,
        ));
        sim.add_updater(
            AmrUpdater3d::new(p, nx, ny, nz, xr, yr, zr, "velocity", 1e9)
                .with_coarsening(1e-1)
                .with_initial_refined(pre.clone()),
            OnStep { step: 2 },
        );
        sim.run(4);
        let v = sim.state.field("velocity");
        let eu = v.component(0).iter().map(|&x| (x - 1.0).abs()).fold(0.0, f64::max);
        let ev = v.component(1).iter().map(|&x| x.abs()).fold(0.0, f64::max);
        let ew = v.component(2).iter().map(|&x| x.abs()).fold(0.0, f64::max);
        let ne = sim.state.mesh.n_elements();
        println!("(A) GPU free-stream + coarsen: {pre_ne}→{ne} elems, |u-(1,0,0)|max=({eu:.2e},{ev:.2e},{ew:.2e})");
        ok &= eu < 1e-7 && ev < 1e-7 && ew < 1e-7 && ne < pre_ne;
    }

    // (B) GPU vs CPU on a static non-conforming mesh (one NS step, non-trivial IC).
    {
        let pre = vec![(0usize, 0usize, 0usize)];
        let ic_u = |x: f64, y: f64, z: f64| (std::f64::consts::PI * x).sin() * (y + 0.5) * (z + 0.3);
        let ic_v = |x: f64, _y: f64, z: f64| 0.2 * (x + 1.0) * (std::f64::consts::PI * z).sin();
        let build = || {
            let mut st = State::new(Mesh3d::cartesian_refined(p, nx, ny, nz, xr, yr, zr, &pre));
            let vid = st.add_field_from(
                "velocity",
                &[
                    Box::new(ic_u) as Box<dyn Fn(f64, f64, f64) -> f64>,
                    Box::new(ic_v),
                    Box::new(|_: f64, _: f64, _: f64| 0.0),
                ],
            );
            (st, vid)
        };
        let zero4 = |_: f64, _: f64, _: f64, _: f64| 0.0;

        let (gst, gvid) = build();
        let mut gsim = Simulation::new(gst);
        gsim.set_integrator(gale_gpu::GpuDualSplitting3d::new(gvid, dt, nu, alpha).boundary(zero4, zero4, zero4));
        gsim.run(1);
        let gv = gsim.state.field("velocity");
        let (gu, gw) = (gv.component(0).to_vec(), gv.component(2).to_vec());
        let gvv = gv.component(1).to_vec();

        let (cst, cvid) = build();
        let mut csim = Simulation::new(cst);
        csim.set_integrator(DualSplitting3d::new(cvid, dt, nu, alpha).boundary(zero4, zero4, zero4));
        csim.run(1);
        let cv = csim.state.field("velocity");
        let n = (cv.component(0).iter().map(|v| v * v).sum::<f64>()).sqrt().max(1e-300);
        let e = (maxd(&gu, cv.component(0)).powi(2) + maxd(&gvv, cv.component(1)).powi(2) + maxd(&gw, cv.component(2)).powi(2)).sqrt();
        let rel = e / n;
        println!("(B) GPU vs CPU static NC (1 step): ||u_gpu-u_cpu||/||u||={rel:.3e}");
        ok &= rel < 1e-7;
    }

    if ok {
        println!("\nPASS: GPU 3D adaptive flow (non-conforming) works and matches CPU.");
        Ok(())
    } else {
        eprintln!("\nFAIL: GPU 3D adaptive flow mismatch.");
        std::process::exit(1);
    }
}
