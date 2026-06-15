//! End-to-end **dynamic AMR + GPU flow** check: a 2D incompressible NS run that refines the mesh
//! mid-simulation (indicator-driven `AmrUpdater`) while the flow is advanced by the GPU integrator
//! (`GpuDualSplitting`), validated against the identical CPU-adaptive run (`DualSplitting`).
//!
//! The refinement fires at step 4 (`OnStep`) — AFTER the integrator has run on the base mesh and
//! built its persistent handles — so it exercises the genuine mid-run path: the mesh changes size,
//! the fields are remapped, and the GPU integrator's handles REBUILD on the new dof count (the
//! conforming `GpuPoisson`/MG slots clear and the non-conforming `GpuPoissonNc` slot builds).
//!
//! Modes (env):
//!   default      — dynamic AMR (refine once at step 4), CPU vs GPU.
//!   STATIC_REFINE=1 — start from a fixed partial-refined (2:1 non-conforming) mesh, NO updater;
//!                     isolates the static-NC flow path on a NON-trivial (blob) flow, CPU vs GPU.
//!   AMR_THRESH=<f> — override the refinement threshold (default 1e-1).
//!
//! Run: cargo oxide run --bin gpu-amr-flow-check

use gale::dg::Mesh2d;
use gale::sim::{AmrUpdater, OnStep, Simulation, State};

fn l2(a: &[f64]) -> f64 {
    a.iter().map(|x| x * x).sum::<f64>().sqrt()
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let p = 3;
    let (nx, ny) = (6, 6);
    let xr = [0.0, 1.0];
    let yr = [0.0, 1.0];
    let nu = 0.02;
    let alpha = 5.0;
    let dt = 2e-3;
    let nsteps = 12u64;
    let adapt_every = 4u64;
    // A threshold giving a STABLE partial refinement (genuine 2:1 non-conforming, not all-refined):
    // the under-resolved vortex refines its region; refine-only + a slowly-diffusing vortex ⇒ the
    // set is unambiguous, so CPU and GPU make identical refinement decisions and must then agree to
    // solver tolerance. (A too-low threshold makes the indicator on the evolving solution
    // threshold-sensitive, so tiny CPU/GPU differences flip a borderline cell ⇒ divergent meshes —
    // a test artifact, not a bug.)
    let threshold = std::env::var("AMR_THRESH").ok().and_then(|v| v.parse().ok()).unwrap_or(1e-1);
    let static_refine = std::env::var("STATIC_REFINE").is_ok();
    // A fixed partial refinement set (genuine 2:1 non-conforming) for the static-isolation mode.
    let static_set: Vec<(usize, usize)> = vec![(2, 2), (3, 2), (2, 3)];

    // A localized, DIVERGENCE-FREE vortex (ψ = exp(−r²/σ²), u = ∂ψ/∂y, v = −∂ψ/∂x): unlike a raw
    // blob it survives the dual-splitting projection, so it stays sharp for several steps and still
    // triggers refinement when the AmrUpdater fires AFTER the integrator has run (a true mid-run
    // remesh + handle rebuild, not just a refine-before-the-first-step).
    let (cxb, cyb, s2) = (0.35, 0.55, 0.008);
    let g = move |x: f64, y: f64| (-((x - cxb).powi(2) + (y - cyb).powi(2)) / s2).exp();
    let vu = move |x: f64, y: f64| -(y - cyb) * g(x, y);
    let vv = move |x: f64, y: f64| (x - cxb) * g(x, y);
    let zero3 = |_: f64, _: f64, _: f64| 0.0;

    let build = || {
        let mesh = if static_refine {
            Mesh2d::cartesian_refined(p, nx, ny, xr, yr, &static_set)
        } else {
            Mesh2d::rectangular(p, nx, ny, xr, yr)
        };
        let mut st = State::new(mesh);
        let vid = st.add_field_from(
            "velocity",
            &[Box::new(vu) as Box<dyn Fn(f64, f64) -> f64>, Box::new(vv)],
        );
        (st, vid)
    };

    let mode = if static_refine { "STATIC non-conforming (no AMR)" } else { "DYNAMIC AMR" };
    println!("=== GPU flow vs CPU — {mode}, base {nx}×{ny} p={p}, {nsteps} steps ===");

    // ---- CPU reference ----
    let (cst, cvid) = build();
    let base_ne = cst.mesh.n_elements();
    let mut csim = Simulation::new(cst);
    csim.set_integrator(gale::sim::DualSplitting::new(cvid, dt, nu, alpha).boundary(zero3, zero3));
    if !static_refine {
        csim.add_updater(AmrUpdater::new(p, nx, ny, xr, yr, "velocity", threshold), OnStep { step: adapt_every });
    }
    csim.run(nsteps);
    let cne = csim.state.mesh.n_elements();
    let cu = csim.state.field("velocity").component(0).to_vec();
    let cv = csim.state.field("velocity").component(1).to_vec();

    // ---- GPU run ----
    let (gst, gvid) = build();
    let mut gsim = Simulation::new(gst);
    gsim.set_integrator(gale_gpu::GpuDualSplitting::new(gvid, dt, nu, alpha).boundary(zero3, zero3));
    if !static_refine {
        gsim.add_updater(AmrUpdater::new(p, nx, ny, xr, yr, "velocity", threshold), OnStep { step: adapt_every });
    }
    gsim.run(nsteps);
    let gne = gsim.state.mesh.n_elements();
    let gu = gsim.state.field("velocity").component(0).to_vec();
    let gv = gsim.state.field("velocity").component(1).to_vec();

    // ---- compare ----
    println!("elements: base {base_ne} → CPU {cne}, GPU {gne}");
    let mesh_match = cne == gne;
    let rel = if mesh_match && cu.len() == gu.len() {
        let du: Vec<f64> = cu.iter().zip(&gu).map(|(a, b)| a - b).collect();
        let dv: Vec<f64> = cv.iter().zip(&gv).map(|(a, b)| a - b).collect();
        let n = (l2(&cu).powi(2) + l2(&cv).powi(2)).sqrt().max(1e-300);
        (l2(&du).powi(2) + l2(&dv).powi(2)).sqrt() / n
    } else {
        f64::INFINITY
    };
    let refined_or_static = static_refine || gne > base_ne;
    println!("mesh match: {mesh_match}   refined/NC: {refined_or_static}   ||u_gpu - u_cpu|| / ||u_cpu|| = {rel:.3e}");

    let ok = refined_or_static && mesh_match && rel < 1e-6;
    if ok {
        println!("\nPASS.");
        Ok(())
    } else {
        eprintln!("\nFAIL (refined/NC={refined_or_static}, mesh_match={mesh_match}, rel={rel:.3e}).");
        std::process::exit(1);
    }
}
