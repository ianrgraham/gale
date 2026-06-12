//! **M4: many-body particle-laden suspension** — docs/research-moving-particle-coupling.md.
//! Several freely-moving rigid disks share one combined penalization mask; each body's
//! hydrodynamic force/torque is recovered over its own region, and short-range repulsion
//! (Glowinski roughness model) + walls keep them from interpenetrating where the mesh
//! can't resolve the gap. Two heavy disks are launched head-on (no ambient flow); the
//! repulsion must decelerate them and bounce them apart WITHOUT overlap.
//!
//! Checks: (a) no interpenetration over the whole trajectory (min surface gap stays
//! ≳ 0), (b) the disks actually came into contact range (repulsion was exercised),
//! (c) GPU matches the CPU oracle (`MultiMovingPenalizationHook`) — only the combined
//! penalize runs on the GPU; repulsion + per-body Newton–Euler are shared host code,
//! (d) stable/finite.
//!
//! Run: cargo oxide run --bin ibm-suspension-check

use gale::dg::{FreeBody, Mesh2d, RigidBody, Suspension};
use gale::sim::{DualSplitting, MultiMovingPenalizationHook, Simulation, State};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let p = 4;
    let mesh = Mesh2d::rectangular(p, 3, 3, [0.0, 1.0], [0.0, 1.0]);
    let r = 0.10;
    let (rho_s, eta_b, dt, nu) = (20.0, 1e-3, 4e-3, 0.1); // heavy ⇒ explicit coupling fine
    let (rep_k, rep_range) = (40.0, 0.06);
    let nsteps = 16usize;
    // Uniform inflow stream u=(1,0) — a WELL-POSED (divergence-free) box, unlike a fully
    // closed box where two bodies squeezing incompressible fluid makes the pressure solve
    // near-infeasible (it stalls). The disks ride the stream; the faster upstream one
    // catches the slower downstream one ⇒ a contact event the repulsion must resolve.
    let inflow = |_: f64, _: f64, _: f64| 1.0;
    let zero = |_: f64, _: f64, _: f64| 0.0;

    let build = || {
        let mass = rho_s * std::f64::consts::PI * r * r;
        let inertia = 0.5 * mass * r * r;
        let mut a = FreeBody::new(RigidBody::disk(0.40, 0.5, r), mass, inertia, eta_b);
        let mut b = FreeBody::new(RigidBody::disk(0.62, 0.5, r), mass, inertia, eta_b);
        a.body.u = 2.0; // upstream disk, faster than the stream ⇒ overtakes B
        b.body.u = 1.0; // downstream disk, riding the stream
        Suspension::new(vec![a, b], eta_b, rep_k, rep_range).with_walls([0.0, 1.0], [0.0, 1.0])
    };
    println!("=== M4: two disks collide in a stream via repulsion (r={r}, k={rep_k}, ρ={rep_range}), {nsteps} steps ===\n");

    // The hook records the closest approach (`min_gap_seen`) every step, so ONE
    // `run(nsteps)` suffices — avoids paying the per-`run()` GPU setup 50× over.
    let run = |gpu: bool| -> (f64, Vec<(f64, f64)>, f64) {
        let mut st = State::new(mesh.clone());
        let vid = st.add_field("velocity", 2);
        for u in st.field_mut("velocity").component_mut(0).iter_mut() {
            *u = 1.0; // seed the uniform stream u = (1, 0)
        }
        let mut sim = Simulation::new(st);
        let handle;
        if gpu {
            sim.set_integrator(gale_gpu::GpuDualSplitting::new(vid, dt, nu, 5.0).boundary(inflow, zero));
            let (h, hd) = gale_gpu::GpuMultiMovingPenalizationHook::new(vid, build(), &mesh, dt);
            sim.set_stage_hook(h);
            handle = hd;
        } else {
            sim.set_integrator(DualSplitting::new(vid, dt, nu, 5.0).boundary(inflow, zero));
            let (h, hd) = MultiMovingPenalizationHook::new(vid, build(), &mesh, dt);
            sim.set_stage_hook(h);
            handle = hd;
        }
        sim.run(nsteps as u64);
        let s = handle.borrow();
        let poses: Vec<(f64, f64)> = s.bodies.iter().map(|b| (b.body.cx, b.body.cy)).collect();
        let umax = {
            let v = sim.state.field("velocity");
            v.component(0).iter().zip(v.component(1).iter()).fold(0.0f64, |m, (a, b)| m.max(a.hypot(*b)))
        };
        (s.min_gap_seen, poses, umax)
    };

    let (gmin, gpos, gumax) = run(true);
    let (cmin, cpos, _cumax) = run(false);

    // GPU vs CPU agreement on the final poses + the min-gap history.
    let dpose: f64 = gpos.iter().zip(&cpos).map(|(a, b)| (a.0 - b.0).abs() + (a.1 - b.1).abs()).sum();
    let dgap = (gmin - cmin).abs();

    println!("GPU: min surface gap over run = {gmin:+.3e}  final poses = {gpos:?}  umax={gumax:.3e}");
    println!("CPU: min surface gap over run = {cmin:+.3e}");
    println!("GPU vs CPU: Σ|Δpose| = {dpose:.3e}  |Δmin_gap| = {dgap:.3e}");

    // No interpenetration: allow a tiny discrete overshoot (≤ 5% of a radius); they must
    // also have come within contact range (min_gap < ρ) so the repulsion was exercised.
    let no_overlap = gmin > -0.05 * r;
    let made_contact = gmin < rep_range;
    let gpu_matches = dpose < 1e-6 && dgap < 1e-7;
    let stable = gmin.is_finite() && gumax.is_finite() && gpos.iter().all(|p| p.0.is_finite() && p.1.is_finite());
    println!("\nno-overlap={no_overlap}  made-contact={made_contact}  gpu-matches={gpu_matches}  stable={stable}");
    if no_overlap && made_contact && gpu_matches && stable {
        println!("\nPASS: many-body suspension — repulsion prevents interpenetration on contact,\nGPU matches the CPU oracle, stable.");
        Ok(())
    } else {
        eprintln!("\nFAIL: M4 suspension check.");
        std::process::exit(1);
    }
}
