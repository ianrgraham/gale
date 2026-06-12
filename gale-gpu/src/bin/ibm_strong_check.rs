//! **M3: strong (implicit) fluid–particle coupling** for light / neutrally-buoyant
//! particles — docs/research-moving-particle-coupling.md. Explicit (weak) coupling
//! (M1/M2) is unstable below a critical solid/fluid density ratio (added-mass effect);
//! strong coupling solves the new rigid velocity simultaneously with the penalization
//! constraint (`FreeBody::strong_solve`, a local 3×3 implicit body solve) and is stable
//! down to zero mass.
//!
//! This bin demonstrates the contrast on gale's DG-SIPG GPU stack (the research's
//! stability results are all Newtonian/non-DG, so this also tests transferability):
//! a LIGHT disk released in a uniform stream is run with (1) explicit and (2) strong
//! coupling. Checks: strong stays bounded/finite, matches the CPU oracle bit-for-bit,
//! and moves; and reports the explicit-vs-strong stability contrast.
//!
//! Run: cargo oxide run --bin ibm-strong-check

use gale::dg::{FreeBody, Mesh2d};
use gale::sim::{DualSplitting, MovingPenalizationHook, Simulation, State};

fn rel_l2(a: &[f64], b: &[f64]) -> f64 {
    let num: f64 = a.iter().zip(b).map(|(x, y)| (x - y).powi(2)).sum();
    let den: f64 = b.iter().map(|y| y * y).sum::<f64>().max(1e-300);
    (num / den).sqrt()
}

/// Peak speed + body velocity magnitude after a run — the stability fingerprint.
struct Fp {
    umax: f64,
    bspeed: f64,
    cx: f64,
    cy: f64,
    ux: Vec<f64>,
    uy: Vec<f64>,
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let p = 4;
    let mesh = Mesh2d::rectangular(p, 3, 3, [0.0, 1.0], [0.0, 1.0]);
    let (cx0, cy0, r) = (0.4, 0.5, 0.18);
    // LIGHT disk: ρ_s = 0.25 < ρ_f (=1) ⇒ explicit coupling is in the added-mass-unstable
    // regime. Stiff penalization + many steps amplify the instability.
    let (rho, eta_b, dt, nu) = (0.25, 5e-4, 4e-3, 0.05);
    let nsteps = 60u64;
    let inflow = |_x: f64, _y: f64, _t: f64| 1.0;
    let zero = |_: f64, _: f64, _: f64| 0.0;
    let make_body = || FreeBody::disk(cx0, cy0, r, rho, eta_b);

    // Run the light disk with a given coupling on a given backend.
    let run = |strong: bool, gpu: bool| -> Fp {
        let mut st = State::new(mesh.clone());
        let vid = st.add_field("velocity", 2);
        for u in st.field_mut("velocity").component_mut(0).iter_mut() {
            *u = 1.0;
        }
        let mut sim = Simulation::new(st);
        let handle;
        if gpu {
            sim.set_integrator(gale_gpu::GpuDualSplitting::new(vid, dt, nu, 5.0).boundary(inflow, zero));
            let (h, b) = gale_gpu::GpuMovingPenalizationHook::new(vid, make_body(), &mesh, dt);
            sim.set_stage_hook(h.strong(strong));
            handle = b;
        } else {
            sim.set_integrator(DualSplitting::new(vid, dt, nu, 5.0).boundary(inflow, zero));
            let (h, b) = MovingPenalizationHook::new(vid, make_body(), &mesh, dt);
            sim.set_stage_hook(h.strong(strong));
            handle = b;
        }
        sim.run(nsteps);
        let ux = sim.state.field("velocity").component(0).to_vec();
        let uy = sim.state.field("velocity").component(1).to_vec();
        let umax = ux.iter().zip(&uy).fold(0.0f64, |m, (a, b)| m.max(a.hypot(*b)));
        let bd = *handle.borrow();
        Fp { umax, bspeed: bd.body.u.hypot(bd.body.v), cx: bd.body.cx, cy: bd.body.cy, ux, uy }
    };

    println!("=== M3: strong vs explicit coupling, LIGHT disk (ρ_s={rho} < ρ_f=1), {nsteps} steps ===\n");

    let expl = run(false, true); // explicit, GPU
    let strong_g = run(true, true); // strong, GPU
    let strong_c = run(true, false); // strong, CPU (oracle)

    let fin = |x: f64| x.is_finite();
    println!("explicit (GPU): umax={:.3e}  body|U|={:.3e}  pos=({:.3},{:.3})  finite={}",
             expl.umax, expl.bspeed, expl.cx, expl.cy, fin(expl.umax) && fin(expl.bspeed));
    println!("strong   (GPU): umax={:.3e}  body|U|={:.3e}  pos=({:.3},{:.3})  finite={}",
             strong_g.umax, strong_g.bspeed, strong_g.cx, strong_g.cy, fin(strong_g.umax) && fin(strong_g.bspeed));

    // Strong vs CPU oracle (Newtonian ⇒ bit-exact: strong_solve is host, penalize bit-exact).
    let rel = rel_l2(&strong_g.ux, &strong_c.ux).max(rel_l2(&strong_g.uy, &strong_c.uy));
    let dpose = (strong_g.cx - strong_c.cx).abs() + (strong_g.cy - strong_c.cy).abs();
    println!("strong GPU vs CPU: velocity rel={rel:.3e}  |Δpose|={dpose:.3e}");

    // Stability contrast: physical scales are O(1) (free stream = 1). The added-mass
    // instability shows in the BODY velocity (the fluid umax is clamped to the body by the
    // penalization, so it can look tame even as the body velocity diverges). Strong must
    // keep BOTH bounded; explicit's body velocity ≫ free stream is the predicted blow-up.
    let bounded = |f: &Fp| fin(f.umax) && fin(f.bspeed) && f.umax < 5.0 && f.bspeed < 5.0;
    let strong_stable = bounded(&strong_g);
    let strong_moved = (strong_g.cx - cx0).hypot(strong_g.cy - cy0) > 1e-5;
    let gpu_matches = rel < 1e-7 && dpose < 1e-9;
    let explicit_unstable = !bounded(&expl);
    println!("\ncontrast: explicit {}  |  strong {} (body |U|: explicit {:.2e} vs strong {:.2e})",
             if explicit_unstable { "UNSTABLE (as predicted)" } else { "stable" },
             if strong_stable { "STABLE" } else { "unstable" },
             expl.bspeed, strong_g.bspeed);

    let pass = strong_stable && strong_moved && gpu_matches;
    if pass {
        let note = if explicit_unstable {
            "strong coupling tames the added-mass instability that breaks explicit"
        } else {
            "strong coupling stable + GPU-exact (explicit happened to survive here too)"
        };
        println!("\nPASS: light-particle strong coupling — stable, GPU matches CPU, moved. ({note})");
        Ok(())
    } else {
        eprintln!("\nFAIL: strong_stable={strong_stable} moved={strong_moved} gpu_matches={gpu_matches}");
        std::process::exit(1);
    }
}
