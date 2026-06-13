//! **Quantitative validation: Wan–Turek single settling disk (2D sedimentation).**
//! A moving-particle external-data benchmark (companion to the fixed-cylinder
//! cylinder-drag-check): a heavy disk released from rest sediments under gravity down a
//! narrow channel, reaching a terminal velocity; we compare the terminal particle
//! Reynolds number to the published reference. This exercises the M1/M3 freely-moving
//! coupling (strong, since ρ_s/ρ_f=1.25 sits near the added-mass threshold) end-to-end.
//!
//! Reference (Wan & Turek, fictitious-boundary DNS): terminal particle Reynolds number
//! Re_T ≈ 17.15–17.45. Setup (CGS): channel width W=2, disk diameter d=0.25 (r=0.125),
//! ρ_s=1.25, ρ_f=1.0, μ=0.1 (ν=0.1), g=981. Re = ρ_f·U·d/μ = 2.5·U.
//!
//! Reduced-gravity formulation: the fluid carries no body force; the particle feels the
//! buoyancy-corrected weight F = (ρ_s−ρ_f)·πr²·g downward. At terminal velocity the
//! penalization (hydrodynamic) drag balances it. As with the fixed cylinder, diffuse
//! penalization under-resolves the drag on coarse meshes ⇒ the disk settles too fast ⇒
//! Re_T over-predicts, converging down toward the reference as `SETTLE_NY` increases.
//!
//! Run: cargo oxide run --bin settling-disk-check   (SETTLE_NY=120 for a finer point)

use gale::dg::{FreeBody, Mesh2d};
use gale::sim::{Simulation, State};
use std::f64::consts::PI;

fn env_usize(k: &str, d: usize) -> usize {
    std::env::var(k).ok().and_then(|v| v.parse().ok()).unwrap_or(d)
}
fn env_f64(k: &str, d: f64) -> f64 {
    std::env::var(k).ok().and_then(|v| v.parse().ok()).unwrap_or(d)
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let p = 4;
    let (w, height, r) = (2.0, 6.0, 0.125); // channel width, height, disk radius (CGS)
    let (rho_s, rho_f, nu, g) = (1.25, 1.0, 0.1, 981.0);
    let d = 2.0 * r;
    let re_ref = 17.15; // Wan-Turek terminal particle Reynolds number (≈17.15–17.45)

    let ny = env_usize("SETTLE_NY", 72);
    let nx = ((w / height) * ny as f64).round().max(1.0) as usize; // ~square elements
    let eta_b = env_f64("SETTLE_ETAB", 1e-3);
    let dt = env_f64("SETTLE_DT", 8e-4);
    let nsteps = env_usize("SETTLE_STEPS", 700);
    let alpha = 5.0;

    let mesh = Mesh2d::rectangular(p, nx, ny, [0.0, w], [0.0, height]);
    let hx = w / nx as f64;
    println!("=== Wan–Turek settling disk (2D sedimentation) ===");
    println!("mesh {nx}×{ny} (p={p}, h≈{hx:.4}, d/h≈{:.1}), η_b={eta_b:.0e}, dt={dt}, steps={nsteps}", d / hx);

    // Heavy disk released from rest, centred near the top; reduced-gravity buoyant weight.
    let mass = rho_s * PI * r * r;
    let inertia = 0.5 * mass * r * r;
    let f_net = (rho_s - rho_f) * PI * r * r * g; // buoyancy-corrected weight (downward)
    let body = FreeBody::new(gale::dg::RigidBody::disk(w / 2.0, height - 1.0, r), mass, inertia, eta_b)
        .with_external_force(0.0, -f_net);

    // Closed quiescent channel (no-slip walls), strong coupling (density ratio 1.25).
    let zero = |_: f64, _: f64, _: f64| 0.0;
    let mut st = State::new(mesh.clone());
    let vid = st.add_field("velocity", 2);
    let mut sim = Simulation::new(st);
    sim.set_integrator(gale_gpu::GpuDualSplitting::new(vid, dt, nu, alpha).boundary(zero, zero));
    let (hook, handle) = gale_gpu::GpuMovingPenalizationHook::new(vid, body, &mesh, dt);
    sim.set_stage_hook(hook.strong(true));

    // March; print the settling speed periodically so the terminal plateau is visible.
    // The disk accelerates to terminal then holds it mid-channel — stop before it nears the
    // bottom wall, so the final speed IS the terminal speed.
    let mut re_max = 0.0f64;
    let mut re_prev = 0.0f64;
    for blk in 0..(nsteps / 50) {
        sim.run(50);
        let b = handle.borrow();
        let speed = b.body.v.abs();
        let re = rho_f * speed * d / nu;
        re_max = re_max.max(re);
        println!("  step {:5}  y={:.3}  |v|={speed:.4}  Re={re:.3}  (ref Re_T≈{re_ref})", (blk + 1) * 50, b.body.cy);
        // Stop once the speed plateaus (terminal reached) — no need to fall to the bottom.
        if blk > 2 && ((re - re_prev) / re.max(1e-30)).abs() < 2e-3 {
            println!("  (terminal velocity plateau reached)");
            break;
        }
        re_prev = re;
        if b.body.cy < 1.0 {
            break; // approaching the bottom wall — stop
        }
    }
    let b = handle.borrow();
    let u_t = b.body.v.abs();
    let re_t = rho_f * u_t * d / nu;
    let err = (re_t - re_ref).abs() / re_ref * 100.0;
    println!("\nRESULT  terminal |v| = {u_t:.4}  Re_T = {re_t:.3} (ref ≈{re_ref}, err {err:.1}%)   [Re_max over fall = {re_max:.3}]");
    let fell = (height - 1.0) - b.body.cy; // distance descended
    println!("disk fell {fell:.3} (from y={:.2} to y={:.2}); v_x={:+.2e} (≈0 by symmetry)", height - 1.0, b.body.cy, b.body.u);

    // Sanity: the disk sediments (moved down), reached a terminal Reynolds number in the
    // ballpark of the reference (penalization over-predicts on coarse meshes ⇒ converges
    // down to ≈17 as SETTLE_NY increases), and stays stable.
    let settled = fell > 0.3 && b.body.v < 0.0;
    let ballpark = re_t > 0.6 * re_ref && re_t < 1.8 * re_ref;
    let stable = re_t.is_finite() && b.body.u.abs() < 0.5 * u_t.max(1e-9) + 1.0;
    if settled && ballpark && stable {
        println!("\nOK: disk sediments to a terminal velocity; Re_T in the ballpark of the\nWan–Turek reference (converges toward it as SETTLE_NY increases).");
        Ok(())
    } else {
        eprintln!("\nFAIL: settled={settled} ballpark={ballpark} stable={stable} (Re_T={re_t})");
        std::process::exit(1);
    }
}
