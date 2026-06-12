//! Validation harness for **M1: a freely-moving rigid particle** (explicit Newton–Euler
//! two-way coupling) in GPU flow. A heavy disk, initially at rest, is dragged downstream
//! by a uniform inflow; the GPU integrator `gale_gpu::GpuDualSplitting` +
//! `gale_gpu::GpuMovingPenalizationHook` advances both the fluid and the body. Checks:
//! (a) the body actually MOVES (downstream displacement > 0, U > 0), (b) the GPU
//! trajectory + velocity field match the CPU oracle (`DualSplitting` +
//! `MovingPenalizationHook`) bit-for-bit (only the penalize step runs on the GPU; the
//! force/torque recovery + Newton–Euler + mask rebuild are shared host code), and (c) it
//! stays finite/stable.
//!
//! This is the M1 deliverable of docs/research-moving-particle-coupling.md — the
//! smallest-diff extension of the FIXED-body IBM (`flow-ibm-check`) to a moving body.
//!
//! Run: cargo oxide run --bin flow-ibm-move-check

use gale::dg::{FreeBody, Mesh2d};
use gale::sim::{DualSplitting, MovingPenalizationHook, Simulation, State};

fn rel_l2(a: &[f64], b: &[f64]) -> f64 {
    let num: f64 = a.iter().zip(b).map(|(x, y)| (x - y).powi(2)).sum();
    let den: f64 = b.iter().map(|y| y * y).sum::<f64>().max(1e-300);
    (num / den).sqrt()
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let p = 4;
    let mesh = Mesh2d::rectangular(p, 3, 3, [0.0, 1.0], [0.0, 1.0]);
    let (cx0, cy0, r) = (0.4, 0.5, 0.18);
    let (rho, eta_b, dt, nu) = (20.0, 1e-3, 5e-3, 0.1); // heavy disk ⇒ explicit coupling stable
    let nsteps = 12u64;
    let inflow = |_x: f64, _y: f64, _t: f64| 1.0;
    let zero = |_: f64, _: f64, _: f64| 0.0;
    let make_body = || FreeBody::disk(cx0, cy0, r, rho, eta_b);
    println!("=== M1: freely-moving rigid disk in GPU flow (GpuDualSplitting + GpuMovingPenalizationHook) ===\n");

    // Identical initial state builder: uniform inflow u = (1, 0). The single "velocity"
    // field gets the same FieldId in both states.
    let init_state = || {
        let mut st = State::new(mesh.clone());
        let vid = st.add_field("velocity", 2);
        for u in st.field_mut("velocity").component_mut(0).iter_mut() {
            *u = 1.0;
        }
        (st, vid)
    };

    // --- GPU run -----------------------------------------------------------------
    let (gst, gvid) = init_state();
    let (ghook, gbody) = gale_gpu::GpuMovingPenalizationHook::new(gvid, make_body(), &mesh, dt);
    let mut gsim = Simulation::new(gst);
    gsim.set_integrator(gale_gpu::GpuDualSplitting::new(gvid, dt, nu, 5.0).boundary(inflow, zero));
    gsim.set_stage_hook(ghook);
    gsim.run(nsteps);
    let gux = gsim.state.field("velocity").component(0).to_vec();
    let guy = gsim.state.field("velocity").component(1).to_vec();
    let gb = *gbody.borrow();

    // --- CPU reference (identical setup) -----------------------------------------
    let (cst, cvid) = init_state();
    let (chook, cbody) = MovingPenalizationHook::new(cvid, make_body(), &mesh, dt);
    let mut csim = Simulation::new(cst);
    csim.set_integrator(DualSplitting::new(cvid, dt, nu, 5.0).boundary(inflow, zero));
    csim.set_stage_hook(chook);
    csim.run(nsteps);
    let cux = csim.state.field("velocity").component(0).to_vec();
    let cuy = csim.state.field("velocity").component(1).to_vec();
    let cb = *cbody.borrow();

    // GPU vs CPU.
    let rel_field = rel_l2(&gux, &cux).max(rel_l2(&guy, &cuy));
    let dpose = (gb.body.cx - cb.body.cx).abs() + (gb.body.cy - cb.body.cy).abs();
    let dvel = (gb.body.u - cb.body.u).abs() + (gb.body.v - cb.body.v).abs();
    let dx = gb.body.cx - cx0; // downstream displacement

    println!("GPU vs CPU: velocity rel = {rel_field:.3e}  |Δpose| = {dpose:.3e}  |Δvel| = {dvel:.3e}");
    println!(
        "body: x {cx0:.4} → {:.4}  (Δx = {dx:+.3e})   U = {:+.4e}  V = {:+.3e}  ω = {:+.3e}",
        gb.body.cx, gb.body.u, gb.body.v, gb.body.omega
    );
    let moved = dx > 1e-5 && gb.body.u > 0.0;
    let matches = rel_field < 1e-7 && dpose < 1e-9 && dvel < 1e-9;
    let stable = gb.body.cx.is_finite() && gb.body.u.is_finite() && rel_field.is_finite();
    if moved && matches && stable {
        println!("\nPASS: freely-moving disk advects downstream, GPU matches the CPU oracle, stays stable.");
        Ok(())
    } else {
        eprintln!("\nFAIL: moved={moved} matches={matches} stable={stable}");
        std::process::exit(1);
    }
}
