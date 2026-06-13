//! **Quantitative validation: Schäfer–Turek 2D-1 cylinder benchmark (DFG, Re=20).**
//! The first check of gale against EXTERNAL published data (not GPU-vs-CPU self-
//! consistency): steady laminar flow past a fixed cylinder in a channel, comparing the
//! drag/lift coefficients to the FEATFLOW reference values.
//!
//! Reference (featflow.de, DFG 2D-1): C_D = 5.57953, C_L = 0.010619, Δp = 0.11752.
//! Domain [0,2.2]×[0,0.41]; cylinder D=0.1 at (0.2,0.2); parabolic inflow
//! u(0,y)=4·Um·y·(H−y)/H², Um=0.3, H=0.41 ⇒ U_mean=0.2; ν=0.001, ρ=1 ⇒ Re=20.
//! Coefficients: C_D = 2·F_x/(ρ·U_mean²·D) = 500·F_x, C_L = 500·F_y.
//!
//! The cylinder is imposed by volume penalization (diffuse, ~1st-order at the interface),
//! so C_D approaches the reference from below as the mesh resolves the cylinder + boundary
//! layer — this bin measures HOW CLOSE penalization gets and how it converges with
//! resolution (set `CYL_NY`, default 16; `CYL_ETAB`, default 1e-3; `CYL_STEPS` cap).
//!
//! Run: cargo oxide run --bin cylinder-drag-check   (CYL_NY=24 for a finer sweep point)

use gale::dg::{BoundaryConditions, Disk, FlowBc, Mesh2d, VolumePenalization};

fn env_usize(k: &str, d: usize) -> usize {
    std::env::var(k).ok().and_then(|v| v.parse().ok()).unwrap_or(d)
}
fn env_f64(k: &str, d: f64) -> f64 {
    std::env::var(k).ok().and_then(|v| v.parse().ok()).unwrap_or(d)
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let p = 4;
    let (h, um, d, nu) = (0.41, 0.3, 0.1, 0.001); // channel height, U_max, diameter, viscosity
    let u_mean = 2.0 / 3.0 * um; // = 0.2 ⇒ Re = u_mean·D/ν = 20
    let norm = 2.0 / (u_mean * u_mean * d); // = 500: C = norm·F (DFG coefficient normalization)
    let (cd_ref, cl_ref) = (5.57953, 0.010619);

    let ny = env_usize("CYL_NY", 16);
    let nx = ((2.2 / h) * ny as f64).round() as usize; // ~square elements (2.2:0.41 aspect)
    let eta_b = env_f64("CYL_ETAB", 1e-3);
    let max_steps = env_usize("CYL_STEPS", 6000);
    let dt = env_f64("CYL_DT", 3e-3);
    let alpha = 5.0;

    let mesh = Mesh2d::rectangular(p, nx, ny, [0.0, 2.2], [0.0, h]);
    let nn = mesh.refq.n_nodes();
    let ndof = mesh.n_elements() * nn;
    let hx = 2.2 / nx as f64;
    println!("=== Schäfer–Turek 2D-1 (Re=20) cylinder drag, penalization ===");
    println!("mesh {nx}×{ny} (p={p}, h≈{hx:.4}, D/h≈{:.1}), η_b={eta_b:.0e}, dt={dt}, ndof={ndof}", d / hx);

    // Parabolic inflow on the west wall (tag 3); traction-free outflow on the east (tag 1);
    // no-slip top/bottom (tags 0,2) are the no_slip() default.
    let parab = move |_x: f64, y: f64, _t: f64| (4.0 * um * y * (h - y) / (h * h), 0.0);
    let bcs = || BoundaryConditions::no_slip().set(3, FlowBc::velocity(parab)).set(1, FlowBc::Outflow);

    // Persistent p-MG-PCG handles (mesh-independent pressure solve + the CUDA-graph
    // default) — without them GpuStokes::with_bcs falls back to plain O(1/h) CG, far too
    // slow to march to steady state. Tags match what with_bcs computes internally.
    let lambda = 1.0 / (nu * dt);
    let b0 = bcs();
    let pres_tags = b0.pressure_neumann_tags(&mesh);
    let velx_tags = b0.velocity_neumann_tags(&mesh, 0);
    let vely_tags = b0.velocity_neumann_tags(&mesh, 1);
    let mgp = gale_gpu::GpuPoissonMg::new(
        gale::dg::PMultigrid::from_mesh(&mesh, alpha, 0.0, pres_tags).ok_or("pressure MG build failed")?,
    )?;
    let mgvx = gale_gpu::GpuPoissonMg::new(
        gale::dg::PMultigrid::from_mesh(&mesh, alpha, lambda, velx_tags).ok_or("velx MG build failed")?,
    )?;
    let mgvy = gale_gpu::GpuPoissonMg::new(
        gale::dg::PMultigrid::from_mesh(&mesh, alpha, lambda, vely_tags).ok_or("vely MG build failed")?,
    )?;
    let mut gst = gale_gpu::GpuStokes::with_bcs(&mesh, alpha, nu, dt, &bcs())
        .with_mg_pressure(&mgp)
        .with_mg_velocity(&mgvx, &mgvy);
    gst.convection_scheme = gale::dg::ConvectionScheme::Nodal;
    let pen = VolumePenalization::new(&mesh, &Disk::new(0.2, 0.2, 0.05), eta_b);
    let zeros = vec![0.0; ndof];

    // Seed the interior with the parabolic profile (closer to steady ⇒ fewer steps).
    let mut ux = vec![0.0; ndof];
    let mut uy = vec![0.0; ndof];
    for (e, el) in mesh.elements.iter().enumerate() {
        for k in 0..nn {
            ux[e * nn + k] = 4.0 * um * el.geom.y[k] * (h - el.geom.y[k]) / (h * h);
        }
    }
    pen.apply(&mut ux, &mut uy, dt); // imprint the cylinder on the seed

    // March to steady state: stop when C_D plateaus (rel change < 1e-4 over a window).
    let bc = bcs();
    let mut t = 0.0;
    let mut cd_prev = 0.0;
    let mut cd = 0.0;
    let mut converged_at = None;
    for step in 1..=max_steps {
        t += dt;
        let (nx_, ny_) = gst.step_ns_forced_bc(&ux, &uy, t, &bc, &zeros, &zeros)?;
        ux = nx_;
        uy = ny_;
        pen.apply(&mut ux, &mut uy, dt);
        let (fx, fy) = pen.force(&ux, &uy, &mesh);
        cd = norm * fx;
        let cl = norm * fy;
        if step % 100 == 0 || step == 1 {
            println!("  step {step:5}  t={t:6.3}  C_D={cd:.4}  C_L={cl:+.4e}  (ref C_D={cd_ref})");
        }
        // Steady when C_D barely changes over 50 steps (after a warmup).
        if step > 200 && step % 50 == 0 {
            if ((cd - cd_prev) / cd.max(1e-30)).abs() < 1e-4 {
                converged_at = Some(step);
                break;
            }
            cd_prev = cd;
        }
    }
    let (fx, fy) = pen.force(&ux, &uy, &mesh);
    let (cd_f, cl_f) = (norm * fx, norm * fy);
    let err_cd = (cd_f - cd_ref).abs() / cd_ref * 100.0;
    let umax = ux.iter().zip(&uy).fold(0.0f64, |m, (a, b)| m.max(a.hypot(*b)));

    println!("\nRESULT  C_D = {cd_f:.4} (ref {cd_ref}, err {err_cd:.1}%)   C_L = {cl_f:+.4e} (ref {cl_ref})");
    match converged_at {
        Some(s) => println!("steady at step {s}; umax={umax:.4} (ref U_max={um})"),
        None => println!("did NOT reach steady tolerance in {max_steps} steps; last C_D={cd:.4}; umax={umax:.4}"),
    }
    // Sanity gates (NOT a tight benchmark pass — penalization accuracy is the thing under test):
    // the solve must be stable and C_D in the right ballpark (penalization under-predicts on
    // coarse meshes). Tighten / sweep CYL_NY to chase the reference.
    let stable = cd_f.is_finite() && umax.is_finite() && umax < 5.0 * um;
    let ballpark = cd_f > 0.5 * cd_ref && cd_f < 1.3 * cd_ref;
    if stable && ballpark {
        println!("\nOK: stable steady cylinder flow; C_D within ballpark of the DFG reference\n(penalization converges to it from below as CYL_NY increases).");
        Ok(())
    } else {
        eprintln!("\nFAIL: stable={stable} ballpark={ballpark} (C_D={cd_f})");
        std::process::exit(1);
    }
}
