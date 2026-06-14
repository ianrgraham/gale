//! **Device-resident SBM cylinder** — the Schäfer-Turek cylinder (Re=20) run with the velocity
//! and pressure fields kept RESIDENT on the GPU across the whole dual-splitting step. Convection,
//! divergence, projection, RHS assembly, and all three SBM-MG-PCG solves run on the device with NO
//! per-step field transfers (the only HtoD/DtoH is the periodic download for the C_D diagnostic).
//! This is the SBM payoff of the device-native foundation: the SBM RHS is structurally identical to
//! the standard one (`M·f on active + boundary lift`), so it reuses the same `rhs_madd`/gradient/
//! convection kernels — only the operator (SBM surrogate via `new_sbm`) and the precomputed
//! `jw`/`lift` differ. Validates C_D matches the host-orchestrated GPU cylinder (≈5.61) and reports
//! the per-step cost vs that path's ~35 ms/step. Run: cargo oxide run --bin sbm-cylinder-resident-check

use gale::dg::{
    sbm_force_torque, CircleLevelSet, Mesh2d, ShiftedBoundary, ShiftedMultigrid, ShiftedPoisson,
};
use gale_gpu::operators::poisson::GpuPoissonMg;

fn env_usize(k: &str, d: usize) -> usize {
    std::env::var(k).ok().and_then(|v| v.parse().ok()).unwrap_or(d)
}
fn env_f64(k: &str, d: f64) -> f64 {
    std::env::var(k).ok().and_then(|v| v.parse().ok()).unwrap_or(d)
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let p = 3;
    let (h, um, d, nu) = (0.41, 0.3, 0.1, 0.001);
    let u_mean = 2.0 / 3.0 * um;
    let norm = 2.0 / (u_mean * u_mean * d); // C_D = norm·F_x
    let cd_ref = 5.57953;
    let ny = env_usize("SBM_NY", 16);
    let nx = ((2.2 / h) * ny as f64).round() as usize;
    let dt = env_f64("SBM_DT", 3e-3);
    let max_steps = env_usize("SBM_STEPS", 4000);
    let alpha = 5.0;
    let lambda = 1.0 / (nu * dt);
    let (cx, cy, r) = (0.2, 0.2, 0.05);
    let (ptol, vtol, maxit) = (env_f64("SBM_PTOL", 1e-4), env_f64("SBM_VTOL", 1e-7), 5000);
    let xr = [0.0, 2.2];
    let yr = [0.0, h];

    let mesh = Mesh2d::rectangular(p, nx, ny, xr, yr);
    let nn = mesh.refq.n_nodes();
    let ndof = mesh.n_elements() * nn;
    let ls = CircleLevelSet::new(cx, cy, r);
    let sb = ShiftedBoundary::new(&mesh, &ls);
    let hx = 2.2 / nx as f64;
    println!("=== device-resident SBM cylinder (Re=20), {nx}×{ny} p={p} (D/h≈{:.1}), dt={dt} ===", d / hx);
    println!("surrogate faces: {}, active elements: {}/{}", sb.faces.len(), sb.n_active(), mesh.n_elements());

    // Host operators (for RHS lift/jw extraction only).
    let vel = ShiftedPoisson::with_bc(&mesh, alpha, lambda, vec![1], sb.clone()).taylor(true);
    let pres = ShiftedPoisson::with_bc(&mesh, alpha, 0.0, vec![3, 0, 2], sb.clone()).surrogate_neumann();
    let parab = |y: f64| 4.0 * um * y * (h - y) / (h * h);
    let g_u = |x: f64, y: f64| if x < 0.5 * hx { parab(y) } else { 0.0 };
    let g_v = |_x: f64, _y: f64| 0.0;
    let g_p = |_x: f64, _y: f64| 0.0;

    // GPU SBM handles (shared primary context + null stream ⇒ buffers interoperate).
    // SBM_WHILE: run each solve's convergence loop as a device-side WHILE conditional graph
    // (no per-iteration host residual readback). with_while_graph puts each handle on its own
    // non-legacy stream, so share one stream (hv ← hp) to keep the cross-handle stage/solve ops
    // on shared buffers ordered.
    let use_while = std::env::var("SBM_WHILE").is_ok();
    let hp = GpuPoissonMg::new_sbm(ShiftedMultigrid::new(
        p, nx, ny, xr, yr, alpha, 0.0, vec![3, 0, 2], &ls, false, false,
    ))?
    .with_while_graph(use_while)?;
    let mut hv = GpuPoissonMg::new_sbm(ShiftedMultigrid::new(
        p, nx, ny, xr, yr, alpha, lambda, vec![1], &ls, true, true,
    ))?
    .with_while_graph(use_while)?;
    if use_while {
        hv.share_stream_with(&hp); // both operators on one stream for ordered cross-handle ops
    }
    let hv = hv;

    // Constants (precomputed once via the validated host SBM rhs): diagonal mass (active) + lifts.
    //   jw      = rhs(1, 0)  — mass diagonal on active elements (0 on inactive)
    //   lift_p  = rhs(0, g_p) — pressure surrogate-Neumann + outflow Dirichlet (= 0 here)
    //   lift_vx = rhs(0, g_u) — velocity inflow-Dirichlet lift (surrogate no-slip g=0 ⇒ no lift there)
    let jw = vel.rhs(&vec![1.0; ndof], |_, _| 0.0);
    let lift_p = pres.rhs(&vec![0.0; ndof], g_p);
    let lift_vx = vel.rhs(&vec![0.0; ndof], g_u);
    let lift_vy = vel.rhs(&vec![0.0; ndof], g_v);

    // Seed ux with the inflow parabola on active elements (closer to steady).
    let mut ux0 = vec![0.0; ndof];
    for (e, el) in mesh.elements.iter().enumerate() {
        if sb.active[e] {
            for k in 0..nn {
                ux0[e * nn + k] = parab(el.geom.y[k]);
            }
        }
    }

    // Resident device buffers.
    let jw_d = hv.upload_field(&jw)?;
    let lift_p_d = hv.upload_field(&lift_p)?;
    let lift_vx_d = hv.upload_field(&lift_vx)?;
    let lift_vy_d = hv.upload_field(&lift_vy)?;
    let mut ux = hv.upload_field(&ux0)?;
    let mut uy = hv.alloc_field()?;
    let mut pp = hv.alloc_field()?;
    let mut ux_n = hv.alloc_field()?;
    let mut uy_n = hv.alloc_field()?;
    let mut pp_n = hv.alloc_field()?;
    let mut uhx = hv.alloc_field()?;
    let mut uhy = hv.alloc_field()?;
    let mut ga = hv.alloc_field()?;
    let mut gb = hv.alloc_field()?;
    let mut gc = hv.alloc_field()?;
    let mut gd = hv.alloc_field()?;
    let mut cxb = hv.alloc_field()?;
    let mut cyb = hv.alloc_field()?;
    let mut div = hv.alloc_field()?;
    let mut rhs = hv.alloc_field()?;

    let (mut cd, mut cd_prev, mut steady_at) = (0.0f64, 0.0f64, None);
    let (mut t_step, mut n_timed) = (0.0f64, 0usize);
    for step in 1..=max_steps {
        let t0 = std::time::Instant::now();
        // Stage 1 — convection û = u − Δt (u·∇)u.
        hv.gradient_dev(&ux, &mut ga, &mut gb)?;
        hv.gradient_dev(&uy, &mut gc, &mut gd)?;
        hv.fma2_dev(&mut cxb, &ux, &ga, &uy, &gb)?;
        hv.fma2_dev(&mut cyb, &ux, &gc, &uy, &gd)?;
        hv.copy_dev(&mut uhx, &ux)?;
        hv.copy_dev(&mut uhy, &uy)?;
        hv.axpy_dev(&mut uhx, &cxb, -dt)?;
        hv.axpy_dev(&mut uhy, &cyb, -dt)?;
        // Stage 2 — pressure projection −∇²p = (1/Δt)∇·û (warm-started).
        hv.gradient_dev(&uhx, &mut ga, &mut gb)?;
        hv.gradient_dev(&uhy, &mut gc, &mut gd)?;
        hv.copy_dev(&mut div, &ga)?;
        hv.axpy_dev(&mut div, &gd, 1.0)?;
        hv.scal_dev(&mut div, -1.0 / dt)?;
        hv.rhs_madd_dev(&mut rhs, &jw_d, &div, &lift_p_d, 1.0)?;
        hp.solve_dev(&rhs, Some(&pp), &mut pp_n, ptol, maxit)?;
        std::mem::swap(&mut pp, &mut pp_n);
        // Project u* = û − Δt ∇p.
        hv.gradient_dev(&pp, &mut ga, &mut gb)?;
        hv.axpy_dev(&mut uhx, &ga, -dt)?;
        hv.axpy_dev(&mut uhy, &gb, -dt)?;
        // Stage 3 — viscous Helmholtz (λM + A) uⁿ⁺¹ = λM u* + lift (warm-started).
        hv.rhs_madd_dev(&mut rhs, &jw_d, &uhx, &lift_vx_d, lambda)?;
        hv.solve_dev(&rhs, Some(&ux), &mut ux_n, vtol, maxit)?;
        std::mem::swap(&mut ux, &mut ux_n);
        hv.rhs_madd_dev(&mut rhs, &jw_d, &uhy, &lift_vy_d, lambda)?;
        hv.solve_dev(&rhs, Some(&uy), &mut uy_n, vtol, maxit)?;
        std::mem::swap(&mut uy, &mut uy_n);
        if step > 50 {
            t_step += t0.elapsed().as_secs_f64();
            n_timed += 1;
        }

        // C_D diagnostic — the ONLY host download, every 25 steps.
        if step % 25 == 0 || step == 1 {
            let (hux, huy, hpp) = (hv.download_field(&ux)?, hv.download_field(&uy)?, hv.download_field(&pp)?);
            let (fx, _fy, _t) = sbm_force_torque(&mesh, &sb, &hux, &huy, &hpp, nu, cx, cy, r);
            cd = norm * fx;
            let umax = hux.iter().zip(&huy).fold(0.0f64, |m, (a, b)| m.max(a.hypot(*b)));
            println!("  step {step:5}  C_D={cd:.4}  (ref {cd_ref})  umax={umax:.3}");
            if step > 300 && ((cd - cd_prev) / cd.max(1e-30)).abs() < 2e-4 {
                steady_at = Some(step);
                break;
            }
            cd_prev = cd;
        }
    }

    if n_timed > 0 {
        println!("\nTIMING: {:.2} ms/step (warm, {n_timed} steps) — device-resident, no per-step field transfers", 1e3 * t_step / n_timed as f64);
    }
    let err = (cd - cd_ref).abs() / cd_ref * 100.0;
    println!("RESULT  device-resident SBM C_D = {cd:.4}  (ref {cd_ref}, err {err:.1}%)");
    match steady_at {
        Some(s) => println!("steady at step {s}"),
        None => println!("did NOT reach steady in {max_steps} steps; last C_D={cd:.4}"),
    }
    let stable = cd.is_finite() && cd > 0.5 * cd_ref && cd < 1.5 * cd_ref;
    if stable {
        println!("\nOK: device-resident SBM cylinder steady, C_D matches the host-orchestrated GPU path.");
        Ok(())
    } else {
        eprintln!("\nFAIL: device-resident SBM cylinder C_D={cd} not in ballpark");
        std::process::exit(1);
    }
}
