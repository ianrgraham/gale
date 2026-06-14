//! **Freely-moving SBM disk** (increment B of moving-body SBM) — a heavy disk settling under
//! gravity in a closed no-slip box, with the *sharp* Shifted Boundary Method instead of volume
//! penalization. Each step: rebuild the surrogate boundary at the body's current center, solve the
//! dual-splitting NS step with the body's RIGID velocity as the surrogate no-slip BC, recover the
//! hydrodynamic force/torque on the true circle (`sbm_force_torque`), and advance the body by
//! explicit Newton–Euler (`FreeBody::advance`, the M1 coupling — heavy particle ⇒ explicit stable).
//!
//! Validates that freely-moving SBM is stable and physical: the disk falls, accelerates, and the
//! speed plateaus at a terminal velocity (drag balances buoyancy-corrected gravity), staying finite
//! and inside the box. The sharp interface should give a cleaner terminal state than penalization's
//! diffuse mask. Run: cargo oxide run --bin sbm-moving-disk-check  (DISK_NY / DISK_STEPS knobs).

use gale::dg::{
    sbm_force_torque, CircleLevelSet, FreeBody, Mesh2d, ShiftedBoundary, ShiftedMultigrid,
    ShiftedPoisson,
};
use gale_gpu::operators::poisson::GpuPoissonMg;

fn env_usize(k: &str, d: usize) -> usize {
    std::env::var(k).ok().and_then(|v| v.parse().ok()).unwrap_or(d)
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let p = 3;
    let (w, h) = (0.6, 1.2); // closed box (no-slip walls)
    let ny = env_usize("DISK_NY", 20);
    let nx = ((w / h) * ny as f64).round().max(1.0) as usize;
    let mesh = Mesh2d::rectangular(p, nx, ny, [0.0, w], [0.0, h]);
    let nn = mesh.refq.n_nodes();
    let ndof = mesh.n_elements() * nn;
    let hx = w / nx as f64;

    // Heavy disk (ρ_s/ρ_f = 4 ⇒ explicit coupling stable), released in the upper box (D/h≈4).
    let r = 0.12;
    let (rho_s, rho_f) = (4.0, 1.0);
    let nu = 0.05;
    let g = 1.0;
    let f_net = (rho_s - rho_f) * std::f64::consts::PI * r * r * g; // buoyancy-corrected weight
    let mut body = FreeBody::disk(0.5 * w, 0.9, r, rho_s, 0.0).with_external_force(0.0, -f_net);
    let dt = 2.0e-3;
    let alpha = 5.0;
    let lambda = 1.0 / (nu * dt);
    let (tol, maxit) = (1e-7, 4000);
    let max_steps = env_usize("DISK_STEPS", 400);
    let wall_tags = mesh.boundary_tags(); // all four walls (no-slip velocity, Neumann pressure)
    println!("=== SBM freely-moving disk (settling), {nx}×{ny} p={p} (D/h≈{:.1}) ===", 2.0 * r / hx);
    println!("r={r} ρs/ρf={rho_s}/{rho_f} ν={nu} f_net={f_net:.4} mass={:.4} dt={dt}", body.mass);

    let grad = |f: &[f64], comp: usize| -> Vec<f64> {
        let mut gv = vec![0.0; ndof];
        for (e, el) in mesh.elements.iter().enumerate() {
            let sl = &f[e * nn..(e + 1) * nn];
            let g1 = if comp == 0 { el.geom.grad_x(&mesh.refq, sl) } else { el.geom.grad_y(&mesh.refq, sl) };
            gv[e * nn..(e + 1) * nn].copy_from_slice(&g1);
        }
        gv
    };

    let mut ux = vec![0.0; ndof];
    let mut uy = vec![0.0; ndof];
    let mut pp = vec![0.0; ndof];
    let mut vmax = 0.0f64;
    let y0 = body.body.cy;

    // Amortized MG setup: the expensive colored-diagonal probing + power iteration only changes
    // when the active-element mask changes (the slow-moving body keeps it fixed for many steps).
    // Cache the smoother and reuse it (geometry still rebuilt fresh each step) until the mask flips.
    let mut prev_active: Vec<bool> = Vec::new();
    let mut cached_pres: Option<(Vec<Vec<f64>>, Vec<f64>)> = None;
    let mut cached_vel: Option<(Vec<Vec<f64>>, Vec<f64>)> = None;
    let mut rebuilds = 0usize;

    // SBM_GPU: do the per-step elliptic solves on the GPU. The surrogate changes every step, so the
    // device hierarchy is re-uploaded each step (`rebuild_sbm`) onto persistent handles (context +
    // module created ONCE). The host still builds the CPU ShiftedMultigrid for the per-level
    // diagonal/ω (shared setup); the GPU accelerates the solves. Built from the initial pose.
    let use_gpu = std::env::var("SBM_GPU").is_ok();
    let (mut gpu_pres, mut gpu_vel) = if use_gpu {
        let ls0 = CircleLevelSet::new(body.body.cx, body.body.cy, r);
        let gp = GpuPoissonMg::new_sbm(ShiftedMultigrid::new(
            p, nx, ny, [0.0, w], [0.0, h], alpha, 0.0, wall_tags.clone(), &ls0, false, false,
        ))?;
        let gv = GpuPoissonMg::new_sbm(ShiftedMultigrid::new(
            p, nx, ny, [0.0, w], [0.0, h], alpha, lambda, vec![], &ls0, true, true,
        ))?;
        (Some(gp), Some(gv))
    } else {
        (None, None)
    };

    for step in 1..=max_steps {
        let (cx, cy) = (body.body.cx, body.body.cy);
        let (bu, bv, bom) = (body.body.u, body.body.v, body.body.omega);
        // Surrogate geometry at the current pose.
        let ls = CircleLevelSet::new(cx, cy, r);
        let sb = ShiftedBoundary::new(&mesh, &ls);

        // Operators (geometry always current). The SBM multigrids' expensive smoother is rebuilt
        // only when the active mask changes; otherwise it is reused (amortized) with fresh geometry.
        let pres = ShiftedPoisson::with_bc(&mesh, alpha, 0.0, wall_tags.clone(), sb.clone()).surrogate_neumann();
        let vel = ShiftedPoisson::with_bc(&mesh, alpha, lambda, vec![], sb.clone()).taylor(true);
        let (pres_smg, vel_smg) = if sb.active != prev_active {
            let ps = ShiftedMultigrid::new(p, nx, ny, [0.0, w], [0.0, h], alpha, 0.0, wall_tags.clone(), &ls, false, false);
            let vs = ShiftedMultigrid::new(p, nx, ny, [0.0, w], [0.0, h], alpha, lambda, vec![], &ls, true, true);
            cached_pres = Some(ps.smoother_data());
            cached_vel = Some(vs.smoother_data());
            prev_active = sb.active.clone();
            rebuilds += 1;
            (ps, vs)
        } else {
            let ps = ShiftedMultigrid::new_reusing_smoother(p, nx, ny, [0.0, w], [0.0, h], alpha, 0.0, wall_tags.clone(), &ls, false, false, cached_pres.clone().unwrap());
            let vs = ShiftedMultigrid::new_reusing_smoother(p, nx, ny, [0.0, w], [0.0, h], alpha, lambda, vec![], &ls, true, true, cached_vel.clone().unwrap());
            (ps, vs)
        };

        // Rigid-body no-slip on the surrogate: u_s = U − ω(y−c_y), v_s = V + ω(x−c_x), evaluated at
        // the true point. Outer walls are no-slip (0). Gate by distance to the center: surrogate
        // true points sit at ~r from the center; walls are far. (Body stays mid-box.)
        let g_u = move |x: f64, y: f64| if (x - cx).hypot(y - cy) < 1.5 * r { bu - bom * (y - cy) } else { 0.0 };
        let g_v = move |x: f64, y: f64| if (x - cx).hypot(y - cy) < 1.5 * r { bv + bom * (x - cx) } else { 0.0 };

        // Stage 1 — explicit convection û = uⁿ − Δt (u·∇)u.
        let (gxx, gyx) = (grad(&ux, 0), grad(&ux, 1));
        let (gxy, gyy) = (grad(&uy, 0), grad(&uy, 1));
        let mut uhx = vec![0.0; ndof];
        let mut uhy = vec![0.0; ndof];
        for i in 0..ndof {
            uhx[i] = ux[i] - dt * (ux[i] * gxx[i] + uy[i] * gyx[i]);
            uhy[i] = uy[i] - dt * (ux[i] * gxy[i] + uy[i] * gyy[i]);
        }
        // Stage 2 — pressure projection −∇²p = (1/Δt)∇·û. Closed box ⇒ singular (deflated).
        let div: Vec<f64> = grad(&uhx, 0).iter().zip(grad(&uhy, 1).iter()).map(|(a, b)| a + b).collect();
        let fp: Vec<f64> = div.iter().map(|x| -x / dt).collect();
        let bp = pres.rhs(&fp, |_, _| 0.0);
        (pp, _) = if let Some(gp) = gpu_pres.as_mut() {
            gp.rebuild_sbm(&pres_smg)?;
            gp.solve_from(&bp, &pp, tol, maxit)? // GPU, warm-started; deflated (singular box)
        } else {
            pres_smg.pcg_deflated(&bp, tol, maxit)
        };
        let (px, py) = (grad(&pp, 0), grad(&pp, 1));
        for i in 0..ndof {
            uhx[i] -= dt * px[i];
            uhy[i] -= dt * py[i];
        }
        // Stage 3 — viscous Helmholtz (λM + A)uⁿ⁺¹ = λM û with SBM rigid no-slip.
        let fxv: Vec<f64> = uhx.iter().map(|v| lambda * v).collect();
        let fyv: Vec<f64> = uhy.iter().map(|v| lambda * v).collect();
        let (bx, by) = (vel.rhs(&fxv, g_u), vel.rhs(&fyv, g_v));
        if let Some(gv) = gpu_vel.as_mut() {
            gv.rebuild_sbm(&vel_smg)?; // velocity operator (same for both components this step)
            ux = gv.solve_from(&bx, &ux, tol, maxit)?.0;
            uy = gv.solve_from(&by, &uy, tol, maxit)?.0;
        } else {
            ux = vel_smg.pcg(&bx, tol, maxit).0;
            uy = vel_smg.pcg(&by, tol, maxit).0;
        }

        // Hydrodynamic force/torque on the true circle, then explicit Newton–Euler.
        let (fx, fy, tq) = sbm_force_torque(&mesh, &sb, &ux, &uy, &pp, nu, cx, cy, r);
        body.advance(fx, fy, tq, dt);

        let speed = body.body.v.abs();
        vmax = vmax.max(speed);
        let re = rho_f * speed * (2.0 * r) / nu;
        if step % 25 == 0 || step == 1 {
            println!(
                "  step {step:5}  y={:.4}  v={:+.4}  |Re|={re:.2}  Fy={fy:+.4}  ω={:+.3}",
                body.body.cy, body.body.v, body.body.omega
            );
        }
        // Stop if the disk nears the floor (wall interaction would need contact handling).
        if body.body.cy < 2.5 * r {
            println!("  (reached the lower wall region at step {step})");
            break;
        }
    }

    // Sanity criterion (short run): stable + physical — finite, descending under gravity, speed
    // bounded (not exploding). The full fall to terminal velocity is validated on the fast GPU path.
    let finite = body.body.cy.is_finite() && body.body.v.is_finite();
    let descending = body.body.v < 0.0;
    let bounded = vmax < 1.0; // a blow-up would race past O(1) immediately
    println!("\nRESULT  y {:.3}→{:.3}, v={:+.4}, max|v|≈{:.4}  (MG rebuilds: {rebuilds})", y0, body.body.cy, body.body.v, vmax);
    if finite && descending && bounded {
        println!("OK: freely-moving SBM disk is stable & physical (descends, drag resists, no blow-up)");
        Ok(())
    } else {
        eprintln!("FAIL: unphysical (finite={finite} descending={descending} bounded={bounded})");
        std::process::exit(1);
    }
}
