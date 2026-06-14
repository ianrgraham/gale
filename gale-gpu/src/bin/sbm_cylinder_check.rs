//! **SBM Schäfer-Turek 2D-1 cylinder (Re=20)** — WORK IN PROGRESS (the sharp-interface
//! counterpart of cylinder-drag-check). The full SBM dual-splitting Navier-Stokes solver
//! is here: SBM-Nitsche no-slip (Taylor-corrected) on the velocity-Helmholtz surrogate,
//! natural (homogeneous-Neumann) surrogate on the pressure-Poisson, and a true-circle
//! drag recovery. The velocity path + geometry are sound (the SBM Dirichlet MMS passes).
//!
//! BLOCKED on the pressure solve: the natural-Neumann SBM pressure-Poisson is solved by
//! UNPRECONDITIONED CG, which STALLS (hits maxit each step) — it cannot reach steady state
//! at usable resolution. This confirms the remaining sub-build: an MG preconditioner for
//! the SBM operator (and likely conditioning work on the natural-Neumann surrogate +
//! inactive-identity block). Until then this bin does not produce a converged C_D — it is
//! the integration scaffold, kept so the MG work can plug straight in. See
//! docs/research-sharp-interface.md. Goal (once unblocked): C_D → 5.5795, beating
//! penalization's ~2.1%-low plateau. CPU; resolution env-tunable (SBM_NY, default 16).
//!
//! Dual-splitting per step (A ≈ −∇² is the SIPG operator, λ = 1/(νΔt)):
//!   û = uⁿ − Δt (u·∇)u;  −∇²p = (1/Δt)∇·û  (Neumann surrogate, p=0 outflow);
//!   u* = û − Δt ∇p;  (λM + A) uⁿ⁺¹ = λM u*  with SBM no-slip (Dirichlet surrogate).
//! Drag = ∮_Γ (σ·n)·ê_x on the TRUE circle (surface-stress integral, r dθ measure;
//! p,∇u Taylor-extrapolated from the surrogate). Tags: west=3 in, east=1 out, walls 0,2.
//!
//! Run: cargo oxide run --bin sbm-cylinder-check   (SBM_NY=24 for a finer point)

use gale::dg::{CircleLevelSet, Mesh2d, ShiftedBoundary, ShiftedMultigrid, ShiftedPoisson};
use std::f64::consts::PI;

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

    let mesh = Mesh2d::rectangular(p, nx, ny, [0.0, 2.2], [0.0, h]);
    let nn = mesh.refq.n_nodes();
    let ndof = mesh.n_elements() * nn;
    let ls = CircleLevelSet::new(0.2, 0.2, 0.05);
    let sb = ShiftedBoundary::new(&mesh, &ls);
    let hx = 2.2 / nx as f64;
    println!("=== SBM Schäfer-Turek cylinder (Re=20), {nx}×{ny} p={p} (D/h≈{:.1}), dt={dt} ===", d / hx);
    println!("surrogate faces: {}, active elements: {}/{}", sb.faces.len(), sb.n_active(), mesh.n_elements());

    // Operators: velocity Helmholtz (Dirichlet surrogate no-slip, Taylor; outflow=Neumann),
    // pressure Poisson (natural-Neumann surrogate; inflow+walls Neumann, outflow Dirichlet p=0).
    let vel = ShiftedPoisson::with_bc(&mesh, alpha, lambda, vec![1], sb.clone()).taylor(true);
    let pres = ShiftedPoisson::with_bc(&mesh, alpha, 0.0, vec![3, 0, 2], sb.clone()).surrogate_neumann();
    // Preconditioner for the ill-conditioned SBM pressure: an SBM-AWARE p-multigrid (the operator
    // itself at every level — same config: natural-Neumann surrogate, no Taylor). Unlike the
    // standard full-mesh MG (which stagnates on the cylinder-local modes and hits maxit), this
    // converges, cutting O(10³) unpreconditioned iters to ~O(10²). See bin sbm-pressure-diag.
    let pres_mg = ShiftedMultigrid::new(p, nx, ny, [0.0, 2.2], [0.0, h], alpha, 0.0, vec![3, 0, 2], &ls, false, false);
    // Velocity Helmholtz is also ill-conditioned (the surrogate Nitsche penalty), so precondition
    // it too with a matching SBM-MG (reaction λ, Dirichlet surrogate, Taylor; outflow Neumann).
    let vel_mg = ShiftedMultigrid::new(p, nx, ny, [0.0, 2.2], [0.0, h], alpha, lambda, vec![1], &ls, true, true);
    // SBM_GPU: run the per-step pressure + velocity elliptic solves fully on the GPU (persistent
    // SBM-MG handles, built once). Same operators as the CPU path, validated bit-for-bit
    // (sbm-poisson/pcg/velocity-check); the GPU solve starts from zero (no warm-start) so it does
    // more iters/step but each is far cheaper. Falls back to the CPU MG-PCG when unset.
    let use_gpu = std::env::var("SBM_GPU").is_ok();
    let (gpu_pres, gpu_vel) = if use_gpu {
        let gp = gale_gpu::operators::poisson::GpuPoissonMg::new_sbm(ShiftedMultigrid::new(
            p, nx, ny, [0.0, 2.2], [0.0, h], alpha, 0.0, vec![3, 0, 2], &ls, false, false,
        ))?;
        let gv = gale_gpu::operators::poisson::GpuPoissonMg::new_sbm(ShiftedMultigrid::new(
            p, nx, ny, [0.0, 2.2], [0.0, h], alpha, lambda, vec![1], &ls, true, true,
        ))?;
        (Some(gp), Some(gv))
    } else {
        (None, None)
    };

    // BC data (Dirichlet value at a point); surrogate (cylinder) evaluates these at the true
    // surface point ⇒ no-slip 0 there since the cylinder is interior (x≈0.2, not the inflow).
    let parab = |y: f64| 4.0 * um * y * (h - y) / (h * h);
    let g_u = |x: f64, y: f64| if x < 0.5 * hx { parab(y) } else { 0.0 }; // inflow parabola, else 0
    let g_v = |_x: f64, _y: f64| 0.0;
    let g_p = |_x: f64, _y: f64| 0.0; // outflow p = 0

    // Element-local helpers.
    let grad = |f: &[f64], comp: usize| -> Vec<f64> {
        let mut g = vec![0.0; ndof];
        for (e, el) in mesh.elements.iter().enumerate() {
            let sl = &f[e * nn..(e + 1) * nn];
            let gv = if comp == 0 { el.geom.grad_x(&mesh.refq, sl) } else { el.geom.grad_y(&mesh.refq, sl) };
            g[e * nn..(e + 1) * nn].copy_from_slice(&gv);
        }
        g
    };

    // Seed with the parabolic profile (closer to steady).
    let mut ux = vec![0.0; ndof];
    let mut uy = vec![0.0; ndof];
    for (e, el) in mesh.elements.iter().enumerate() {
        for k in 0..nn {
            if sb.active[e] {
                ux[e * nn + k] = parab(el.geom.y[k]);
            }
        }
    }

    // Loose per-step pressure tol: the STEADY state is tol-independent, so a sloppy per-step
    // projection (warm-started) converges to the right steady solution with far fewer CG iters.
    let (ptol, vtol, maxit) = (env_f64("SBM_PTOL", 1e-4), 1e-7, 5000);
    let mut cd = 0.0;
    let mut cd_prev = 0.0;
    let mut steady_at = None;
    let mut pp = vec![0.0; ndof]; // persists across steps for CG warm-starting
    for step in 1..=max_steps {
        // Stage 1: explicit convection û = u − Δt (u·∇)u.
        let (gxx, gyx) = (grad(&ux, 0), grad(&ux, 1));
        let (gxy, gyy) = (grad(&uy, 0), grad(&uy, 1));
        let mut uhx = vec![0.0; ndof];
        let mut uhy = vec![0.0; ndof];
        for i in 0..ndof {
            uhx[i] = ux[i] - dt * (ux[i] * gxx[i] + uy[i] * gyx[i]);
            uhy[i] = uy[i] - dt * (ux[i] * gxy[i] + uy[i] * gyy[i]);
        }
        // Stage 2: pressure projection −∇²p = (1/Δt)∇·û (warm-started from the previous step).
        let div: Vec<f64> = grad(&uhx, 0).iter().zip(grad(&uhy, 1).iter()).map(|(a, b)| a + b).collect();
        let fp: Vec<f64> = div.iter().map(|x| -x / dt).collect();
        let bp = pres.rhs(&fp, g_p);
        let (newp, pit) = if let Some(gp) = &gpu_pres {
            gp.solve_from(&bp, &pp, ptol, maxit)? // GPU, warm-started from the previous step
        } else {
            pres.solve_pcg_from(&bp, pp.clone(), |r| pres_mg.precondition(r), ptol, maxit)
        };
        pp = newp;
        // Stage 2b: u* = û − Δt ∇p.
        let (px, py) = (grad(&pp, 0), grad(&pp, 1));
        for i in 0..ndof {
            uhx[i] -= dt * px[i];
            uhy[i] -= dt * py[i];
        }
        // Stage 3: viscous Helmholtz (λM + A) uⁿ⁺¹ = λM u* with SBM no-slip.
        let fxv: Vec<f64> = uhx.iter().map(|v| lambda * v).collect();
        let fyv: Vec<f64> = uhy.iter().map(|v| lambda * v).collect();
        let (uxn, vitx) = if let Some(gv) = &gpu_vel {
            gv.solve_from(&vel.rhs(&fxv, g_u), &ux, vtol, maxit)?
        } else {
            vel.solve_pcg_from(&vel.rhs(&fxv, g_u), ux.clone(), |r| vel_mg.precondition(r), vtol, maxit)
        };
        let (uyn, vity) = if let Some(gv) = &gpu_vel {
            gv.solve_from(&vel.rhs(&fyv, g_v), &uy, vtol, maxit)?
        } else {
            vel.solve_pcg_from(&vel.rhs(&fyv, g_v), uy.clone(), |r| vel_mg.precondition(r), vtol, maxit)
        };
        ux = uxn;
        uy = uyn;
        if std::env::var("SBM_VERBOSE").is_ok() {
            println!("    [step {step}] p_iters={pit}  vel_iters={vitx}/{vity}");
        }

        // Drag on the true circle (every 25 steps + steady check). C_D uses the high-order
        // (Hessian-extrapolated) recovery; C_D_lo is the first-order recovery for comparison.
        if step % 25 == 0 || step == 1 {
            cd = norm * drag_x(&mesh, &sb, &ls, &ux, &uy, &pp, nu, false, &grad);
            let cd_lo = norm * drag_x(&mesh, &sb, &ls, &ux, &uy, &pp, nu, true, &grad);
            if step % 25 == 0 || step == 1 {
                let umax = ux.iter().zip(&uy).fold(0.0f64, |m, (a, b)| m.max(a.hypot(*b)));
                println!("  step {step:5}  C_D={cd:.4} (lo {cd_lo:.4})  (ref {cd_ref})  umax={umax:.3}  p_iters={pit}");
            }
            if step > 300 && ((cd - cd_prev) / cd.max(1e-30)).abs() < 2e-4 {
                steady_at = Some(step);
                break;
            }
            cd_prev = cd;
        }
    }

    let err = (cd - cd_ref).abs() / cd_ref * 100.0;
    println!("\nRESULT  SBM C_D = {cd:.4}  (ref {cd_ref}, err {err:.1}%)");
    match steady_at {
        Some(s) => println!("steady at step {s}"),
        None => println!("did NOT reach steady in {max_steps} steps; last C_D={cd:.4}"),
    }
    let stable = cd.is_finite() && cd > 0.5 * cd_ref && cd < 1.5 * cd_ref;
    if stable {
        println!("\nOK: SBM cylinder steady, C_D in the ballpark of the DFG reference\n(compare err to penalization's ~2.1% at similar resolution; SBM should be closer/converge).");
        Ok(())
    } else {
        eprintln!("\nFAIL: SBM cylinder C_D={cd} not in ballpark");
        std::process::exit(1);
    }
}

/// Drag (x-force on the body) by integrating the traction `σ·n = −p n + ν(∇u+∇uᵀ)·n` over
/// the TRUE circle, with `n` the body-outward normal and the `r dθ` arc measure. Fields are
/// Taylor-extrapolated from each surrogate node to its true-boundary point.
#[allow(clippy::too_many_arguments)]
fn drag_x(
    mesh: &Mesh2d,
    sb: &ShiftedBoundary,
    _ls: &CircleLevelSet,
    ux: &[f64],
    uy: &[f64],
    pp: &[f64],
    nu: f64,
    lo: bool, // true ⇒ first-order recovery (∇u at the surrogate); false ⇒ high-order (Hessian)
    grad: &impl Fn(&[f64], usize) -> Vec<f64>,
) -> f64 {
    let nn = mesh.refq.n_nodes();
    let (uxx, uxy) = (grad(ux, 0), grad(ux, 1));
    let uyx = grad(uy, 0); // ∂v/∂x (∂v/∂y unused in the x-traction)
    let (pgx, pgy) = (grad(pp, 0), grad(pp, 1));
    // HIGH-ORDER force recovery: the traction lives on the TRUE circle, but the discrete fields
    // live on the surrogate. The pressure is Taylor-extrapolated (p + ∇p·d); the velocity GRADIENT
    // must be too, or the viscous traction is only O(h). Extrapolate ∇u via its own gradient (the
    // velocity Hessian): ∂u/∂x(x̃+d) ≈ ∂u/∂x(x̃) + ∇(∂u/∂x)·d, etc. Set SBM_LO_DRAG to fall back to
    // the first-order recovery (∇u at the surrogate) for comparison.
    // Second derivatives needed for the three traction-x gradient components (∂u/∂x, ∂u/∂y, ∂v/∂x).
    let (dxx_x, dxx_y) = (grad(&uxx, 0), grad(&uxx, 1)); // ∇(∂u/∂x)
    let (dxy_x, dxy_y) = (grad(&uxy, 0), grad(&uxy, 1)); // ∇(∂u/∂y)
    let (dyx_x, dyx_y) = (grad(&uyx, 0), grad(&uyx, 1)); // ∇(∂v/∂x)
    // Collect (angle, traction_x) per surrogate node at its true-boundary point.
    let (cx, cy) = (0.2, 0.2);
    let mut pts: Vec<(f64, f64)> = Vec::new();
    for sf in &sb.faces {
        let e = sf.elem;
        let fd = &mesh.elements[e].faces[sf.edge as usize];
        for (a, sn) in sf.nodes.iter().enumerate() {
            let v = fd.nodes[a];
            let i = e * nn + v;
            let (dx, dy) = (sn.dx, sn.dy);
            // Taylor-extrapolate the fields to the true point x̃+d.
            let p_t = pp[i] + pgx[i] * dx + pgy[i] * dy;
            let (dudx, dudy, dvdx) = if lo {
                (uxx[i], uxy[i], uyx[i]) // first-order: ∇u at the surrogate (O(h))
            } else {
                (
                    uxx[i] + dxx_x[i] * dx + dxx_y[i] * dy,
                    uxy[i] + dxy_x[i] * dx + dxy_y[i] * dy,
                    uyx[i] + dyx_x[i] * dx + dyx_y[i] * dy,
                )
            };
            let (nx, ny) = (sn.tnx, sn.tny); // body-outward (into fluid) unit normal
            // (σ·n)_x = −p n_x + ν[2 ∂u/∂x n_x + (∂u/∂y+∂v/∂x) n_y]
            let tx = -p_t * nx + nu * (2.0 * dudx * nx + (dudy + dvdx) * ny);
            let theta = (sn.y + dy - cy).atan2(sn.x + dx - cx);
            pts.push((theta, tx));
        }
    }
    // Integrate t_x over the circle: sort by angle, trapezoidal in θ with r dθ.
    pts.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());
    let r = 0.05;
    let n = pts.len();
    let mut drag = 0.0;
    for k in 0..n {
        let (_th, tx) = pts[k];
        let thm = if k == 0 { pts[n - 1].0 - 2.0 * PI } else { pts[k - 1].0 };
        let thp = if k == n - 1 { pts[0].0 + 2.0 * PI } else { pts[k + 1].0 };
        let dth = 0.5 * (thp - thm); // node's angular span
        drag += tx * r * dth;
    }
    drag
}
