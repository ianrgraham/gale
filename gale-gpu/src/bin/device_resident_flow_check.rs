//! Validates a **fully device-resident dual-splitting NS step** against a host-orchestrated
//! reference that uses the SAME GPU solver. This is the foundation increment for eliminating
//! per-step host↔device memcopies: in the device path the velocity and pressure fields stay
//! resident on the GPU across the whole step — convection, divergence, projection, RHS assembly,
//! and all three elliptic solves run on the device with NO field transfers (only the final
//! download for the comparison). Lid-driven cavity (closed box, moving top lid ⇒ nonzero velocity
//! Nitsche lift; singular all-Neumann pressure, deflated).
//!
//! Both paths use the identical GPU MG-PCG solver (the host reference via the host-slice wrapper
//! `solve`, the device path via the device-buffer `solve_dev`), so the ONLY difference is WHERE
//! the per-step stages run — host `Vec` math vs device kernels. They must agree to solver
//! tolerance, which isolates the new stage kernels. Run: cargo oxide run --bin device-resident-flow-check

use gale::dg::{Mesh2d, PMultigrid, Poisson};
use gale_gpu::operators::poisson::GpuPoissonMg;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let (p, alpha) = (4usize, 5.0);
    let (nx, ny) = (16usize, 16usize);
    let (nu, dt) = (0.05, 5e-3);
    let lambda = 1.0 / (nu * dt);
    let (tol, maxit) = (1e-9, 2000);
    let steps = 12usize;
    let xr = [0.0, 1.0];
    let yr = [0.0, 1.0];

    let mesh = Mesh2d::rectangular(p, nx, ny, xr, yr);
    let nn = mesh.refq.n_nodes();
    let ndof = mesh.n_elements() * nn;
    let tags = mesh.boundary_tags();
    println!("=== device-resident dual-splitting vs host (same GPU solver) — lid cavity {nx}×{ny} p={p}, {steps} steps ===");

    // Host operators (for RHS assembly + lift/jw extraction).
    let pres_op = Poisson::with_bc(&mesh, alpha, 0.0, tags.clone());
    let vel_op = Poisson::with_reaction(&mesh, alpha, lambda);
    let g_u = |_x: f64, y: f64| if y > 1.0 - 1e-9 { 1.0 } else { 0.0 }; // moving lid (top)
    let g_v = |_x: f64, _y: f64| 0.0;

    // GPU handles (shared primary context + null stream ⇒ buffers interoperate). Used by BOTH
    // paths so the solver is identical; only the stage math differs.
    let hp = GpuPoissonMg::new(PMultigrid::with_bc(p, nx, ny, xr, yr, alpha, 0.0, tags.clone()))?;
    let hv = GpuPoissonMg::new(PMultigrid::with_reaction(p, nx, ny, xr, yr, alpha, lambda))?;

    // Constants (precomputed once): diagonal mass + boundary lifts.
    let jw = vel_op.rhs(&vec![1.0; ndof], |_, _| 0.0); // M·1
    let lift_vx = vel_op.rhs(&vec![0.0; ndof], g_u);
    let lift_vy = vel_op.rhs(&vec![0.0; ndof], g_v);

    // Element-local host gradient (matches the device gradient kernel).
    let grad = |f: &[f64], comp: usize| -> Vec<f64> {
        let mut g = vec![0.0; ndof];
        for (e, el) in mesh.elements.iter().enumerate() {
            let sl = &f[e * nn..(e + 1) * nn];
            let gv = if comp == 0 { el.geom.grad_x(&mesh.refq, sl) } else { el.geom.grad_y(&mesh.refq, sl) };
            g[e * nn..(e + 1) * nn].copy_from_slice(&gv);
        }
        g
    };

    // ---- Host-orchestrated reference (host stages + GPU solver via the host-slice wrapper) -----
    let (mut hux, mut huy) = (vec![0.0; ndof], vec![0.0; ndof]);
    for _ in 0..steps {
        let (gxx, gyx) = (grad(&hux, 0), grad(&hux, 1));
        let (gxy, gyy) = (grad(&huy, 0), grad(&huy, 1));
        let mut uhx = vec![0.0; ndof];
        let mut uhy = vec![0.0; ndof];
        for i in 0..ndof {
            uhx[i] = hux[i] - dt * (hux[i] * gxx[i] + huy[i] * gyx[i]);
            uhy[i] = huy[i] - dt * (hux[i] * gxy[i] + huy[i] * gyy[i]);
        }
        let div: Vec<f64> = grad(&uhx, 0).iter().zip(grad(&uhy, 1).iter()).map(|(a, b)| a + b).collect();
        let fp: Vec<f64> = div.iter().map(|d| -d / dt).collect();
        let bp = pres_op.rhs(&fp, |_, _| 0.0);
        let pp = hp.solve(&bp, tol, maxit)?.0;
        let (px, py) = (grad(&pp, 0), grad(&pp, 1));
        for i in 0..ndof {
            uhx[i] -= dt * px[i];
            uhy[i] -= dt * py[i];
        }
        let fxv: Vec<f64> = uhx.iter().map(|v| lambda * v).collect();
        let fyv: Vec<f64> = uhy.iter().map(|v| lambda * v).collect();
        hux = hv.solve(&vel_op.rhs(&fxv, g_u), tol, maxit)?.0;
        huy = hv.solve(&vel_op.rhs(&fyv, g_v), tol, maxit)?.0;
    }
    let humax = hux.iter().zip(&huy).fold(0.0f64, |m, (a, b)| m.max(a.hypot(*b)));

    // ---- Device-resident path (stages + solves all on the GPU; fields never leave) -------------
    let jw_d = hv.upload_field(&jw)?;
    let lift_vx_d = hv.upload_field(&lift_vx)?;
    let lift_vy_d = hv.upload_field(&lift_vy)?;
    let lift_p_d = hv.upload_field(&vec![0.0; ndof])?; // homogeneous-Neumann pressure ⇒ no lift
    let mut ux = hv.alloc_field()?;
    let mut uy = hv.alloc_field()?;
    let mut uhx = hv.alloc_field()?;
    let mut uhy = hv.alloc_field()?;
    let mut ga = hv.alloc_field()?;
    let mut gb = hv.alloc_field()?;
    let mut gc = hv.alloc_field()?;
    let mut gd = hv.alloc_field()?;
    let mut cx = hv.alloc_field()?;
    let mut cy = hv.alloc_field()?;
    let mut div = hv.alloc_field()?;
    let mut pp = hv.alloc_field()?;
    let mut rhs = hv.alloc_field()?;

    for _ in 0..steps {
        // Stage 1 — convection û = u − Δt (u·∇)u.
        hv.gradient_dev(&ux, &mut ga, &mut gb)?; // ∂ux/∂x, ∂ux/∂y
        hv.gradient_dev(&uy, &mut gc, &mut gd)?; // ∂uy/∂x, ∂uy/∂y
        hv.fma2_dev(&mut cx, &ux, &ga, &uy, &gb)?; // u·∂ₓu + v·∂ᵧu
        hv.fma2_dev(&mut cy, &ux, &gc, &uy, &gd)?;
        hv.copy_dev(&mut uhx, &ux)?;
        hv.copy_dev(&mut uhy, &uy)?;
        hv.axpy_dev(&mut uhx, &cx, -dt)?;
        hv.axpy_dev(&mut uhy, &cy, -dt)?;
        // Stage 2 — pressure projection −∇²p = (1/Δt)∇·û (singular ⇒ deflated).
        hv.gradient_dev(&uhx, &mut ga, &mut gb)?; // ∂ûx/∂x in ga
        hv.gradient_dev(&uhy, &mut gc, &mut gd)?; // ∂ûy/∂y in gd
        hv.copy_dev(&mut div, &ga)?;
        hv.axpy_dev(&mut div, &gd, 1.0)?;
        hv.scal_dev(&mut div, -1.0 / dt)?; // fp = −div/Δt
        hv.rhs_madd_dev(&mut rhs, &jw_d, &div, &lift_p_d, 1.0)?; // bp = M·fp
        hp.solve_dev(&rhs, None, &mut pp, tol, maxit)?; // singular pressure ⇒ pressure handle
        // Project: u* = û − Δt ∇p.
        hv.gradient_dev(&pp, &mut ga, &mut gb)?;
        hv.axpy_dev(&mut uhx, &ga, -dt)?;
        hv.axpy_dev(&mut uhy, &gb, -dt)?;
        // Stage 3 — viscous Helmholtz (λM + A) uⁿ⁺¹ = λM u* + lift.
        hv.rhs_madd_dev(&mut rhs, &jw_d, &uhx, &lift_vx_d, lambda)?;
        hv.solve_dev(&rhs, None, &mut ux, tol, maxit)?;
        hv.rhs_madd_dev(&mut rhs, &jw_d, &uhy, &lift_vy_d, lambda)?;
        hv.solve_dev(&rhs, None, &mut uy, tol, maxit)?;
    }
    let gux = hv.download_field(&ux)?;
    let guy = hv.download_field(&uy)?;
    let gumax = gux.iter().zip(&guy).fold(0.0f64, |m, (a, b)| m.max(a.hypot(*b)));

    // ---- Compare --------------------------------------------------------------------
    let rel = |a: &[f64], b: &[f64]| -> f64 {
        let d: f64 = a.iter().zip(b).map(|(x, y)| (x - y).powi(2)).sum();
        let n: f64 = b.iter().map(|x| x * x).sum::<f64>().max(1e-300);
        (d / n).sqrt()
    };
    let ru = rel(&gux, &hux);
    let rv = rel(&guy, &huy);
    println!("  host umax={humax:.4}  device umax={gumax:.4}  rel ux={ru:.3e}  rel uy={rv:.3e}");
    if humax.is_finite() && ru < 1e-6 && rv < 1e-6 {
        println!("OK: device-resident dual-splitting matches the host trajectory (no per-step field transfers).");
        Ok(())
    } else {
        eprintln!("FAIL: device-resident trajectory diverges from host.");
        std::process::exit(1);
    }
}
