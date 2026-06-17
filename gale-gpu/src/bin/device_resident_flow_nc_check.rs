//! Stage 3c.3b validation: a **fully device-resident dual-splitting NS step on a NON-CONFORMING
//! (2:1 refined) mesh**, vs a host-orchestrated reference using the SAME GPU NC solver. Both run on
//! a `cartesian_refined` mesh; the device path keeps velocity/pressure resident on the GPU and does
//! every stage (convection, divergence, projection, RHS assembly) + both elliptic solves with the
//! NC primitives (`GpuPoissonNc::{gradient_dev,fma2_dev,rhs_madd_dev,…,solve_dev}`) — no field
//! transfers. The host reference uses host-`Vec` stages + `GpuPoissonNc::solve`. Same solver ⇒ they
//! must agree to round-off. This is the resident NC integrator step behind GPU-resident AMR.
//! Run: cargo oxide run --bin device-resident-flow-nc-check

use gale::dg::{Mesh2d, Poisson};
use gale_gpu::operators::poisson_nc::GpuPoissonNc;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let (p, alpha) = (3usize, 5.0);
    let (nx, ny) = (8usize, 8usize);
    let (nu, dt) = (0.05, 5e-3);
    let lambda = 1.0 / (nu * dt);
    let (tol, maxit) = (1e-9, 5000);
    let steps = 8usize;
    let xr = [0.0, 1.0];
    let yr = [0.0, 1.0];
    let refine: Vec<(usize, usize)> = vec![(2, 2), (3, 2), (5, 4), (1, 6)];
    let mesh = Mesh2d::cartesian_refined(p, nx, ny, xr, yr, &refine);
    let nn = mesh.refq.n_nodes();
    let ndof = mesh.n_elements() * nn;
    let tags = mesh.boundary_tags();
    println!("=== Stage 3c.3b: device-resident NC dual-splitting vs host (same GPU NC solver) — refined {nx}×{ny} p={p}, {} elems, {steps} steps ===", mesh.n_elements());

    // Host operators (RHS/lift/jw assembly only).
    let vel_op = Poisson::with_reaction(&mesh, alpha, lambda);
    let g_u = |_x: f64, y: f64| if y > 1.0 - 1e-9 { 1.0 } else { 0.0 }; // moving lid (top)
    let g_v = |_x: f64, _y: f64| 0.0;
    let jw = vel_op.rhs(&vec![1.0; ndof], |_, _| 0.0);
    let lift_vx = vel_op.rhs(&vec![0.0; ndof], g_u);
    let lift_vy = vel_op.rhs(&vec![0.0; ndof], g_v);
    let lift_p = vec![0.0; ndof];

    let h = GpuPoissonNc::new(&mesh, alpha)?;

    // Element-local host gradient (matches gradient_nc / gradient_dev).
    let grad = |f: &[f64], comp: usize| -> Vec<f64> {
        let mut g = vec![0.0; ndof];
        for (e, el) in mesh.elements.iter().enumerate() {
            let sl = &f[e * nn..(e + 1) * nn];
            let gv = if comp == 0 { el.geom.grad_x(&mesh.refq, sl) } else { el.geom.grad_y(&mesh.refq, sl) };
            g[e * nn..(e + 1) * nn].copy_from_slice(&gv);
        }
        g
    };

    // ---- Host-orchestrated reference (host stages + GPU NC solver) ----
    let (mut hux, mut huy) = (vec![0.0; ndof], vec![0.0; ndof]);
    for _ in 0..steps {
        let (gxx, gyx) = (grad(&hux, 0), grad(&hux, 1));
        let (gxy, gyy) = (grad(&huy, 0), grad(&huy, 1));
        let (mut uhx, mut uhy) = (vec![0.0; ndof], vec![0.0; ndof]);
        for i in 0..ndof {
            uhx[i] = hux[i] - dt * (hux[i] * gxx[i] + huy[i] * gyx[i]);
            uhy[i] = huy[i] - dt * (hux[i] * gxy[i] + huy[i] * gyy[i]);
        }
        let div: Vec<f64> = grad(&uhx, 0).iter().zip(grad(&uhy, 1).iter()).map(|(a, b)| a + b).collect();
        let fp: Vec<f64> = div.iter().map(|d| -d / dt).collect();
        let bp: Vec<f64> = (0..ndof).map(|i| jw[i] * fp[i] + lift_p[i]).collect();
        let pp = h.solve(&bp, 0.0, &tags, true, tol, maxit)?.0;
        let (px, py) = (grad(&pp, 0), grad(&pp, 1));
        for i in 0..ndof {
            uhx[i] -= dt * px[i];
            uhy[i] -= dt * py[i];
        }
        let bx: Vec<f64> = (0..ndof).map(|i| lambda * jw[i] * uhx[i] + lift_vx[i]).collect();
        let by: Vec<f64> = (0..ndof).map(|i| lambda * jw[i] * uhy[i] + lift_vy[i]).collect();
        hux = h.solve(&bx, lambda, &[], false, tol, maxit)?.0;
        huy = h.solve(&by, lambda, &[], false, tol, maxit)?.0;
    }
    let humax = hux.iter().zip(&huy).fold(0.0f64, |m, (a, b)| m.max(a.hypot(*b)));

    // ---- Device-resident path (all stages + solves on the GPU; fields never leave) ----
    let jw_d = h.upload(&jw)?;
    let lift_vx_d = h.upload(&lift_vx)?;
    let lift_vy_d = h.upload(&lift_vy)?;
    let lift_p_d = h.upload(&lift_p)?;
    let (mut ux, mut uy) = (h.alloc()?, h.alloc()?);
    let (mut uhx, mut uhy) = (h.alloc()?, h.alloc()?);
    let (mut ga, mut gb, mut gc, mut gd) = (h.alloc()?, h.alloc()?, h.alloc()?, h.alloc()?);
    let (mut cx, mut cy, mut div, mut pp, mut rhs) = (h.alloc()?, h.alloc()?, h.alloc()?, h.alloc()?, h.alloc()?);
    for _ in 0..steps {
        // convection û = u − Δt (u·∇)u
        h.gradient_dev(&ux, &mut ga, &mut gb)?;
        h.gradient_dev(&uy, &mut gc, &mut gd)?;
        h.fma2_dev(&mut cx, &ux, &ga, &uy, &gb)?;
        h.fma2_dev(&mut cy, &ux, &gc, &uy, &gd)?;
        h.copy_dev(&mut uhx, &ux)?;
        h.copy_dev(&mut uhy, &uy)?;
        h.axpy_dev(&mut uhx, &cx, -dt)?;
        h.axpy_dev(&mut uhy, &cy, -dt)?;
        // pressure projection
        h.gradient_dev(&uhx, &mut ga, &mut gb)?;
        h.gradient_dev(&uhy, &mut gc, &mut gd)?;
        h.copy_dev(&mut div, &ga)?;
        h.axpy_dev(&mut div, &gd, 1.0)?;
        h.scal_dev(&mut div, -1.0 / dt)?;
        h.rhs_madd_dev(&mut rhs, &jw_d, &div, &lift_p_d, 1.0)?;
        h.solve_dev(&rhs, None, &mut pp, 0.0, &tags, true, tol, maxit)?;
        // correction
        h.gradient_dev(&pp, &mut ga, &mut gb)?;
        h.axpy_dev(&mut uhx, &ga, -dt)?;
        h.axpy_dev(&mut uhy, &gb, -dt)?;
        // viscous
        h.rhs_madd_dev(&mut rhs, &jw_d, &uhx, &lift_vx_d, lambda)?;
        h.solve_dev(&rhs, None, &mut ux, lambda, &[], false, tol, maxit)?;
        h.rhs_madd_dev(&mut rhs, &jw_d, &uhy, &lift_vy_d, lambda)?;
        h.solve_dev(&rhs, None, &mut uy, lambda, &[], false, tol, maxit)?;
    }
    let (gux, guy) = (h.download(&ux)?, h.download(&uy)?);
    let gumax = gux.iter().zip(&guy).fold(0.0f64, |m, (a, b)| m.max(a.hypot(*b)));

    let rel = |a: &[f64], b: &[f64]| -> f64 {
        let d: f64 = a.iter().zip(b).map(|(x, y)| (x - y).powi(2)).sum();
        let n: f64 = b.iter().map(|x| x * x).sum::<f64>().max(1e-300);
        (d / n).sqrt()
    };
    let (ru, rv) = (rel(&gux, &hux), rel(&guy, &huy));
    println!("  host umax={humax:.5}  device umax={gumax:.5}  rel ux={ru:.3e}  rel uy={rv:.3e}");
    if humax.is_finite() && ru < 1e-6 && rv < 1e-6 {
        println!("OK: device-resident NC dual-splitting matches the host trajectory (no per-step field transfers).");
        Ok(())
    } else {
        eprintln!("FAIL: device-resident NC step diverges from host.");
        std::process::exit(1);
    }
}
