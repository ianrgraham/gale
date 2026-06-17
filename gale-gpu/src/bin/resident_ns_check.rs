//! Validates `GpuResidentNs` (the device-resident, self-stepping NS integrator) against a
//! host-orchestrated reference that runs the IDENTICAL dual-splitting math through the SAME GPU
//! MG-PCG solver. The only difference is WHERE the per-step stages run — host `Vec` math vs the
//! resident device kernels — so they must agree to solver tolerance. Kolmogorov body forcing
//! (`fx = F·sin(2πn y)`), closed zero-velocity walls, a small symmetry-breaking IC.
//! Run: cargo oxide run --bin resident-ns-check

use gale::dg::{Mesh2d, PMultigrid, Poisson};
use gale_gpu::operators::poisson::GpuPoissonMg;
use gale_gpu::resident::GpuResidentNs;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let (p, alpha) = (3usize, 5.0);
    let (nx, ny) = (16usize, 16usize);
    let (nu, dt) = (0.08, 2e-3);
    let lambda = 1.0 / (nu * dt);
    let (tol, maxit) = (1e-9, 4000);
    let steps = 30usize;
    let xr = [0.0, 1.0];
    let yr = [0.0, 1.0];
    let two_pi = std::f64::consts::TAU;
    let (force, nk) = (4.0, 2.0);

    let mesh = Mesh2d::rectangular(p, nx, ny, xr, yr);
    let nn = mesh.refq.n_nodes();
    let ndof = mesh.n_elements() * nn;
    let tags = mesh.boundary_tags();
    println!("=== GpuResidentNs vs host (same GPU solver) — Kolmogorov NS {nx}×{ny} p={p}, {steps} steps ===");

    let fx = |_x: f64, y: f64| force * (two_pi * nk * y).sin();
    let fy = |_x: f64, _y: f64| 0.0;
    let zero = |_x: f64, _y: f64| 0.0;
    let u_est = force / (nu * (two_pi * nk).powi(2));
    let u_ic = |_x: f64, y: f64| u_est * (two_pi * nk * y).sin();
    let v_ic = |x: f64, y: f64| 0.02 * u_est * (two_pi * x).sin() * (two_pi * y).sin();

    // Nodal IC + nodal body force.
    let (mut ux0, mut uy0) = (vec![0.0; ndof], vec![0.0; ndof]);
    let (mut bx, mut by) = (vec![0.0; ndof], vec![0.0; ndof]);
    for (e, el) in mesh.elements.iter().enumerate() {
        for k in 0..nn {
            let (x, y) = (el.geom.x[k], el.geom.y[k]);
            ux0[e * nn + k] = u_ic(x, y);
            uy0[e * nn + k] = v_ic(x, y);
            bx[e * nn + k] = fx(x, y);
            by[e * nn + k] = fy(x, y);
        }
    }

    // ---- Host-orchestrated reference (host stages + GPU MG-PCG solver) ----
    let pres_op = Poisson::with_bc(&mesh, alpha, 0.0, tags.clone());
    let vel_op = Poisson::with_reaction(&mesh, alpha, lambda);
    let hp = GpuPoissonMg::new(PMultigrid::from_mesh(&mesh, alpha, 0.0, tags.clone()).unwrap())?;
    let hv = GpuPoissonMg::new(PMultigrid::from_mesh(&mesh, alpha, lambda, Vec::new()).unwrap())?;
    let grad = |f: &[f64], comp: usize| -> Vec<f64> {
        let mut g = vec![0.0; ndof];
        for (e, el) in mesh.elements.iter().enumerate() {
            let sl = &f[e * nn..(e + 1) * nn];
            let gv = if comp == 0 { el.geom.grad_x(&mesh.refq, sl) } else { el.geom.grad_y(&mesh.refq, sl) };
            g[e * nn..(e + 1) * nn].copy_from_slice(&gv);
        }
        g
    };
    let (mut hux, mut huy) = (ux0.clone(), uy0.clone());
    for _ in 0..steps {
        let (gxx, gyx) = (grad(&hux, 0), grad(&hux, 1));
        let (gxy, gyy) = (grad(&huy, 0), grad(&huy, 1));
        let mut uhx = vec![0.0; ndof];
        let mut uhy = vec![0.0; ndof];
        for i in 0..ndof {
            uhx[i] = hux[i] + dt * (bx[i] - (hux[i] * gxx[i] + huy[i] * gyx[i]));
            uhy[i] = huy[i] + dt * (by[i] - (hux[i] * gxy[i] + huy[i] * gyy[i]));
        }
        let div: Vec<f64> = grad(&uhx, 0).iter().zip(grad(&uhy, 1).iter()).map(|(a, b)| a + b).collect();
        let fp: Vec<f64> = div.iter().map(|d| -d / dt).collect();
        let pp = hp.solve(&pres_op.rhs(&fp, |_, _| 0.0), tol, maxit)?.0;
        let (px, py) = (grad(&pp, 0), grad(&pp, 1));
        for i in 0..ndof {
            uhx[i] -= dt * px[i];
            uhy[i] -= dt * py[i];
        }
        let fxv: Vec<f64> = uhx.iter().map(|v| lambda * v).collect();
        let fyv: Vec<f64> = uhy.iter().map(|v| lambda * v).collect();
        hux = hv.solve(&vel_op.rhs(&fxv, zero), tol, maxit)?.0;
        huy = hv.solve(&vel_op.rhs(&fyv, zero), tol, maxit)?.0;
    }
    let humax = hux.iter().zip(&huy).fold(0.0f64, |m, (a, b)| m.max(a.hypot(*b)));

    // ---- Device-resident path (GpuResidentNs: every stage on the GPU, no per-step transfers) ----
    let mut sim = GpuResidentNs::new(&mesh, nu, dt, alpha, fx, fy, zero, zero)?.with_tol(tol, maxit);
    sim.set_velocity(&ux0, &uy0)?;
    sim.run(steps)?;
    let (gux, guy) = sim.velocity()?;
    let gumax = gux.iter().zip(&guy).fold(0.0f64, |m, (a, b)| m.max(a.hypot(*b)));

    let rel = |a: &[f64], b: &[f64]| -> f64 {
        let d: f64 = a.iter().zip(b).map(|(x, y)| (x - y).powi(2)).sum();
        let n: f64 = b.iter().map(|x| x * x).sum::<f64>().max(1e-300);
        (d / n).sqrt()
    };
    let (ru, rv) = (rel(&gux, &hux), rel(&guy, &huy));
    println!("  host umax={humax:.5}  device umax={gumax:.5}  rel ux={ru:.3e}  rel uy={rv:.3e}");
    if humax.is_finite() && ru < 1e-6 && rv < 1e-6 {
        println!("OK: GpuResidentNs matches the host trajectory (fully device-resident, no per-step field transfers).");
        Ok(())
    } else {
        eprintln!("FAIL: GpuResidentNs diverges from host.");
        std::process::exit(1);
    }
}
