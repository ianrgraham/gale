//! Perf gate for the **device-resident NC dual-splitting NS step** — the per-iteration solver cost of
//! an adaptive run. Runs the validated device-resident step (assembly + both elliptic solves via
//! `GpuPoissonNc`, NO per-step field transfer) on a single-level refined mesh, and the host-orchestrated
//! reference (host stages + GPU solve, per-step transfers) for the BEFORE/AFTER comparison. Reports
//! ms/step for each and the speedup. This is the dominant cost of a simulation iteration; the adapt
//! cycle (see amr-adapt-perf) is amortized every N steps on top.
//!
//! Env: NS_NX (base nx=ny, default 32), NS_STEPS (default 20), NS_TOL (default 1e-6), NS_REFINE (band
//! stride; cells with (cx+cy)%stride==0 refined; default 4; 0 = uniform).
//! Run: cargo oxide run --bin resident-ns-step-perf
//! Profile: nsys profile -o /tmp/nsys_ns ./target/release/resident-ns-step-perf

use gale::dg::{Mesh2d, Poisson};
use gale_gpu::operators::multigrid_nc::GpuPMultigridNc;
use gale_gpu::operators::poisson_nc::GpuPoissonNc;
use std::time::Instant;

const P: usize = 3;

fn lid(_x: f64, y: f64) -> f64 {
    if y > 1.0 - 1e-9 { 1.0 } else { 0.0 }
}

fn consts(mesh: &Mesh2d, alpha: f64, lambda: f64) -> (Vec<f64>, Vec<f64>, Vec<f64>, Vec<f64>) {
    let ndof = mesh.n_elements() * mesh.refq.n_nodes();
    let vel = Poisson::with_reaction(mesh, alpha, lambda);
    (
        vel.rhs(&vec![1.0; ndof], |_, _| 0.0),
        vel.rhs(&vec![0.0; ndof], lid),
        vel.rhs(&vec![0.0; ndof], |_, _| 0.0),
        vec![0.0; ndof],
    )
}

fn grad(mesh: &Mesh2d, f: &[f64], comp: usize) -> Vec<f64> {
    let nn = mesh.refq.n_nodes();
    let mut g = vec![0.0; f.len()];
    for (e, el) in mesh.elements.iter().enumerate() {
        let sl = &f[e * nn..(e + 1) * nn];
        let gv = if comp == 0 { el.geom.grad_x(&mesh.refq, sl) } else { el.geom.grad_y(&mesh.refq, sl) };
        g[e * nn..(e + 1) * nn].copy_from_slice(&gv);
    }
    g
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let nx: usize = std::env::var("NS_NX").ok().and_then(|s| s.parse().ok()).unwrap_or(32);
    let steps: usize = std::env::var("NS_STEPS").ok().and_then(|s| s.parse().ok()).unwrap_or(20);
    let tol: f64 = std::env::var("NS_TOL").ok().and_then(|s| s.parse().ok()).unwrap_or(1e-6);
    let stride: usize = std::env::var("NS_REFINE").ok().and_then(|s| s.parse().ok()).unwrap_or(4);
    let (alpha, nu, dt, maxit) = (5.0, 0.05, 5e-3, 5000usize);
    let lambda = 1.0 / (nu * dt);

    let refine: Vec<(usize, usize)> = if stride == 0 {
        vec![]
    } else {
        (0..nx).flat_map(|cy| (0..nx).map(move |cx| (cx, cy))).filter(|(cx, cy)| (cx + cy) % stride == 0).collect()
    };
    let mesh = Mesh2d::cartesian_refined(P, nx, nx, [0.0, 1.0], [0.0, 1.0], &refine);
    let ne = mesh.n_elements();
    let nn = mesh.refq.n_nodes();
    let ndof = ne * nn;
    let tags = mesh.boundary_tags();
    println!("=== device-resident NC NS step perf — base {nx}×{nx} p={P}, {} refined ⇒ {ne} elems ({ndof} dof), tol {tol:.0e} ===", refine.len());

    let (jw, lift_vx, lift_vy, lift_p) = consts(&mesh, alpha, lambda);
    let h = GpuPoissonNc::new(&mesh, alpha)?;
    // p-multigrid preconditioners: viscous (Helmholtz, Dirichlet) and pressure (singular Neumann).
    let mg_v = GpuPMultigridNc::new(P, nx, nx, [0.0, 1.0], [0.0, 1.0], &refine, alpha, lambda, vec![], false)?;
    let mg_p = GpuPMultigridNc::new(P, nx, nx, [0.0, 1.0], [0.0, 1.0], &refine, alpha, 0.0, tags.clone(), true)?;

    // ---- Device-resident step (no per-step transfer); shared scratch reused for CG & MG paths. ----
    let jw_d = h.upload(&jw)?;
    let (lvx, lvy, lp) = (h.upload(&lift_vx)?, h.upload(&lift_vy)?, h.upload(&lift_p)?);
    let (mut dux, mut duy) = (h.alloc()?, h.alloc()?);
    let (mut uhx, mut uhy) = (h.alloc()?, h.alloc()?);
    let (mut ga, mut gb, mut gc, mut gd) = (h.alloc()?, h.alloc()?, h.alloc()?, h.alloc()?);
    let (mut cx, mut cy, mut div, mut pp, mut rhs) = (h.alloc()?, h.alloc()?, h.alloc()?, h.alloc()?, h.alloc()?);

    // The assembly is shared; only the three elliptic solves differ (plain CG vs p-MG-PCG). `mg` selects.
    macro_rules! ns_step {
        ($n:expr, $solve_p:expr, $solve_vx:expr, $solve_vy:expr) => {{
            for _ in 0..$n {
                h.gradient_dev(&dux, &mut ga, &mut gb)?;
                h.gradient_dev(&duy, &mut gc, &mut gd)?;
                h.fma2_dev(&mut cx, &dux, &ga, &duy, &gb)?;
                h.fma2_dev(&mut cy, &dux, &gc, &duy, &gd)?;
                h.copy_dev(&mut uhx, &dux)?;
                h.copy_dev(&mut uhy, &duy)?;
                h.axpy_dev(&mut uhx, &cx, -dt)?;
                h.axpy_dev(&mut uhy, &cy, -dt)?;
                h.gradient_dev(&uhx, &mut ga, &mut gb)?;
                h.gradient_dev(&uhy, &mut gc, &mut gd)?;
                h.copy_dev(&mut div, &ga)?;
                h.axpy_dev(&mut div, &gd, 1.0)?;
                h.scal_dev(&mut div, -1.0 / dt)?;
                h.rhs_madd_dev(&mut rhs, &jw_d, &div, &lp, 1.0)?;
                $solve_p(&rhs, &mut pp)?;
                h.gradient_dev(&pp, &mut ga, &mut gb)?;
                h.axpy_dev(&mut uhx, &ga, -dt)?;
                h.axpy_dev(&mut uhy, &gb, -dt)?;
                h.rhs_madd_dev(&mut rhs, &jw_d, &uhx, &lvx, lambda)?;
                $solve_vx(&rhs, &mut dux)?;
                h.rhs_madd_dev(&mut rhs, &jw_d, &uhy, &lvy, lambda)?;
                $solve_vy(&rhs, &mut duy)?;
            }
        }};
    }

    // NS_ONLY_MG=1 ⇒ run only the p-MG-PCG path (clean ncu capture of the V-cycle kernels).
    let only_mg = std::env::var("NS_ONLY_MG").is_ok();

    // --- plain-CG device-resident path ---
    let mut dev_ms = f64::NAN;
    if !only_mg {
        let cg_p = |rhs: &_, out: &mut _| h.solve_dev(rhs, None, out, 0.0, &tags, true, tol, maxit).map(|_| ());
        let cg_v = |rhs: &_, out: &mut _| h.solve_dev(rhs, None, out, lambda, &[], false, tol, maxit).map(|_| ());
        ns_step!(2, cg_p, cg_v, cg_v); // warm-up + JIT
        let t = Instant::now();
        ns_step!(steps, cg_p, cg_v, cg_v);
        let _ = (h.download(&dux)?, h.download(&duy)?); // single sync at the end
        dev_ms = t.elapsed().as_secs_f64() * 1e3 / steps as f64;
    }

    // --- p-MG-PCG device-resident path (reset fields first) ---
    h.scal_dev(&mut dux, 0.0)?;
    h.scal_dev(&mut duy, 0.0)?;
    let mg_solve_p = |rhs: &_, out: &mut _| mg_p.solve_dev(rhs, out, tol, maxit).map(|_| ());
    let mg_solve_v = |rhs: &_, out: &mut _| mg_v.solve_dev(rhs, out, tol, maxit).map(|_| ());
    ns_step!(2, mg_solve_p, mg_solve_v, mg_solve_v); // warm-up
    let t = Instant::now();
    ns_step!(steps, mg_solve_p, mg_solve_v, mg_solve_v);
    let _ = (h.download(&dux)?, h.download(&duy)?);
    let dev_mg_ms = t.elapsed().as_secs_f64() * 1e3 / steps as f64;

    // ---- Host-orchestrated step (per-step transfers) ----
    let mut host_ms = f64::NAN;
    if !only_mg {
    let (mut hux, mut huy) = (vec![0.0; ndof], vec![0.0; ndof]);
    let mut host_step = |hux: &mut Vec<f64>, huy: &mut Vec<f64>, n: usize| -> Result<(), Box<dyn std::error::Error>> {
        for _ in 0..n {
            let (gxx, gyx) = (grad(&mesh, hux, 0), grad(&mesh, hux, 1));
            let (gxy, gyy) = (grad(&mesh, huy, 0), grad(&mesh, huy, 1));
            let (mut bhx, mut bhy) = (vec![0.0; ndof], vec![0.0; ndof]);
            for i in 0..ndof {
                bhx[i] = hux[i] - dt * (hux[i] * gxx[i] + huy[i] * gyx[i]);
                bhy[i] = huy[i] - dt * (hux[i] * gxy[i] + huy[i] * gyy[i]);
            }
            let dv: Vec<f64> = grad(&mesh, &bhx, 0).iter().zip(grad(&mesh, &bhy, 1).iter()).map(|(a, b)| a + b).collect();
            let bp: Vec<f64> = (0..ndof).map(|i| jw[i] * (-dv[i] / dt) + lift_p[i]).collect();
            let pp = h.solve(&bp, 0.0, &tags, true, tol, maxit)?.0;
            let (px, py) = (grad(&mesh, &pp, 0), grad(&mesh, &pp, 1));
            for i in 0..ndof {
                bhx[i] -= dt * px[i];
                bhy[i] -= dt * py[i];
            }
            let bx: Vec<f64> = (0..ndof).map(|i| lambda * jw[i] * bhx[i] + lift_vx[i]).collect();
            let by: Vec<f64> = (0..ndof).map(|i| lambda * jw[i] * bhy[i] + lift_vy[i]).collect();
            *hux = h.solve(&bx, lambda, &[], false, tol, maxit)?.0;
            *huy = h.solve(&by, lambda, &[], false, tol, maxit)?.0;
        }
        Ok(())
    };
    host_step(&mut hux, &mut huy, 2)?; // warm-up
    let t = Instant::now();
    host_step(&mut hux, &mut huy, steps)?;
    host_ms = t.elapsed().as_secs_f64() * 1e3 / steps as f64;
    }

    println!("  host-orchestrated step (plain CG):     {host_ms:.2} ms/step");
    println!("  device-resident step (plain CG):       {dev_ms:.2} ms/step");
    println!("  device-resident step (p-MG-PCG):       {dev_mg_ms:.2} ms/step");
    println!("  p-MG-PCG speedup: {:.1}× vs device CG, {:.1}× vs host CG", dev_ms / dev_mg_ms, host_ms / dev_mg_ms);
    Ok(())
}
