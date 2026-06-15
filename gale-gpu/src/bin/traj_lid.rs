//! Trajectory dump of a lid-driven cavity (device-resident dual-splitting) → one HDF5 file.
//! Closed unit box, top lid moving at u=1 (singular all-Neumann pressure, deflated). Velocity is
//! downloaded every `TRAJ_EVERY` steps and written as a frame; the recirculating vortex develops
//! over the run.
//!
//! Run: cargo oxide run --features traj --bin traj-lid
//! View: python gale-traj/python/view_traj.py /tmp/lid.h5 --field u --comp mag --gif

use gale::dg::{Mesh2d, PMultigrid, Poisson};
use gale_gpu::operators::poisson::GpuPoissonMg;
use gale_traj::TrajectoryWriter;

fn env_usize(k: &str, d: usize) -> usize {
    std::env::var(k).ok().and_then(|v| v.parse().ok()).unwrap_or(d)
}
fn env_f64(k: &str, d: f64) -> f64 {
    std::env::var(k).ok().and_then(|v| v.parse().ok()).unwrap_or(d)
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let (p, alpha) = (4usize, 5.0);
    let n = env_usize("TRAJ_N", 32); // square mesh n×n (TRAJ_N to override)
    let (nx, ny) = (n, n);
    let nu = 0.05;
    let dt = env_f64("TRAJ_DT", 2e-3);
    let lambda = 1.0 / (nu * dt);
    let (tol, maxit) = (1e-7, 2000);
    let steps = env_usize("TRAJ_STEPS", 1000);
    let every = env_usize("TRAJ_EVERY", 12);
    let out = std::env::args().nth(1).unwrap_or_else(|| "/tmp/lid.h5".to_string());
    let xr = [0.0, 1.0];
    let yr = [0.0, 1.0];

    let mesh = Mesh2d::rectangular(p, nx, ny, xr, yr);
    let nn = mesh.refq.n_nodes();
    let ne = mesh.n_elements();
    let ndof = ne * nn;
    let tags = mesh.boundary_tags();
    println!("=== traj lid cavity {nx}×{ny} p={p}, dt={dt}, {steps} steps, dump every {every} ===");

    let pres_op = Poisson::with_bc(&mesh, alpha, 0.0, tags.clone());
    let vel_op = Poisson::with_reaction(&mesh, alpha, lambda);
    let g_u = |_x: f64, y: f64| if y > 1.0 - 1e-9 { 1.0 } else { 0.0 }; // moving top lid
    let g_v = |_x: f64, _y: f64| 0.0;

    let hp = GpuPoissonMg::new(PMultigrid::with_bc(p, nx, ny, xr, yr, alpha, 0.0, tags.clone()))?;
    let hv = GpuPoissonMg::new(PMultigrid::with_reaction(p, nx, ny, xr, yr, alpha, lambda))?;

    let jw = vel_op.rhs(&vec![1.0; ndof], |_, _| 0.0);
    let lift_vx = vel_op.rhs(&vec![0.0; ndof], g_u);
    let lift_vy = vel_op.rhs(&vec![0.0; ndof], g_v);

    let jw_d = hv.upload_field(&jw)?;
    let lift_vx_d = hv.upload_field(&lift_vx)?;
    let lift_vy_d = hv.upload_field(&lift_vy)?;
    let lift_p_d = hv.upload_field(&vec![0.0; ndof])?;
    let mut ux = hv.alloc_field()?;
    let mut uy = hv.alloc_field()?;
    let (mut ux_n, mut uy_n, mut pp, mut pp_n) = (hv.alloc_field()?, hv.alloc_field()?, hv.alloc_field()?, hv.alloc_field()?);
    let (mut uhx, mut uhy) = (hv.alloc_field()?, hv.alloc_field()?);
    let (mut ga, mut gb, mut gc, mut gd) = (hv.alloc_field()?, hv.alloc_field()?, hv.alloc_field()?, hv.alloc_field()?);
    let (mut cx, mut cy, mut div, mut rhs) = (hv.alloc_field()?, hv.alloc_field()?, hv.alloc_field()?, hv.alloc_field()?);

    let mut tw = TrajectoryWriter::create(&out, p, 2)?;
    let topo = tw.write_mesh2d(&mesh)?;
    let pack = |hux: &[f64], huy: &[f64]| -> Vec<f32> {
        let mut v = vec![0f32; ndof * 2];
        for g in 0..ndof {
            v[g * 2] = hux[g] as f32;
            v[g * 2 + 1] = huy[g] as f32;
        }
        v
    };
    tw.write_frame(0.0, 0, topo, ne, nn, &[("u", pack(&vec![0.0; ndof], &vec![0.0; ndof]), 2)], None)?;

    for step in 1..=steps {
        hv.gradient_dev(&ux, &mut ga, &mut gb)?;
        hv.gradient_dev(&uy, &mut gc, &mut gd)?;
        hv.fma2_dev(&mut cx, &ux, &ga, &uy, &gb)?;
        hv.fma2_dev(&mut cy, &ux, &gc, &uy, &gd)?;
        hv.copy_dev(&mut uhx, &ux)?;
        hv.copy_dev(&mut uhy, &uy)?;
        hv.axpy_dev(&mut uhx, &cx, -dt)?;
        hv.axpy_dev(&mut uhy, &cy, -dt)?;
        hv.gradient_dev(&uhx, &mut ga, &mut gb)?;
        hv.gradient_dev(&uhy, &mut gc, &mut gd)?;
        hv.copy_dev(&mut div, &ga)?;
        hv.axpy_dev(&mut div, &gd, 1.0)?;
        hv.scal_dev(&mut div, -1.0 / dt)?;
        hv.rhs_madd_dev(&mut rhs, &jw_d, &div, &lift_p_d, 1.0)?;
        hp.solve_dev(&rhs, Some(&pp), &mut pp_n, tol, maxit)?;
        std::mem::swap(&mut pp, &mut pp_n);
        hv.gradient_dev(&pp, &mut ga, &mut gb)?;
        hv.axpy_dev(&mut uhx, &ga, -dt)?;
        hv.axpy_dev(&mut uhy, &gb, -dt)?;
        hv.rhs_madd_dev(&mut rhs, &jw_d, &uhx, &lift_vx_d, lambda)?;
        hv.solve_dev(&rhs, Some(&ux), &mut ux_n, tol, maxit)?;
        std::mem::swap(&mut ux, &mut ux_n);
        hv.rhs_madd_dev(&mut rhs, &jw_d, &uhy, &lift_vy_d, lambda)?;
        hv.solve_dev(&rhs, Some(&uy), &mut uy_n, tol, maxit)?;
        std::mem::swap(&mut uy, &mut uy_n);

        if step % every == 0 {
            let (hux, huy) = (hv.download_field(&ux)?, hv.download_field(&uy)?);
            tw.write_frame(step as f64 * dt, step as u64, topo, ne, nn, &[("u", pack(&hux, &huy), 2)], None)?;
        }
    }
    println!("wrote {} frames → {out}", tw.n_frames());
    Ok(())
}
