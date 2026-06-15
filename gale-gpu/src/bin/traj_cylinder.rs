//! Trajectory dump of the device-resident SBM cylinder (Schäfer-Turek Re=20) → one HDF5 file.
//! Same device-resident dual-splitting step as `sbm_cylinder_resident_check`, but instead of just
//! the C_D diagnostic it downloads the velocity every `TRAJ_EVERY` steps and writes a frame. Inside
//! the cylinder (inactive SBM elements) the field is set to NaN so the viewer renders it as a hole.
//!
//! Run: cargo oxide run --features traj --bin traj-cylinder
//! View: python gale-traj/python/view_traj.py /tmp/cylinder.h5 --field u --comp mag --gif

use gale::dg::{
    sbm_reconstruct, CircleLevelSet, Mesh2d, ShiftedBoundary, ShiftedMultigrid, ShiftedPoisson,
};
use gale_gpu::operators::poisson::GpuPoissonMg;
use gale_traj::TrajectoryWriter;

fn env_usize(k: &str, d: usize) -> usize {
    std::env::var(k).ok().and_then(|v| v.parse().ok()).unwrap_or(d)
}
fn env_f64(k: &str, d: f64) -> f64 {
    std::env::var(k).ok().and_then(|v| v.parse().ok()).unwrap_or(d)
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let p = 3;
    let (h, um, nu) = (0.41, 0.3, 0.001);
    let ny = env_usize("SBM_NY", 32); // finer than the validation default (16) for a smoother cylinder
    let nx = ((2.2 / h) * ny as f64).round() as usize;
    let dt = env_f64("SBM_DT", 2e-3);
    let steps = env_usize("TRAJ_STEPS", 1000);
    let every = env_usize("TRAJ_EVERY", 10);
    let out = std::env::args().nth(1).unwrap_or_else(|| "/tmp/cylinder.h5".to_string());
    let alpha = 5.0;
    let lambda = 1.0 / (nu * dt);
    let (cx, cy, r) = (0.2, 0.2, 0.05);
    let (ptol, vtol, maxit) = (env_f64("SBM_PTOL", 1e-4), env_f64("SBM_VTOL", 1e-7), 5000);
    let xr = [0.0, 2.2];
    let yr = [0.0, h];

    let mesh = Mesh2d::rectangular(p, nx, ny, xr, yr);
    let nn = mesh.refq.n_nodes();
    let ne = mesh.n_elements();
    let ndof = ne * nn;
    let ls = CircleLevelSet::new(cx, cy, r);
    let sb = ShiftedBoundary::new(&mesh, &ls);
    let hx = 2.2 / nx as f64;
    println!("=== traj SBM cylinder {nx}×{ny} p={p}, dt={dt}, {steps} steps, dump every {every} ===");

    let vel = ShiftedPoisson::with_bc(&mesh, alpha, lambda, vec![1], sb.clone()).taylor(true);
    let pres = ShiftedPoisson::with_bc(&mesh, alpha, 0.0, vec![3, 0, 2], sb.clone()).surrogate_neumann();
    let parab = |y: f64| 4.0 * um * y * (h - y) / (h * h);
    let g_u = |x: f64, y: f64| if x < 0.5 * hx { parab(y) } else { 0.0 };
    let g_v = |_x: f64, _y: f64| 0.0;
    let g_p = |_x: f64, _y: f64| 0.0;

    let hp = GpuPoissonMg::new_sbm(ShiftedMultigrid::new(
        p, nx, ny, xr, yr, alpha, 0.0, vec![3, 0, 2], &ls, false, false,
    ))?;
    let hv = GpuPoissonMg::new_sbm(ShiftedMultigrid::new(
        p, nx, ny, xr, yr, alpha, lambda, vec![1], &ls, true, true,
    ))?;

    let jw = vel.rhs(&vec![1.0; ndof], |_, _| 0.0);
    let lift_p = pres.rhs(&vec![0.0; ndof], g_p);
    let lift_vx = vel.rhs(&vec![0.0; ndof], g_u);
    let lift_vy = vel.rhs(&vec![0.0; ndof], g_v);

    let mut ux0 = vec![0.0; ndof];
    for (e, el) in mesh.elements.iter().enumerate() {
        if sb.active[e] {
            for k in 0..nn {
                ux0[e * nn + k] = parab(el.geom.y[k]);
            }
        }
    }

    let jw_d = hv.upload_field(&jw)?;
    let lift_p_d = hv.upload_field(&lift_p)?;
    let lift_vx_d = hv.upload_field(&lift_vx)?;
    let lift_vy_d = hv.upload_field(&lift_vy)?;
    let mut ux = hv.upload_field(&ux0)?;
    let mut uy = hv.alloc_field()?;
    let mut pp = hv.alloc_field()?;
    let (mut ux_n, mut uy_n, mut pp_n) = (hv.alloc_field()?, hv.alloc_field()?, hv.alloc_field()?);
    let (mut uhx, mut uhy) = (hv.alloc_field()?, hv.alloc_field()?);
    let (mut ga, mut gb, mut gc, mut gd) = (hv.alloc_field()?, hv.alloc_field()?, hv.alloc_field()?, hv.alloc_field()?);
    let (mut cxb, mut cyb, mut div, mut rhs) = (hv.alloc_field()?, hv.alloc_field()?, hv.alloc_field()?, hv.alloc_field()?);

    // TRAJ_MAX_MB caps each file and rolls to <stem>.NNNN.h5; otherwise one file at `out`.
    let mut tw = match std::env::var("TRAJ_MAX_MB").ok().and_then(|v| v.parse::<u64>().ok()) {
        Some(mb) => TrajectoryWriter::create_split(out.strip_suffix(".h5").unwrap_or(&out), p, 2, mb)?,
        None => TrajectoryWriter::create(&out, p, 2)?,
    };
    let topo = tw.write_mesh2d(&mesh)?;
    // The cylinder is a fixed embedded boundary — record it at its TRUE radius so the viewer masks a
    // smooth circle of the correct size. (The near-wall velocity → 0 at the no-slip boundary is real
    // physics, not an artifact, so it is shown rather than hidden behind an inflated disk.)
    tw.write_body_radii(&[r])?;
    let pose = [[cx, cy, 0.0]];
    // Pack (ux,uy) → [ne,nn,2] f32. ACTIVE (fluid) elements keep their real SBM values. For the
    // INACTIVE elements that straddle the boundary, `sbm_reconstruct` fills the FLUID side of the gap
    // (between the staircased surrogate edge and the true circle) with the field the SBM itself
    // implies there: a 2nd-order Taylor extrapolation (value + ∇ + Hessian) from the nearest
    // surrogate node — the same high-order reconstruction `sbm_force_torque` uses for drag. The SOLID
    // side (φ<0) stays NaN ⇒ blank. The viewer then cuts a clean circle of the TRUE radius in the
    // shader. Net effect: the boundary layer renders smoothly right up to the real cylinder surface,
    // with no staircase sliver and no fudged geometry — the gap shows the SBM's own implied field.
    let pack = |hux: &[f64], huy: &[f64]| -> Vec<f32> {
        let rux = sbm_reconstruct(&mesh, &sb, &ls, hux);
        let ruy = sbm_reconstruct(&mesh, &sb, &ls, huy);
        let mut v = vec![0f32; ndof * 2];
        for g in 0..ndof {
            v[g * 2] = rux[g] as f32;
            v[g * 2 + 1] = ruy[g] as f32;
        }
        v
    };
    tw.write_frame(0.0, 0, topo, ne, nn, &[("u", pack(&hv.download_field(&ux)?, &hv.download_field(&uy)?), 2)], Some(&pose))?;

    for step in 1..=steps {
        hv.gradient_dev(&ux, &mut ga, &mut gb)?;
        hv.gradient_dev(&uy, &mut gc, &mut gd)?;
        hv.fma2_dev(&mut cxb, &ux, &ga, &uy, &gb)?;
        hv.fma2_dev(&mut cyb, &ux, &gc, &uy, &gd)?;
        hv.copy_dev(&mut uhx, &ux)?;
        hv.copy_dev(&mut uhy, &uy)?;
        hv.axpy_dev(&mut uhx, &cxb, -dt)?;
        hv.axpy_dev(&mut uhy, &cyb, -dt)?;
        hv.gradient_dev(&uhx, &mut ga, &mut gb)?;
        hv.gradient_dev(&uhy, &mut gc, &mut gd)?;
        hv.copy_dev(&mut div, &ga)?;
        hv.axpy_dev(&mut div, &gd, 1.0)?;
        hv.scal_dev(&mut div, -1.0 / dt)?;
        hv.rhs_madd_dev(&mut rhs, &jw_d, &div, &lift_p_d, 1.0)?;
        hp.solve_dev(&rhs, Some(&pp), &mut pp_n, ptol, maxit)?;
        std::mem::swap(&mut pp, &mut pp_n);
        hv.gradient_dev(&pp, &mut ga, &mut gb)?;
        hv.axpy_dev(&mut uhx, &ga, -dt)?;
        hv.axpy_dev(&mut uhy, &gb, -dt)?;
        hv.rhs_madd_dev(&mut rhs, &jw_d, &uhx, &lift_vx_d, lambda)?;
        hv.solve_dev(&rhs, Some(&ux), &mut ux_n, vtol, maxit)?;
        std::mem::swap(&mut ux, &mut ux_n);
        hv.rhs_madd_dev(&mut rhs, &jw_d, &uhy, &lift_vy_d, lambda)?;
        hv.solve_dev(&rhs, Some(&uy), &mut uy_n, vtol, maxit)?;
        std::mem::swap(&mut uy, &mut uy_n);

        if step % every == 0 {
            let (hux, huy) = (hv.download_field(&ux)?, hv.download_field(&uy)?);
            tw.write_frame(step as f64 * dt, step as u64, topo, ne, nn, &[("u", pack(&hux, &huy), 2)], Some(&pose))?;
        }
    }
    println!("wrote {} frames → {out}", tw.n_frames());
    Ok(())
}
