//! Trajectory dump of a VISCOELASTIC (Oldroyd-B) SBM cylinder (Schäfer-Turek geometry, Re≈20) →
//! one HDF5 file. Same device-resident dual-splitting flow step as `traj_cylinder`, but the
//! momentum balance is now driven by the polymer stress divergence `∇·τ_p` of a conformation
//! tensor `C` that is transported alongside the flow:
//!
//!   ∂u/∂t + (u·∇)u = −∇p + η_s∇²u + ∇·τ_p,     τ_p = (η_p/λ)(C − I)
//!   ∂C/∂t + (u·∇)C = L·C + C·Lᵀ − (1/λ)(C − I),  L = ∇u
//!
//! The flow (advection + pressure projection + viscous Helmholtz) stays DEVICE-RESIDENT on the
//! SBM-aware `GpuPoissonMg`; the constitutive part is host-orchestrated each step (download C →
//! `OldroydB::stress_divergence` → upload `∇·τ_p` as the momentum body force; after the velocity
//! update, download u → bound-preserving SSP-RK3 transport of C → keep on host). The polymer
//! force is masked to the active (fluid) elements and `C` is reset to `I` inside the cylinder, so
//! no spurious solid-region stress leaks into the SBM solve. Both `u` and the polymer-stretch
//! diagnostic `tr C` are written, each reconstructed up to the true boundary via `sbm_reconstruct`.
//!
//! The companion Newtonian run is `traj-cylinder` (same geometry/Re) — compare side by side to see
//! the viscoelastic signature: a birefringent strand of stretched polymer (high `tr C`) trailing
//! the cylinder and a fore-aft asymmetric wake.
//!
//! Run:  cargo oxide run --features traj --bin traj-ve-cylinder
//! View: python gale-traj/python/view_traj.py /tmp/ve_cylinder.h5 --field trC --comp 0 --gif
//!   (or --field u --comp mag for the velocity magnitude)

use gale::dg::{
    sbm_reconstruct, CircleLevelSet, ConformationInflow, Mesh2d, OldroydB, ShiftedBoundary,
    ShiftedMultigrid, ShiftedPoisson,
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
    let (h, um) = (0.41, 0.3);
    let ny = env_usize("SBM_NY", 32);
    let nx = ((2.2 / h) * ny as f64).round() as usize;
    let dt = env_f64("SBM_DT", 1e-3); // smaller than the Newtonian dump for the explicit C transport
    let steps = env_usize("TRAJ_STEPS", 2000);
    let every = env_usize("TRAJ_EVERY", 20);
    let out = std::env::args().nth(1).unwrap_or_else(|| "/tmp/ve_cylinder.h5".to_string());

    // Oldroyd-B: total zero-shear kinematic viscosity nu0 (fixes Re≈20 as in the Newtonian case),
    // split into solvent η_s = β·nu0 and polymer η_p = (1−β)·nu0; λ is the polymer relaxation time.
    let nu0 = env_f64("VE_NU0", 0.001);
    let beta = env_f64("VE_BETA", 0.5);
    let relax = env_f64("VE_LAMBDA", 0.15); // De = λ·U_mean/R = λ·(2/3·um)/r  (≈0.6 at default)
    let eta_s = beta * nu0;
    let eta_p = (1.0 - beta) * nu0;
    // Bound-preserving (Zhang–Shu) limiter knobs keep C SPD (det≥eps) and trace≤b_max through the
    // high-shear stress concentration at the cylinder — the HWNP guard for the direct form.
    let (lim_eps, lim_bmax) = (env_f64("VE_EPS", 1e-8), env_f64("VE_BMAX", 1e4));

    let alpha = 5.0;
    let lambda = 1.0 / (eta_s * dt); // Helmholtz shift for the SOLVENT-viscosity velocity solve
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
    let de = relax * (2.0 / 3.0 * um) / r;
    println!(
        "=== traj VE SBM cylinder {nx}×{ny} p={p}, dt={dt}, {steps} steps, dump every {every} ===\n\
         Oldroyd-B: β={beta} (η_s={eta_s:.2e}, η_p={eta_p:.2e}), λ={relax}, De≈{de:.2}, Re≈20"
    );

    // Velocity solve: Neumann at outflow (tag 1), Dirichlet (parabola / no-slip) elsewhere + SBM.
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

    // Oldroyd-B constitutive transport: relaxed fluid (C=I) enters at the inlet (left, tag 3).
    let ob = OldroydB::new(&mesh, relax, eta_p)
        .with_inflow(ConformationInflow::equilibrium(vec![3]));
    let mut c = ob.identity(); // [Cxx, Cxy, Cyy] = I everywhere
    // Per-DoF fluid mask (1.0 in active SBM elements, 0.0 inside the solid) for the body force.
    let active: Vec<f64> =
        (0..ndof).map(|g| if sb.active[g / nn] { 1.0 } else { 0.0 }).collect();
    let reset_solid = |c: &mut [Vec<f64>; 3]| {
        for e in 0..ne {
            if !sb.active[e] {
                for k in 0..nn {
                    let g = e * nn + k;
                    c[0][g] = 1.0;
                    c[1][g] = 0.0;
                    c[2][g] = 1.0;
                }
            }
        }
    };

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

    let mut tw = match std::env::var("TRAJ_MAX_MB").ok().and_then(|v| v.parse::<u64>().ok()) {
        Some(mb) => TrajectoryWriter::create_split(out.strip_suffix(".h5").unwrap_or(&out), p, 2, mb)?,
        None => TrajectoryWriter::create(&out, p, 2)?,
    };
    let topo = tw.write_mesh2d(&mesh)?;
    tw.write_body_radii(&[r])?; // true radius — viewer masks the smooth circle in-shader
    let pose = [[cx, cy, 0.0]];

    // Pack velocity (2-comp) + tr C (1-comp), each reconstructed up to the true boundary so the
    // near-wall layer and the polymer-stretch strand render smoothly to the cylinder surface.
    let pack_u = |hux: &[f64], huy: &[f64]| -> Vec<f32> {
        let rux = sbm_reconstruct(&mesh, &sb, &ls, hux);
        let ruy = sbm_reconstruct(&mesh, &sb, &ls, huy);
        (0..ndof).flat_map(|g| [rux[g] as f32, ruy[g] as f32]).collect()
    };
    let pack_trc = |c: &[Vec<f64>; 3]| -> Vec<f32> {
        let trc: Vec<f64> = (0..ndof).map(|g| c[0][g] + c[2][g]).collect();
        let r = sbm_reconstruct(&mesh, &sb, &ls, &trc);
        r.iter().map(|&v| v as f32).collect()
    };
    let dump = |tw: &mut TrajectoryWriter,
                t: f64,
                step: u64,
                hux: &[f64],
                huy: &[f64],
                c: &[Vec<f64>; 3]|
     -> Result<(), Box<dyn std::error::Error>> {
        tw.write_frame(
            t,
            step,
            topo,
            ne,
            nn,
            &[("u", pack_u(hux, huy), 2), ("trC", pack_trc(c), 1)],
            Some(&pose),
        )?;
        Ok(())
    };

    dump(&mut tw, 0.0, 0, &hv.download_field(&ux)?, &hv.download_field(&uy)?, &c)?;

    for step in 1..=steps {
        // Polymer body force from the CURRENT conformation: ∇·τ_p, masked to the fluid.
        let (mut tx, mut ty) = ob.stress_divergence(&c);
        for g in 0..ndof {
            tx[g] *= active[g];
            ty[g] *= active[g];
        }
        let tx_d = hv.upload_field(&tx)?;
        let ty_d = hv.upload_field(&ty)?;

        // --- device-resident dual-splitting momentum step (with the polymer body force) ---
        hv.gradient_dev(&ux, &mut ga, &mut gb)?;
        hv.gradient_dev(&uy, &mut gc, &mut gd)?;
        hv.fma2_dev(&mut cxb, &ux, &ga, &uy, &gb)?;
        hv.fma2_dev(&mut cyb, &ux, &gc, &uy, &gd)?;
        hv.copy_dev(&mut uhx, &ux)?;
        hv.copy_dev(&mut uhy, &uy)?;
        hv.axpy_dev(&mut uhx, &cxb, -dt)?; // û = u − dt (u·∇)u
        hv.axpy_dev(&mut uhy, &cyb, -dt)?;
        hv.axpy_dev(&mut uhx, &tx_d, dt)?; //      + dt ∇·τ_p
        hv.axpy_dev(&mut uhy, &ty_d, dt)?;
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

        // --- constitutive transport: advance C with the NEW velocity (bound-preserving) ---
        let (hux, huy) = (hv.download_field(&ux)?, hv.download_field(&uy)?);
        c = ob.step_ssp_rk3_bounded(&c, &hux, &huy, dt, lim_eps, lim_bmax);
        reset_solid(&mut c); // keep the solid region at equilibrium (no spurious stress)

        if step % every == 0 {
            dump(&mut tw, step as f64 * dt, step as u64, &hux, &huy, &c)?;
            let max_trc = (0..ndof)
                .filter(|&g| active[g] > 0.5)
                .map(|g| c[0][g] + c[2][g])
                .fold(0.0_f64, f64::max);
            println!("step {step:>5}/{steps}  max tr C = {max_trc:.3}");
        }
    }
    println!("wrote {} frames → {out}", tw.n_frames());
    Ok(())
}
