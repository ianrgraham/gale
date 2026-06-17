//! Validates the device-resident `GpuResidentVe` viscoelastic integrator against the host-
//! orchestrated framework integrator `GpuViscoelasticDualSplitting` (LogConf) on a Kolmogorov VE
//! flow, for BOTH κ = 0 (pure stress + SSP-RK3 transport) and κ > 0 (adds the implicit polymer
//! stress-diffusion solve). Both use the same GPU MG-PCG solver and constitutive math; the resident
//! path keeps velocity + Ψ on the device across the whole step (incl. the diffusion solve). They
//! must agree to solver tolerance. No trace cap (short run stays finite ⇒ isolates the math).
//! Run: cargo oxide run --bin resident-ve-check

use gale::dg::Mesh2d;
use gale::sim::{Simulation, State, ViscoModel};
use gale_gpu::resident::GpuResidentVe;
use gale_gpu::GpuViscoelasticDualSplitting;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let (p, alpha) = (3usize, 5.0);
    let (nx, ny) = (12usize, 12usize);
    let (nu0, beta) = (0.08, 0.5);
    let (eta_s, eta_p) = (beta * nu0, (1.0 - beta) * nu0);
    let lambda_p = 4.0;
    let dt = 2e-3;
    let steps = 30usize;
    let (tol, maxit) = (1e-10, 4000);
    let xr = [0.0, 1.0];
    let yr = [0.0, 1.0];
    let two_pi = std::f64::consts::TAU;
    let (force, nk) = (4.0, 2.0);
    println!("=== GpuResidentVe vs host GpuViscoelasticDualSplitting — Kolmogorov VE {nx}×{ny} p={p}, {steps} steps ===");

    let u_est = force / (nu0 * (two_pi * nk).powi(2));
    let u_ic = move |_x: f64, y: f64| u_est * (two_pi * nk * y).sin();
    let v_ic = move |x: f64, y: f64| 0.02 * u_est * (two_pi * x).sin() * (two_pi * y).sin();
    let rel = |a: &[f64], b: &[f64]| -> f64 {
        let d: f64 = a.iter().zip(b).map(|(x, y)| (x - y).powi(2)).sum();
        let n: f64 = b.iter().map(|x| x * x).sum::<f64>().max(1e-300);
        (d / n).sqrt()
    };

    let mut ok = true;
    for &kappa in &[0.0_f64, 0.002] {
        // ---- Host framework integrator ----
        let mut st = State::new(Mesh2d::rectangular(p, nx, ny, xr, yr));
        let vid = st.add_field_from(
            "velocity",
            &[Box::new(u_ic) as Box<dyn Fn(f64, f64) -> f64>, Box::new(v_ic)],
        );
        let cid = st.add_field_from(
            "conformation",
            &[
                Box::new(|_: f64, _: f64| 0.0) as Box<dyn Fn(f64, f64) -> f64>,
                Box::new(|_: f64, _: f64| 0.0),
                Box::new(|_: f64, _: f64| 0.0),
            ],
        );
        let fx = move |_x: f64, y: f64, _t: f64| force * (two_pi * nk * y).sin();
        let fy = |_x: f64, _y: f64, _t: f64| 0.0;
        let zero = |_: f64, _: f64, _: f64| 0.0;
        let mut integ = GpuViscoelasticDualSplitting::new(vid, cid, dt, eta_s, eta_p, lambda_p, alpha, ViscoModel::LogConf)
            .boundary(zero, zero)
            .drive(fx, fy);
        if kappa > 0.0 {
            integ = integ.with_stress_diffusion_implicit(kappa);
        }
        let mut sim = Simulation::new(st);
        sim.set_integrator(integ);
        sim.run(steps as u64);
        let hv = sim.state.field("velocity");
        let (hux, huy) = (hv.component(0).to_vec(), hv.component(1).to_vec());
        let hc = sim.state.field("conformation");
        let hpsi = [hc.component(0).to_vec(), hc.component(1).to_vec(), hc.component(2).to_vec()];

        // ---- Device-resident integrator ----
        let mesh = Mesh2d::rectangular(p, nx, ny, xr, yr);
        let nn = mesh.refq.n_nodes();
        let ndof = mesh.n_elements() * nn;
        let (mut ux0, mut uy0) = (vec![0.0; ndof], vec![0.0; ndof]);
        for (e, el) in mesh.elements.iter().enumerate() {
            for k in 0..nn {
                ux0[e * nn + k] = u_ic(el.geom.x[k], el.geom.y[k]);
                uy0[e * nn + k] = v_ic(el.geom.x[k], el.geom.y[k]);
            }
        }
        let fx2 = move |_x: f64, y: f64| force * (two_pi * nk * y).sin();
        let fy2 = |_x: f64, _y: f64| 0.0;
        let mut ve = GpuResidentVe::new(&mesh, eta_s, eta_p, lambda_p, dt, alpha, kappa, fx2, fy2)?.with_tol(tol, maxit);
        ve.set_velocity(&ux0, &uy0)?;
        ve.run(steps)?;
        let (dux, duy) = ve.velocity()?;
        let dpsi = ve.psi()?;

        let ru = rel(&dux, &hux).max(rel(&duy, &huy));
        let rp = (0..3).map(|i| rel(&dpsi[i], &hpsi[i])).fold(0.0f64, f64::max);
        let pmax = hpsi[0].iter().chain(&hpsi[2]).fold(0.0f64, |m, &x| m.max(x.abs()));
        let pass = ru < 1e-6 && rp < 1e-6;
        ok &= pass;
        println!("  κ={kappa:<6} |Ψ|max={pmax:.3}  rel velocity={ru:.3e}  rel Ψ={rp:.3e}  {}", if pass { "ok" } else { "FAIL" });
    }

    if ok {
        println!("OK: GpuResidentVe matches the host viscoelastic trajectory (fully device-resident, with & without κ-diffusion).");
        Ok(())
    } else {
        eprintln!("FAIL: GpuResidentVe diverges from host VE.");
        std::process::exit(1);
    }
}
