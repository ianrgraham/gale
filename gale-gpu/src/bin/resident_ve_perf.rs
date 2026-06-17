//! Perf/util harness for the device-resident VE integrator (no host comparison): runs
//! `GpuResidentVe` for N steps so `nvidia-smi dmon`/`nsys` can measure sustained GPU utilization and
//! per-step host sync. Grounds the Stage-2 (self-driving graph) decision per the gale-gpu-perf skill.
//! Run: cargo oxide run --bin resident-ve-perf   (env: RVP_N res, RVP_STEPS)

use gale::dg::Mesh2d;
use gale_gpu::resident::GpuResidentVe;

fn env(k: &str, d: usize) -> usize {
    std::env::var(k).ok().and_then(|v| v.parse().ok()).unwrap_or(d)
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let p = 3usize;
    let n = env("RVP_N", 24);
    let steps = env("RVP_STEPS", 2000);
    let (nu0, beta) = (0.08, 0.5);
    let (eta_s, eta_p) = (beta * nu0, (1.0 - beta) * nu0);
    let (lambda_p, dt, alpha, kappa) = (4.0, 2e-3, 5.0, 0.002);
    let two_pi = std::f64::consts::TAU;
    let (force, nk) = (4.0, 2.0);
    let mesh = Mesh2d::rectangular(p, n, n, [0.0, 1.0], [0.0, 1.0]);
    let nn = mesh.refq.n_nodes();
    let ndof = mesh.n_elements() * nn;
    let u_est = force / (nu0 * (two_pi * nk).powi(2));
    let (mut ux0, mut uy0) = (vec![0.0; ndof], vec![0.0; ndof]);
    for (e, el) in mesh.elements.iter().enumerate() {
        for k in 0..nn {
            ux0[e * nn + k] = u_est * (two_pi * nk * el.geom.y[k]).sin();
            uy0[e * nn + k] = 0.02 * u_est * (two_pi * el.geom.x[k]).sin() * (two_pi * el.geom.y[k]).sin();
        }
    }
    let fx = move |_x: f64, y: f64| force * (two_pi * nk * y).sin();
    let fy = |_x: f64, _y: f64| 0.0;
    let mut ve = GpuResidentVe::new(&mesh, eta_s, eta_p, lambda_p, dt, alpha, kappa, fx, fy)?
        .with_trace_bound(2000.0)
        .with_tol(1e-9, 4000);
    ve.set_velocity(&ux0, &uy0)?;
    println!("resident-ve-perf: {n}×{n} p={p}, {steps} steps, κ={kappa} — stepping (profile now)");
    ve.run(steps)?;
    let psi = ve.psi()?;
    let lc = gale::dg::LogConfOldroydB::new(&mesh, lambda_p, eta_p);
    let cc = lc.conformation(&psi);
    let trmax = (0..ndof).fold(0.0f64, |m, g| m.max(cc[0][g] + cc[2][g]));
    println!("done. max tr C = {trmax:.3}");
    Ok(())
}
