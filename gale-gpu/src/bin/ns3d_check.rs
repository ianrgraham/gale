//! Validation harness for the GPU 3D Navier–Stokes path ([`gale_gpu::GpuStokes3d`] +
//! the [`gale_gpu::GpuDualSplitting3d`] framework integrator). Integrates an (x,z)-plane
//! Taylor–Green vortex embedded in 3D (uniform in y, v≡0 — an exact decaying NS
//! solution exercising the 3-component divergence, grad_z, and all three viscous
//! solves) and checks against the CPU `gale::dg::Stokes3d` and the analytic solution;
//! the framework run must reproduce the direct loop.
//!
//! Run: cargo oxide run --bin ns3d-check

use gale::dg::{Mesh3d, Stokes3d};
use gale::sim::{Simulation, State};
use std::f64::consts::PI;

fn nodal(mesh: &Mesh3d, f: impl Fn(f64, f64, f64) -> f64) -> Vec<f64> {
    let nn = mesh.refh.n_nodes();
    let mut v = vec![0.0; mesh.n_elements() * nn];
    for (e, el) in mesh.elements.iter().enumerate() {
        for k in 0..nn {
            v[e * nn + k] = f(el.geom.x[k], el.geom.y[k], el.geom.z[k]);
        }
    }
    v
}

fn l2(mesh: &Mesh3d, v: &[f64]) -> f64 {
    let nn = mesh.refh.n_nodes();
    let mut s = 0.0;
    for (e, el) in mesh.elements.iter().enumerate() {
        for k in 0..nn {
            s += el.geom.jw[k] * v[e * nn + k].powi(2);
        }
    }
    s.sqrt()
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let nu = 1.0;
    let p = 3;
    let alpha = 5.0;
    let mesh = Mesh3d::rectangular(p, 2, 2, 2, [0.0, 1.0], [0.0, 1.0], [0.0, 1.0]);
    let decay = move |t: f64| (-2.0 * PI * PI * nu * t).exp();
    let eu = move |x: f64, _y: f64, z: f64, t: f64| -(PI * x).cos() * (PI * z).sin() * decay(t);
    let ew = move |x: f64, _y: f64, z: f64, t: f64| (PI * x).sin() * (PI * z).cos() * decay(t);
    let ez = |_: f64, _: f64, _: f64, _: f64| 0.0;
    let zero = |_: f64, _: f64, _: f64, _: f64| 0.0;
    let t_end = 0.05;
    let nsteps = 10usize;
    let dt = t_end / nsteps as f64;
    println!("=== GPU 3D Navier–Stokes (Taylor–Green), {nsteps} steps (p={p}) ===\n");

    // Direct GPU loop + CPU reference.
    let gst = gale_gpu::GpuStokes3d::new(&mesh, alpha, nu, dt);
    let cst = Stokes3d::new(&mesh, alpha, nu, dt);
    let mut gx = nodal(&mesh, |x, y, z| eu(x, y, z, 0.0));
    let mut gy = nodal(&mesh, |_, _, _| 0.0);
    let mut gz = nodal(&mesh, |x, y, z| ew(x, y, z, 0.0));
    let (mut cx, mut cy, mut cz) = (gx.clone(), gy.clone(), gz.clone());
    let mut t = 0.0;
    for _ in 0..nsteps {
        t += dt;
        let (nx, ny, nz) = gst.step_ns(&gx, &gy, &gz, t, eu, ez, ew, zero, zero, zero)?;
        gx = nx;
        gy = ny;
        gz = nz;
        let (mx, my, mz) = cst.step_ns(&cx, &cy, &cz, t, eu, ez, ew, zero, zero, zero);
        cx = mx;
        cy = my;
        cz = mz;
    }

    // GPU vs CPU.
    let dnum = (l2(&mesh, &sub(&gx, &cx)).powi(2) + l2(&mesh, &sub(&gy, &cy)).powi(2) + l2(&mesh, &sub(&gz, &cz)).powi(2)).sqrt();
    let dden = (l2(&mesh, &cx).powi(2) + l2(&mesh, &cz).powi(2)).sqrt().max(1e-300);
    let rel_cpu = dnum / dden;

    // GPU vs analytic.
    let exu = nodal(&mesh, |x, y, z| eu(x, y, z, t_end));
    let exw = nodal(&mesh, |x, y, z| ew(x, y, z, t_end));
    let err_exact = (l2(&mesh, &sub(&gx, &exu)).powi(2) + l2(&mesh, &gy).powi(2) + l2(&mesh, &sub(&gz, &exw)).powi(2)).sqrt()
        / (l2(&mesh, &exu).powi(2) + l2(&mesh, &exw).powi(2)).sqrt().max(1e-300);

    // Framework: Simulation + GpuDualSplitting3d must reproduce the direct loop.
    let mut st: State<Mesh3d> = State::new(mesh.clone());
    let vid = st.add_field("velocity", 3);
    {
        let nn = mesh.refh.n_nodes();
        let v = st.fields.by_id_mut(vid);
        for (e, el) in mesh.elements.iter().enumerate() {
            for k in 0..nn {
                v.component_mut(0)[e * nn + k] = eu(el.geom.x[k], el.geom.y[k], el.geom.z[k], 0.0);
                v.component_mut(2)[e * nn + k] = ew(el.geom.x[k], el.geom.y[k], el.geom.z[k], 0.0);
            }
        }
    }
    let mut sim = Simulation::new(st);
    sim.set_integrator(gale_gpu::GpuDualSplitting3d::new(vid, dt, nu, alpha).boundary(eu, ez, ew));
    sim.run(nsteps as u64);
    let fx = sim.state.field("velocity").component(0).to_vec();
    let fy = sim.state.field("velocity").component(1).to_vec();
    let fz = sim.state.field("velocity").component(2).to_vec();
    let rel_fw = (l2(&mesh, &sub(&fx, &gx)).powi(2) + l2(&mesh, &sub(&fy, &gy)).powi(2) + l2(&mesh, &sub(&fz, &gz)).powi(2)).sqrt() / dden;

    println!("GPU vs CPU Stokes3d  = {rel_cpu:.3e}");
    println!("GPU vs analytic TG   = {err_exact:.3e}");
    println!("framework vs direct  = {rel_fw:.3e}");
    let pass = rel_cpu < 1e-7 && err_exact < 2e-2 && rel_fw < 1e-12;
    if pass {
        println!("\nPASS: GPU 3D Navier–Stokes matches the CPU solver, the analytic vortex, and the framework path.");
        Ok(())
    } else {
        eprintln!("\nFAIL: GPU 3D NS mismatch.");
        std::process::exit(1);
    }
}

fn sub(a: &[f64], b: &[f64]) -> Vec<f64> {
    a.iter().zip(b).map(|(x, y)| x - y).collect()
}
