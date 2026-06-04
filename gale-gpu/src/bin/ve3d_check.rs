//! Validation harness for the GPU 3D viscoelastic path: (1) the standalone Oldroyd-B
//! conformation rhs [`gale_gpu::oldroyd3d_conf_rhs`] vs `gale::dg::OldroydB3d::conformation_rhs`,
//! and (2) the coupled [`gale_gpu::GpuViscoelasticDualSplitting3d`] driving a 3D
//! Oldroyd-B channel through `Simulation<Mesh3d>`, vs a CPU reference dual-split loop
//! (OldroydB3d stress-divergence → Stokes3d.step_ns_forced → OldroydB3d.step_ssp_rk3).
//!
//! Run: cargo oxide run --bin ve3d-check

use gale::dg::{Mesh3d, OldroydB3d, Stokes3d};
use gale::sim::{Simulation, State};

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

fn rel_l2(a: &[f64], b: &[f64]) -> f64 {
    let num: f64 = a.iter().zip(b).map(|(x, y)| (x - y).powi(2)).sum();
    let den: f64 = b.iter().map(|y| y * y).sum::<f64>().max(1e-300);
    (num / den).sqrt()
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let p = 3;
    let mesh = Mesh3d::rectangular(p, 2, 2, 2, [0.0, 1.0], [0.0, 1.0], [0.0, 1.0]);
    let (eta_s, eta_p, lambda, alpha) = (0.5, 0.5, 0.5, 5.0);
    println!("=== GPU 3D viscoelastic (Oldroyd-B) vs CPU oracle (p={p}) ===\n");
    let mut ok = true;

    // (1) Standalone conformation rhs on a smooth velocity + SPD-ish conformation.
    let model = OldroydB3d::new(&mesh, lambda, eta_p);
    let ux = nodal(&mesh, |x, y, z| 0.3 * (x + 0.5 * y - 0.2 * z));
    let uy = nodal(&mesh, |x, y, z| -0.2 * (y - 0.3 * x + 0.1 * z));
    let uz = nodal(&mesh, |x, y, z| 0.15 * (z + 0.2 * x - 0.4 * y));
    let c0 = nodal(&mesh, |x, _, _| 1.0 + 0.1 * x);
    let c3 = nodal(&mesh, |_, y, _| 1.0 + 0.1 * y);
    let c5 = nodal(&mesh, |_, _, z| 1.0 + 0.1 * z);
    let c1 = nodal(&mesh, |x, y, _| 0.05 * (x * y));
    let c2 = nodal(&mesh, |x, _, z| 0.04 * (x * z));
    let c4 = nodal(&mesh, |_, y, z| 0.03 * (y * z));
    let c = [c0, c1, c2, c3, c4, c5];
    let r_cpu = model.conformation_rhs(&c, &ux, &uy, &uz);
    let r_gpu = gale_gpu::oldroyd3d_conf_rhs(&mesh, &c, &ux, &uy, &uz, lambda)?;
    let rhs_err = (0..6).fold(0.0f64, |a, o| a.max(rel_l2(&r_gpu[o], &r_cpu[o])));
    let rhs_ok = rhs_err < 1e-12;
    ok &= rhs_ok;
    println!("conformation rhs: max rel = {rhs_err:.3e}   {}", if rhs_ok { "OK" } else { "FAIL" });

    // (2) Coupled Oldroyd-B channel: GPU framework vs CPU reference loop.
    let (g, dt, nsteps) = (1.0, 0.02, 20u64);
    let bc = |_: f64, _: f64, _: f64, _: f64| 0.0;
    let drive = move |_: f64, _: f64, _: f64, _: f64| g;
    let zero = |_: f64, _: f64, _: f64, _: f64| 0.0;

    // GPU framework run.
    let mut st: State<Mesh3d> = State::new(mesh.clone());
    let vid = st.add_field("velocity", 3);
    let cid = st.add_field("conformation", 6);
    let integ = gale_gpu::GpuViscoelasticDualSplitting3d::new(vid, cid, dt, eta_s, eta_p, lambda, alpha)
        .boundary(bc, bc, bc)
        .drive(drive, zero, zero);
    let eq = integ.equilibrium(&st);
    for o in 0..6 {
        st.fields.by_id_mut(cid).component_mut(o).copy_from_slice(&eq[o]);
    }
    let mut sim = Simulation::new(st);
    sim.set_integrator(integ);
    sim.run(nsteps);
    let gux = sim.state.field("velocity").component(0).to_vec();
    let gc: Vec<Vec<f64>> = (0..6).map(|o| sim.state.field("conformation").component(o).to_vec()).collect();

    // CPU reference dual-split loop (no CPU 3D-VE-flow integrator exists, so assemble it).
    let stokes = Stokes3d::new(&mesh, alpha, eta_s, dt);
    let m = OldroydB3d::new(&mesh, lambda, eta_p);
    let nn = mesh.refh.n_nodes();
    let ndof = mesh.n_elements() * nn;
    let (mut rux, mut ruy, mut ruz) = (vec![0.0; ndof], vec![0.0; ndof], vec![0.0; ndof]);
    let mut rc = m.identity();
    let mut t = 0.0;
    for _ in 0..nsteps {
        t += dt;
        let (mut bx, by, bz) = m.stress_divergence(&rc);
        for v in bx.iter_mut() {
            *v += g; // x-direction pressure-gradient drive
        }
        let (nx, ny, nz) = stokes.step_ns_forced(&rux, &ruy, &ruz, t, bc, bc, bc, &bx, &by, &bz);
        rc = m.step_ssp_rk3(&rc, &nx, &ny, &nz, dt);
        rux = nx;
        ruy = ny;
        ruz = nz;
    }

    let vel_err = rel_l2(&gux, &rux);
    let conf_err = (0..6).fold(0.0f64, |a, o| a.max(rel_l2(&gc[o], &rc[o])));
    let umax = gux.iter().fold(0.0f64, |a, &v| a.max(v.abs()));
    let coupled_ok = vel_err < 1e-6 && conf_err < 1e-6 && umax > 1e-3;
    ok &= coupled_ok;
    println!("coupled channel: vel rel = {vel_err:.3e}  conf rel = {conf_err:.3e}  umax = {umax:.4}  {}", if coupled_ok { "OK" } else { "FAIL" });

    if ok {
        println!("\nPASS: GPU 3D Oldroyd-B viscoelastic rhs and coupled flow match the CPU oracle.");
        Ok(())
    } else {
        eprintln!("\nFAIL: GPU 3D viscoelastic mismatch.");
        std::process::exit(1);
    }
}
