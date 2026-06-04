//! 3D capstone: an immersed rigid **sphere** in **3D viscoelastic flow**, entirely on
//! the GPU — the 3D analogue of `ve-ibm-check`. Composes, through
//! `gale::sim::Simulation<Mesh3d>`, the coupled GPU 3D viscoelastic integrator
//! [`gale_gpu::GpuViscoelasticDualSplitting3d`] (Oldroyd-B) with the GPU 3D
//! immersed-body stage hook [`gale_gpu::GpuPenalization3dHook`]. Validated against the
//! identical CPU reference loop (OldroydB3d stress-divergence → Stokes3d.step_ns_forced
//! → OldroydB3d.step_ssp_rk3 → VolumePenalization3d.apply).
//!
//! Run: cargo oxide run --bin ve3d-ibm-check

use gale::dg::immersed3d::{Sphere, VolumePenalization3d};
use gale::dg::{Mesh3d, OldroydB3d, Stokes3d};
use gale::sim::{Penalization3dDrag, Simulation, State, ViscoModel};

fn rel_l2(a: &[f64], b: &[f64]) -> f64 {
    let num: f64 = a.iter().zip(b).map(|(x, y)| (x - y).powi(2)).sum();
    let den: f64 = b.iter().map(|y| y * y).sum::<f64>().max(1e-300);
    (num / den).sqrt()
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let p = 3;
    let mesh = Mesh3d::rectangular(p, 3, 3, 3, [0.0, 1.0], [0.0, 1.0], [0.0, 1.0]);
    let sphere = Sphere::new(0.5, 0.5, 0.5, 0.2);
    let (eta_s, eta_p, lambda, alpha) = (0.5, 0.5, 0.5, 5.0);
    let (eta_b, dt, g) = (1e-3, 0.02, 1.0);
    let nsteps = 4u64;
    let bc = |_: f64, _: f64, _: f64, _: f64| 0.0;
    let drive = move |_: f64, _: f64, _: f64, _: f64| g;
    let zero = |_: f64, _: f64, _: f64, _: f64| 0.0;
    println!("=== 3D capstone: immersed sphere in GPU viscoelastic flow (Oldroyd-B), {nsteps} steps ===\n");

    // --- GPU run: coupled 3D VE integrator + GPU 3D penalization hook ------------
    let penal_g = VolumePenalization3d::new(&mesh, &sphere, eta_b);
    let mut gst: State<Mesh3d> = State::new(mesh.clone());
    let gvid = gst.add_field("velocity", 3);
    let gcid = gst.add_field("conformation", 6);
    let ginteg = gale_gpu::GpuViscoelasticDualSplitting3d::new(gvid, gcid, dt, eta_s, eta_p, lambda, alpha, ViscoModel::OldroydB)
        .boundary(bc, bc, bc)
        .drive(drive, zero, zero);
    let eq = ginteg.equilibrium(&gst);
    for o in 0..6 {
        gst.fields.by_id_mut(gcid).component_mut(o).copy_from_slice(&eq[o]);
    }
    let mut gsim = Simulation::new(gst);
    gsim.set_integrator(ginteg);
    gsim.set_stage_hook(gale_gpu::GpuPenalization3dHook::new(gvid, penal_g.clone(), dt));
    gsim.add_compute(Penalization3dDrag::new("drag_x", gvid, penal_g.clone(), 0));
    gsim.run(nsteps);
    let gux = gsim.state.field("velocity").component(0).to_vec();
    let gc: Vec<Vec<f64>> = (0..6).map(|o| gsim.state.field("conformation").component(o).to_vec()).collect();
    let gdrag = gsim.compute("drag_x").unwrap();

    // --- CPU reference loop (same split order; penalize after the conformation advance) -
    let stokes = Stokes3d::new(&mesh, alpha, eta_s, dt);
    let m = OldroydB3d::new(&mesh, lambda, eta_p);
    let penal_c = VolumePenalization3d::new(&mesh, &sphere, eta_b);
    let nn = mesh.refh.n_nodes();
    let ndof = mesh.n_elements() * nn;
    let (mut rux, mut ruy, mut ruz) = (vec![0.0; ndof], vec![0.0; ndof], vec![0.0; ndof]);
    let mut rc = m.identity();
    let mut t = 0.0;
    for _ in 0..nsteps {
        t += dt;
        let (mut bx, by, bz) = m.stress_divergence(&rc);
        for v in bx.iter_mut() {
            *v += g;
        }
        let (mut nx, mut ny, mut nz) = stokes.step_ns_forced(&rux, &ruy, &ruz, t, bc, bc, bc, &bx, &by, &bz);
        rc = m.step_ssp_rk3(&rc, &nx, &ny, &nz, dt); // conformation with un-penalized velocity
        penal_c.apply(&mut nx, &mut ny, &mut nz, dt); // then IBM no-slip
        rux = nx;
        ruy = ny;
        ruz = nz;
    }
    let cdrag = {
        let (fx, _, _) = penal_c.force(&rux, &ruy, &ruz, &mesh);
        fx
    };

    let vel_err = rel_l2(&gux, &rux);
    let conf_err = (0..6).fold(0.0f64, |a, o| a.max(rel_l2(&gc[o], &rc[o])));
    let umax = gux.iter().fold(0.0f64, |a, &v| a.max(v.abs()));
    let (mut s, mut n) = (0.0f64, 0.0f64);
    for i in 0..penal_g.mask.len() {
        if penal_g.mask[i] > 0.5 {
            s += gux[i].abs();
            n += 1.0;
        }
    }
    let interior_mean = s / n.max(1.0);

    println!("GPU vs CPU: velocity rel = {vel_err:.3e}   conformation rel = {conf_err:.3e}");
    println!("flow: max|u| = {umax:.4}   sphere interior mean |u_x| = {interior_mean:.4}");
    println!("drag_x: gpu={gdrag:.4}  cpu={cdrag:.4}");
    let pass = vel_err < 1e-6
        && conf_err < 1e-6
        && umax > 1e-3
        && interior_mean < 0.5 * umax
        && (gdrag - cdrag).abs() / cdrag.abs().max(1e-30) < 1e-5;
    if pass {
        println!("\nPASS: immersed sphere in 3D viscoelastic flow runs end-to-end on the GPU,\nmatching the CPU oracle — the 3D headline capability.");
        Ok(())
    } else {
        eprintln!("\nFAIL: 3D capstone mismatch.");
        std::process::exit(1);
    }
}
