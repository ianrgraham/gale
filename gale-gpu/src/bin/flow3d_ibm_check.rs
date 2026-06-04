//! Validation harness for 3D IBM-in-GPU-flow: flow past an immersed **sphere**
//! assembled through `gale::sim::Simulation<Mesh3d>` with the GPU 3D NS integrator
//! [`gale_gpu::GpuDualSplitting3d`] + the GPU 3D volume-penalization stage hook
//! [`gale_gpu::GpuPenalization3dHook`], checked against the CPU equivalent
//! (`DualSplitting3d` + `Penalization3dHook`).
//!
//! Run: cargo oxide run --bin flow3d-ibm-check

use gale::dg::immersed3d::{Sphere, VolumePenalization3d};
use gale::dg::Mesh3d;
use gale::sim::{
    DualSplitting3d, Penalization3dDrag, Penalization3dHook, Simulation, State,
};

fn rel_l2(a: &[f64], b: &[f64]) -> f64 {
    let num: f64 = a.iter().zip(b).map(|(x, y)| (x - y).powi(2)).sum();
    let den: f64 = b.iter().map(|y| y * y).sum::<f64>().max(1e-300);
    (num / den).sqrt()
}

fn build_velocity(st: &mut State<Mesh3d>) -> gale::sim::FieldId {
    let vid = st.add_field("velocity", 3);
    for u in st.field_mut("velocity").component_mut(0).iter_mut() {
        *u = 1.0; // uniform inflow (1,0,0)
    }
    vid
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mesh = Mesh3d::rectangular(3, 3, 3, 3, [0.0, 1.0], [0.0, 1.0], [0.0, 1.0]);
    let sphere = Sphere::new(0.5, 0.5, 0.5, 0.22);
    let (eta_b, dt, nu) = (1e-3, 5e-3, 0.1);
    let nsteps = 4u64;
    let inflow = |_x: f64, _y: f64, _z: f64, _t: f64| 1.0;
    let zero = |_: f64, _: f64, _: f64, _: f64| 0.0;
    println!("=== GPU 3D flow past immersed sphere (Simulation<Mesh3d> + GpuDualSplitting3d + GpuPenalization3dHook) ===\n");

    // GPU run.
    let penal_g = VolumePenalization3d::new(&mesh, &sphere, eta_b);
    let mut gst: State<Mesh3d> = State::new(mesh.clone());
    let gvid = build_velocity(&mut gst);
    let mut gsim = Simulation::new(gst);
    gsim.set_integrator(gale_gpu::GpuDualSplitting3d::new(gvid, dt, nu, 5.0).boundary(inflow, zero, zero));
    gsim.set_stage_hook(gale_gpu::GpuPenalization3dHook::new(gvid, penal_g.clone(), dt));
    gsim.add_compute(Penalization3dDrag::new("drag_x", gvid, penal_g.clone(), 0));
    gsim.run(nsteps);
    let gux = gsim.state.field("velocity").component(0).to_vec();
    let guy = gsim.state.field("velocity").component(1).to_vec();
    let guz = gsim.state.field("velocity").component(2).to_vec();
    let gdrag = gsim.compute("drag_x").unwrap();

    // CPU reference.
    let penal_c = VolumePenalization3d::new(&mesh, &sphere, eta_b);
    let mut cst: State<Mesh3d> = State::new(mesh.clone());
    let cvid = build_velocity(&mut cst);
    let mut csim = Simulation::new(cst);
    csim.set_integrator(DualSplitting3d::new(cvid, dt, nu, 5.0).boundary(inflow, zero, zero));
    csim.set_stage_hook(Penalization3dHook::new(cvid, penal_c.clone(), dt));
    csim.add_compute(Penalization3dDrag::new("drag_x", cvid, penal_c.clone(), 0));
    csim.run(nsteps);
    let cux = csim.state.field("velocity").component(0).to_vec();
    let cuy = csim.state.field("velocity").component(1).to_vec();
    let cuz = csim.state.field("velocity").component(2).to_vec();
    let cdrag = csim.compute("drag_x").unwrap();

    let rel = rel_l2(&gux, &cux).max(rel_l2(&guy, &cuy)).max(rel_l2(&guz, &cuz));

    let (mut s, mut n) = (0.0f64, 0.0f64);
    for i in 0..penal_g.mask.len() {
        if penal_g.mask[i] > 0.5 {
            s += gux[i].abs();
            n += 1.0;
        }
    }
    let interior_mean = s / n.max(1.0);

    println!("GPU vs CPU velocity rel = {rel:.3e}");
    println!("sphere interior mean |u_x| = {interior_mean:.4}   (free stream = 1.0)");
    println!("drag_x: gpu={gdrag:.4}  cpu={cdrag:.4}");
    let pass = rel < 1e-7
        && interior_mean < 0.6
        && gdrag.is_finite()
        && gdrag > 0.0
        && (gdrag - cdrag).abs() / cdrag.abs().max(1e-30) < 1e-6;
    if pass {
        println!("\nPASS: GPU 3D flow + IBM penalization matches CPU, damps the sphere, reports drag.");
        Ok(())
    } else {
        eprintln!("\nFAIL: GPU 3D IBM-flow mismatch.");
        std::process::exit(1);
    }
}
