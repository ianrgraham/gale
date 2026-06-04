//! Validation harness for IBM-in-GPU-flow: a flow past an immersed disk assembled
//! through `gale::sim::Simulation` with the GPU Navier–Stokes integrator
//! [`gale_gpu::GpuDualSplitting`] + the GPU volume-penalization stage hook
//! [`gale_gpu::GpuPenalizationHook`]. Checks (a) the GPU trajectory matches the CPU
//! equivalent (`DualSplitting` + `PenalizationHook`) and (b) the disk interior is
//! damped below the free stream with a finite, positive drag — i.e. the immersed
//! body and the GPU flow compose through the HOOMD-style framework.
//!
//! Run: cargo oxide run --bin flow-ibm-check

use gale::dg::{Disk, Mesh2d, VolumePenalization};
use gale::sim::{DualSplitting, PenalizationDrag, PenalizationHook, Simulation, State};

fn rel_l2(a: &[f64], b: &[f64]) -> f64 {
    let num: f64 = a.iter().zip(b).map(|(x, y)| (x - y).powi(2)).sum();
    let den: f64 = b.iter().map(|y| y * y).sum::<f64>().max(1e-300);
    (num / den).sqrt()
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let p = 4;
    let mesh = Mesh2d::rectangular(p, 3, 3, [0.0, 1.0], [0.0, 1.0]);
    let disk = Disk::new(0.5, 0.5, 0.22);
    let (eta_b, dt, nu) = (1e-3, 5e-3, 0.1);
    let nsteps = 4u64;
    let inflow = |_x: f64, _y: f64, _t: f64| 1.0;
    let zero = |_: f64, _: f64, _: f64| 0.0;
    println!("=== GPU flow past immersed disk (Simulation + GpuDualSplitting + GpuPenalizationHook) ===\n");

    // --- GPU run -----------------------------------------------------------------
    let penal_g = VolumePenalization::new(&mesh, &disk, eta_b);
    let mut gst = State::new(mesh.clone());
    let gvid = gst.add_field("velocity", 2);
    for u in gst.field_mut("velocity").component_mut(0).iter_mut() {
        *u = 1.0; // uniform inflow u = (1, 0)
    }
    let mut gsim = Simulation::new(gst);
    gsim.set_integrator(gale_gpu::GpuDualSplitting::new(gvid, dt, nu, 5.0).boundary(inflow, zero));
    gsim.set_stage_hook(gale_gpu::GpuPenalizationHook::new(gvid, penal_g.clone(), dt));
    gsim.add_compute(PenalizationDrag::new("drag_x", gvid, penal_g.clone(), 0));
    gsim.run(nsteps);
    let gux = gsim.state.field("velocity").component(0).to_vec();
    let guy = gsim.state.field("velocity").component(1).to_vec();
    let gdrag = gsim.compute("drag_x").unwrap();

    // --- CPU reference (identical setup) -----------------------------------------
    let penal_c = VolumePenalization::new(&mesh, &disk, eta_b);
    let mut cst = State::new(mesh.clone());
    let cvid = cst.add_field("velocity", 2);
    for u in cst.field_mut("velocity").component_mut(0).iter_mut() {
        *u = 1.0;
    }
    let mut csim = Simulation::new(cst);
    csim.set_integrator(DualSplitting::new(cvid, dt, nu, 5.0).boundary(inflow, zero));
    csim.set_stage_hook(PenalizationHook::new(cvid, penal_c.clone(), dt));
    csim.add_compute(PenalizationDrag::new("drag_x", cvid, penal_c.clone(), 0));
    csim.run(nsteps);
    let cux = csim.state.field("velocity").component(0).to_vec();
    let cuy = csim.state.field("velocity").component(1).to_vec();
    let cdrag = csim.compute("drag_x").unwrap();

    // GPU vs CPU.
    let rel = rel_l2(&gux, &cux).max(rel_l2(&guy, &cuy));

    // Interior-damping diagnostic on the GPU field.
    let (mut s, mut n) = (0.0f64, 0.0f64);
    for i in 0..penal_g.mask.len() {
        if penal_g.mask[i] > 0.5 {
            s += gux[i].abs();
            n += 1.0;
        }
    }
    let interior_mean = s / n.max(1.0);

    println!("GPU vs CPU velocity rel = {rel:.3e}");
    println!("disk interior mean |u| = {interior_mean:.4}   (free stream = 1.0)");
    println!("drag_x: gpu={gdrag:.4}  cpu={cdrag:.4}");
    let pass = rel < 1e-7
        && interior_mean < 0.6
        && gdrag.is_finite()
        && gdrag > 0.0
        && (gdrag - cdrag).abs() / cdrag.abs().max(1e-30) < 1e-6;
    if pass {
        println!("\nPASS: GPU flow + IBM penalization matches CPU, damps the disk, reports drag.");
        Ok(())
    } else {
        eprintln!("\nFAIL: GPU IBM-flow mismatch.");
        std::process::exit(1);
    }
}
