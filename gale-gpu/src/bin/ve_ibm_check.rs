//! Capstone: an immersed rigid particle in **viscoelastic flow**, entirely on the GPU
//! — the project's headline capability (viscoelastic particle-laden microfluidics).
//! Composes, through `gale::sim::Simulation`, the coupled GPU viscoelastic integrator
//! [`gale_gpu::GpuViscoelasticDualSplitting`] (velocity + Oldroyd-B conformation, GPU
//! solves) with the GPU immersed-body stage hook [`gale_gpu::GpuPenalizationHook`].
//! Validated against the identical CPU composition (`ViscoelasticDualSplitting` +
//! `PenalizationHook`): velocity, conformation, disk damping, and drag.
//!
//! Run: cargo oxide run --bin ve-ibm-check

use gale::dg::{Disk, Mesh2d, VolumePenalization};
use gale::sim::{
    PenalizationDrag, PenalizationHook, Simulation, State, ViscoModel, ViscoelasticDualSplitting,
};

fn rel_l2(a: &[f64], b: &[f64]) -> f64 {
    let num: f64 = a.iter().zip(b).map(|(x, y)| (x - y).powi(2)).sum();
    let den: f64 = b.iter().map(|y| y * y).sum::<f64>().max(1e-300);
    (num / den).sqrt()
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let p = 4;
    let mesh = Mesh2d::rectangular(p, 3, 3, [0.0, 1.0], [0.0, 1.0]);
    let disk = Disk::new(0.5, 0.5, 0.2);
    let (eta_s, eta_p, lambda, alpha) = (0.5, 0.5, 0.5, 5.0);
    let (eta_b, dt, g) = (1e-3, 0.02, 1.0);
    let nsteps = 20u64;
    let bc_u = |_: f64, _: f64, _: f64| 0.0; // no-slip channel walls
    let bc_v = |_: f64, _: f64, _: f64| 0.0;
    let drive_x = move |_: f64, _: f64, _: f64| g; // pressure-gradient drive
    let zero_f = |_: f64, _: f64, _: f64| 0.0;
    println!("=== Capstone: immersed disk in GPU viscoelastic flow (Oldroyd-B), {nsteps} steps ===\n");

    // --- GPU run: coupled VE integrator + GPU penalization hook ------------------
    let penal_g = VolumePenalization::new(&mesh, &disk, eta_b);
    let mut gst = State::new(mesh.clone());
    let gvid = gst.add_field("velocity", 2);
    let gcid = gst.add_field("conformation", 3);
    let ginteg = gale_gpu::GpuViscoelasticDualSplitting::new(
        gvid, gcid, dt, eta_s, eta_p, lambda, alpha, ViscoModel::OldroydB,
    )
    .boundary(bc_u, bc_v)
    .drive(drive_x, zero_f);
    let eq = ginteg.equilibrium(&gst);
    for j in 0..3 {
        gst.fields.by_id_mut(gcid).component_mut(j).copy_from_slice(&eq[j]);
    }
    let mut gsim = Simulation::new(gst);
    gsim.set_integrator(ginteg);
    gsim.set_stage_hook(gale_gpu::GpuPenalizationHook::new(gvid, penal_g.clone(), dt));
    gsim.add_compute(PenalizationDrag::new("drag_x", gvid, penal_g.clone(), 0));
    gsim.run(nsteps);
    let gux = gsim.state.field("velocity").component(0).to_vec();
    let guy = gsim.state.field("velocity").component(1).to_vec();
    let gc: Vec<Vec<f64>> = (0..3).map(|j| gsim.state.field("conformation").component(j).to_vec()).collect();
    let gdrag = gsim.compute("drag_x").unwrap();

    // --- CPU reference: identical composition ------------------------------------
    let penal_c = VolumePenalization::new(&mesh, &disk, eta_b);
    let mut cst = State::new(mesh.clone());
    let cvid = cst.add_field("velocity", 2);
    let ccid = cst.add_field("conformation", 3);
    let cinteg = ViscoelasticDualSplitting::new(
        cvid, ccid, dt, eta_s, eta_p, lambda, alpha, ViscoModel::OldroydB,
    )
    .boundary(bc_u, bc_v)
    .drive(drive_x, zero_f);
    let eqc = cinteg.equilibrium(&cst);
    for j in 0..3 {
        cst.fields.by_id_mut(ccid).component_mut(j).copy_from_slice(&eqc[j]);
    }
    let mut csim = Simulation::new(cst);
    csim.set_integrator(cinteg);
    csim.set_stage_hook(PenalizationHook::new(cvid, penal_c.clone(), dt));
    csim.add_compute(PenalizationDrag::new("drag_x", cvid, penal_c.clone(), 0));
    csim.run(nsteps);
    let cux = csim.state.field("velocity").component(0).to_vec();
    let cuy = csim.state.field("velocity").component(1).to_vec();
    let cc: Vec<Vec<f64>> = (0..3).map(|j| csim.state.field("conformation").component(j).to_vec()).collect();
    let cdrag = csim.compute("drag_x").unwrap();

    // GPU vs CPU.
    let rel_u = rel_l2(&gux, &cux).max(rel_l2(&guy, &cuy));
    let rel_c = (0..3).fold(0.0f64, |a, j| a.max(rel_l2(&gc[j], &cc[j])));

    // Disk-interior damping on the GPU field.
    let (mut s, mut n) = (0.0f64, 0.0f64);
    let mut umax = 0.0f64;
    for i in 0..penal_g.mask.len() {
        umax = umax.max(gux[i].hypot(guy[i]));
        if penal_g.mask[i] > 0.5 {
            s += gux[i].hypot(guy[i]);
            n += 1.0;
        }
    }
    let interior_mean = s / n.max(1.0);

    println!("GPU vs CPU: velocity rel = {rel_u:.3e}   conformation rel = {rel_c:.3e}");
    println!("flow speed: max |u| = {umax:.4}   disk interior mean |u| = {interior_mean:.4}");
    println!("drag_x: gpu={gdrag:.4}  cpu={cdrag:.4}");
    let pass = rel_u < 1e-6
        && rel_c < 1e-6
        && umax > 1e-3
        && interior_mean < 0.5 * umax
        && gdrag.is_finite()
        && (gdrag - cdrag).abs() / cdrag.abs().max(1e-30) < 1e-5;
    if pass {
        println!("\nPASS: immersed particle in viscoelastic flow runs end-to-end on the GPU,\nmatching the CPU oracle — the headline capability.");
        Ok(())
    } else {
        eprintln!("\nFAIL: capstone mismatch.");
        std::process::exit(1);
    }
}
