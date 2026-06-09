//! Validates the GPU ARK2 / ARS(2,2,2) IMEX conformation advance
//! (`gale_gpu::logconf_ark2_advance_gpu`, Phase 3) against the CPU oracle
//! `gale::dg::LogConfOldroydB::step_ark2_imex`, for both Oldroyd-B (α=0) and Giesekus (α>0).

use gale::dg::{LogConfOldroydB, Mesh2d};

fn run(name: &str, alpha: f64, ext: f64) -> Result<bool, Box<dyn std::error::Error>> {
    let p = 4;
    let mesh = Mesh2d::rectangular(p, 3, 3, [0.0, 1.0], [0.0, 1.0]);
    let nn = mesh.refq.n_nodes();
    let ne = mesh.n_elements();
    let ndof = ne * nn;
    let (lambda, gdot) = (0.5, 4.0);
    let lc = LogConfOldroydB::new(&mesh, lambda, 1.0).with_mobility(alpha).with_extensibility(ext);

    let mut ux = vec![0.0; ndof];
    for (e, el) in mesh.elements.iter().enumerate() {
        for k in 0..nn {
            ux[e * nn + k] = gdot * el.geom.y[k]; // simple shear
        }
    }
    let uy = vec![0.0; ndof];
    let dt = 0.02;
    let nsteps = 20;

    let mut cpu = lc.identity();
    let mut gpu = lc.identity();
    for _ in 0..nsteps {
        cpu = lc.step_ark2_imex(&cpu, &ux, &uy, dt);
        gpu = gale_gpu::logconf_ark2_advance_gpu(&mesh, &lc, &gpu, &ux, &uy, dt, None)?;
    }

    let cc = lc.conformation(&cpu);
    let gc = lc.conformation(&gpu);
    let mut max_abs = 0.0f64;
    let mut scale = 1e-300f64;
    let mut spd = true;
    for v in 0..3 {
        for i in 0..ndof {
            max_abs = max_abs.max((gc[v][i] - cc[v][i]).abs());
            scale = scale.max(cc[v][i].abs());
        }
    }
    for i in 0..ndof {
        let det = gc[0][i] * gc[2][i] - gc[1][i] * gc[1][i];
        if !(gc[0][i] > 0.0 && det > 0.0) {
            spd = false;
        }
    }
    let trc = cc[0][0] + cc[2][0];
    let rel = max_abs / scale;
    println!(
        "{name:<10} {nsteps} ARK2 steps, Wi={}: max|gpu−cpu|/|C| = {rel:.3e}  Cxx={:.3}  trC={:.3}  SPD: {spd}",
        lambda * gdot,
        cc[0][0],
        trc
    );
    Ok(rel < 1e-9 && spd)
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("=== GPU ARK2 IMEX conformation advance vs CPU oracle ===\n");
    let mut ok = true;
    ok &= run("Oldroyd-B", 0.0, f64::INFINITY)?;
    ok &= run("Giesekus", 0.4, f64::INFINITY)?;
    ok &= run("FENE-P", 0.0, 20.0)?;
    if ok {
        println!("\nPASS: GPU ARK2 IMEX matches the CPU oracle for Oldroyd-B, Giesekus, and FENE-P.");
        Ok(())
    } else {
        eprintln!("\nFAIL: GPU/CPU mismatch or non-SPD conformation.");
        std::process::exit(1);
    }
}
