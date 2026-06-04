//! Validation harness for the 3D GPU SIPG elliptic foundation: the matrix-free
//! operator [`gale_gpu::poisson3d_apply`], the viscous [`gale_gpu::helmholtz3d_cg_solve`],
//! and the deflated pure-Neumann [`gale_gpu::pressure3d_cg_solve`] — each checked
//! against the CPU oracle `gale::dg::Poisson3d`.
//!
//! Run: cargo oxide run --bin poisson3d-check

use gale::dg::{Mesh3d, Poisson3d};
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

fn demean(v: &mut [f64]) {
    let m = v.iter().sum::<f64>() / v.len() as f64;
    v.iter_mut().for_each(|x| *x -= m);
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let p = 4;
    let alpha = 8.0;
    let mesh = Mesh3d::rectangular(p, 2, 2, 2, [0.0, 1.0], [0.0, 1.0], [0.0, 1.0]);
    let nn = mesh.refh.n_nodes();
    let ndof = mesh.n_elements() * nn;
    println!("=== GPU 3D SIPG elliptic (p={p}, dofs={ndof}) vs CPU Poisson3d ===\n");
    let mut ok = true;

    // (1) Operator action A·u (+ reaction) on a pseudo-random vector.
    let lambda = 2.5;
    let u: Vec<f64> = (0..ndof).map(|i| ((i * 7 + 3) % 13) as f64 - 6.0).collect();
    let cpu_op = Poisson3d::with_reaction(&mesh, alpha, lambda);
    let a_cpu = cpu_op.apply(&u);
    let a_gpu = gale_gpu::poisson3d_apply(&mesh, &u, alpha, lambda)?;
    let scale = a_cpu.iter().fold(0.0f64, |a, &v| a.max(v.abs())).max(1e-300);
    let op_err = a_gpu.iter().zip(&a_cpu).fold(0.0f64, |a, (g, c)| a.max((g - c).abs())) / scale;
    let op_ok = op_err < 1e-12;
    ok &= op_ok;
    println!("operator (λM+A)·u: max|gpu−cpu|/|op| = {op_err:.3e}   {}", if op_ok { "OK" } else { "FAIL" });

    // (2) Helmholtz solve: u = sin(πx)sin(πy)sin(πz), (λ−Δ)u ↔ (λ+3π²)u = f.
    let exact = |x: f64, y: f64, z: f64| (PI * x).sin() * (PI * y).sin() * (PI * z).sin();
    let l = 50.0;
    let hop = Poisson3d::with_reaction(&mesh, alpha, l);
    let f = nodal(&mesh, |x, y, z| (l + 3.0 * PI * PI) * exact(x, y, z));
    let b = hop.rhs(&f, |_, _, _| 0.0);
    let (uc, itc, _) = hop.cg(&b, 1e-10, 20000);
    let (ug, itg) = gale_gpu::helmholtz3d_cg_solve(&mesh, &b, alpha, l, 1e-10, 20000)?;
    let hs = uc.iter().fold(0.0f64, |a, &v| a.max(v.abs())).max(1e-300);
    let h_err = ug.iter().zip(&uc).fold(0.0f64, |a, (g, c)| a.max((g - c).abs())) / hs;
    let h_ok = h_err < 1e-8;
    ok &= h_ok;
    println!("helmholtz CG: iters cpu={itc} gpu={itg}, ‖gpu−cpu‖∞/|u| = {h_err:.3e}   {}", if h_ok { "OK" } else { "FAIL" });

    // (3) Deflated pressure (pure Neumann): u = cos(πx)cos(πy)cos(πz), −Δu = 3π²u.
    let exn = |x: f64, y: f64, z: f64| (PI * x).cos() * (PI * y).cos() * (PI * z).cos();
    let pop = Poisson3d::with_bc(&mesh, alpha, 0.0, vec![0, 1, 2, 3, 4, 5]);
    let fp = nodal(&mesh, |x, y, z| 3.0 * PI * PI * exn(x, y, z));
    let bp = pop.rhs_mixed(&fp, |_, _, _| 0.0, |_, _, _| 0.0);
    let (mut pc, itpc) = pop.cg_deflated(&bp, 1e-10, 20000);
    let (mut pg, itpg) = gale_gpu::pressure3d_cg_solve(&mesh, &bp, alpha, 1e-10, 20000)?;
    demean(&mut pc);
    demean(&mut pg);
    let ps = pc.iter().fold(0.0f64, |a, &v| a.max(v.abs())).max(1e-300);
    let p_err = pg.iter().zip(&pc).fold(0.0f64, |a, (g, c)| a.max((g - c).abs())) / ps;
    let p_ok = p_err < 1e-8;
    ok &= p_ok;
    println!("pressure deflated CG: iters cpu={itpc} gpu={itpg}, ‖gpu−cpu‖∞/|p| = {p_err:.3e}   {}", if p_ok { "OK" } else { "FAIL" });

    if ok {
        println!("\nPASS: 3D GPU SIPG operator, Helmholtz, and deflated pressure all match the CPU oracle.");
        Ok(())
    } else {
        eprintln!("\nFAIL: 3D GPU elliptic mismatch.");
        std::process::exit(1);
    }
}
