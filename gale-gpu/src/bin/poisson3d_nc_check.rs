//! Validation of the GPU **3D non-conforming (2:1 octree)** SIPG Poisson operator
//! (`poisson3d_nc_apply`) against the CPU oracle `gale::dg::Poisson3d::apply`.
//!
//! Checks: (a) operator vs CPU on a refined hex mesh (Dirichlet, reaction 0); (b) operator symmetry
//! ⟨Au,v⟩=⟨Av,u⟩; (c) Helmholtz + mixed Dirichlet/Neumann vs CPU; (d) a conforming mesh still
//! matches CPU (the NC machinery's conforming fast-path).
//!
//! Run: cargo oxide run --bin poisson3d-nc-check

use gale::dg::{Mesh3d, Poisson3d};
use gale_gpu::operators::poisson3d_nc::{
    helmholtz3d_nc_cg_solve_tags, poisson3d_nc_apply, pressure3d_nc_cg_solve,
};

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
fn dot(a: &[f64], b: &[f64]) -> f64 {
    a.iter().zip(b).map(|(x, y)| x * y).sum()
}
fn maxabs(a: &[f64], b: &[f64]) -> f64 {
    a.iter().zip(b).map(|(x, y)| (x - y).abs()).fold(0.0, f64::max)
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let alpha = 8.0;
    let mut ok = true;

    // (a) refined hex mesh, Dirichlet, reaction 0 — operator vs CPU.
    {
        let p = 3;
        let mesh =
            Mesh3d::cartesian_refined(p, 2, 2, 2, [0.0, 1.0], [0.0, 1.0], [0.0, 1.0], &[(0, 0, 0), (1, 1, 1)]);
        let u = nodal(&mesh, |x, y, z| (1.7 * x + 0.3).sin() * (1.1 * y).cos() * (0.9 * z + 0.2).sin());
        let cpu = Poisson3d::new(&mesh, alpha).apply(&u);
        let gpu = poisson3d_nc_apply(&mesh, &u, alpha, 0.0, &[])?;
        let e = maxabs(&gpu, &cpu);
        let rel = e / cpu.iter().fold(0.0_f64, |m, &v| m.max(v.abs())).max(1e-300);
        println!("(a) refined p={p} ne={}: ||gpu-cpu||={e:.3e} rel={rel:.3e}", mesh.n_elements());
        ok &= rel < 1e-11;
    }

    // (b) symmetry of the GPU operator on a refined mesh.
    {
        let p = 3;
        let mesh =
            Mesh3d::cartesian_refined(p, 2, 2, 2, [0.0, 1.0], [0.0, 1.0], [0.0, 1.0], &[(0, 0, 0), (1, 0, 1)]);
        let u = nodal(&mesh, |x, y, z| (2.1 * x).sin() * (1.3 * y + 0.5).cos() * (0.7 * z).sin());
        let v = nodal(&mesh, |x, y, z| (1.1 * x + 0.2).cos() * (1.9 * y).sin() * (1.4 * z).cos());
        let au = poisson3d_nc_apply(&mesh, &u, alpha, 0.0, &[])?;
        let av = poisson3d_nc_apply(&mesh, &v, alpha, 0.0, &[])?;
        let (uav, vau) = (dot(&u, &av), dot(&v, &au));
        let rel = (uav - vau).abs() / uav.abs().max(1e-300);
        println!("(b) symmetry: ⟨u,Av⟩={uav:.6} ⟨v,Au⟩={vau:.6} rel={rel:.3e}");
        ok &= rel < 1e-11;
    }

    // (c) Helmholtz + mixed Dirichlet/Neumann (bottom/top Neumann) vs CPU on a refined mesh.
    {
        let p = 3;
        let mesh = Mesh3d::cartesian_refined(p, 2, 2, 2, [0.0, 1.0], [0.0, 1.0], [0.0, 1.0], &[(1, 1, 0)]);
        let u = nodal(&mesh, |x, y, z| (1.3 * x).cos() * (1.7 * y).sin() * (1.1 * z + 0.4).cos());
        let (reaction, neumann) = (4.0, vec![0u32, 1u32]); // Bottom=0, Top=1
        let cpu = Poisson3d::with_bc(&mesh, alpha, reaction, neumann.clone()).apply(&u);
        let gpu = poisson3d_nc_apply(&mesh, &u, alpha, reaction, &neumann)?;
        let e = maxabs(&gpu, &cpu);
        let rel = e / cpu.iter().fold(0.0_f64, |m, &v| m.max(v.abs())).max(1e-300);
        println!("(c) refined Helmholtz+Neumann p={p}: ||gpu-cpu||={e:.3e} rel={rel:.3e}");
        ok &= rel < 1e-11;
    }

    // (d) conforming mesh (no NC faces) still matches CPU.
    {
        let p = 4;
        let mesh = Mesh3d::rectangular(p, 3, 2, 2, [0.0, 1.0], [0.0, 1.0], [0.0, 1.0]);
        let u = nodal(&mesh, |x, y, z| (1.0 + x) * (2.0 - y) * (0.5 + z) + x * y * z);
        let cpu = Poisson3d::new(&mesh, alpha).apply(&u);
        let gpu = poisson3d_nc_apply(&mesh, &u, alpha, 0.0, &[])?;
        let e = maxabs(&gpu, &cpu);
        let rel = e / cpu.iter().fold(0.0_f64, |m, &v| m.max(v.abs())).max(1e-300);
        println!("(d) conforming p={p}: ||gpu-cpu||={e:.3e} rel={rel:.3e}");
        ok &= rel < 1e-11;
    }

    // (e) Helmholtz CG solve on a refined mesh vs CPU Poisson3d::cg.
    {
        let p = 3;
        let mesh = Mesh3d::cartesian_refined(p, 2, 2, 2, [0.0, 1.0], [0.0, 1.0], [0.0, 1.0], &[(0, 0, 0), (1, 1, 1)]);
        let ndof = mesh.n_elements() * mesh.refh.n_nodes();
        let b: Vec<f64> = (0..ndof).map(|i| ((i as f64) * 0.13).sin()).collect();
        let (reaction, tol) = (10.0, 1e-10);
        let (gpu, gi) = helmholtz3d_nc_cg_solve_tags(&mesh, &b, alpha, reaction, &[], tol, 5000)?;
        let (cpu, _ci, _r) = Poisson3d::with_reaction(&mesh, alpha, reaction).cg(&b, tol, 5000);
        let e = maxabs(&gpu, &cpu);
        let rel = e / cpu.iter().fold(0.0_f64, |m, &v| m.max(v.abs())).max(1e-300);
        println!("(e) NC Helmholtz CG p={p}: {gi} iters, ||gpu-cpu||={e:.3e} rel={rel:.3e}");
        ok &= rel < 1e-7;
    }

    // (f) deflated pure-Neumann pressure CG on a refined mesh vs CPU cg_deflated.
    {
        let p = 3;
        let mesh = Mesh3d::cartesian_refined(p, 2, 2, 2, [0.0, 1.0], [0.0, 1.0], [0.0, 1.0], &[(1, 0, 1)]);
        let ndof = mesh.n_elements() * mesh.refh.n_nodes();
        // compatible RHS (zero mean) for the singular system.
        let mut b: Vec<f64> = (0..ndof).map(|i| ((i as f64) * 0.21).cos()).collect();
        let mean = b.iter().sum::<f64>() / ndof as f64;
        b.iter_mut().for_each(|v| *v -= mean);
        let tol = 1e-10;
        let (gpu, gi) = pressure3d_nc_cg_solve(&mesh, &b, alpha, tol, 5000)?;
        let (cpu, _ci) = Poisson3d::with_bc(&mesh, alpha, 0.0, mesh.boundary_tags()).cg_deflated(&b, tol, 5000);
        // both are determined up to a constant ⇒ compare after removing the mean.
        let gm = gpu.iter().sum::<f64>() / ndof as f64;
        let cm = cpu.iter().sum::<f64>() / ndof as f64;
        let e = gpu.iter().zip(&cpu).map(|(a, b)| ((a - gm) - (b - cm)).abs()).fold(0.0, f64::max);
        let rel = e / cpu.iter().map(|v| (v - cm).abs()).fold(0.0_f64, f64::max).max(1e-300);
        println!("(f) NC deflated pressure CG p={p}: {gi} iters, ||gpu-cpu||={e:.3e} rel={rel:.3e}");
        ok &= rel < 1e-6;
    }

    if ok {
        println!("\nPASS: GPU 3D non-conforming Poisson operator matches the CPU oracle.");
        Ok(())
    } else {
        eprintln!("\nFAIL: GPU 3D NC operator mismatch.");
        std::process::exit(1);
    }
}
