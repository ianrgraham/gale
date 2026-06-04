//! Validation harness for the GPU SIPG-Poisson operator on **2:1 non-conforming
//! (mortar)** quad meshes: [`gale_gpu::poisson_nc_apply`] checked against the CPU
//! oracle `gale::dg::Poisson::apply` on a refined mesh (so `CoarseToFine` /
//! `FineToCoarse` interfaces are exercised), plus an operator-symmetry check.
//!
//! Run: cargo oxide run --bin poisson-nc-check

use gale::dg::{Mesh2d, Poisson};

fn dot(a: &[f64], b: &[f64]) -> f64 {
    a.iter().zip(b).map(|(x, y)| x * y).sum()
}

/// Deterministic pseudo-random vector in roughly [-1, 1].
fn rand_vec(n: usize, seed: u64) -> Vec<f64> {
    let mut s = seed;
    (0..n)
        .map(|_| {
            s = s.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            ((s >> 33) as f64) / (1u64 << 31) as f64 - 1.0
        })
        .collect()
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let p = 4;
    let alpha = 5.0;
    let reaction = 0.0;
    // Centre cell refined → CoarseToFine / FineToCoarse mortar interfaces.
    let mesh = Mesh2d::cartesian_refined(p, 3, 3, [0.0, 1.0], [0.0, 1.0], &[(1, 1)]);
    let cpu = Poisson::with_reaction(&mesh, alpha, reaction);
    let ndof = cpu.ndof();
    println!(
        "non-conforming mesh: p={p}, {} elements, {ndof} dof",
        mesh.n_elements()
    );

    let mut ok = true;

    // (a) operator vs CPU oracle on a random field.
    let u = rand_vec(ndof, 1);
    let r_gpu = gale_gpu::poisson_nc_apply(&mesh, &u, alpha, reaction)?;
    let r_cpu = cpu.apply(&u);
    let mut max_rel = 0.0f64;
    let mut max_abs = 0.0f64;
    for i in 0..ndof {
        let abs = (r_gpu[i] - r_cpu[i]).abs();
        let rel = abs / (r_cpu[i].abs() + 1e-300);
        max_abs = max_abs.max(abs);
        max_rel = max_rel.max(rel);
    }
    let tol_a = 1e-12;
    let pass_a = max_rel < tol_a;
    ok &= pass_a;
    println!(
        "(a) operator vs CPU: max_abs={max_abs:.3e} max_rel={max_rel:.3e}  [{}]",
        if pass_a { "PASS" } else { "FAIL" }
    );

    // (b) symmetry ⟨A u, v⟩ ≈ ⟨A v, u⟩ on random u, v.
    let v = rand_vec(ndof, 2);
    let au = gale_gpu::poisson_nc_apply(&mesh, &u, alpha, reaction)?;
    let av = gale_gpu::poisson_nc_apply(&mesh, &v, alpha, reaction)?;
    let uav = dot(&u, &av);
    let vau = dot(&v, &au);
    let sym = (uav - vau).abs() / (uav.abs() + 1.0);
    let tol_b = 1e-9;
    let pass_b = sym < tol_b;
    ok &= pass_b;
    println!(
        "(b) symmetry: uᵀAv={uav:.6e} vᵀAu={vau:.6e} rel={sym:.3e}  [{}]",
        if pass_b { "PASS" } else { "FAIL" }
    );

    // Nodal sampler over the (refined) mesh.
    let nn = mesh.refq.n_nodes();
    let nodal = |f: &dyn Fn(f64, f64) -> f64| -> Vec<f64> {
        let mut out = vec![0.0; ndof];
        for (e, el) in mesh.elements.iter().enumerate() {
            for k in 0..nn {
                out[e * nn + k] = f(el.geom.x[k], el.geom.y[k]);
            }
        }
        out
    };
    use std::f64::consts::PI;

    // (c) Dirichlet CG solve vs CPU Poisson::cg. MMS u=sin(πx)sin(πy) (0 on boundary).
    let exact = |x: f64, y: f64| (PI * x).sin() * (PI * y).sin();
    let f = nodal(&|x, y| 2.0 * PI * PI * exact(x, y));
    let b = cpu.rhs(&f, |_, _| 0.0);
    let (uc, itc, _) = cpu.cg(&b, 1e-10, 20000);
    let (ug, itg) = gale_gpu::poisson_nc_cg_solve(&mesh, &b, alpha, 0.0, 1e-10, 20000)?;
    let scale = uc.iter().fold(0.0f64, |a, &v| a.max(v.abs())).max(1e-300);
    let rel_c = ug.iter().zip(&uc).fold(0.0f64, |a, (g, c)| a.max((g - c).abs())) / scale;
    let pass_c = rel_c < 1e-8;
    ok &= pass_c;
    println!("(c) Dirichlet CG: iters cpu={itc} gpu={itg}, rel={rel_c:.3e}  [{}]", if pass_c { "PASS" } else { "FAIL" });

    // (d) deflated pure-Neumann pressure CG vs CPU cg_deflated. MMS cos·cos.
    let pres = Poisson::with_bc(&mesh, alpha, 0.0, vec![0, 1, 2, 3]);
    let exn = |x: f64, y: f64| (PI * x).cos() * (PI * y).cos();
    let fp = nodal(&|x, y| 2.0 * PI * PI * exn(x, y));
    let bp = pres.rhs_mixed(&fp, |_, _| 0.0, |_, _| 0.0);
    let (mut pc, itpc) = pres.cg_deflated(&bp, 1e-10, 20000);
    let (mut pg, itpg) = gale_gpu::pressure_nc_cg_solve(&mesh, &bp, alpha, 1e-10, 20000)?;
    let demean = |v: &mut [f64]| { let m = v.iter().sum::<f64>() / v.len() as f64; v.iter_mut().for_each(|x| *x -= m); };
    demean(&mut pc);
    demean(&mut pg);
    let ps = pc.iter().fold(0.0f64, |a, &v| a.max(v.abs())).max(1e-300);
    let rel_d = pg.iter().zip(&pc).fold(0.0f64, |a, (g, c)| a.max((g - c).abs())) / ps;
    let pass_d = rel_d < 1e-8;
    ok &= pass_d;
    println!("(d) deflated pressure CG: iters cpu={itpc} gpu={itpg}, rel={rel_d:.3e}  [{}]", if pass_d { "PASS" } else { "FAIL" });

    if ok {
        println!("ALL PASS");
        Ok(())
    } else {
        eprintln!("FAIL");
        std::process::exit(1);
    }
}
