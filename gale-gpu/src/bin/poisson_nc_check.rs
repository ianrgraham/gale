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

    if ok {
        println!("ALL PASS");
        Ok(())
    } else {
        eprintln!("FAIL");
        std::process::exit(1);
    }
}
