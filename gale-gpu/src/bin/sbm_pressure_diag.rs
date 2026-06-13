//! **SBM pressure-operator diagnostic** — is the stalling natural-Neumann SBM pressure-Poisson
//! (the step-2c blocker, `docs/sbm-status.md`) merely ILL-CONDITIONED (⇒ a preconditioner fixes
//! it) or genuinely ILL-POSED / asymmetric / near-singular (⇒ an operator bug)?
//!
//! Same operator as `sbm-cylinder-check`: `ShiftedPoisson::with_bc(.., 0.0, [3,0,2], sb)
//! .surrogate_neumann()` — natural-Neumann surrogate (cylinder), inflow+walls Neumann, outflow
//! (tag 1) Dirichlet p=0 ⇒ non-singular. Checks: (1) symmetry `<Au,v>==<u,Av>`; (2) nullspace
//! probe `A·1`; (3) CG residual HISTORY — steady-but-slow decay = conditioning; a plateau =
//! RHS-out-of-range / breakdown. Run: `cargo run --bin sbm-pressure-diag` (SBM_NY sets the grid).

use gale::dg::{CircleLevelSet, Mesh2d, PMultigrid, ShiftedBoundary, ShiftedPoisson};

fn dot(a: &[f64], b: &[f64]) -> f64 {
    a.iter().zip(b).map(|(x, y)| x * y).sum()
}
fn norm(a: &[f64]) -> f64 {
    dot(a, a).sqrt()
}

fn broadband(n: usize) -> Vec<f64> {
    (0..n)
        .map(|i| {
            let h = (i as u64).wrapping_mul(2862933555777941757).wrapping_add(3037000493);
            ((h >> 11) as f64 / (1u64 << 53) as f64) * 2.0 - 1.0
        })
        .collect()
}

fn main() {
    let p = 3;
    let ny = std::env::var("SBM_NY").ok().and_then(|v| v.parse().ok()).unwrap_or(8usize);
    let h = 0.41;
    let nx = ((2.2 / h) * ny as f64).round() as usize;
    let alpha = 5.0;
    let mesh = Mesh2d::rectangular(p, nx, ny, [0.0, 2.2], [0.0, h]);
    let nn = mesh.refq.n_nodes();
    let ndof = mesh.n_elements() * nn;
    let ls = CircleLevelSet::new(0.2, 0.2, 0.05);
    let sb = ShiftedBoundary::new(&mesh, &ls);
    let pres = ShiftedPoisson::with_bc(&mesh, alpha, 0.0, vec![3, 0, 2], sb.clone()).surrogate_neumann();
    let active = pres.active().to_vec();
    let n_active_dof: usize = (0..mesh.n_elements()).filter(|&e| active[e]).count() * nn;
    println!("=== SBM pressure diagnostic: {nx}×{ny} p={p}, ndof={ndof}, active dofs={n_active_dof} ===");
    println!("is_singular (should be false — outflow Dirichlet pins it): {}", pres.is_singular());

    // (1) SYMMETRY: <A u, v> vs <u, A v> for random u, v. SPD CG requires symmetry.
    let u = broadband(ndof);
    let mut v = broadband(ndof);
    v.iter_mut().enumerate().for_each(|(i, x)| *x *= ((i % 7) as f64 - 3.0) * 0.3);
    let au = pres.apply(&u);
    let av = pres.apply(&v);
    let (uav, vau) = (dot(&u, &av), dot(&v, &au));
    let sym_rel = (uav - vau).abs() / uav.abs().max(1e-300);
    println!("\n(1) symmetry  <u,Av>={uav:.6e}  <v,Au>={vau:.6e}  rel asym={sym_rel:.2e}  {}",
             if sym_rel < 1e-10 { "OK (symmetric)" } else { "*** ASYMMETRIC — CG invalid ***" });

    // (2) NULLSPACE probe: A·1 on the active dofs. Pure-Neumann ⇒ ~0 (constant nullspace); the
    // outflow Dirichlet should make it clearly NON-zero. ||A·1|| near 0 ⇒ near-singular.
    let mut ones = vec![0.0; ndof];
    for e in 0..mesh.n_elements() {
        if active[e] {
            for k in 0..nn {
                ones[e * nn + k] = 1.0;
            }
        }
    }
    let a_ones = pres.apply(&ones);
    // restrict the norm to active dofs (inactive identity makes A·1=1 there trivially)
    let mut a_ones_active = 0.0;
    for e in 0..mesh.n_elements() {
        if active[e] {
            for k in 0..nn {
                a_ones_active += a_ones[e * nn + k] * a_ones[e * nn + k];
            }
        }
    }
    let a_ones_active = a_ones_active.sqrt();
    println!("(2) nullspace ||A·1||_active = {a_ones_active:.6e}  (≫0 ⇒ outflow breaks the constant mode; ≈0 ⇒ near-singular)");

    // (3) CG residual HISTORY on a representative (range-consistent) RHS b = A·x_true.
    let x_true = broadband(ndof);
    let b = pres.apply(&x_true); // guaranteed in range
    let bn = norm(&b).max(1e-300);
    let maxit = std::env::var("SBM_MAXIT").ok().and_then(|v| v.parse().ok()).unwrap_or(8000usize);
    let mut x = vec![0.0; ndof];
    let mut r: Vec<f64> = b.clone();
    let mut pp = r.clone();
    let mut rs = dot(&r, &r);
    println!("\n(3) CG residual history (rel ||r||/||b||), b = A·x_true (in range):");
    let mut last = f64::INFINITY;
    for it in 0..maxit {
        let ap = pres.apply(&pp);
        let pap = dot(&pp, &ap);
        let al = rs / pap;
        for i in 0..ndof {
            x[i] += al * pp[i];
            r[i] -= al * ap[i];
        }
        let rn = norm(&r) / bn;
        if it == 0 || (it + 1) % 250 == 0 || rn < 1e-8 {
            println!("   iter {:5}: {:.3e}  (Δ/250it factor {:.3})", it + 1, rn, rn / last);
            last = rn;
        }
        if rn < 1e-8 {
            println!("   CONVERGED in {} iters ⇒ well-posed, just needed iterations", it + 1);
            break;
        }
        let be = dot(&r, &r) / rs;
        rs = dot(&r, &r);
        for i in 0..ndof {
            pp[i] = r[i] + be * pp[i];
        }
    }
    let final_err = {
        let e: Vec<f64> = x.iter().zip(&x_true).map(|(a, b)| a - b).collect();
        norm(&e) / norm(&x_true).max(1e-300)
    };
    println!("\nVERDICT: steady (even if slow) decay + symmetric + ||A·1||≫0  ⇒ ILL-CONDITIONED");
    println!("         (needs a preconditioner: MG-for-SBM). A plateau ⇒ operator/RHS bug.");
    println!("         final solution rel error vs x_true: {final_err:.2e}");

    // (4) EXPERIMENT: can the STANDARD full-mesh p-MG-PCG (same outer Neumann tags, ignoring the
    // active mask + surrogate) precondition the SBM operator? The dominant ill-conditioning is the
    // global smooth (channel-length) mode, shared by both operators — MG kills low-freq error
    // regardless of the local cylinder treatment. If iters collapse, this is the cheap fix.
    println!("\n(4) PCG with the standard full-mesh PMultigrid V-cycle as preconditioner:");
    let Some(mg) = PMultigrid::from_mesh(&mesh, alpha, 0.0, vec![3, 0, 2]) else {
        println!("   from_mesh returned None (non-tensor mesh) — skipped");
        return;
    };
    // Preconditioned CG: A = SBM apply, M^-1 = one standard V-cycle.
    let mut x = vec![0.0; ndof];
    let mut r = b.clone();
    let mut z = mg.precondition(&r);
    let mut pp = z.clone();
    let mut rz = dot(&r, &z);
    let mut pcg_iters = 0;
    for it in 0..maxit {
        let ap = pres.apply(&pp);
        let al = rz / dot(&pp, &ap);
        for i in 0..ndof {
            x[i] += al * pp[i];
            r[i] -= al * ap[i];
        }
        let rn = norm(&r) / bn;
        pcg_iters = it + 1;
        if it == 0 || (it + 1) % 10 == 0 || rn < 1e-8 {
            println!("   iter {:5}: {:.3e}", it + 1, rn);
        }
        if rn < 1e-8 {
            break;
        }
        z = mg.precondition(&r);
        let rz_new = dot(&r, &z);
        let be = rz_new / rz;
        for i in 0..ndof {
            pp[i] = z[i] + be * pp[i];
        }
        rz = rz_new;
    }
    let pcg_err = {
        let e: Vec<f64> = x.iter().zip(&x_true).map(|(a, b)| a - b).collect();
        norm(&e) / norm(&x_true).max(1e-300)
    };
    println!("   standard-MG-PCG: {pcg_iters} iters (vs {} unpreconditioned), rel err {pcg_err:.2e}",
             "861-ish");
}
