//! Stage 3c.3a validation: the device-resident non-conforming solve `GpuPoissonNc::solve_dev`
//! (rhs/x on the device, no field transfer) vs the host-slice `GpuPoissonNc::solve`, on a 2:1
//! refined (non-conforming) mesh. Same operator + CG, so they must agree bit-for-bit. This is the
//! device-resident NC operator that lets the masked AMR structure drive a resident step (the missing
//! piece behind Stage 3c.3). Helmholtz (reaction>0, non-singular) and deflated pure-Neumann pressure.
//! Run: cargo oxide run --bin amr-nc-resident-check

use gale::dg::Mesh2d;
use gale_gpu::operators::poisson_nc::GpuPoissonNc;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let (p, alpha) = (3usize, 5.0);
    let (nx, ny) = (8usize, 8usize);
    let xr = [0.0, 1.0];
    let yr = [0.0, 1.0];
    let (tol, maxit) = (1e-10, 5000);
    let refine: Vec<(usize, usize)> = vec![(2, 2), (3, 2), (5, 4), (1, 6)];
    let mesh = Mesh2d::cartesian_refined(p, nx, ny, xr, yr, &refine);
    let nn = mesh.refq.n_nodes();
    let ndof = mesh.n_elements() * nn;
    let tags = mesh.boundary_tags();
    println!("=== Stage 3c.3a: GpuPoissonNc::solve_dev vs host solve — refined {nx}×{ny} p={p}, {} elems ===", mesh.n_elements());

    // A smooth-ish RHS field.
    let two_pi = std::f64::consts::TAU;
    let mut b = vec![0.0; ndof];
    for (e, el) in mesh.elements.iter().enumerate() {
        for k in 0..nn {
            b[e * nn + k] = (two_pi * el.geom.x[k]).cos() * (two_pi * el.geom.y[k]).sin();
        }
    }

    let h = GpuPoissonNc::new(&mesh, alpha)?;
    let rel = |a: &[f64], c: &[f64]| -> f64 {
        let d: f64 = a.iter().zip(c).map(|(x, y)| (x - y).powi(2)).sum();
        let n: f64 = c.iter().map(|x| x * x).sum::<f64>().max(1e-300);
        (d / n).sqrt()
    };
    let mut ok = true;

    // Case 1: Helmholtz (reaction λ, all-Dirichlet ⇒ no Neumann tags, no deflation).
    {
        let lambda = 1.0 / (0.05 * 2e-3);
        let (x_host, it_h) = h.solve(&b, lambda, &[], false, tol, maxit)?;
        let rhs_d = h.upload(&b)?;
        let mut out = h.alloc()?;
        let it_d = h.solve_dev(&rhs_d, None, &mut out, lambda, &[], false, tol, maxit)?;
        let x_dev = h.download(&out)?;
        let r = rel(&x_dev, &x_host);
        let pass = r < 1e-12;
        ok &= pass;
        println!("  Helmholtz: host {it_h} it / dev {it_d} it, rel = {r:.3e} [{}]", tag(pass));
    }

    // Case 2: deflated pure-Neumann pressure (reaction 0, all tags Neumann, deflate). Compare
    // mean-removed (solution defined up to a constant). Needs a compatible (zero-mean) RHS.
    {
        let mean = b.iter().sum::<f64>() / ndof as f64;
        let bc: Vec<f64> = b.iter().map(|v| v - mean).collect();
        let (x_host, it_h) = h.solve(&bc, 0.0, &tags, true, tol, maxit)?;
        let rhs_d = h.upload(&bc)?;
        let mut out = h.alloc()?;
        let it_d = h.solve_dev(&rhs_d, None, &mut out, 0.0, &tags, true, tol, maxit)?;
        let x_dev = h.download(&out)?;
        let mr = |v: &[f64]| -> Vec<f64> {
            let m = v.iter().sum::<f64>() / v.len() as f64;
            v.iter().map(|x| x - m).collect()
        };
        let r = rel(&mr(&x_dev), &mr(&x_host));
        let pass = r < 1e-10;
        ok &= pass;
        println!("  pressure (deflated): host {it_h} it / dev {it_d} it, rel = {r:.3e} [{}]", tag(pass));
    }

    if ok {
        println!("OK: device-resident NC solve matches the host-slice NC solve (no field transfer).");
        Ok(())
    } else {
        eprintln!("FAIL: solve_dev diverges from host solve.");
        std::process::exit(1);
    }
}

fn tag(b: bool) -> &'static str {
    if b { "ok" } else { "FAIL" }
}
