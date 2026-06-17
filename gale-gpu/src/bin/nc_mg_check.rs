//! Validate the **GPU p-multigrid-preconditioned CG** for the non-conforming SIPG operator
//! (`GpuPMultigridNc`) against GPU plain CG (`GpuPoissonNc`) on a 2:1-refined mesh: same solution,
//! dramatically fewer iterations. Both the Helmholtz (viscous, Dirichlet) and the deflated singular
//! pressure (pure-Neumann) cases — the two solves in the dual-splitting step. Mirrors the CPU
//! `multigrid_nc` tests on the device.
//! Run: cargo oxide run --bin nc-mg-check

use gale::dg::{Mesh2d, Poisson};
use gale_gpu::operators::multigrid_nc::GpuPMultigridNc;
use gale_gpu::operators::poisson_nc::GpuPoissonNc;

const P: usize = 4;

fn rel(a: &[f64], b: &[f64]) -> f64 {
    let d: f64 = a.iter().zip(b).map(|(x, y)| (x - y).powi(2)).sum();
    let n: f64 = b.iter().map(|x| x * x).sum::<f64>().max(1e-300);
    (d / n).sqrt()
}
fn demean(v: &[f64]) -> Vec<f64> {
    let m = v.iter().sum::<f64>() / v.len() as f64;
    v.iter().map(|x| x - m).collect()
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let (nx, ny) = (16usize, 16usize);
    let (xr, yr) = ([0.0, 1.0], [0.0, 1.0]);
    let alpha = 5.0;
    let refine: Vec<(usize, usize)> =
        (0..nx).flat_map(|cy| (0..nx).map(move |cx| (cx, cy))).filter(|(cx, cy)| (cx + cy) % 4 == 0).collect();
    let mesh = Mesh2d::cartesian_refined(P, nx, ny, xr, yr, &refine);
    let nn = mesh.refq.n_nodes();
    let ndof = mesh.n_elements() * nn;
    println!("=== GPU NC p-MG-PCG vs GPU plain CG — base {nx}×{ny} p={P}, {} refined ⇒ {} elems ({ndof} dof) ===", refine.len(), mesh.n_elements());

    // ---- (1) Helmholtz (Dirichlet, non-singular): the viscous solve ----
    {
        let reaction = 50.0;
        let op = Poisson::with_reaction(&mesh, alpha, reaction);
        let mut xt = vec![0.0; ndof];
        for (e, el) in mesh.elements.iter().enumerate() {
            for k in 0..nn {
                xt[e * nn + k] = (3.0 * el.geom.x[k]).sin() * (2.0 * el.geom.y[k]).cos();
            }
        }
        let b = op.apply(&xt);

        let h = GpuPoissonNc::new(&mesh, alpha)?;
        let (x_cg, it_cg) = h.solve(&b, reaction, &[], false, 1e-10, 20000)?;
        let mg = GpuPMultigridNc::new(P, nx, ny, xr, yr, &refine, alpha, reaction, vec![], false)?;
        let bd = mg.upload(&b)?;
        let mut xd = mg.upload(&vec![0.0; ndof])?;
        let it_mg = mg.solve_dev(&bd, &mut xd, 1e-10, 2000)?;
        let x_mg = mg.download(&xd)?;
        let r = rel(&x_mg, &x_cg);
        println!("  (1) Helmholtz: GPU CG {it_cg} iters | GPU p-MG-PCG {it_mg} iters ({:.0}× fewer) | rel {r:.2e}", it_cg as f64 / it_mg.max(1) as f64);
        if !(r < 1e-6 && it_mg * 4 < it_cg) {
            eprintln!("FAIL (Helmholtz): rel {r:.2e}, CG {it_cg}, MG {it_mg}");
            std::process::exit(1);
        }
    }

    // ---- (2) Singular pure-Neumann pressure (deflated): the projection solve ----
    {
        let tags = mesh.boundary_tags();
        let op = Poisson::with_bc(&mesh, alpha, 0.0, tags.clone());
        let mut xt = vec![0.0; ndof];
        for (e, el) in mesh.elements.iter().enumerate() {
            for k in 0..nn {
                xt[e * nn + k] = (std::f64::consts::PI * el.geom.x[k]).cos() * (std::f64::consts::PI * el.geom.y[k]).cos();
            }
        }
        let mut b = op.apply(&xt);
        let m = b.iter().sum::<f64>() / ndof as f64;
        b.iter_mut().for_each(|v| *v -= m);

        let h = GpuPoissonNc::new(&mesh, alpha)?;
        let (x_cg, it_cg) = h.solve(&b, 0.0, &tags, true, 1e-8, 30000)?;
        let mg = GpuPMultigridNc::new(P, nx, ny, xr, yr, &refine, alpha, 0.0, tags, true)?;
        let bd = mg.upload(&b)?;
        let mut xd = mg.upload(&vec![0.0; ndof])?;
        let it_mg = mg.solve_dev(&bd, &mut xd, 1e-8, 2000)?;
        let x_mg = mg.download(&xd)?;
        let r = rel(&demean(&x_mg), &demean(&x_cg));
        println!("  (2) singular pressure: GPU CG {it_cg} iters | GPU p-MG-PCG {it_mg} iters ({:.0}× fewer) | rel {r:.2e}", it_cg as f64 / it_mg.max(1) as f64);
        if !(r < 1e-5 && it_mg * 4 < it_cg) {
            eprintln!("FAIL (pressure): rel {r:.2e}, CG {it_cg}, MG {it_mg}");
            std::process::exit(1);
        }
    }

    println!("OK: GPU p-MG-PCG matches GPU plain CG and cuts iterations on both NC solves.");
    Ok(())
}
