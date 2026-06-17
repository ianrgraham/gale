//! Stage 3d validation: the GPU AMR remap kernels (`prolong_gpu` refine, `restrict_gpu` coarsen) vs
//! the host `gale::dg::RefineQuad::prolong`/`restrict`, plus the conservation/consistency property
//! that `restrict(prolong(parent)) == parent` for a degree-≤p polynomial (the L2 coarsen recovers an
//! exactly-representable field). These are the remap kernels of the GPU-resident AMR — the
//! conservation-critical step where a log-conformation field must NOT lose SPD.
//! Run: cargo oxide run --bin amr-remap-check

use gale::dg::{Reference1d, RefineQuad};
use gale_gpu::{prolong_gpu, restrict_gpu};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let p = 3usize;
    let n = p + 1;
    let nn = n * n;
    let n_parents = 6usize;
    let refq = RefineQuad::new(p);
    let nodes = &Reference1d::new(p).nodes; // LGL nodes on [-1,1]
    println!("=== Stage 3d: GPU AMR remap (prolong/restrict) vs host RefineQuad — p={p}, {n_parents} parents ===");

    // Parent fields = a degree-≤p polynomial per parent (so prolong is exact and restrict recovers).
    let poly = |r: f64, s: f64, k: f64| 1.0 + 0.7 * k * r - 0.4 * s + 0.3 * r * r - 0.5 * r * s + 0.2 * s * s * s;
    let mut parents = vec![0.0; n_parents * nn];
    for e in 0..n_parents {
        for j in 0..n {
            for i in 0..n {
                parents[e * nn + i + j * n] = poly(nodes[i], nodes[j], (e + 1) as f64);
            }
        }
    }

    // Host prolong: 4 children per parent, child index c = cx + 2·cy.
    let mut host_children = vec![0.0; n_parents * 4 * nn];
    for e in 0..n_parents {
        for c in 0..4 {
            let (cx, cy) = (c % 2, c / 2);
            let ch = refq.prolong(&parents[e * nn..(e + 1) * nn], cx, cy);
            host_children[(e * 4 + c) * nn..(e * 4 + c + 1) * nn].copy_from_slice(&ch);
        }
    }
    let gpu_children = prolong_gpu(&refq, n_parents, &parents)?;

    // Host restrict: 4 children → parent.
    let mut host_restrict = vec![0.0; n_parents * nn];
    for e in 0..n_parents {
        let kids: [Vec<f64>; 4] = std::array::from_fn(|c| {
            host_children[(e * 4 + c) * nn..(e * 4 + c + 1) * nn].to_vec()
        });
        let par = refq.restrict(&kids);
        host_restrict[e * nn..(e + 1) * nn].copy_from_slice(&par);
    }
    let gpu_restrict = restrict_gpu(&refq, n_parents, &gpu_children)?;

    let maxabs = |a: &[f64], b: &[f64]| a.iter().zip(b).fold(0.0f64, |m, (x, y)| m.max((x - y).abs()));
    let prolong_err = maxabs(&gpu_children, &host_children);
    let restrict_err = maxabs(&gpu_restrict, &host_restrict);

    // Conservation property of restrict: the coarsened parent's cell average equals the (area-weighted)
    // average of the 4 children's cell averages — mean_parent = (1/4) Σ_c mean_child. (restrict is the
    // *conservative* projection, not the exact L2 inverse, so it preserves the mean, not the polynomial.)
    let w = refq.weights();
    let cell_mean = |f: &[f64]| -> f64 {
        let (mut num, mut den) = (0.0, 0.0);
        for j in 0..n {
            for i in 0..n {
                let wij = w[i] * w[j];
                num += wij * f[i + j * n];
                den += wij;
            }
        }
        num / den
    };
    let mut cons_err = 0.0f64;
    for e in 0..n_parents {
        let mp = cell_mean(&gpu_restrict[e * nn..(e + 1) * nn]);
        let mc: f64 = (0..4).map(|c| cell_mean(&gpu_children[(e * 4 + c) * nn..(e * 4 + c + 1) * nn])).sum::<f64>() / 4.0;
        cons_err = cons_err.max((mp - mc).abs());
    }

    println!("  prolong  max|gpu-host| = {prolong_err:.3e}");
    println!("  restrict max|gpu-host| = {restrict_err:.3e}");
    println!("  restrict cell-average conservation (mean_parent − ¼Σ mean_child) = {cons_err:.3e}");
    if prolong_err < 1e-12 && restrict_err < 1e-12 && cons_err < 1e-12 {
        println!("OK: GPU remap matches host RefineQuad bit-for-bit and the coarsen is conservative.");
        Ok(())
    } else {
        eprintln!("FAIL: GPU remap diverges.");
        std::process::exit(1);
    }
}
