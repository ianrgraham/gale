//! Stage 3c.2 validation: the `GpuAmrMesh` 2:1 face connectivity (`build_neighbors`) reproduces the
//! host `Mesh2d::cartesian_refined` connectivity. For a single-level refinement set (so no 2:1
//! balancing is triggered — that's 3b), build both representations and check that every active
//! block's per-face neighbour CLASSIFICATION (same-level / coarser / finer / boundary) matches the
//! host element's `Neighbor` arm on the geometrically-corresponding edge.
//! Run: cargo oxide run --bin amr-connectivity-check

use gale::dg::{Mesh2d, Neighbor};
use gale_gpu::amr_mesh::{FaceKind, GpuAmrMesh};

fn main() {
    let (p, nx, ny) = (3usize, 8usize, 8usize);
    let xr = [0.0, 1.0];
    let yr = [0.0, 1.0];
    // Single-level refine set: adjacent pair (fine-fine interface), an interior cell, and a
    // boundary cell (left edge) — exercises same/coarse/fine/boundary faces. No 2:1 balance needed.
    let refine: Vec<(usize, usize)> = vec![(1, 1), (2, 1), (4, 3), (0, 5)];
    println!("=== Stage 3c.2: GpuAmrMesh connectivity vs host cartesian_refined — {nx}×{ny}, refine {refine:?} ===");

    // GpuAmrMesh.
    let mut m = GpuAmrMesh::new(nx, ny, 1);
    for &(cx, cy) in &refine {
        assert!(m.refine(cx + cy * nx));
    }
    m.build_neighbors();

    // Host mesh.
    let mesh = Mesh2d::cartesian_refined(p, nx, ny, xr, yr, &refine);
    let nn = mesh.refq.n_nodes();

    let host_kind = |nb: &Neighbor| -> FaceKind {
        match nb {
            Neighbor::Interior { .. } => FaceKind::Same,
            Neighbor::CoarseToFine { .. } => FaceKind::Finer,
            Neighbor::FineToCoarse { .. } => FaceKind::Coarser,
            Neighbor::Boundary { .. } => FaceKind::Boundary,
        }
    };

    let mut checked = 0usize;
    let mut mism = 0usize;
    for el in mesh.elements.iter() {
        // Derive (level, ix, iy) from element geometry on [0,1]².
        let (mut xmin, mut xmax, mut ymin, mut ymax) = (f64::MAX, f64::MIN, f64::MAX, f64::MIN);
        for k in 0..nn {
            xmin = xmin.min(el.geom.x[k]);
            xmax = xmax.max(el.geom.x[k]);
            ymin = ymin.min(el.geom.y[k]);
            ymax = ymax.max(el.geom.y[k]);
        }
        let sizex = xmax - xmin;
        let level = (1.0 / (nx as f64 * sizex)).log2().round() as u32;
        let ix = (xmin / sizex).round() as u32;
        let iy = (ymin / sizex).round() as u32;
        let slot = m.slot_at(level, ix, iy).unwrap_or_else(|| panic!("no block for host elem at L{level} ({ix},{iy})"));
        let kinds = m.face_kinds(slot);

        for (edge, nb) in el.neighbors.iter().enumerate() {
            let f = &el.faces[edge];
            // Direction from the face's outward normal.
            let dir = if f.nx[0] < -0.5 {
                0
            } else if f.nx[0] > 0.5 {
                1
            } else if f.ny[0] < -0.5 {
                2
            } else {
                3
            };
            let hk = host_kind(nb);
            if hk != kinds[dir] {
                mism += 1;
                if mism <= 5 {
                    eprintln!("  mismatch elem L{level}({ix},{iy}) edge{edge} dir{dir}: host={hk:?} block={:?}", kinds[dir]);
                }
            }
            checked += 1;
        }
    }

    // Bridge primitive: recover the refine-set from the block structure (drives cartesian_refined).
    let derived: std::collections::BTreeSet<(usize, usize)> = m.refined_base_cells().into_iter().collect();
    let want: std::collections::BTreeSet<(usize, usize)> = refine.iter().copied().collect();
    let bridge_ok = derived == want;
    println!("  refined_base_cells recovered refine-set: {bridge_ok}");

    println!("  active blocks={} host elems={} faces checked={checked} mismatches={mism}", m.n_active(), mesh.n_elements());
    if m.n_active() == mesh.n_elements() && mism == 0 && bridge_ok {
        println!("OK: GpuAmrMesh 2:1 connectivity matches host cartesian_refined (all faces).");
    } else {
        eprintln!("FAIL: connectivity mismatch.");
        std::process::exit(1);
    }
}
