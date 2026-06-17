//! Stage 3f: validate **device refine on the masked block structure** — `device_refine` runs the
//! whole adapt-allocation pipeline on the GPU (mark → prefix-scan/compact → slot-alloc + activate,
//! fixed/graph-legal launch dims) with no host decision in the loop. The ORACLE is the host
//! `GpuAmrMesh::refine` (validated round-tripping in `amr-mesh-check`): we refine the SAME set of
//! active leaves on a host clone and a device clone and require the resulting ACTIVE SET
//! `(level,ix,iy)` to match exactly. Slot ids may differ (the mesh is defined by its active leaves).
//! Covers single-level AND multi-level (a second pass refines freshly-created children).
//! Run: cargo oxide run --bin amr-refine-device-check

use gale_gpu::amr_mesh::GpuAmrMesh;
use gale_gpu::operators::amr::{device_coarsen, device_refine};

/// Per-slot flag (length cap) with `val` on the active leaves at the given `(level,ix,iy)` cells.
fn flags_for(m: &GpuAmrMesh, cells: &[(u32, u32, u32)], val: i32) -> Vec<i32> {
    let mut f = vec![0i32; m.cap];
    for &(l, ix, iy) in cells {
        let s = m.slot_at(l, ix, iy).expect("cell must be an active leaf");
        f[s] = val;
    }
    f
}

/// Find ANY slot (active or internal) at `(level,ix,iy)` — for picking host parents to coarsen.
fn slot_any(m: &GpuAmrMesh, l: u32, ix: u32, iy: u32) -> usize {
    (0..m.cap)
        .find(|&s| m.level[s] == l && m.ix[s] == ix && m.iy[s] == iy && (m.active[s] == 1 || m.children[s * 4] != u32::MAX))
        .expect("slot must exist")
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let (nx, ny, l_max) = (4usize, 4usize, 2usize);
    println!("=== Stage 3f: device refine on masked structure vs host GpuAmrMesh::refine — {nx}×{ny}, l_max={l_max} ===");

    let mut hostm = GpuAmrMesh::new(nx, ny, l_max);
    let mut devm = GpuAmrMesh::new(nx, ny, l_max);

    // ---- Round 1: refine a set of level-0 base cells ----
    let r1: Vec<(u32, u32, u32)> = vec![(0, 1, 1), (0, 2, 2), (0, 2, 1), (0, 0, 3)];
    for &(l, ix, iy) in &r1 {
        let s = hostm.slot_at(l, ix, iy).unwrap();
        assert!(hostm.refine(s));
    }
    let f1 = flags_for(&devm, &r1, 1);
    device_refine(&mut devm, &f1)?;

    let (ha, da) = (hostm.active_cells(), devm.active_cells());
    let ok1 = ha == da;
    println!("  round 1 (single-level, {} refines): host {} active, device {} active — {}", r1.len(), ha.len(), da.len(), if ok1 { "MATCH" } else { "MISMATCH" });

    // ---- Round 2: refine freshly-created children (level 1) → multi-level ----
    // Children of base (1,1) live at level 1, ix∈{2,3}, iy∈{2,3}. Refine two of them.
    let r2: Vec<(u32, u32, u32)> = vec![(1, 2, 2), (1, 3, 3)];
    for &(l, ix, iy) in &r2 {
        let s = hostm.slot_at(l, ix, iy).unwrap();
        assert!(hostm.refine(s));
    }
    let f2 = flags_for(&devm, &r2, 1);
    device_refine(&mut devm, &f2)?;

    let (ha2, da2) = (hostm.active_cells(), devm.active_cells());
    let ok2 = ha2 == da2;
    println!("  round 2 (multi-level, {} refines): host {} active, device {} active — {}", r2.len(), ha2.len(), da2.len(), if ok2 { "MATCH" } else { "MISMATCH" });

    // Levels present (sanity: we actually reached level 2).
    let max_lvl = da2.iter().map(|c| c.0).max().unwrap_or(0);
    println!("  deepest active level reached: {max_lvl}");

    // ---- Round 3: coarsen — flag all 4 level-2 children of parent (1,2,2) and (1,3,3) back ----
    let coarsen_parents: Vec<(u32, u32, u32)> = vec![(1, 2, 2), (1, 3, 3)];
    // The 4 level-2 children of a level-1 parent (ix,iy) are at (2, 2ix+sx, 2iy+sy).
    let mut leaves: Vec<(u32, u32, u32)> = Vec::new();
    for &(_, ix, iy) in &coarsen_parents {
        for sy in 0..2 {
            for sx in 0..2 {
                leaves.push((2, 2 * ix + sx, 2 * iy + sy));
            }
        }
    }
    for &(_, ix, iy) in &coarsen_parents {
        let s = slot_any(&hostm, 1, ix, iy);
        assert!(hostm.coarsen(s));
    }
    let f3 = flags_for(&devm, &leaves, -1);
    device_coarsen(&mut devm, &f3)?;

    let (ha3, da3) = (hostm.active_cells(), devm.active_cells());
    let ok3 = ha3 == da3;
    println!("  round 3 (coarsen {} sibling groups): host {} active, device {} active — {}", coarsen_parents.len(), ha3.len(), da3.len(), if ok3 { "MATCH" } else { "MISMATCH" });

    if ok1 && ok2 && ok3 && max_lvl == 2 {
        println!("OK: device refine+coarsen reproduce the host masked-structure active set (single- and multi-level).");
        Ok(())
    } else {
        // Show first divergence for debugging.
        if !ok3 {
            for (i, (h, d)) in ha3.iter().zip(&da3).enumerate() {
                if h != d {
                    eprintln!("  first diff at idx {i}: host {h:?} device {d:?}");
                    break;
                }
            }
        }
        eprintln!("FAIL: device refine diverges from host.");
        std::process::exit(1);
    }
}
