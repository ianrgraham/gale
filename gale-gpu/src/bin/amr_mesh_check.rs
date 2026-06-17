//! Stage 3c (foundation) validation: the masked block-pool `GpuAmrMesh` refine/coarsen mechanics.
//! Checks that activating child slots from the free-list reproduces the correct active leaf set and
//! levels for a refinement pattern, supports a second level, and that refine→coarsen round-trips
//! exactly (active set + free-pool fully restored — no slot leaks). This validates the AGAL
//! gap_set/active-set machinery before connectivity + operator wiring.
//! Run: cargo oxide run --bin amr-mesh-check

use gale_gpu::amr_mesh::GpuAmrMesh;
use std::collections::BTreeSet;

fn main() {
    let (nx, ny, l_max) = (6usize, 6usize, 2usize);
    let mut m = GpuAmrMesh::new(nx, ny, l_max);
    let nbase = nx * ny;
    println!("=== Stage 3c: GpuAmrMesh masked block-pool refine/coarsen — {nx}×{ny} base, l_max={l_max} ===");

    let expect_uniform: BTreeSet<(u32, u32, u32)> =
        (0..ny).flat_map(|cy| (0..nx).map(move |cx| (0u32, cx as u32, cy as u32))).collect();
    let mut ok = m.active_cells().into_iter().collect::<BTreeSet<_>>() == expect_uniform
        && m.n_active() == nbase
        && m.n_free() == m.cap - nbase;
    println!("  initial uniform: active={} free={} [{}]", m.n_active(), m.n_free(), tag(ok));

    // Refine a set of base cells (one level).
    let refine_set: Vec<(usize, usize)> = vec![(1, 1), (2, 1), (4, 3), (0, 5)];
    for &(cx, cy) in &refine_set {
        assert!(m.refine(cx + cy * nx), "refine base ({cx},{cy}) failed");
    }
    let rset: BTreeSet<(usize, usize)> = refine_set.iter().copied().collect();
    let mut expect1: BTreeSet<(u32, u32, u32)> = BTreeSet::new();
    for cy in 0..ny {
        for cx in 0..nx {
            if rset.contains(&(cx, cy)) {
                for sy in 0..2 {
                    for sx in 0..2 {
                        expect1.insert((1, (cx * 2 + sx) as u32, (cy * 2 + sy) as u32));
                    }
                }
            } else {
                expect1.insert((0, cx as u32, cy as u32));
            }
        }
    }
    let got1 = m.active_cells().into_iter().collect::<BTreeSet<_>>();
    let pass1 = got1 == expect1 && m.n_active() == nbase + 3 * refine_set.len();
    ok &= pass1;
    println!("  after L1 refine of {} cells: active={} (expect {}) [{}]", refine_set.len(), m.n_active(), nbase + 3 * refine_set.len(), tag(pass1));

    // Refine one level-1 block to level 2 (find a child slot of the first refined base cell).
    let first = refine_set[0];
    let parent_slot = first.0 + first.1 * nx;
    let child0 = m.children[parent_slot * 4]; // first child slot
    assert!(m.refine(child0 as usize), "refine L1 child failed");
    let pass2 = m.n_active() == nbase + 3 * refine_set.len() + 3
        && m.active_cells().iter().any(|&(l, _, _)| l == 2);
    ok &= pass2;
    println!("  after one L2 refine: active={} (has level-2 cells: {}) [{}]", m.n_active(), m.active_cells().iter().any(|c| c.0 == 2), tag(pass2));

    // Coarsen back: the L2 block first, then all the L1 blocks → must return to uniform with the
    // free pool fully restored (no leaked slots).
    assert!(m.coarsen(child0 as usize), "coarsen L2 failed");
    for &(cx, cy) in &refine_set {
        assert!(m.coarsen(cx + cy * nx), "coarsen base ({cx},{cy}) failed");
    }
    let back = m.active_cells().into_iter().collect::<BTreeSet<_>>();
    let pass3 = back == expect_uniform && m.n_active() == nbase && m.n_free() == m.cap - nbase;
    ok &= pass3;
    println!("  round-trip coarsen→uniform: active={} free={} (cap-nbase={}) [{}]", m.n_active(), m.n_free(), m.cap - nbase, tag(pass3));

    if ok {
        println!("OK: masked block-pool refine/coarsen mechanics correct (active set + free-list round-trip).");
    } else {
        eprintln!("FAIL: GpuAmrMesh mechanics wrong.");
        std::process::exit(1);
    }
}

fn tag(b: bool) -> &'static str {
    if b { "ok" } else { "FAIL" }
}
