//! **Masked fixed-capacity quadtree block pool** — the Stage-3c foundation for GPU-resident AMR
//! (`docs/plan-stage3-gpu-resident-amr.md`). Instead of rebuilding mesh connectivity on each adapt
//! (illegal inside a CUDA conditional graph node), we pre-allocate `N_max` block slots once and
//! "remesh" by flipping activation bits: refine = pop 4 child slots from a free-list (`gap`) and
//! activate them while deactivating the parent; coarsen = the reverse. This is the AGAL
//! `id_set`/`gap_set` pattern. The authoritative mesh at any instant is the set of **active leaf
//! blocks** + their levels.
//!
//! This first increment is the host-side structure + the refine/coarsen *mechanics* (validated by
//! `amr-mesh-check`). The device upload of these flat arrays, the neighbour-connectivity build, and
//! turning refine/coarsen into device kernels inside the conditional graph come in the following
//! 3c/3f increments. Arrays are laid out exactly as they will be uploaded (flat, `u32`/`u8`).

const NONE: u32 = u32::MAX;

/// Masked quadtree block pool over an `nx×ny` base mesh, up to `l_max` refinement levels. A "block"
/// is a quadtree cell; `active=1` marks the current leaves (the live mesh). Slots are recycled
/// through `gap`. Capacity is the full quadtree to depth `l_max`: `Σ_{l=0..l_max} nx·ny·4^l`.
pub struct GpuAmrMesh {
    pub base_nx: usize,
    pub base_ny: usize,
    pub l_max: usize,
    pub cap: usize,
    /// Per-slot metadata (length `cap`). Coords `ix,iy` are the cell index at the block's own level
    /// (grid `nx·2^level × ny·2^level`).
    pub level: Vec<u32>,
    pub ix: Vec<u32>,
    pub iy: Vec<u32>,
    pub active: Vec<u8>,
    pub parent: Vec<u32>,
    /// `[cap*4]` child slot ids in order (sx + 2·sy); `NONE` if this block is a leaf.
    pub children: Vec<u32>,
    /// Face connectivity (filled by [`build_neighbors`](Self::build_neighbors)), `[cap*4]` in face
    /// order [−x, +x, −y, +y]. `block_nbr_kind`: 0=boundary, 1=same-level, 2=coarser, 3=finer.
    /// `block_nbr` = neighbour slot (same/coarse) or the FIRST fine slot; `block_nbr2` = the second
    /// fine slot (kind 3 only), else `NONE`.
    pub block_nbr: Vec<u32>,
    pub block_nbr2: Vec<u32>,
    pub block_nbr_kind: Vec<u8>,
    /// Free-slot stack (ids not currently in the tree).
    gap: Vec<u32>,
    n_active: usize,
}

/// Face-neighbour classification (for validation / readability).
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum FaceKind {
    Boundary,
    Same,
    Coarser,
    Finer,
}

impl GpuAmrMesh {
    /// Build the pool for an `nx×ny` base with `l_max` levels: slots `0..nx*ny` are the active level-0
    /// roots (row-major `ix + iy*nx`); all higher-level slots start free and inactive.
    pub fn new(base_nx: usize, base_ny: usize, l_max: usize) -> Self {
        let nbase = base_nx * base_ny;
        let mut cap = 0usize;
        for l in 0..=l_max {
            cap += nbase * 4usize.pow(l as u32);
        }
        let mut m = GpuAmrMesh {
            base_nx,
            base_ny,
            l_max,
            cap,
            level: vec![0; cap],
            ix: vec![0; cap],
            iy: vec![0; cap],
            active: vec![0; cap],
            parent: vec![NONE; cap],
            children: vec![NONE; cap * 4],
            block_nbr: vec![NONE; cap * 4],
            block_nbr2: vec![NONE; cap * 4],
            block_nbr_kind: vec![0; cap * 4],
            gap: Vec::with_capacity(cap - nbase),
            n_active: 0,
        };
        // Roots: slots 0..nbase, active, level 0.
        for cy in 0..base_ny {
            for cx in 0..base_nx {
                let s = cx + cy * base_nx;
                m.level[s] = 0;
                m.ix[s] = cx as u32;
                m.iy[s] = cy as u32;
                m.active[s] = 1;
            }
        }
        m.n_active = nbase;
        // Free pool: the rest, pushed so low ids are popped first (deterministic).
        for s in (nbase..cap).rev() {
            m.gap.push(s as u32);
        }
        m
    }

    pub fn n_active(&self) -> usize {
        self.n_active
    }
    pub fn n_free(&self) -> usize {
        self.gap.len()
    }

    /// Refine an active leaf block: claim 4 child slots, activate them (level+1, child coords), and
    /// deactivate the parent. No-op (returns false) if the slot isn't an active leaf, is already at
    /// `l_max`, or the free pool is exhausted.
    pub fn refine(&mut self, slot: usize) -> bool {
        if self.active[slot] == 0 || (self.level[slot] as usize) >= self.l_max || self.gap.len() < 4 {
            return false;
        }
        let (pl, pix, piy) = (self.level[slot], self.ix[slot], self.iy[slot]);
        for c in 0..4 {
            let (sx, sy) = (c % 2, c / 2);
            let child = self.gap.pop().unwrap() as usize;
            self.level[child] = pl + 1;
            self.ix[child] = pix * 2 + sx as u32;
            self.iy[child] = piy * 2 + sy as u32;
            self.active[child] = 1;
            self.parent[child] = slot as u32;
            self.children[child * 4..child * 4 + 4].fill(NONE);
            self.children[slot * 4 + c] = child as u32;
        }
        self.active[slot] = 0;
        self.n_active += 3; // -1 parent + 4 children
        true
    }

    /// Coarsen a block whose 4 children are all active leaves: deactivate the children (return their
    /// slots to the pool) and reactivate the parent. No-op if `slot` has no children or any child is
    /// itself refined (not a leaf).
    pub fn coarsen(&mut self, slot: usize) -> bool {
        if self.children[slot * 4] == NONE {
            return false;
        }
        for c in 0..4 {
            let child = self.children[slot * 4 + c] as usize;
            if self.active[child] == 0 {
                return false; // a grandchild exists ⇒ not coarsenable yet
            }
        }
        for c in 0..4 {
            let child = self.children[slot * 4 + c] as usize;
            self.active[child] = 0;
            self.gap.push(child as u32);
            self.children[slot * 4 + c] = NONE;
        }
        self.active[slot] = 1;
        self.n_active -= 3;
        true
    }

    /// Build the 2:1 face connectivity of the active leaves into `block_nbr*`. For each active block
    /// at `(L,ix,iy)` and face dir d∈{−x,+x,−y,+y}: the same-level adjacent cell is the conforming
    /// neighbour if active; else its parent `(L−1, …)` if active ⇒ this block is the FINE side of a
    /// 2:1 interface (neighbour COARSER); else the same-level cell is itself refined ⇒ the two
    /// `(L+1)` children on the shared face are the neighbours (FINER). Out-of-range ⇒ boundary.
    /// Assumes a 2:1-balanced active set (neighbour level differs by ≤1).
    pub fn build_neighbors(&mut self) {
        use std::collections::HashMap;
        let key = |l: u32, x: u32, y: u32| (l as u64) << 48 | (x as u64) << 24 | y as u64;
        let mut act: HashMap<u64, usize> = HashMap::with_capacity(self.n_active);
        for s in 0..self.cap {
            if self.active[s] == 1 {
                act.insert(key(self.level[s], self.ix[s], self.iy[s]), s);
            }
        }
        // dir order: -x, +x, -y, +y.
        const D: [(i64, i64); 4] = [(-1, 0), (1, 0), (0, -1), (0, 1)];
        for s in 0..self.cap {
            if self.active[s] == 0 {
                continue;
            }
            let (l, ix, iy) = (self.level[s], self.ix[s], self.iy[s]);
            let nxl = (self.base_nx as i64) << l;
            let nyl = (self.base_ny as i64) << l;
            for (d, &(dx, dy)) in D.iter().enumerate() {
                let (jx, jy) = (ix as i64 + dx, iy as i64 + dy);
                let o = s * 4 + d;
                if jx < 0 || jx >= nxl || jy < 0 || jy >= nyl {
                    self.block_nbr_kind[o] = 0; // boundary
                    continue;
                }
                let (jx, jy) = (jx as u32, jy as u32);
                if let Some(&nb) = act.get(&key(l, jx, jy)) {
                    self.block_nbr[o] = nb as u32;
                    self.block_nbr_kind[o] = 1; // same level
                } else if l > 0 && act.get(&key(l - 1, jx / 2, jy / 2)).is_some() {
                    self.block_nbr[o] = act[&key(l - 1, jx / 2, jy / 2)] as u32;
                    self.block_nbr_kind[o] = 2; // coarser (we are the fine side)
                } else {
                    // Neighbour is refined: the two (L+1) children on the shared face.
                    let (c0, c1) = fine_face_children(d, jx, jy);
                    let f0 = act.get(&key(l + 1, c0.0, c0.1)).copied();
                    let f1 = act.get(&key(l + 1, c1.0, c1.1)).copied();
                    match (f0, f1) {
                        (Some(a), Some(b)) => {
                            self.block_nbr[o] = a as u32;
                            self.block_nbr2[o] = b as u32;
                            self.block_nbr_kind[o] = 3; // finer
                        }
                        _ => self.block_nbr_kind[o] = 0, // inconsistent (unbalanced) ⇒ treat as boundary
                    }
                }
            }
        }
    }

    /// Face-kind classification of an active `slot` (dir order −x,+x,−y,+y) — read from the arrays
    /// built by [`build_neighbors`](Self::build_neighbors).
    pub fn face_kinds(&self, slot: usize) -> [FaceKind; 4] {
        std::array::from_fn(|d| match self.block_nbr_kind[slot * 4 + d] {
            1 => FaceKind::Same,
            2 => FaceKind::Coarser,
            3 => FaceKind::Finer,
            _ => FaceKind::Boundary,
        })
    }

    /// Base (level-0) cells that are currently refined (have children), as `(cx, cy)` — the
    /// refine-set that reconstructs a SINGLE-LEVEL mesh via `Mesh2d::cartesian_refined`. The
    /// host-AMR bridge for driving the existing solver from the masked structure while a fully
    /// device-resident non-conforming operator is built. (Single-level only; multi-level needs the
    /// general block→mesh builder.)
    pub fn refined_base_cells(&self) -> Vec<(usize, usize)> {
        let nbase = self.base_nx * self.base_ny;
        (0..nbase)
            .filter(|&s| self.children[s * 4] != NONE)
            .map(|s| (self.ix[s] as usize, self.iy[s] as usize))
            .collect()
    }

    /// Active slot id at `(level, ix, iy)`, or `None`.
    pub fn slot_at(&self, level: u32, ix: u32, iy: u32) -> Option<usize> {
        (0..self.cap).find(|&s| self.active[s] == 1 && self.level[s] == level && self.ix[s] == ix && self.iy[s] == iy)
    }

    /// Canonical full-quadtree position of cell `(level,ix,iy)`: `offset(level) + iy·(nx<<level) + ix`
    /// with `offset(L) = Σ_{l<L} nx·ny·4^l`. A dense bijection cell↔`0..cap` independent of slot ids —
    /// the index the device `pos2slot` scatter uses for O(1) neighbour lookup.
    pub fn pos_of(&self, level: u32, ix: u32, iy: u32) -> usize {
        let mut off = 0usize;
        for l in 0..level {
            off += self.base_nx * self.base_ny * 4usize.pow(l);
        }
        let nxl = self.base_nx << level;
        off + (iy as usize) * nxl + ix as usize
    }

    /// **Host 2:1-balance oracle (one pass):** the active leaves that must refine to restore 2:1
    /// balance — a leaf at `(L,ix,iy)` whose face-adjacent cell is refined two-or-more levels deeper
    /// (its same-level neighbour is internal AND that neighbour's child toward us is also internal, so
    /// an `L+2` leaf touches the shared face). Returns the slot ids. Iterating refine→this until empty
    /// gives a 2:1-balanced mesh. Independently written (simple coordinate logic) ⇒ a trustworthy
    /// oracle for the device balance kernel (host AMR proper is single-level, has no multi-level balance).
    pub fn balance_refine_flags(&self) -> Vec<usize> {
        use std::collections::HashMap;
        let key = |l: u32, x: u32, y: u32| (l as u64) << 48 | (x as u64) << 24 | y as u64;
        // Map every tree cell (active leaf OR internal) → has_children.
        let mut internal: HashMap<u64, bool> = HashMap::with_capacity(self.cap);
        for s in 0..self.cap {
            if self.active[s] == 1 || self.children[s * 4] != NONE {
                internal.insert(key(self.level[s], self.ix[s], self.iy[s]), self.children[s * 4] != NONE);
            }
        }
        const D: [(i64, i64); 4] = [(-1, 0), (1, 0), (0, -1), (0, 1)];
        let mut out = Vec::new();
        for s in 0..self.cap {
            if self.active[s] != 1 {
                continue;
            }
            let (l, ix, iy) = (self.level[s], self.ix[s], self.iy[s]);
            let (nxl, nyl) = ((self.base_nx as i64) << l, (self.base_ny as i64) << l);
            let mut unbalanced = false;
            for (d, &(dx, dy)) in D.iter().enumerate() {
                let (jx, jy) = (ix as i64 + dx, iy as i64 + dy);
                if jx < 0 || jx >= nxl || jy < 0 || jy >= nyl {
                    continue;
                }
                let (jx, jy) = (jx as u32, jy as u32);
                // same-level neighbour internal?
                if internal.get(&key(l, jx, jy)).copied() == Some(true) {
                    // its child toward us (the shared face) — also internal ⇒ L+2 leaf adjacent.
                    let (c0, c1) = fine_face_children(d, jx, jy);
                    if internal.get(&key(l + 1, c0.0, c0.1)).copied() == Some(true)
                        || internal.get(&key(l + 1, c1.0, c1.1)).copied() == Some(true)
                    {
                        unbalanced = true;
                    }
                }
            }
            if unbalanced {
                out.push(s);
            }
        }
        out
    }

    /// The active leaf blocks as `(level, ix, iy)`, sorted — the authoritative current mesh, for
    /// validation / connectivity build.
    pub fn active_cells(&self) -> Vec<(u32, u32, u32)> {
        let mut v: Vec<(u32, u32, u32)> = (0..self.cap)
            .filter(|&s| self.active[s] == 1)
            .map(|s| (self.level[s], self.ix[s], self.iy[s]))
            .collect();
        v.sort_unstable();
        v
    }
}

/// The two level-(L+1) child cells of a refined same-level neighbour cell `(jx,jy)` that lie on the
/// face shared with the querying block, for face dir d∈{−x,+x,−y,+y}. (The neighbour presents its
/// opposite face: dir −x ⇒ neighbour's +x children (sx=1); +x ⇒ sx=0; −y ⇒ sy=1; +y ⇒ sy=0.)
fn fine_face_children(d: usize, jx: u32, jy: u32) -> ((u32, u32), (u32, u32)) {
    let (bx, by) = (2 * jx, 2 * jy);
    match d {
        0 => ((bx + 1, by), (bx + 1, by + 1)), // -x: neighbour +x face, vary sy
        1 => ((bx, by), (bx, by + 1)),         // +x: neighbour -x face, vary sy
        2 => ((bx, by + 1), (bx + 1, by + 1)), // -y: neighbour +y face, vary sx
        _ => ((bx, by), (bx + 1, by)),         // +y: neighbour -y face, vary sx
    }
}
