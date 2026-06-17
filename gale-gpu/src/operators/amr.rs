//! GPU-resident **adaptive mesh refinement** kernels — Stage 3 of the device-residency plan
//! (`docs/plan-stage3-gpu-resident-amr.md`). The goal is to run the whole adapt cycle (indicator →
//! flag → 2:1 balance → slot-alloc → metric write → mortar rebuild → remap) on the GPU with no host
//! synchronization, using a masked fixed-max-level representation, so an adaptive sim self-drives on
//! the device. Each device kernel is validated against the existing HOST `gale::dg::amr` oracle.
//!
//! Stage 3a (here): the **Persson–Peraire smoothness indicator** `Se` per element — the modal-decay
//! refinement sensor. One block per element; matches `gale::dg::SmoothnessIndicator::indicator`.

use cuda_core::{CudaContext, CudaStream, DeviceBuffer, LaunchConfig};
use cuda_device::atomic::{AtomicOrdering, DeviceAtomicU32};
use cuda_device::{kernel, thread, DisjointSlice, SharedArray};
use cuda_host::cuda_module;
use gale::dg::{Mesh2d, RefineQuad, SmoothnessIndicator};
use std::sync::Arc;

const NN_MAX: usize = 81;
const CN_MAX: usize = 4 * NN_MAX; // 4 children in shared for the conservative restrict
const CBLK: usize = 256; // block size for the block-aggregated stream compaction (power of two)

#[cuda_module]
mod kernels {
    use super::*;

    /// Per-element Persson–Peraire smoothness indicator `Se = (Σ top-mode energy)/(Σ energy)` of a
    /// scalar field. One block per element, one thread per node `m=(a,b)` computes the modal
    /// coefficient `ĉ[a,b] = Σ_ij V⁻¹[a,i] V⁻¹[b,j] u[i,j]` and its Parseval energy `ĉ²γ_aγ_b`;
    /// the block reduces total vs highest-mode (`a==p || b==p`) energy. Writes `Se` to every node
    /// slot of the element (host reads stride `nn`). Mirrors `SmoothnessIndicator::indicator`.
    #[kernel]
    pub fn smoothness_se(
        field: &[f64], vinv: &[f64], gamma: &[f64], n1: u32, mut out: DisjointSlice<f64>,
    ) {
        static mut FS: SharedArray<f64, NN_MAX> = SharedArray::UNINIT;
        static mut E: SharedArray<f64, NN_MAX> = SharedArray::UNINIT;
        static mut TE: SharedArray<f64, NN_MAX> = SharedArray::UNINIT;
        let n = n1 as usize;
        let nn = n * n;
        let e = thread::blockIdx_x() as usize;
        let m = thread::threadIdx_x() as usize;
        let g = e * nn + m;
        unsafe {
            FS[m] = field[g];
        }
        thread::sync_threads();
        let a = m % n;
        let bb = m / n;
        // ĉ[a,bb] = Σ_j V⁻¹[bb,j] (Σ_i V⁻¹[a,i] u[i,j]).
        let mut c = 0.0f64;
        let mut j = 0usize;
        while j < n {
            let mut row = 0.0f64;
            let mut i = 0usize;
            while i < n {
                row += vinv[a * n + i] * unsafe { FS[i + j * n] };
                i += 1;
            }
            c += vinv[bb * n + j] * row;
            j += 1;
        }
        let energy = c * c * gamma[a] * gamma[bb];
        let is_top = a == n - 1 || bb == n - 1;
        unsafe {
            E[m] = energy;
            TE[m] = if is_top { energy } else { 0.0 };
        }
        thread::sync_threads();
        // Redundant per-thread reduction over the element's modes (nn ≤ NN_MAX, small).
        let (mut e_total, mut e_top) = (0.0f64, 0.0f64);
        let mut k = 0usize;
        while k < nn {
            e_total += unsafe { E[k] };
            e_top += unsafe { TE[k] };
            k += 1;
        }
        let se = if e_total > 0.0 { e_top / e_total } else { 0.0 };
        if let Some(o) = out.get_mut(thread::index_1d()) {
            *o = se;
        }
    }

    /// **Adapt flag** per element from the smoothness indicator: `+1` refine (`Se>refine_thr` and not
    /// at `l_max`), `−1` coarsen (`Se<coarsen_thr` and not at level 0), else `0`. One thread per
    /// element — the on-device adapt DECISION (no host readback of the field). Sibling-agreement for
    /// coarsen + 2:1 balance are applied afterwards on the block structure.
    #[kernel]
    pub fn amr_flag(
        se: &[f64], level: &[u32], refine_thr: f64, coarsen_thr: f64, l_max: u32, ne: u32,
        mut flag: DisjointSlice<i32>,
    ) {
        let idx = thread::index_1d();
        let i = idx.get();
        if i >= ne as usize {
            return;
        }
        let f = if se[i] > refine_thr && level[i] < l_max {
            1
        } else if se[i] < coarsen_thr && level[i] > 0 {
            -1
        } else {
            0
        };
        if let Some(o) = flag.get_mut(idx) {
            *o = f;
        }
    }

    /// **Prolong** (refine): parent nodal field → its 4 children, `child[i,j] = Σ_ab pr[i,a] ps[j,b]
    /// parent[a,b]` with `pr=axis(cx)`, `ps=axis(cy)`. Grid = (n_parents·4) blocks, one per child;
    /// `blockIdx = parent·4 + child`, `child = cx + 2·cy`. Output `out[blockIdx·nn + node]`.
    /// Exact for degree ≤ p — matches `RefineQuad::prolong`.
    #[kernel]
    pub fn prolong2d(
        parent_f: &[f64], p_left: &[f64], p_right: &[f64], n1: u32, mut out: DisjointSlice<f64>,
    ) {
        static mut PF: SharedArray<f64, NN_MAX> = SharedArray::UNINIT;
        let n = n1 as usize;
        let nn = n * n;
        let blk = thread::blockIdx_x() as usize;
        let parent = blk / 4;
        let child = blk % 4;
        let (cx, cy) = (child % 2, child / 2);
        let m = thread::threadIdx_x() as usize;
        unsafe {
            PF[m] = parent_f[parent * nn + m];
        }
        thread::sync_threads();
        let i = m % n;
        let j = m / n;
        let mut s = 0.0f64;
        let mut b = 0usize;
        while b < n {
            let mut row = 0.0f64;
            let mut a = 0usize;
            while a < n {
                let pra = if cx == 0 { p_left[i * n + a] } else { p_right[i * n + a] };
                row += pra * unsafe { PF[a + b * n] };
                a += 1;
            }
            let psb = if cy == 0 { p_left[j * n + b] } else { p_right[j * n + b] };
            s += psb * row;
            b += 1;
        }
        if let Some(o) = out.get_mut(thread::index_1d()) {
            *o = s;
        }
    }

    /// **Restrict** (coarsen): 4 children → parent, the conservative L2 projection
    /// `parent[i,j] = Σ_children 0.25·(Σ_ab pr[a,i] ps[b,j] w_a w_b uc[a,b]) / (w_i w_j)`. Grid =
    /// (n_parents) blocks, one per parent; children laid out as `children[(parent·4+child)·nn+node]`
    /// (the `prolong2d` output layout). Preserves the cell average. Matches `RefineQuad::restrict`.
    #[kernel]
    pub fn restrict2d(
        children: &[f64], p_left: &[f64], p_right: &[f64], w: &[f64], n1: u32, mut out: DisjointSlice<f64>,
    ) {
        static mut CH: SharedArray<f64, CN_MAX> = SharedArray::UNINIT;
        let n = n1 as usize;
        let nn = n * n;
        let parent = thread::blockIdx_x() as usize;
        let m = thread::threadIdx_x() as usize;
        let mut c = 0usize;
        while c < 4 {
            unsafe {
                CH[c * nn + m] = children[(parent * 4 + c) * nn + m];
            }
            c += 1;
        }
        thread::sync_threads();
        let i = m % n;
        let j = m / n;
        let mut acc = 0.0f64;
        let mut cy = 0usize;
        while cy < 2 {
            let mut cx = 0usize;
            while cx < 2 {
                let cc = cx + 2 * cy;
                let mut s = 0.0f64;
                let mut b = 0usize;
                while b < n {
                    let wb = w[b];
                    let psb = if cy == 0 { p_left[b * n + j] } else { p_right[b * n + j] };
                    let mut a = 0usize;
                    while a < n {
                        let pra = if cx == 0 { p_left[a * n + i] } else { p_right[a * n + i] };
                        s += pra * psb * w[a] * wb * unsafe { CH[cc * nn + a + b * n] };
                        a += 1;
                    }
                    b += 1;
                }
                acc += 0.25 * s / (w[i] * w[j]);
                cx += 1;
            }
            cy += 1;
        }
        if let Some(o) = out.get_mut(thread::index_1d()) {
            *o = acc;
        }
    }

    /// **Mark** pass for device refine: per slot, `is_refine = active & flag==1 & level<l_max` and
    /// `is_free = !active & is-leaf(children==NONE) & non-root(s>=nbase)` — the inputs the prefix-scan
    /// compaction consumes. One thread per slot. `NONE = u32::MAX`.
    #[kernel]
    pub fn amr_mark(
        active: &[u8], children: &[u32], level: &[u32], flag: &[i32], nbase: u32, l_max: u32, cap: u32,
        mut is_refine: DisjointSlice<i32>, mut is_free: DisjointSlice<i32>,
    ) {
        let idx = thread::index_1d();
        let s = idx.get();
        if s >= cap as usize {
            return;
        }
        let refine = if active[s] == 1 && flag[s] == 1 && (level[s] as u32) < l_max { 1 } else { 0 };
        let free = if active[s] == 0 && children[s * 4] == u32::MAX && (s as u32) >= nbase { 1 } else { 0 };
        if let Some(o) = is_refine.get_mut(thread::index_1d()) {
            *o = refine;
        }
        if let Some(o) = is_free.get_mut(thread::index_1d()) {
            *o = free;
        }
    }

    /// **Block-aggregated stream compaction** (fully parallel): compact the slots where `pred[s]!=0`
    /// into a dense `list`, and accumulate the total into `counts[counter_idx]` (which must be
    /// pre-zeroed). Each block of `CBLK` threads does a shared-memory inclusive scan of its predicates
    /// → per-thread local rank + block total; thread 0 does ONE `atomicAdd` to claim a contiguous base
    /// offset; matching threads scatter `list[base+rank] = s` in parallel. Replaces the old O(cap)
    /// single-thread scan — `cap/CBLK` global atomics instead of one serial thread. Slot ORDER within
    /// the list is block-interleaved (not ascending), which is fine: the adapt result (active cell set
    /// + per-cell field) is invariant to which free slot a child is assigned. `counts[counter_idx]`
    /// ends at the total (read by the apply kernels as the entry count).
    #[kernel]
    pub fn amr_compact_blocked(pred: &[i32], counts: &[u32], counter_idx: u32, cap: u32, mut list: DisjointSlice<u32>) {
        static mut SC: SharedArray<u32, CBLK> = SharedArray::UNINIT;
        static mut BASE: SharedArray<u32, 1> = SharedArray::UNINIT;
        let tid = thread::threadIdx_x() as usize;
        let gid = thread::blockIdx_x() as usize * CBLK + tid;
        let f: u32 = if gid < cap as usize && pred[gid] != 0 { 1 } else { 0 };
        unsafe {
            SC[tid] = f;
        }
        thread::sync_threads();
        // Hillis–Steele inclusive scan over the block (CBLK power of two).
        let mut offset = 1usize;
        while offset < CBLK {
            let add = if tid >= offset { unsafe { SC[tid - offset] } } else { 0 };
            thread::sync_threads();
            unsafe {
                SC[tid] += add;
            }
            thread::sync_threads();
            offset <<= 1;
        }
        let total = unsafe { SC[CBLK - 1] };
        if tid == 0 && total > 0 {
            // SAFETY: single global counter; atomicAdd returns a unique base per block.
            let counter = unsafe { &*(counts.as_ptr().add(counter_idx as usize) as *const DeviceAtomicU32) };
            unsafe {
                BASE[0] = counter.fetch_add(total, AtomicOrdering::Relaxed);
            }
        }
        thread::sync_threads();
        if f == 1 {
            let rank = unsafe { SC[tid] } - 1; // exclusive rank = inclusive − own contribution
            let pos = unsafe { BASE[0] } + rank;
            // SAFETY: base unique per block, rank unique within block ⇒ pos globally unique.
            unsafe {
                *list.get_unchecked_mut(pos as usize) = gid as u32;
            }
        }
    }

    /// **Apply refine** on the masked structure: one thread per refine entry `k` (grid over-provisioned
    /// to `cap`, guarded `k<n_refine` and `4·k+3<n_free`). Claims the 4 free slots `free_list[4k..4k+4]`
    /// as the children of `refine_list[k]` (level+1, child coords `sx+2·sy`), activates them, sets
    /// parent/children links, and deactivates the parent. Scatter writes are to provably-disjoint slots
    /// (distinct refine entries, uniquely partitioned free slots) ⇒ `get_unchecked_mut` is sound.
    #[kernel]
    pub fn amr_apply_refine(
        refine_list: &[u32], free_list: &[u32], counts: &[u32],
        mut active: DisjointSlice<u8>, mut level: DisjointSlice<u32>, mut ix: DisjointSlice<u32>,
        mut iy: DisjointSlice<u32>, mut parent: DisjointSlice<u32>, mut children: DisjointSlice<u32>,
    ) {
        let idx = thread::index_1d();
        let k = idx.get();
        let n_refine = counts[0] as usize;
        let n_free = counts[1] as usize;
        if k >= n_refine || 4 * k + 3 >= n_free {
            return;
        }
        let r = refine_list[k] as usize;
        let pl = unsafe { *level.get_unchecked_mut(r) };
        let pix = unsafe { *ix.get_unchecked_mut(r) };
        let piy = unsafe { *iy.get_unchecked_mut(r) };
        let mut c = 0usize;
        while c < 4 {
            let child = free_list[4 * k + c] as usize;
            let sx = (c % 2) as u32;
            let sy = (c / 2) as u32;
            unsafe {
                *level.get_unchecked_mut(child) = pl + 1;
                *ix.get_unchecked_mut(child) = pix * 2 + sx;
                *iy.get_unchecked_mut(child) = piy * 2 + sy;
                *active.get_unchecked_mut(child) = 1;
                *parent.get_unchecked_mut(child) = r as u32;
                // child becomes a leaf
                *children.get_unchecked_mut(child * 4) = u32::MAX;
                *children.get_unchecked_mut(child * 4 + 1) = u32::MAX;
                *children.get_unchecked_mut(child * 4 + 2) = u32::MAX;
                *children.get_unchecked_mut(child * 4 + 3) = u32::MAX;
                *children.get_unchecked_mut(r * 4 + c) = child as u32;
            }
            c += 1;
        }
        unsafe { *active.get_unchecked_mut(r) = 0; }
    }

    /// **Mark coarsen**: per slot `p`, `is_coarsen = p has children AND all 4 children are active leaves
    /// AND all 4 are flagged −1` (sibling agreement). One thread per slot. The unit that coarsens is the
    /// PARENT, gated on its whole sibling group agreeing — mirrors the host `GpuAmrMesh::coarsen`.
    #[kernel]
    pub fn amr_mark_coarsen(
        active: &[u8], children: &[u32], flag: &[i32], cap: u32, mut is_coarsen: DisjointSlice<i32>,
    ) {
        let idx = thread::index_1d();
        let p = idx.get();
        if p >= cap as usize {
            return;
        }
        let mut ok = children[p * 4] != u32::MAX;
        let mut c = 0usize;
        while c < 4 {
            let ch = children[p * 4 + c];
            if ch == u32::MAX || active[ch as usize] != 1 || flag[ch as usize] != -1 {
                ok = false;
            }
            c += 1;
        }
        if let Some(o) = is_coarsen.get_mut(thread::index_1d()) {
            *o = if ok { 1 } else { 0 };
        }
    }

    /// **Apply coarsen**: one thread per coarsen entry `k` (grid over-provisioned to `cap`, guarded).
    /// Reactivates parent `coarsen_list[k]`, deactivates its 4 children and clears their leaf status
    /// (the children become free for the next `amr_mark`: `active=0 & children==NONE & non-root`), and
    /// detaches the parent's child links. Distinct parents ⇒ disjoint child groups ⇒ scatter is sound.
    #[kernel]
    pub fn amr_apply_coarsen(
        coarsen_list: &[u32], count: &[u32],
        mut active: DisjointSlice<u8>, mut children: DisjointSlice<u32>,
    ) {
        let idx = thread::index_1d();
        let k = idx.get();
        if k >= count[0] as usize {
            return;
        }
        let p = coarsen_list[k] as usize;
        let mut c = 0usize;
        while c < 4 {
            let child = unsafe { *children.get_unchecked_mut(p * 4 + c) } as usize;
            unsafe {
                *active.get_unchecked_mut(child) = 0;
                // child is already a leaf (children==NONE); leave its links so it reads as free.
                *children.get_unchecked_mut(p * 4 + c) = u32::MAX;
            }
            c += 1;
        }
        unsafe { *active.get_unchecked_mut(p) = 1; }
    }

    /// **Build the positional index** `pos2slot`: for every tree cell (active leaf OR internal node)
    /// scatter its slot id to its canonical full-quadtree position `offsets[level] + iy·(nx<<level) +
    /// ix`. `pos2slot` must be pre-filled with `NONE`. One thread per slot; positions are unique per
    /// cell ⇒ scatter is race-free. Gives O(1) device neighbour lookup without a hash.
    #[kernel]
    pub fn amr_build_pos2slot(
        active: &[u8], children: &[u32], level: &[u32], ix: &[u32], iy: &[u32], offsets: &[u32],
        base_nx: u32, cap: u32, mut pos2slot: DisjointSlice<u32>,
    ) {
        let idx = thread::index_1d();
        let s = idx.get();
        if s >= cap as usize {
            return;
        }
        let in_tree = active[s] == 1 || children[s * 4] != u32::MAX;
        if !in_tree {
            return;
        }
        let l = level[s];
        let nxl = base_nx << l;
        let pos = offsets[l as usize] as usize + (iy[s] as usize) * (nxl as usize) + ix[s] as usize;
        unsafe { *pos2slot.get_unchecked_mut(pos) = s as u32; }
    }

    /// **2:1-balance flag pass** on device: for each active leaf, set `flag=1` if a face-adjacent cell
    /// is refined ≥2 levels deeper (same-level neighbour internal AND its child toward us internal ⇒ an
    /// `L+2` leaf touches the face). Uses `pos2slot` for O(1) neighbour/child lookup. One thread per
    /// slot. Mirrors host `GpuAmrMesh::balance_refine_flags`. Iterate refine→this until no flags.
    #[kernel]
    pub fn amr_balance_flag(
        active: &[u8], children: &[u32], level: &[u32], ix: &[u32], iy: &[u32], pos2slot: &[u32],
        offsets: &[u32], base_nx: u32, base_ny: u32, cap: u32, mut flag: DisjointSlice<i32>,
    ) {
        let idx = thread::index_1d();
        let s = idx.get();
        if s >= cap as usize {
            return;
        }
        let mut f = 0i32;
        if active[s] == 1 {
            let l = level[s];
            let (cix, ciy) = (ix[s] as i64, iy[s] as i64);
            let nxl = (base_nx as i64) << l;
            let nyl = (base_ny as i64) << l;
            // dir order: -x,+x,-y,+y ; deltas
            let dxs = [-1i64, 1, 0, 0];
            let dys = [0i64, 0, -1, 1];
            let mut d = 0usize;
            while d < 4 {
                let jx = cix + dxs[d];
                let jy = ciy + dys[d];
                if jx >= 0 && jx < nxl && jy >= 0 && jy < nyl {
                    let npos = offsets[l as usize] as usize + (jy as usize) * (nxl as usize) + jx as usize;
                    let nslot = pos2slot[npos];
                    if nslot != u32::MAX && children[nslot as usize * 4] != u32::MAX {
                        // neighbour internal: its two children toward us (L+1).
                        let (bx, by) = (2 * jx, 2 * jy);
                        // (c0,c1) per dir — opposite face of the neighbour.
                        let (c0x, c0y, c1x, c1y) = match d {
                            0 => (bx + 1, by, bx + 1, by + 1), // -x
                            1 => (bx, by, bx, by + 1),         // +x
                            2 => (bx, by + 1, bx + 1, by + 1), // -y
                            _ => (bx, by, bx + 1, by),         // +y
                        };
                        let nxl1 = (base_nx as i64) << (l + 1);
                        let off1 = offsets[(l + 1) as usize] as usize;
                        let p0 = off1 + (c0y as usize) * (nxl1 as usize) + c0x as usize;
                        let p1 = off1 + (c1y as usize) * (nxl1 as usize) + c1x as usize;
                        let cs0 = pos2slot[p0];
                        let cs1 = pos2slot[p1];
                        let c0_int = cs0 != u32::MAX && children[cs0 as usize * 4] != u32::MAX;
                        let c1_int = cs1 != u32::MAX && children[cs1 as usize * 4] != u32::MAX;
                        if c0_int || c1_int {
                            f = 1;
                        }
                    }
                }
                d += 1;
            }
        }
        if let Some(o) = flag.get_mut(thread::index_1d()) {
            *o = f;
        }
    }

    /// **Connectivity rebuild** on device: for each active leaf, fill its 4 face neighbours
    /// (`block_nbr`/`block_nbr2`/`block_nbr_kind`, dir −x,+x,−y,+y; kind 0/1/2/3 = boundary/same/
    /// coarser/finer) using the `pos2slot` index for O(1) lookup. Mirrors host
    /// `GpuAmrMesh::build_neighbors` exactly. One thread per slot, writing its own `s*4+d` entries
    /// (disjoint per slot). Assumes a 2:1-balanced active set.
    #[kernel]
    pub fn amr_connectivity(
        active: &[u8], level: &[u32], ix: &[u32], iy: &[u32], pos2slot: &[u32], offsets: &[u32],
        base_nx: u32, base_ny: u32, cap: u32,
        mut block_nbr: DisjointSlice<u32>, mut block_nbr2: DisjointSlice<u32>, mut block_nbr_kind: DisjointSlice<u8>,
    ) {
        let idx = thread::index_1d();
        let s = idx.get();
        if s >= cap as usize || active[s] != 1 {
            return;
        }
        let l = level[s];
        let (cix, ciy) = (ix[s] as i64, iy[s] as i64);
        let nxl = (base_nx as i64) << l;
        let nyl = (base_ny as i64) << l;
        let dxs = [-1i64, 1, 0, 0];
        let dys = [0i64, 0, -1, 1];
        let off_l = offsets[l as usize] as usize;
        let mut d = 0usize;
        while d < 4 {
            let o = s * 4 + d;
            let jx = cix + dxs[d];
            let jy = ciy + dys[d];
            let mut kind = 0u8;
            let mut nbr = u32::MAX;
            let mut nbr2 = u32::MAX;
            if jx >= 0 && jx < nxl && jy >= 0 && jy < nyl {
                // same level
                let npos = off_l + (jy as usize) * (nxl as usize) + jx as usize;
                let ns = pos2slot[npos];
                if ns != u32::MAX && active[ns as usize] == 1 {
                    nbr = ns;
                    kind = 1;
                } else if l > 0 {
                    // coarser parent (l-1, jx/2, jy/2)
                    let nxlm = (base_nx as i64) << (l - 1);
                    let ppos = offsets[(l - 1) as usize] as usize + ((jy / 2) as usize) * (nxlm as usize) + (jx / 2) as usize;
                    let ps = pos2slot[ppos];
                    if ps != u32::MAX && active[ps as usize] == 1 {
                        nbr = ps;
                        kind = 2;
                    }
                }
                if kind == 0 {
                    // finer: the two (l+1) children on the shared face
                    let (bx, by) = (2 * jx, 2 * jy);
                    let (c0x, c0y, c1x, c1y) = match d {
                        0 => (bx + 1, by, bx + 1, by + 1),
                        1 => (bx, by, bx, by + 1),
                        2 => (bx, by + 1, bx + 1, by + 1),
                        _ => (bx, by, bx + 1, by),
                    };
                    let nxl1 = (base_nx as i64) << (l + 1);
                    let off1 = offsets[(l + 1) as usize] as usize;
                    let p0 = off1 + (c0y as usize) * (nxl1 as usize) + c0x as usize;
                    let p1 = off1 + (c1y as usize) * (nxl1 as usize) + c1x as usize;
                    let f0 = pos2slot[p0];
                    let f1 = pos2slot[p1];
                    let f0a = f0 != u32::MAX && active[f0 as usize] == 1;
                    let f1a = f1 != u32::MAX && active[f1 as usize] == 1;
                    if f0a && f1a {
                        nbr = f0;
                        nbr2 = f1;
                        kind = 3;
                    }
                }
            }
            unsafe {
                *block_nbr.get_unchecked_mut(o) = nbr;
                *block_nbr2.get_unchecked_mut(o) = nbr2;
                *block_nbr_kind.get_unchecked_mut(o) = kind;
            }
            d += 1;
        }
    }

    /// Fill a `u32` buffer with `val` (device memset) — used to clear `pos2slot` to NONE before a
    /// resident rebuild without a host transfer.
    #[kernel]
    pub fn amr_fill_u32(val: u32, n: u32, mut out: DisjointSlice<u32>) {
        let idx = thread::index_1d();
        let i = idx.get();
        if i >= n as usize {
            return;
        }
        if let Some(o) = out.get_mut(idx) {
            *o = val;
        }
    }

    /// Whole-buffer copy `out = src` (cap·nn). One thread per dof — used to seed the remap output so
    /// unchanged slots keep their field while refine/coarsen overwrite the touched slots.
    #[kernel]
    pub fn amr_copy_f64(src: &[f64], n: u32, mut out: DisjointSlice<f64>) {
        let idx = thread::index_1d();
        let i = idx.get();
        if i >= n as usize {
            return;
        }
        if let Some(o) = out.get_mut(idx) {
            *o = src[i];
        }
    }

    /// **Remap prolong** (slot-indexed): for each refine entry, prolong the parent slot's field into its
    /// 4 child slots. Grid = `cap·4` blocks (`blk = k·4 + child`, guarded `k<n_refine`), `nn` threads.
    /// Reads `field_in[parent]`, writes `field_out[child]` — distinct buffers, distinct child slots ⇒
    /// race-free. Same exact prolong as `prolong2d` (degree ≤ p exact). Also drives balance-induced
    /// refines (they go through the same refine list).
    #[kernel]
    pub fn amr_remap_prolong(
        field_in: &[f64], refine_list: &[u32], free_list: &[u32], counts: &[u32], p_left: &[f64], p_right: &[f64],
        n1: u32, mut field_out: DisjointSlice<f64>,
    ) {
        static mut PF: SharedArray<f64, NN_MAX> = SharedArray::UNINIT;
        let n = n1 as usize;
        let nn = n * n;
        let blk = thread::blockIdx_x() as usize;
        let k = blk / 4;
        let child = blk % 4;
        let nr = counts[0] as usize;
        let nf = counts[1] as usize;
        if k >= nr || 4 * k + 3 >= nf {
            return;
        }
        let r = refine_list[k] as usize;
        let cs = free_list[4 * k + child] as usize;
        let m = thread::threadIdx_x() as usize;
        unsafe {
            PF[m] = field_in[r * nn + m];
        }
        thread::sync_threads();
        let (cx, cy) = (child % 2, child / 2);
        let i = m % n;
        let j = m / n;
        let mut s = 0.0f64;
        let mut b = 0usize;
        while b < n {
            let mut row = 0.0f64;
            let mut a = 0usize;
            while a < n {
                let pra = if cx == 0 { p_left[i * n + a] } else { p_right[i * n + a] };
                row += pra * unsafe { PF[a + b * n] };
                a += 1;
            }
            let psb = if cy == 0 { p_left[j * n + b] } else { p_right[j * n + b] };
            s += psb * row;
            b += 1;
        }
        unsafe {
            *field_out.get_unchecked_mut(cs * nn + m) = s;
        }
    }

    /// **Remap restrict** (slot-indexed): for each coarsen entry, conservatively restrict the 4 child
    /// slots' field into the parent slot. Grid = `cap` blocks (guarded `blk<n_coarsen`), `nn` threads.
    /// `children_in` MUST be the pre-coarsen child links (read before `apply_coarsen` detaches them);
    /// reads `field_in[child]`, writes `field_out[parent]`. Same conservative L2 projection as
    /// `restrict2d` (preserves the cell average).
    #[kernel]
    pub fn amr_remap_restrict(
        field_in: &[f64], coarsen_list: &[u32], count: &[u32], children_in: &[u32], p_left: &[f64], p_right: &[f64],
        w: &[f64], n1: u32, mut field_out: DisjointSlice<f64>,
    ) {
        static mut CH: SharedArray<f64, CN_MAX> = SharedArray::UNINIT;
        let n = n1 as usize;
        let nn = n * n;
        let blk = thread::blockIdx_x() as usize;
        if blk >= count[0] as usize {
            return;
        }
        let p = coarsen_list[blk] as usize;
        let m = thread::threadIdx_x() as usize;
        let mut c = 0usize;
        while c < 4 {
            let cs = children_in[p * 4 + c] as usize;
            unsafe {
                CH[c * nn + m] = field_in[cs * nn + m];
            }
            c += 1;
        }
        thread::sync_threads();
        let i = m % n;
        let j = m / n;
        let mut acc = 0.0f64;
        let mut cy = 0usize;
        while cy < 2 {
            let mut cx = 0usize;
            while cx < 2 {
                let cc = cx + 2 * cy;
                let mut s = 0.0f64;
                let mut b = 0usize;
                while b < n {
                    let wb = w[b];
                    let psb = if cy == 0 { p_left[b * n + j] } else { p_right[b * n + j] };
                    let mut a = 0usize;
                    while a < n {
                        let pra = if cx == 0 { p_left[a * n + i] } else { p_right[a * n + i] };
                        s += pra * psb * w[a] * wb * unsafe { CH[cc * nn + a + b * n] };
                        a += 1;
                    }
                    b += 1;
                }
                acc += 0.25 * s / (w[i] * w[j]);
                cx += 1;
            }
            cy += 1;
        }
        unsafe {
            *field_out.get_unchecked_mut(p * nn + m) = acc;
        }
    }
}

/// Per-element smoothness indicator `Se` on the GPU for a scalar `field` (`n_elements·n_nodes`),
/// returning one value per element. `si` supplies the SAME nodal→modal transform + Legendre norms
/// as the host indicator, so the result matches `SmoothnessIndicator::indicator` to round-off.
/// One-shot (loads the module + uploads) — the device-resident handle version comes with the masked
/// AMR data structures (Stage 3c).
pub fn smoothness_se_gpu(
    mesh: &Mesh2d, field: &[f64], si: &SmoothnessIndicator,
) -> Result<Vec<f64>, Box<dyn std::error::Error>> {
    let nn = mesh.refq.n_nodes();
    let ne = mesh.n_elements();
    let ndof = ne * nn;
    let n1 = (mesh.order + 1) as u32;
    assert_eq!(field.len(), ndof, "field length must be n_elements·n_nodes");
    assert!(nn <= NN_MAX, "p too large for NN_MAX={NN_MAX}");

    let ctx = CudaContext::new(0)?;
    let stream = ctx.default_stream();
    let module = kernels::load(&ctx)?;
    let up = |v: &[f64]| DeviceBuffer::from_host(&stream, v);
    let f_d = up(field)?;
    let vinv_d = up(si.vinv())?;
    let gamma_d = up(si.gamma())?;
    let mut out = DeviceBuffer::<f64>::zeroed(&stream, ndof)?;
    let cfg = LaunchConfig { grid_dim: (ne as u32, 1, 1), block_dim: (nn as u32, 1, 1), shared_mem_bytes: 0 };
    module.smoothness_se(&stream, cfg, &f_d, &vinv_d, &gamma_d, n1, &mut out)?;
    let full = out.to_host_vec(&stream)?;
    Ok((0..ne).map(|e| full[e * nn]).collect())
}

/// GPU **adapt flag**: per-element refine(+1)/coarsen(−1)/keep(0) from a per-element smoothness `se`
/// (e.g. the [`smoothness_se_gpu`] output) + per-element `level`, thresholds, and `l_max`. This is the
/// on-device adapt DECISION — no host readback of the field. Sibling-agreement for coarsen and 2:1
/// balance are applied afterwards on the block structure (host or device). One-shot wrapper.
pub fn amr_flag_gpu(
    se: &[f64], level: &[u32], refine_thr: f64, coarsen_thr: f64, l_max: u32,
) -> Result<Vec<i32>, Box<dyn std::error::Error>> {
    let ne = se.len();
    assert_eq!(level.len(), ne, "se and level must have one entry per element");
    let ctx = CudaContext::new(0)?;
    let stream = ctx.default_stream();
    let module = kernels::load(&ctx)?;
    let se_d = DeviceBuffer::from_host(&stream, se)?;
    let lvl_d = DeviceBuffer::from_host(&stream, level)?;
    let mut out = DeviceBuffer::<i32>::zeroed(&stream, ne)?;
    let block = 256u32;
    let grid = ((ne as u32) + block - 1) / block;
    let cfg = LaunchConfig { grid_dim: (grid, 1, 1), block_dim: (block, 1, 1), shared_mem_bytes: 0 };
    module.amr_flag(&stream, cfg, &se_d, &lvl_d, refine_thr, coarsen_thr, l_max, ne as u32, &mut out)?;
    Ok(out.to_host_vec(&stream)?)
}

/// GPU **prolong** (refine): each parent element's nodal field → its 4 children. `parents` is
/// `n_parents·nn`; returns `n_parents·4·nn` laid out `[(parent·4+child)·nn + node]`, child index
/// `cx + 2·cy`. `refq` supplies the same 1D half-child matrices as the host. Matches
/// `RefineQuad::prolong` to round-off. One-shot wrapper (handle version comes with Stage 3c).
pub fn prolong_gpu(refq: &RefineQuad, n_parents: usize, parents: &[f64]) -> Result<Vec<f64>, Box<dyn std::error::Error>> {
    let n1 = refq.order + 1;
    let nn = n1 * n1;
    assert_eq!(parents.len(), n_parents * nn);
    let ctx = CudaContext::new(0)?;
    let stream = ctx.default_stream();
    let module = kernels::load(&ctx)?;
    let up = |v: &[f64]| DeviceBuffer::from_host(&stream, v);
    let pf = up(parents)?;
    let pl = up(refq.axis_matrix(0))?;
    let pr = up(refq.axis_matrix(1))?;
    let mut out = DeviceBuffer::<f64>::zeroed(&stream, n_parents * 4 * nn)?;
    let cfg = LaunchConfig { grid_dim: ((n_parents * 4) as u32, 1, 1), block_dim: (nn as u32, 1, 1), shared_mem_bytes: 0 };
    module.prolong2d(&stream, cfg, &pf, &pl, &pr, n1 as u32, &mut out)?;
    Ok(out.to_host_vec(&stream)?)
}

/// GPU **restrict** (coarsen): 4 children per parent → parent, conservative L2 projection. `children`
/// is `n_parents·4·nn` (the [`prolong_gpu`] layout); returns `n_parents·nn`. Matches
/// `RefineQuad::restrict` to round-off (preserves cell averages).
pub fn restrict_gpu(refq: &RefineQuad, n_parents: usize, children: &[f64]) -> Result<Vec<f64>, Box<dyn std::error::Error>> {
    let n1 = refq.order + 1;
    let nn = n1 * n1;
    assert_eq!(children.len(), n_parents * 4 * nn);
    let ctx = CudaContext::new(0)?;
    let stream = ctx.default_stream();
    let module = kernels::load(&ctx)?;
    let up = |v: &[f64]| DeviceBuffer::from_host(&stream, v);
    let ch = up(children)?;
    let pl = up(refq.axis_matrix(0))?;
    let pr = up(refq.axis_matrix(1))?;
    let w = up(refq.weights())?;
    let mut out = DeviceBuffer::<f64>::zeroed(&stream, n_parents * nn)?;
    let cfg = LaunchConfig { grid_dim: (n_parents as u32, 1, 1), block_dim: (nn as u32, 1, 1), shared_mem_bytes: 0 };
    module.restrict2d(&stream, cfg, &ch, &pl, &pr, &w, n1 as u32, &mut out)?;
    Ok(out.to_host_vec(&stream)?)
}

/// GPU **refine on the masked block structure** (Stage 3f). Given per-slot refine `flag` (length
/// `cap`; only active leaves are honored), runs the on-device adapt pipeline — `amr_mark` →
/// `amr_compact` (prefix-scan/compaction) → `amr_apply_refine` (slot allocation + activation) — and
/// writes the updated `active/level/ix/iy/parent/children` arrays back into `m`. NO host decision in
/// the loop; the launch dims are fixed (over-provisioned + guarded) so this is CUDA-graph-legal. The
/// resulting ACTIVE SET matches the host [`GpuAmrMesh::refine`] applied to the flagged active leaves
/// (slot ids may differ; the mesh is defined by the active `(level,ix,iy)` set). Leaves `m.gap`/
/// `m.n_active` STALE — the device path is authoritative through the flat arrays; callers that mix
/// host + device adapt must rebuild those. Validated by `amr-refine-device-check`.
pub fn device_refine(m: &mut crate::amr_mesh::GpuAmrMesh, flag: &[i32]) -> Result<(), Box<dyn std::error::Error>> {
    let cap = m.cap;
    assert_eq!(flag.len(), cap, "flag must have one entry per slot (cap)");
    let nbase = (m.base_nx * m.base_ny) as u32;
    let ctx = CudaContext::new(0)?;
    let stream = ctx.default_stream();
    let module = kernels::load(&ctx)?;

    let active_d = DeviceBuffer::from_host(&stream, &m.active)?;
    let children_d = DeviceBuffer::from_host(&stream, &m.children)?;
    let level_d = DeviceBuffer::from_host(&stream, &m.level)?;
    let flag_d = DeviceBuffer::from_host(&stream, flag)?;
    let mut is_refine = DeviceBuffer::<i32>::zeroed(&stream, cap)?;
    let mut is_free = DeviceBuffer::<i32>::zeroed(&stream, cap)?;

    let block = 256u32;
    let grid = (cap as u32 + block - 1) / block;
    let cfg = LaunchConfig { grid_dim: (grid, 1, 1), block_dim: (block, 1, 1), shared_mem_bytes: 0 };
    module.amr_mark(
        &stream, cfg, &active_d, &children_d, &level_d, &flag_d, nbase, m.l_max as u32, cap as u32,
        &mut is_refine, &mut is_free,
    )?;

    let mut refine_list = DeviceBuffer::<u32>::zeroed(&stream, cap)?;
    let mut free_list = DeviceBuffer::<u32>::zeroed(&stream, cap)?;
    let counts = DeviceBuffer::<u32>::zeroed(&stream, 2)?; // pre-zeroed atomic counters [n_refine, n_free]
    module.amr_compact_blocked(&stream, cfg, &is_refine, &counts, 0, cap as u32, &mut refine_list)?;
    module.amr_compact_blocked(&stream, cfg, &is_free, &counts, 1, cap as u32, &mut free_list)?;

    // Mutated structure lives on the device for the apply pass.
    let mut active_m = DeviceBuffer::from_host(&stream, &m.active)?;
    let mut level_m = DeviceBuffer::from_host(&stream, &m.level)?;
    let mut ix_m = DeviceBuffer::from_host(&stream, &m.ix)?;
    let mut iy_m = DeviceBuffer::from_host(&stream, &m.iy)?;
    let mut parent_m = DeviceBuffer::from_host(&stream, &m.parent)?;
    let mut children_m = DeviceBuffer::from_host(&stream, &m.children)?;
    module.amr_apply_refine(
        &stream, cfg, &refine_list, &free_list, &counts,
        &mut active_m, &mut level_m, &mut ix_m, &mut iy_m, &mut parent_m, &mut children_m,
    )?;

    m.active = active_m.to_host_vec(&stream)?;
    m.level = level_m.to_host_vec(&stream)?;
    m.ix = ix_m.to_host_vec(&stream)?;
    m.iy = iy_m.to_host_vec(&stream)?;
    m.parent = parent_m.to_host_vec(&stream)?;
    m.children = children_m.to_host_vec(&stream)?;
    Ok(())
}

/// GPU **coarsen on the masked block structure** (Stage 3f). Given per-slot coarsen `flag` (length
/// `cap`; `−1` on active leaves to coarsen), runs `amr_mark_coarsen` → `amr_compact1` →
/// `amr_apply_coarsen` on-device: every parent whose 4 children are active leaves all flagged `−1`
/// (sibling agreement) is reactivated and its children freed. Matches host [`GpuAmrMesh::coarsen`] on
/// the agreeing sibling groups. Like [`device_refine`], leaves `m.gap`/`m.n_active` STALE.
pub fn device_coarsen(m: &mut crate::amr_mesh::GpuAmrMesh, flag: &[i32]) -> Result<(), Box<dyn std::error::Error>> {
    let cap = m.cap;
    assert_eq!(flag.len(), cap, "flag must have one entry per slot (cap)");
    let ctx = CudaContext::new(0)?;
    let stream = ctx.default_stream();
    let module = kernels::load(&ctx)?;

    let active_d = DeviceBuffer::from_host(&stream, &m.active)?;
    let children_d = DeviceBuffer::from_host(&stream, &m.children)?;
    let flag_d = DeviceBuffer::from_host(&stream, flag)?;
    let mut is_coarsen = DeviceBuffer::<i32>::zeroed(&stream, cap)?;

    let block = 256u32;
    let grid = (cap as u32 + block - 1) / block;
    let cfg = LaunchConfig { grid_dim: (grid, 1, 1), block_dim: (block, 1, 1), shared_mem_bytes: 0 };
    module.amr_mark_coarsen(&stream, cfg, &active_d, &children_d, &flag_d, cap as u32, &mut is_coarsen)?;

    let mut list = DeviceBuffer::<u32>::zeroed(&stream, cap)?;
    let count = DeviceBuffer::<u32>::zeroed(&stream, 1)?; // pre-zeroed atomic counter
    module.amr_compact_blocked(&stream, cfg, &is_coarsen, &count, 0, cap as u32, &mut list)?;

    let mut active_m = DeviceBuffer::from_host(&stream, &m.active)?;
    let mut children_m = DeviceBuffer::from_host(&stream, &m.children)?;
    module.amr_apply_coarsen(&stream, cfg, &list, &count, &mut active_m, &mut children_m)?;

    m.active = active_m.to_host_vec(&stream)?;
    m.children = children_m.to_host_vec(&stream)?;
    Ok(())
}

/// Per-level canonical-position offsets `offsets[l] = Σ_{k<l} nbase·4^k` (length `l_max+1`).
fn pos_offsets(m: &crate::amr_mesh::GpuAmrMesh) -> Vec<u32> {
    let nbase = m.base_nx * m.base_ny;
    let mut offs = Vec::with_capacity(m.l_max + 1);
    let mut acc = 0u32;
    for l in 0..=m.l_max {
        offs.push(acc);
        acc += (nbase * 4usize.pow(l as u32)) as u32;
    }
    offs
}

/// GPU **one 2:1-balance pass**: returns the per-slot refine flags (`1` on active leaves that must
/// refine to restore 2:1 balance) computed entirely on-device via the `pos2slot` positional index +
/// `amr_balance_flag`. Matches host [`GpuAmrMesh::balance_refine_flags`]. (`device_balance` iterates
/// this with `device_refine`.)
pub fn device_balance_flags(m: &crate::amr_mesh::GpuAmrMesh) -> Result<Vec<i32>, Box<dyn std::error::Error>> {
    let cap = m.cap;
    let offsets = pos_offsets(m);
    let ctx = CudaContext::new(0)?;
    let stream = ctx.default_stream();
    let module = kernels::load(&ctx)?;

    let active_d = DeviceBuffer::from_host(&stream, &m.active)?;
    let children_d = DeviceBuffer::from_host(&stream, &m.children)?;
    let level_d = DeviceBuffer::from_host(&stream, &m.level)?;
    let ix_d = DeviceBuffer::from_host(&stream, &m.ix)?;
    let iy_d = DeviceBuffer::from_host(&stream, &m.iy)?;
    let offs_d = DeviceBuffer::from_host(&stream, &offsets)?;

    // pos2slot pre-filled with NONE (u32::MAX).
    let mut pos2slot = DeviceBuffer::from_host(&stream, &vec![u32::MAX; cap])?;
    let block = 256u32;
    let grid = (cap as u32 + block - 1) / block;
    let cfg = LaunchConfig { grid_dim: (grid, 1, 1), block_dim: (block, 1, 1), shared_mem_bytes: 0 };
    module.amr_build_pos2slot(
        &stream, cfg, &active_d, &children_d, &level_d, &ix_d, &iy_d, &offs_d, m.base_nx as u32, cap as u32, &mut pos2slot,
    )?;

    let mut flag = DeviceBuffer::<i32>::zeroed(&stream, cap)?;
    module.amr_balance_flag(
        &stream, cfg, &active_d, &children_d, &level_d, &ix_d, &iy_d, &pos2slot, &offs_d,
        m.base_nx as u32, m.base_ny as u32, cap as u32, &mut flag,
    )?;
    Ok(flag.to_host_vec(&stream)?)
}

/// GPU **2:1 balance**: iterate `device_balance_flags` → `device_refine` until no leaf needs refining.
/// Each pass is fully on-device; the host only checks the (infrequent, O(L)-bounded) loop-exit count.
/// Leaves the mesh 2:1-balanced. Returns the number of passes taken.
pub fn device_balance(m: &mut crate::amr_mesh::GpuAmrMesh) -> Result<usize, Box<dyn std::error::Error>> {
    let mut passes = 0usize;
    loop {
        let flags = device_balance_flags(m)?;
        if !flags.iter().any(|&f| f == 1) {
            break;
        }
        device_refine(m, &flags)?;
        passes += 1;
        if passes > m.l_max + 2 {
            return Err("2:1 balance failed to converge".into());
        }
    }
    Ok(passes)
}

/// GPU **connectivity rebuild**: fill `m.block_nbr`/`block_nbr2`/`block_nbr_kind` on-device (via the
/// `pos2slot` index + `amr_connectivity`), writing the result back into `m`. Matches host
/// [`GpuAmrMesh::build_neighbors`]. This is the face/mortar list the non-conforming operator consumes,
/// produced with no host loop. Assumes `m` is 2:1-balanced (run [`device_balance`] first).
pub fn device_connectivity(m: &mut crate::amr_mesh::GpuAmrMesh) -> Result<(), Box<dyn std::error::Error>> {
    let cap = m.cap;
    let offsets = pos_offsets(m);
    let ctx = CudaContext::new(0)?;
    let stream = ctx.default_stream();
    let module = kernels::load(&ctx)?;

    let active_d = DeviceBuffer::from_host(&stream, &m.active)?;
    let children_d = DeviceBuffer::from_host(&stream, &m.children)?;
    let level_d = DeviceBuffer::from_host(&stream, &m.level)?;
    let ix_d = DeviceBuffer::from_host(&stream, &m.ix)?;
    let iy_d = DeviceBuffer::from_host(&stream, &m.iy)?;
    let offs_d = DeviceBuffer::from_host(&stream, &offsets)?;

    let block = 256u32;
    let grid = (cap as u32 + block - 1) / block;
    let cfg = LaunchConfig { grid_dim: (grid, 1, 1), block_dim: (block, 1, 1), shared_mem_bytes: 0 };
    let mut pos2slot = DeviceBuffer::from_host(&stream, &vec![u32::MAX; cap])?;
    module.amr_build_pos2slot(
        &stream, cfg, &active_d, &children_d, &level_d, &ix_d, &iy_d, &offs_d, m.base_nx as u32, cap as u32, &mut pos2slot,
    )?;

    // Initialize outputs to NONE/boundary (kernel only writes active-leaf slots).
    let mut nbr_d = DeviceBuffer::from_host(&stream, &vec![u32::MAX; cap * 4])?;
    let mut nbr2_d = DeviceBuffer::from_host(&stream, &vec![u32::MAX; cap * 4])?;
    let mut kind_d = DeviceBuffer::from_host(&stream, &vec![0u8; cap * 4])?;
    module.amr_connectivity(
        &stream, cfg, &active_d, &level_d, &ix_d, &iy_d, &pos2slot, &offs_d,
        m.base_nx as u32, m.base_ny as u32, cap as u32, &mut nbr_d, &mut nbr2_d, &mut kind_d,
    )?;

    m.block_nbr = nbr_d.to_host_vec(&stream)?;
    m.block_nbr2 = nbr2_d.to_host_vec(&stream)?;
    m.block_nbr_kind = kind_d.to_host_vec(&stream)?;
    Ok(())
}

/// GPU **refine WITH field remap** (Stage 3f): structural [`device_refine`] plus carry of a per-slot
/// field `field_in` (`cap·nn`) across the adapt — each newly-created child gets the exact prolong of
/// its parent's field (`amr_remap_prolong`, degree ≤ p exact), unchanged slots keep their values.
/// Returns the remapped field. The solution carry-over for the on-device adaptive loop. (When carrying
/// the log-conformation Ψ, prolong in log space stays SPD — apply the Zhang–Shu pre-limiter upstream.)
pub fn device_refine_remap(
    m: &mut crate::amr_mesh::GpuAmrMesh, flag: &[i32], refq: &RefineQuad, field_in: &[f64],
) -> Result<Vec<f64>, Box<dyn std::error::Error>> {
    let cap = m.cap;
    let n1 = refq.order + 1;
    let nn = n1 * n1;
    assert_eq!(flag.len(), cap);
    assert_eq!(field_in.len(), cap * nn, "field_in must be cap·nn");
    let nbase = (m.base_nx * m.base_ny) as u32;
    let ctx = CudaContext::new(0)?;
    let stream = ctx.default_stream();
    let module = kernels::load(&ctx)?;

    let active_d = DeviceBuffer::from_host(&stream, &m.active)?;
    let children_d = DeviceBuffer::from_host(&stream, &m.children)?;
    let level_d = DeviceBuffer::from_host(&stream, &m.level)?;
    let flag_d = DeviceBuffer::from_host(&stream, flag)?;
    let mut is_refine = DeviceBuffer::<i32>::zeroed(&stream, cap)?;
    let mut is_free = DeviceBuffer::<i32>::zeroed(&stream, cap)?;

    let block = 256u32;
    let grid = (cap as u32 + block - 1) / block;
    let cfg = LaunchConfig { grid_dim: (grid, 1, 1), block_dim: (block, 1, 1), shared_mem_bytes: 0 };
    module.amr_mark(&stream, cfg, &active_d, &children_d, &level_d, &flag_d, nbase, m.l_max as u32, cap as u32, &mut is_refine, &mut is_free)?;

    let mut refine_list = DeviceBuffer::<u32>::zeroed(&stream, cap)?;
    let mut free_list = DeviceBuffer::<u32>::zeroed(&stream, cap)?;
    let counts = DeviceBuffer::<u32>::zeroed(&stream, 2)?;
    module.amr_compact_blocked(&stream, cfg, &is_refine, &counts, 0, cap as u32, &mut refine_list)?;
    module.amr_compact_blocked(&stream, cfg, &is_free, &counts, 1, cap as u32, &mut free_list)?;

    // Field remap: copy then prolong children from parents.
    let field_in_d = DeviceBuffer::from_host(&stream, field_in)?;
    let mut field_out = DeviceBuffer::from_host(&stream, field_in)?; // seed = copy
    let pl = DeviceBuffer::from_host(&stream, refq.axis_matrix(0))?;
    let pr = DeviceBuffer::from_host(&stream, refq.axis_matrix(1))?;
    let n_refine = counts.to_host_vec(&stream)?[0] as usize; // size the grid to actual work, not cap·4
    if n_refine > 0 {
        let cfg_pl = LaunchConfig { grid_dim: ((n_refine * 4) as u32, 1, 1), block_dim: (nn as u32, 1, 1), shared_mem_bytes: 0 };
        module.amr_remap_prolong(&stream, cfg_pl, &field_in_d, &refine_list, &free_list, &counts, &pl, &pr, n1 as u32, &mut field_out)?;
    }

    // Structural refine.
    let mut active_m = DeviceBuffer::from_host(&stream, &m.active)?;
    let mut level_m = DeviceBuffer::from_host(&stream, &m.level)?;
    let mut ix_m = DeviceBuffer::from_host(&stream, &m.ix)?;
    let mut iy_m = DeviceBuffer::from_host(&stream, &m.iy)?;
    let mut parent_m = DeviceBuffer::from_host(&stream, &m.parent)?;
    let mut children_m = DeviceBuffer::from_host(&stream, &m.children)?;
    module.amr_apply_refine(&stream, cfg, &refine_list, &free_list, &counts, &mut active_m, &mut level_m, &mut ix_m, &mut iy_m, &mut parent_m, &mut children_m)?;

    m.active = active_m.to_host_vec(&stream)?;
    m.level = level_m.to_host_vec(&stream)?;
    m.ix = ix_m.to_host_vec(&stream)?;
    m.iy = iy_m.to_host_vec(&stream)?;
    m.parent = parent_m.to_host_vec(&stream)?;
    m.children = children_m.to_host_vec(&stream)?;
    Ok(field_out.to_host_vec(&stream)?)
}

/// GPU **coarsen WITH field remap** (Stage 3f): structural [`device_coarsen`] plus conservative
/// restrict of each coarsened sibling group's field into the parent (`amr_remap_restrict`, preserves
/// the cell average). Returns the remapped field. Reads the child links BEFORE detaching them.
pub fn device_coarsen_remap(
    m: &mut crate::amr_mesh::GpuAmrMesh, flag: &[i32], refq: &RefineQuad, field_in: &[f64],
) -> Result<Vec<f64>, Box<dyn std::error::Error>> {
    let cap = m.cap;
    let n1 = refq.order + 1;
    let nn = n1 * n1;
    assert_eq!(flag.len(), cap);
    assert_eq!(field_in.len(), cap * nn, "field_in must be cap·nn");
    let ctx = CudaContext::new(0)?;
    let stream = ctx.default_stream();
    let module = kernels::load(&ctx)?;

    let active_d = DeviceBuffer::from_host(&stream, &m.active)?;
    let children_d = DeviceBuffer::from_host(&stream, &m.children)?; // pre-detach links (read by remap)
    let flag_d = DeviceBuffer::from_host(&stream, flag)?;
    let mut is_coarsen = DeviceBuffer::<i32>::zeroed(&stream, cap)?;

    let block = 256u32;
    let grid = (cap as u32 + block - 1) / block;
    let cfg = LaunchConfig { grid_dim: (grid, 1, 1), block_dim: (block, 1, 1), shared_mem_bytes: 0 };
    module.amr_mark_coarsen(&stream, cfg, &active_d, &children_d, &flag_d, cap as u32, &mut is_coarsen)?;

    let mut list = DeviceBuffer::<u32>::zeroed(&stream, cap)?;
    let count = DeviceBuffer::<u32>::zeroed(&stream, 1)?;
    module.amr_compact_blocked(&stream, cfg, &is_coarsen, &count, 0, cap as u32, &mut list)?;

    // Field remap: copy then restrict children → parents (children_d still intact).
    let field_in_d = DeviceBuffer::from_host(&stream, field_in)?;
    let mut field_out = DeviceBuffer::from_host(&stream, field_in)?;
    let pl = DeviceBuffer::from_host(&stream, refq.axis_matrix(0))?;
    let pr = DeviceBuffer::from_host(&stream, refq.axis_matrix(1))?;
    let w = DeviceBuffer::from_host(&stream, refq.weights())?;
    let n_coarsen = count.to_host_vec(&stream)?[0] as usize; // size the grid to actual work, not cap
    if n_coarsen > 0 {
        let cfg_rs = LaunchConfig { grid_dim: (n_coarsen as u32, 1, 1), block_dim: (nn as u32, 1, 1), shared_mem_bytes: 0 };
        module.amr_remap_restrict(&stream, cfg_rs, &field_in_d, &list, &count, &children_d, &pl, &pr, &w, n1 as u32, &mut field_out)?;
    }

    // Structural coarsen.
    let mut active_m = DeviceBuffer::from_host(&stream, &m.active)?;
    let mut children_m = DeviceBuffer::from_host(&stream, &m.children)?;
    module.amr_apply_coarsen(&stream, cfg, &list, &count, &mut active_m, &mut children_m)?;

    m.active = active_m.to_host_vec(&stream)?;
    m.children = children_m.to_host_vec(&stream)?;
    Ok(field_out.to_host_vec(&stream)?)
}

/// **Persistent device-resident adaptive mesh** (Stage 3f capstone). Holds the masked block structure
/// AND the per-slot solution field as resident `DeviceBuffer`s on one shared stream, and runs the whole
/// adapt cycle — refine/coarsen (+ field remap) → 2:1 balance (+ remap) → connectivity rebuild — as a
/// sequence of kernel launches with **no host↔device transfers** (the only host read is the tiny
/// `counts` scalar for the balance loop-exit test, which a CUDA conditional graph would remove). This
/// is the residency-compliant composition of the six validated one-shot primitives. `download_*` exist
/// for inspection/validation only. The flat arrays are authoritative (host `gap`/`n_active` not tracked).
pub struct GpuAdaptiveMesh {
    stream: Arc<CudaStream>,
    module: kernels::LoadedModule,
    pub base_nx: usize,
    pub base_ny: usize,
    pub l_max: usize,
    pub cap: usize,
    nn: usize,
    n1: usize,
    // structure (resident)
    active: DeviceBuffer<u8>,
    level: DeviceBuffer<u32>,
    ix: DeviceBuffer<u32>,
    iy: DeviceBuffer<u32>,
    parent: DeviceBuffer<u32>,
    children: DeviceBuffer<u32>,
    block_nbr: DeviceBuffer<u32>,
    block_nbr2: DeviceBuffer<u32>,
    block_nbr_kind: DeviceBuffer<u8>,
    pos2slot: DeviceBuffer<u32>,
    offsets: DeviceBuffer<u32>,
    // scratch (resident)
    is_refine: DeviceBuffer<i32>,
    is_free: DeviceBuffer<i32>,
    is_coarsen: DeviceBuffer<i32>,
    refine_list: DeviceBuffer<u32>,
    free_list: DeviceBuffer<u32>,
    coarsen_list: DeviceBuffer<u32>,
    counts: DeviceBuffer<u32>,
    ccount: DeviceBuffer<u32>,
    bflag: DeviceBuffer<i32>,
    fscratch: DeviceBuffer<f64>,
    // constant remap operators (resident)
    pl: DeviceBuffer<f64>,
    pr: DeviceBuffer<f64>,
    w: DeviceBuffer<f64>,
}

impl GpuAdaptiveMesh {
    /// Upload a host [`GpuAmrMesh`] + the constant `RefineQuad` operators once; everything stays
    /// resident thereafter.
    pub fn from_host(m: &crate::amr_mesh::GpuAmrMesh, refq: &RefineQuad) -> Result<Self, Box<dyn std::error::Error>> {
        let cap = m.cap;
        let n1 = refq.order + 1;
        let nn = n1 * n1;
        let ctx = CudaContext::new(0)?;
        let stream = ctx.default_stream();
        let offsets = {
            let nbase = m.base_nx * m.base_ny;
            let mut acc = 0u32;
            let mut v = Vec::with_capacity(m.l_max + 1);
            for l in 0..=m.l_max {
                v.push(acc);
                acc += (nbase * 4usize.pow(l as u32)) as u32;
            }
            v
        };
        Ok(Self {
            module: kernels::load(&ctx)?,
            base_nx: m.base_nx,
            base_ny: m.base_ny,
            l_max: m.l_max,
            cap,
            nn,
            n1,
            active: DeviceBuffer::from_host(&stream, &m.active)?,
            level: DeviceBuffer::from_host(&stream, &m.level)?,
            ix: DeviceBuffer::from_host(&stream, &m.ix)?,
            iy: DeviceBuffer::from_host(&stream, &m.iy)?,
            parent: DeviceBuffer::from_host(&stream, &m.parent)?,
            children: DeviceBuffer::from_host(&stream, &m.children)?,
            block_nbr: DeviceBuffer::from_host(&stream, &vec![u32::MAX; cap * 4])?,
            block_nbr2: DeviceBuffer::from_host(&stream, &vec![u32::MAX; cap * 4])?,
            block_nbr_kind: DeviceBuffer::from_host(&stream, &vec![0u8; cap * 4])?,
            pos2slot: DeviceBuffer::from_host(&stream, &vec![u32::MAX; cap])?,
            offsets: DeviceBuffer::from_host(&stream, &offsets)?,
            is_refine: DeviceBuffer::zeroed(&stream, cap)?,
            is_free: DeviceBuffer::zeroed(&stream, cap)?,
            is_coarsen: DeviceBuffer::zeroed(&stream, cap)?,
            refine_list: DeviceBuffer::zeroed(&stream, cap)?,
            free_list: DeviceBuffer::zeroed(&stream, cap)?,
            coarsen_list: DeviceBuffer::zeroed(&stream, cap)?,
            counts: DeviceBuffer::zeroed(&stream, 2)?,
            ccount: DeviceBuffer::zeroed(&stream, 1)?,
            bflag: DeviceBuffer::zeroed(&stream, cap)?,
            fscratch: DeviceBuffer::zeroed(&stream, cap * nn)?,
            pl: DeviceBuffer::from_host(&stream, refq.axis_matrix(0))?,
            pr: DeviceBuffer::from_host(&stream, refq.axis_matrix(1))?,
            w: DeviceBuffer::from_host(&stream, refq.weights())?,
            stream,
        })
    }

    fn cfg_slots(&self) -> LaunchConfig {
        let block = 256u32;
        let grid = (self.cap as u32 + block - 1) / block;
        LaunchConfig { grid_dim: (grid, 1, 1), block_dim: (block, 1, 1), shared_mem_bytes: 0 }
    }

    /// Upload a per-slot field (`cap·nn`) to a resident buffer on this handle's stream.
    pub fn upload_field(&self, field: &[f64]) -> Result<DeviceBuffer<f64>, Box<dyn std::error::Error>> {
        assert_eq!(field.len(), self.cap * self.nn);
        Ok(DeviceBuffer::from_host(&self.stream, field)?)
    }
    pub fn download_field(&self, f: &DeviceBuffer<f64>) -> Result<Vec<f64>, Box<dyn std::error::Error>> {
        Ok(f.to_host_vec(&self.stream)?)
    }

    /// Internal: rebuild `pos2slot` (NONE-clear then scatter) from the current resident structure.
    fn build_pos2slot(&mut self) -> Result<(), Box<dyn std::error::Error>> {
        // clear to NONE via re-upload of a NONE vector is a transfer; instead reuse a fill kernel?
        // We re-seed by uploading once is avoided: amr_build_pos2slot only writes tree cells, so stale
        // entries for freed positions could linger. Positions are a bijection on the FULL quadtree and
        // every position that can be queried (in-range neighbours of active leaves) is overwritten when
        // its cell is in-tree; freed cells leave their slot but are guarded by an `active` check at the
        // read site. To be safe we clear with a device memset-style fill.
        let cfg = self.cfg_slots();
        self.module.amr_fill_u32(&self.stream, cfg, u32::MAX, self.cap as u32, &mut self.pos2slot)?;
        self.module.amr_build_pos2slot(
            &self.stream, cfg, &self.active, &self.children, &self.level, &self.ix, &self.iy, &self.offsets,
            self.base_nx as u32, self.cap as u32, &mut self.pos2slot,
        )?;
        Ok(())
    }

    fn cfg_field(&self) -> LaunchConfig {
        LaunchConfig { grid_dim: (((self.cap * self.nn) as u32 + 255) / 256, 1, 1), block_dim: (256, 1, 1), shared_mem_bytes: 0 }
    }

    /// One resident refine pass driven by `self.refine_list`/`free_list`/`counts` (already computed):
    /// remap `field`→`fscratch` (copy + prolong), apply the structural refine, swap field←fscratch.
    /// The prolong grid is sized to the ACTUAL work (`n_refine·4` child-blocks), not `cap·4` — so we
    /// launch ~`n_refine·4` blocks instead of millions that early-return.
    fn refine_pass(&mut self, field: &mut DeviceBuffer<f64>, n_refine: usize) -> Result<(), Box<dyn std::error::Error>> {
        if n_refine == 0 {
            return Ok(());
        }
        let cfg = self.cfg_slots();
        let cfg_pl = LaunchConfig { grid_dim: ((n_refine * 4) as u32, 1, 1), block_dim: (self.nn as u32, 1, 1), shared_mem_bytes: 0 };
        self.module.amr_copy_f64(&self.stream, self.cfg_field(), field, (self.cap * self.nn) as u32, &mut self.fscratch)?;
        self.module.amr_remap_prolong(&self.stream, cfg_pl, field, &self.refine_list, &self.free_list, &self.counts, &self.pl, &self.pr, self.n1 as u32, &mut self.fscratch)?;
        self.module.amr_apply_refine(&self.stream, cfg, &self.refine_list, &self.free_list, &self.counts, &mut self.active, &mut self.level, &mut self.ix, &mut self.iy, &mut self.parent, &mut self.children)?;
        std::mem::swap(field, &mut self.fscratch);
        Ok(())
    }

    /// Run the full resident adapt cycle for a host-provided per-slot `flag` (+1 refine / −1 coarsen)
    /// against a resident `field`: refine+coarsen with remap, then 2:1 balance (with remap), then
    /// connectivity rebuild. Returns the number of balance passes. NO host↔device field transfers; the
    /// only host reads are the `counts`/`ccount` scalars (loop control), which a conditional graph elides.
    pub fn adapt(&mut self, flag_host: &[i32], field: &mut DeviceBuffer<f64>) -> Result<usize, Box<dyn std::error::Error>> {
        assert_eq!(flag_host.len(), self.cap);
        let flag = DeviceBuffer::from_host(&self.stream, flag_host)?;
        let nbase = (self.base_nx * self.base_ny) as u32;
        let cfg = self.cfg_slots();
        let cfg_cnt = LaunchConfig { grid_dim: (1, 1, 1), block_dim: (256, 1, 1), shared_mem_bytes: 0 };

        // --- refine + coarsen marks/lists (remap reads pre-detach children) ---
        self.module.amr_mark(&self.stream, cfg, &self.active, &self.children, &self.level, &flag, nbase, self.l_max as u32, self.cap as u32, &mut self.is_refine, &mut self.is_free)?;
        self.module.amr_mark_coarsen(&self.stream, cfg, &self.active, &self.children, &flag, self.cap as u32, &mut self.is_coarsen)?;
        // zero the reused atomic counters, then block-aggregated parallel compaction (no serial scan)
        self.module.amr_fill_u32(&self.stream, cfg_cnt, 0, 2, &mut self.counts)?;
        self.module.amr_fill_u32(&self.stream, cfg_cnt, 0, 1, &mut self.ccount)?;
        self.module.amr_compact_blocked(&self.stream, cfg, &self.is_refine, &self.counts, 0, self.cap as u32, &mut self.refine_list)?;
        self.module.amr_compact_blocked(&self.stream, cfg, &self.is_free, &self.counts, 1, self.cap as u32, &mut self.free_list)?;
        self.module.amr_compact_blocked(&self.stream, cfg, &self.is_coarsen, &self.ccount, 0, self.cap as u32, &mut self.coarsen_list)?;

        // Read the actual refine/coarsen counts so the (block-per-entry) remap kernels launch only the
        // work that exists — n_refine·4 / n_coarsen blocks, NOT cap·4 / cap (millions of empty blocks).
        // Two tiny scalar reads per adapt (adapt is infrequent); a graph capture would use a fixed bound.
        let cnt = self.counts.to_host_vec(&self.stream)?;
        let (n_refine, n_coarsen) = (cnt[0] as usize, self.ccount.to_host_vec(&self.stream)?[0] as usize);

        // Only touch the field + structure if there's actual work — a no-op adapt (nothing flagged)
        // skips the full-field copy/swap and the apply passes entirely (the 350 MB copy was 50%+ of a
        // no-op adapt). When there IS work: seed scratch = field, prolong (refine) + restrict (coarsen).
        if n_refine > 0 || n_coarsen > 0 {
            self.module.amr_copy_f64(&self.stream, self.cfg_field(), field, (self.cap * self.nn) as u32, &mut self.fscratch)?;
            if n_refine > 0 {
                let cfg_pl = LaunchConfig { grid_dim: ((n_refine * 4) as u32, 1, 1), block_dim: (self.nn as u32, 1, 1), shared_mem_bytes: 0 };
                self.module.amr_remap_prolong(&self.stream, cfg_pl, field, &self.refine_list, &self.free_list, &self.counts, &self.pl, &self.pr, self.n1 as u32, &mut self.fscratch)?;
            }
            if n_coarsen > 0 {
                let cfg_rs = LaunchConfig { grid_dim: (n_coarsen as u32, 1, 1), block_dim: (self.nn as u32, 1, 1), shared_mem_bytes: 0 };
                self.module.amr_remap_restrict(&self.stream, cfg_rs, field, &self.coarsen_list, &self.ccount, &self.children, &self.pl, &self.pr, &self.w, self.n1 as u32, &mut self.fscratch)?;
            }
            self.module.amr_apply_refine(&self.stream, cfg, &self.refine_list, &self.free_list, &self.counts, &mut self.active, &mut self.level, &mut self.ix, &mut self.iy, &mut self.parent, &mut self.children)?;
            self.module.amr_apply_coarsen(&self.stream, cfg, &self.coarsen_list, &self.ccount, &mut self.active, &mut self.children)?;
            std::mem::swap(field, &mut self.fscratch);
        }

        // --- 2:1 balance: iterate flag→refine until no leaf needs it ---
        let mut passes = 0usize;
        loop {
            self.build_pos2slot()?;
            self.module.amr_balance_flag(&self.stream, cfg, &self.active, &self.children, &self.level, &self.ix, &self.iy, &self.pos2slot, &self.offsets, self.base_nx as u32, self.base_ny as u32, self.cap as u32, &mut self.bflag)?;
            // compact bflag into the refine list (reuse), and free list from is_free
            self.module.amr_mark(&self.stream, cfg, &self.active, &self.children, &self.level, &self.bflag, nbase, self.l_max as u32, self.cap as u32, &mut self.is_refine, &mut self.is_free)?;
            self.module.amr_fill_u32(&self.stream, cfg_cnt, 0, 2, &mut self.counts)?;
            self.module.amr_compact_blocked(&self.stream, cfg, &self.is_refine, &self.counts, 0, self.cap as u32, &mut self.refine_list)?;
            self.module.amr_compact_blocked(&self.stream, cfg, &self.is_free, &self.counts, 1, self.cap as u32, &mut self.free_list)?;
            let nr = self.counts.to_host_vec(&self.stream)?[0] as usize; // tiny loop-control read
            if nr == 0 {
                break;
            }
            self.refine_pass(field, nr)?;
            passes += 1;
            if passes > self.l_max + 2 {
                return Err("resident 2:1 balance failed to converge".into());
            }
        }

        // --- connectivity rebuild ---
        self.build_pos2slot()?;
        self.module.amr_connectivity(&self.stream, cfg, &self.active, &self.level, &self.ix, &self.iy, &self.pos2slot, &self.offsets, self.base_nx as u32, self.base_ny as u32, self.cap as u32, &mut self.block_nbr, &mut self.block_nbr2, &mut self.block_nbr_kind)?;
        Ok(passes)
    }

    /// Gather a resident field's per-cell node values for the requested `cells`, looked up via the
    /// handle's OWN active structure — so comparisons are independent of (block-interleaved) slot
    /// assignment. Returns the `nn`-blocks concatenated in `cells` order. Validation/I-O use.
    pub fn download_field_on(
        &self, field: &DeviceBuffer<f64>, cells: &[(u32, u32, u32)],
    ) -> Result<Vec<f64>, Box<dyn std::error::Error>> {
        use std::collections::HashMap;
        let active = self.active.to_host_vec(&self.stream)?;
        let level = self.level.to_host_vec(&self.stream)?;
        let ix = self.ix.to_host_vec(&self.stream)?;
        let iy = self.iy.to_host_vec(&self.stream)?;
        let f = field.to_host_vec(&self.stream)?;
        let mut map: HashMap<(u32, u32, u32), usize> = HashMap::with_capacity(self.cap);
        for s in 0..self.cap {
            if active[s] == 1 {
                map.insert((level[s], ix[s], iy[s]), s);
            }
        }
        let mut out = Vec::with_capacity(cells.len() * self.nn);
        for &c in cells {
            let s = *map.get(&c).expect("requested cell not active");
            out.extend_from_slice(&f[s * self.nn..(s + 1) * self.nn]);
        }
        Ok(out)
    }

    /// Active leaf cells `(level,ix,iy)` sorted — for validation/inspection (downloads the structure).
    pub fn download_active_cells(&self) -> Result<Vec<(u32, u32, u32)>, Box<dyn std::error::Error>> {
        let active = self.active.to_host_vec(&self.stream)?;
        let level = self.level.to_host_vec(&self.stream)?;
        let ix = self.ix.to_host_vec(&self.stream)?;
        let iy = self.iy.to_host_vec(&self.stream)?;
        let mut v: Vec<(u32, u32, u32)> = (0..self.cap).filter(|&s| active[s] == 1).map(|s| (level[s], ix[s], iy[s])).collect();
        v.sort_unstable();
        Ok(v)
    }
}
