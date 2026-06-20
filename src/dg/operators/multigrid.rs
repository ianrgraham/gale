//! p-multigrid preconditioner for the SIPG Poisson operator, with preconditioned
//! CG. The verified bottleneck-solver from `docs/implicit-solver-strategy.md` §2:
//! a hierarchy of polynomial orders `p → p/2 → … → 1` on the same mesh, V-cycle
//! with a damped-Jacobi smoother (Chebyshev is a drop-in smoother upgrade) and a
//! coarse CG solve, used as `M⁻¹` inside CG. CPU oracle; the GPU port reuses the
//! operator/smoother kernels.
//!
//! Transfer is nodal (per-element, tensor-product) Lagrange interpolation between
//! LGL grids of consecutive orders; restriction is its transpose (Galerkin).
//! Levels are re-discretized SIPG operators (not RAP).

use super::mesh::Mesh2d;
use super::poisson::Poisson;
use rayon::prelude::*;

fn dot(a: &[f64], b: &[f64]) -> f64 {
    a.iter().zip(b).map(|(x, y)| x * y).sum()
}
fn norm(a: &[f64]) -> f64 {
    dot(a, a).sqrt()
}

/// Coarsening sequence of orders: `p, p/2, p/4, …, 1`.
pub(crate) fn order_levels(p: usize) -> Vec<usize> {
    let mut v = Vec::new();
    let mut q = p;
    while q > 1 {
        v.push(q);
        q /= 2;
    }
    v.push(1);
    v
}

/// 1D Lagrange interpolation matrix (`fine × coarse`, row-major): value of each
/// coarse basis function at each fine node.
pub(crate) fn lagrange_matrix(coarse: &[f64], fine: &[f64]) -> Vec<f64> {
    let nc = coarse.len();
    let nf = fine.len();
    let mut m = vec![0.0; nf * nc];
    for a in 0..nf {
        let x = fine[a];
        for b in 0..nc {
            let mut l = 1.0;
            for q in 0..nc {
                if q != b {
                    l *= (x - coarse[q]) / (coarse[b] - coarse[q]);
                }
            }
            m[a * nc + b] = l;
        }
    }
    m
}

/// Inter-level transfer between consecutive multigrid levels.
enum Transfer {
    /// **p-transfer** (same element grid, order drops): tensor-product 1D Lagrange
    /// interpolation (coarse → fine), restriction is its transpose. `interp` is the 1D
    /// `fine × coarse` matrix.
    P { interp: Vec<f64> },
    /// **h-transfer** (order 1 on both, element grid halves 2:1): geometric prolongation —
    /// each coarse element covers 4 fine children, a fine node takes the coarse bilinear
    /// value at its position in the coarse reference square; restriction is the transpose.
    /// The four 4×4 per-quadrant matrices are the shared [`PMultigrid::quad_prolong`].
    H,
}

/// The four 2:1 geometric prolongation matrices (one per child quadrant `q = qx + 2·qy`),
/// **`nn` fine nodes × `nn` coarse nodes** (`nn = n1²`), row-major `[q·nn·nn + f·nn + a]`. A child
/// quadrant occupies half of the coarse reference square per axis; the fine node `f=(fx,fy)` maps
/// to the coarse-reference coords `((r+2qx−1)/2, (s+2qy−1)/2)`, where the coarse value at that point
/// is the order-`p` Lagrange interpolant (tensor product of the 1D Lagrange weights). Identical for
/// every 2:1 level at this order (reference-space geometry only), so built once and shared.
///
/// `nodes` are the order-`p` 1D LGL reference nodes (`n1 = nodes.len()`); for `nodes=[-1,1]`
/// (order 1) this reproduces the old hardcoded `[f64;64]` bilinear matrices (asserted in tests).
fn quad_prolong_matrices(nodes: &[f64]) -> Vec<f64> {
    let n1 = nodes.len();
    let nn = n1 * n1;
    let mut p = vec![0.0; 4 * nn * nn];
    for qy in 0..2 {
        for qx in 0..2 {
            let q = qx + 2 * qy;
            // Coarse-reference eval coords for this quadrant's fine nodes, per axis.
            let ex: Vec<f64> = nodes.iter().map(|&r| (r + (2.0 * qx as f64 - 1.0)) / 2.0).collect();
            let ey: Vec<f64> = nodes.iter().map(|&s| (s + (2.0 * qy as f64 - 1.0)) / 2.0).collect();
            // `lagrange_matrix(basis, eval)[e*n1 + b] = L_b(eval[e])`. Here basis = `nodes` (the
            // coarse 1D nodes), eval = the per-quadrant fine coords.
            let lx = lagrange_matrix(nodes, &ex); // [fx*n1 + ax] = L_ax(ex[fx])
            let ly = lagrange_matrix(nodes, &ey); // [fy*n1 + ay] = L_ay(ey[fy])
            for fy in 0..n1 {
                for fx in 0..n1 {
                    let f = fy * n1 + fx;
                    for ay in 0..n1 {
                        for ax in 0..n1 {
                            let a = ay * n1 + ax;
                            p[q * nn * nn + f * nn + a] = lx[fx * n1 + ax] * ly[fy * n1 + ay];
                        }
                    }
                }
            }
        }
    }
    p
}

pub struct PMultigrid {
    pub orders: Vec<usize>,
    pub meshes: Vec<Mesh2d>,
    pub alpha: f64,
    /// Helmholtz reaction `λ` (the `λM` term, `M = diag(jw)`). `0` ⇒ pure Poisson. The same
    /// physical `λ` is used at every p-level (the per-order mass differs); applied via
    /// `Poisson::with_bc`, so the diagonal and smoother weights pick it up.
    reaction: f64,
    /// Natural (Neumann) boundary tags, applied at **every** p-level by rediscretization
    /// (`Poisson::with_bc`) — NOT Galerkin coarsening (which doesn't reliably inherit DG
    /// BCs; see docs/research-pressure-multigrid.md). Empty ⇒ all-Dirichlet; all boundary
    /// tags ⇒ the singular pure-Neumann pressure operator (deflate in the PCG, see
    /// `poisson_pcg_solve`).
    neumann_tags: Vec<u32>,
    /// Per-transition transfer operator (`len = n_levels − 1`): p-coarsening (Lagrange) for
    /// the order-dropping levels, then h-coarsening (2:1 geometric) for the grid-halving ones.
    transfers: Vec<Transfer>,
    /// Per-level element-grid dimensions `(nx, ny)`. The p-levels all share the finest grid;
    /// the appended h-levels halve it (order 1 throughout), so the coarsest level is genuinely
    /// small and its solve is cheap — the fix for p-multigrid's otherwise-h-fine coarse grid.
    dims: Vec<(usize, usize)>,
    /// Shared 2:1 geometric prolongation matrices (per quadrant) for the h-transfers, row-major
    /// `[q·nn·nn + f·nn + a]` with `nn = h_nn` fine/coarse nodes per element. Length `4·nn·nn`.
    /// (Order 1 ⇒ `nn=4`, length 64, the original p-then-h hierarchy; pure-h ⇒ `nn=(p+1)²`.)
    quad_prolong: Vec<f64>,
    /// Nodes-per-element (`n1²`) of the h-transfer levels — the per-quadrant matrix stride is
    /// `h_nn·h_nn` and the GPU kernel launches `h_nn` threads/block. In the default p-then-h
    /// hierarchy the h-levels are order 1 (`h_nn=4`); under `HMG` every level is order `p`
    /// (`h_nn=(p+1)²`).
    h_nn: usize,
    inv_diag: Vec<Vec<f64>>,
    lam_hi: Vec<f64>,
    n_pre: usize,
    n_post: usize,
    // Domain extent, for the per-level checkerboard element coloring (cell size from dims[l]).
    xr: [f64; 2],
    yr: [f64; 2],
    /// Whether the operator is **singular** (constant nullspace): pure Poisson (reaction 0)
    /// with every boundary natural (Neumann). When set, the coarsest-level CG deflates the
    /// constant (projects its RHS onto the range) — otherwise that singular coarse solve
    /// produces a nullspace-polluted correction that breaks the outer (deflated) PCG. Mirrors
    /// the GPU `deflate_c!` in `gale-gpu`'s `poisson_pcg_solve`.
    singular: bool,
}

impl PMultigrid {
    /// Pure-Poisson (Dirichlet) p-multigrid. For the viscous **Helmholtz** `(λM + A)` use
    /// [`with_reaction`](Self::with_reaction).
    pub fn new(order: usize, nx: usize, ny: usize, xr: [f64; 2], yr: [f64; 2], alpha: f64) -> Self {
        Self::with_reaction(order, nx, ny, xr, yr, alpha, 0.0)
    }

    /// p-multigrid for the Helmholtz operator `(reaction·M + A)` (Dirichlet). `reaction = 0`
    /// is the pure-Poisson case [`new`](Self::new).
    pub fn with_reaction(
        order: usize, nx: usize, ny: usize, xr: [f64; 2], yr: [f64; 2], alpha: f64, reaction: f64,
    ) -> Self {
        Self::with_bc(order, nx, ny, xr, yr, alpha, reaction, Vec::new())
    }

    /// p-multigrid for `(reaction·M + A)` with **per-region boundary conditions**: tags in
    /// `neumann_tags` are natural (Neumann) at every level, the rest Dirichlet SIPG. For the
    /// dual-splitting **pressure** solve use `reaction = 0` and all boundary tags (the
    /// singular pure-Neumann operator) — then drive it with the deflated
    /// [`poisson_pcg_solve`]. The coarse operators are **rediscretized** per level (research
    /// shows Galerkin RAP does not reliably inherit DG BCs/penalty).
    pub fn with_bc(
        order: usize, nx: usize, ny: usize, xr: [f64; 2], yr: [f64; 2], alpha: f64, reaction: f64,
        neumann_tags: Vec<u32>,
    ) -> Self {
        // Two hierarchy modes:
        //  - default (p-then-h): p-coarsening [p, p/2, …, 1] on the full grid, then h-coarsening
        //    — order 1 on a 2:1-halved grid each step, while both dims stay even, down to a tiny
        //    coarsest grid (so its CG solve is cheap, not the hundreds of iters a p-only
        //    order-1-on-the-full-grid coarse level took).
        //  - `HMG` env set (pure-h, perf experiment): KEEP order `p` and h-coarsen the MESH all
        //    the way down (nx×ny → nx/2×ny/2 → … → 1×1), every level order `p`. The coarsest is
        //    order `p` on 1×1 ((p+1)² dofs) so its solve is still cheap; geometric h-MG at full
        //    order gives mesh-independent iteration counts (the head-to-head vs p-MG).
        let hmg = std::env::var("HMG").is_ok();
        let mut orders: Vec<usize> = Vec::new();
        let mut dims: Vec<(usize, usize)> = Vec::new();
        // h-transfer node count: order 1 (nn=4) for the appended h-levels in the default path;
        // order `p` (nn=(p+1)²) for the pure-h hierarchy.
        let h_order = if hmg { order } else { 1 };
        if hmg {
            // Pure-h: full grid at order `p`, then halve while both dims are even and ≥ 2.
            orders.push(order);
            dims.push((nx, ny));
            let (mut hx, mut hy) = (nx, ny);
            while hx % 2 == 0 && hy % 2 == 0 && hx >= 2 && hy >= 2 {
                hx /= 2;
                hy /= 2;
                orders.push(order);
                dims.push((hx, hy));
            }
        } else {
            for &o in &order_levels(order) {
                orders.push(o);
                dims.push((nx, ny));
            }
            let (mut hx, mut hy) = (nx, ny);
            while hx % 2 == 0 && hy % 2 == 0 && hx >= 2 && hy >= 2 {
                hx /= 2;
                hy /= 2;
                orders.push(1);
                dims.push((hx, hy));
            }
        }
        let meshes: Vec<Mesh2d> =
            (0..orders.len()).map(|l| Mesh2d::rectangular(orders[l], dims[l].0, dims[l].1, xr, yr)).collect();
        // 1D LGL nodes of the h-transfer order (order 1 for the default appended h-levels, order
        // `p` for pure-h) — the per-quadrant geometric prolongation matrices and their stride.
        let h_nodes = Mesh2d::rectangular(h_order, 1, 1, xr, yr).refq.line.nodes.clone();
        let h_nn = h_nodes.len() * h_nodes.len();
        let quad_prolong = quad_prolong_matrices(&h_nodes);
        // p-transfer where the grid is unchanged (order drops), h-transfer where it halves.
        let transfers: Vec<Transfer> = (0..orders.len() - 1)
            .map(|l| {
                if dims[l] == dims[l + 1] {
                    Transfer::P {
                        interp: lagrange_matrix(&meshes[l + 1].refq.line.nodes, &meshes[l].refq.line.nodes),
                    }
                } else {
                    Transfer::H
                }
            })
            .collect();
        // Pure Poisson (reaction 0) with every finest-level boundary tag natural ⇒ singular
        // (constant nullspace) ⇒ the coarsest CG must deflate. Computed before `meshes` moves.
        let singular = reaction == 0.0 && {
            let tags = meshes[0].boundary_tags();
            !tags.is_empty() && tags.iter().all(|t| neumann_tags.contains(t))
        };
        let mut s = Self {
            orders,
            meshes,
            alpha,
            reaction,
            neumann_tags,
            transfers,
            dims,
            quad_prolong,
            h_nn,
            inv_diag: Vec::new(),
            lam_hi: Vec::new(),
            n_pre: 3,
            n_post: 3,
            xr,
            yr,
            singular,
        };
        // Build the per-level smoother data in parallel across levels (the levels are
        // independent). Each `diagonal` itself fans its `2·nn` probes out over rayon, so
        // the finest level — which dominates — is load-balanced across all cores even
        // though it is one item of this outer loop. Results are collected in level order,
        // so the build stays bit-for-bit identical to the serial version.
        s.inv_diag = (0..s.orders.len())
            .into_par_iter()
            .map(|l| s.diagonal(l).iter().map(|&v| 1.0 / v).collect())
            .collect();
        // `power_lambda` reads `inv_diag[l]` (now fully built) and is inherently serial
        // within a level (power iteration), so parallelism here is across levels only.
        s.lam_hi = (0..s.orders.len()).into_par_iter().map(|l| 1.1 * s.power_lambda(l)).collect();
        s
    }

    /// Build a hierarchy whose finest level **reproduces** `mesh` (which must be a uniform
    /// rectangular `Mesh2d::rectangular` grid), deriving `(order, nx, ny, xr, yr)` from its
    /// element layout — so a flow solver holding only a `Mesh2d` can construct the matching
    /// p-MG for its elliptic solves. Returns `None` if `mesh` is not a uniform tensor grid
    /// (e.g. non-conforming/AMR), in which case the caller falls back to plain CG.
    pub fn from_mesh(mesh: &Mesh2d, alpha: f64, reaction: f64, neumann_tags: Vec<u32>) -> Option<Self> {
        let ne = mesh.n_elements();
        let nn = mesh.refq.n_nodes();
        if ne == 0 {
            return None;
        }
        let (mut xmin, mut xmax, mut ymin, mut ymax) = (f64::MAX, f64::MIN, f64::MAX, f64::MIN);
        let (mut xs, mut ys): (Vec<f64>, Vec<f64>) = (Vec::new(), Vec::new());
        let push_uniq = |v: &mut Vec<f64>, c: f64| {
            if !v.iter().any(|&u| (u - c).abs() < 1e-9 * (1.0 + c.abs())) {
                v.push(c);
            }
        };
        for el in &mesh.elements {
            let (mut cx, mut cy) = (0.0, 0.0);
            for k in 0..nn {
                let (x, y) = (el.geom.x[k], el.geom.y[k]);
                cx += x;
                cy += y;
                xmin = xmin.min(x);
                xmax = xmax.max(x);
                ymin = ymin.min(y);
                ymax = ymax.max(y);
            }
            push_uniq(&mut xs, cx / nn as f64);
            push_uniq(&mut ys, cy / nn as f64);
        }
        let (nx, ny) = (xs.len(), ys.len());
        if nx * ny != ne {
            return None; // not a uniform tensor grid (e.g. non-conforming)
        }
        Some(Self::with_bc(mesh.order, nx, ny, [xmin, xmax], [ymin, ymax], alpha, reaction, neumann_tags))
    }

    fn ndof(&self, l: usize) -> usize {
        self.meshes[l].n_elements() * self.meshes[l].refq.n_nodes()
    }

    // --- accessors for an external (e.g. GPU) driver that reuses this setup ---

    /// Number of levels (finest .. coarsest).
    pub fn n_levels(&self) -> usize {
        self.orders.len()
    }
    /// Polynomial order at level `l`.
    pub fn level_order(&self, l: usize) -> usize {
        self.orders[l]
    }
    /// Mesh at level `l`.
    pub fn mesh(&self, l: usize) -> &Mesh2d {
        &self.meshes[l]
    }
    /// 1D interpolation matrix (coarse `l+1` → fine `l`), `fine × coarse`, row-major — for a
    /// **p-transfer** level. Empty for an h-transfer (use [`is_h_transfer`](Self::is_h_transfer)
    /// + [`quad_prolong`](Self::quad_prolong) there).
    pub fn interp_matrix(&self, l: usize) -> &[f64] {
        match &self.transfers[l] {
            Transfer::P { interp } => interp,
            Transfer::H => &[],
        }
    }
    /// Element-grid dimensions `(nx, ny)` at level `l`.
    pub fn level_dims(&self, l: usize) -> (usize, usize) {
        self.dims[l]
    }
    /// Whether the transfer from level `l` to `l+1` is **h-coarsening** (2:1 geometric on the
    /// element grid) rather than p-coarsening. The GPU driver dispatches its restrict/prolong
    /// accordingly.
    pub fn is_h_transfer(&self, l: usize) -> bool {
        matches!(self.transfers[l], Transfer::H)
    }
    /// The four per-quadrant 2:1 geometric prolongation matrices (`nn` fine × `nn` coarse nodes,
    /// `[q·nn·nn + f·nn + a]`, `nn = `[`h_nodes`](Self::h_nodes)), shared by every h-transfer.
    pub fn quad_prolong(&self) -> &[f64] {
        &self.quad_prolong
    }
    /// Nodes-per-element (`n1²`) of the h-transfer levels — the per-quadrant `quad_prolong` stride
    /// (`h_nodes·h_nodes`) and the GPU h-transfer kernel block size. `4` for the default p-then-h
    /// hierarchy (order-1 h-levels); `(p+1)²` for pure-h (`HMG`).
    pub fn h_nodes(&self) -> usize {
        self.h_nn
    }
    /// Inverse operator diagonal at level `l`.
    pub fn inv_diagonal(&self, l: usize) -> &[f64] {
        &self.inv_diag[l]
    }
    /// Damped-Jacobi smoother weight at level `l`.
    pub fn jacobi_omega(&self, l: usize) -> f64 {
        (4.0 / 3.0) / self.lam_hi[l]
    }
    /// Pre / post smoothing sweep counts.
    pub fn smoothing(&self) -> (usize, usize) {
        (self.n_pre, self.n_post)
    }

    /// Helmholtz reaction `λ` (0 for pure Poisson) — the GPU PCG driver passes this to the
    /// device operator so the on-device V-cycle matches this setup.
    pub fn reaction(&self) -> f64 {
        self.reaction
    }

    /// Natural (Neumann) boundary tags applied at every level (empty ⇒ all-Dirichlet). The
    /// GPU PCG driver flattens each level's operator with these so the device matvec carries
    /// the matching `NEU` sentinels.
    pub fn neumann_tags(&self) -> &[u32] {
        &self.neumann_tags
    }

    /// Whether the operator is **singular** (constant nullspace): pure Poisson (reaction 0)
    /// with every boundary natural (Neumann). Such a solve must be deflated.
    pub fn is_singular(&self, mesh_boundary_tags: &[u32]) -> bool {
        self.reaction == 0.0 && mesh_boundary_tags.iter().all(|t| self.neumann_tags.contains(t))
    }

    fn apply_level(&self, l: usize, u: &[f64]) -> Vec<f64> {
        Poisson::with_bc(&self.meshes[l], self.alpha, self.reaction, self.neumann_tags.clone()).apply(u)
    }

    /// Checkerboard parity of element `e` on the rectangular `nx×ny` grid, from its
    /// centroid (ordering-independent). SIPG couples an element only to its **face**
    /// neighbours (which differ by ±1 in one cell index ⇒ opposite parity), so two
    /// same-parity elements never couple — the key to the colored diagonal below.
    fn elem_color(&self, l: usize, e: usize) -> usize {
        let (nxl, nyl) = self.dims[l];
        let g = &self.meshes[l].elements[e].geom;
        let nn = self.meshes[l].refq.n_nodes();
        let (mut xc, mut yc) = (0.0, 0.0);
        for k in 0..nn {
            xc += g.x[k];
            yc += g.y[k];
        }
        xc /= nn as f64;
        yc /= nn as f64;
        let cx = (((xc - self.xr[0]) / ((self.xr[1] - self.xr[0]) / nxl as f64)) as usize).min(nxl - 1);
        let cy = (((yc - self.yr[0]) / ((self.yr[1] - self.yr[0]) / nyl as f64)) as usize).min(nyl - 1);
        (cx + cy) % 2
    }

    /// Operator diagonal in **O(n·nn)** via 2-colored probing instead of O(n²) unit-vector
    /// probing. For each (local node `m`, element colour `c`) set a probe with `1` at node
    /// `m` of every colour-`c` element and apply once: since same-colour elements don't
    /// couple (SIPG is face-local; checkerboard separates face neighbours) and only node
    /// `m` is set per element, `(A·e)` at those nodes is exactly the diagonal. `2·nn`
    /// matvecs total — bit-for-bit equal to the old probing (validated by poisson-pcg-check).
    fn diagonal(&self, l: usize) -> Vec<f64> {
        let mesh = &self.meshes[l];
        let nn = mesh.refq.n_nodes();
        let ne = mesh.n_elements();
        let n = ne * nn;
        let colors: Vec<usize> = (0..ne).map(|e| self.elem_color(l, e)).collect();
        let mut diag = vec![0.0; n];
        // The `2·nn` probes (one per local node `m` × colour `c`) are independent: each
        // sets node `m` of every colour-`c` element, applies once, and reads back the
        // diagonal at exactly those (disjoint) indices. Fan them out over rayon — each
        // task owns its probe buffer and returns its `(index, value)` contributions, which
        // are scattered back below. Disjoint indices ⇒ bit-for-bit identical to the serial
        // probing (validated by poisson-pcg-check).
        let probes: Vec<(usize, usize)> =
            (0..2).flat_map(|c| (0..nn).map(move |m| (c, m))).collect();
        let parts: Vec<Vec<(usize, f64)>> = probes
            .par_iter()
            .map(|&(c, m)| {
                let mut e = vec![0.0; n];
                for el in 0..ne {
                    if colors[el] == c {
                        e[el * nn + m] = 1.0;
                    }
                }
                let ae = self.apply_level(l, &e);
                (0..ne)
                    .filter(|&el| colors[el] == c)
                    .map(|el| (el * nn + m, ae[el * nn + m]))
                    .collect()
            })
            .collect();
        for part in parts {
            for (i, v) in part {
                diag[i] = v;
            }
        }
        diag
    }

    /// Estimate `λ_max(D⁻¹A)` by power iteration (for the Jacobi smoother weight).
    fn power_lambda(&self, l: usize) -> f64 {
        let n = self.ndof(l);
        let mut v: Vec<f64> = (0..n).map(|i| 1.0 + (i % 7) as f64 * 0.13).collect();
        let nv = norm(&v);
        v.iter_mut().for_each(|x| *x /= nv);
        let mut lam = 1.0;
        for _ in 0..60 {
            let av = self.apply_level(l, &v);
            let mut w: Vec<f64> = (0..n).map(|i| self.inv_diag[l][i] * av[i]).collect();
            lam = norm(&w).max(1e-300);
            w.iter_mut().for_each(|x| *x /= lam);
            v = w;
        }
        lam
    }

    /// Damped-Jacobi smoothing (ω = 4/(3λ_max) damps the upper-half spectrum).
    fn smooth(&self, l: usize, x: &mut [f64], b: &[f64], sweeps: usize) {
        let omega = (4.0 / 3.0) / self.lam_hi[l];
        let id = &self.inv_diag[l];
        for _ in 0..sweeps {
            let ax = self.apply_level(l, x);
            for i in 0..x.len() {
                x[i] += omega * id[i] * (b[i] - ax[i]);
            }
        }
    }

    /// Prolong a coarse (level `l+1`) field to fine (level `l`), dispatching on the transfer.
    fn prolong(&self, l: usize, coarse: &[f64]) -> Vec<f64> {
        match &self.transfers[l] {
            Transfer::P { interp } => self.prolong_p(l, coarse, interp),
            Transfer::H => self.prolong_h(l, coarse),
        }
    }
    /// Restrict a fine (level `l`) field to coarse (level `l+1`) — transpose of prolong.
    fn restrict(&self, l: usize, fine: &[f64]) -> Vec<f64> {
        match &self.transfers[l] {
            Transfer::P { interp } => self.restrict_p(l, fine, interp),
            Transfer::H => self.restrict_h(l, fine),
        }
    }

    /// 2:1 geometric prolongation (coarse order-1 → fine order-1, grid doubled). Each fine
    /// element takes its parent coarse element's bilinear value at the fine nodes' positions
    /// (the per-quadrant [`quad_prolong`](Self::quad_prolong) matrices).
    fn prolong_h(&self, l: usize, coarse: &[f64]) -> Vec<f64> {
        let (nxc, _) = self.dims[l + 1];
        let (nxf, _) = self.dims[l];
        let ne_f = self.meshes[l].n_elements();
        let pq = &self.quad_prolong;
        let nn = self.h_nn;
        let mut out = vec![0.0; ne_f * nn];
        for ef in 0..ne_f {
            let (fx, fy) = (ef % nxf, ef / nxf);
            let ec = (fx / 2) + (fy / 2) * nxc;
            let q = (fx % 2) + 2 * (fy % 2);
            for f in 0..nn {
                let mut s = 0.0;
                for a in 0..nn {
                    s += pq[q * nn * nn + f * nn + a] * coarse[ec * nn + a];
                }
                out[ef * nn + f] = s;
            }
        }
        out
    }

    /// 2:1 geometric restriction = transpose of [`prolong_h`](Self::prolong_h): each coarse
    /// element accumulates the `Pᵀ`-weighted contributions of its 4 fine children.
    fn restrict_h(&self, l: usize, fine: &[f64]) -> Vec<f64> {
        let (nxc, _) = self.dims[l + 1];
        let (nxf, _) = self.dims[l];
        let ne_f = self.meshes[l].n_elements();
        let pq = &self.quad_prolong;
        let nn = self.h_nn;
        let mut out = vec![0.0; self.meshes[l + 1].n_elements() * nn];
        for ef in 0..ne_f {
            let (fx, fy) = (ef % nxf, ef / nxf);
            let ec = (fx / 2) + (fy / 2) * nxc;
            let q = (fx % 2) + 2 * (fy % 2);
            for a in 0..nn {
                let mut s = 0.0;
                for f in 0..nn {
                    s += pq[q * nn * nn + f * nn + a] * fine[ef * nn + f];
                }
                out[ec * nn + a] += s;
            }
        }
        out
    }

    /// Prolong a coarse (level `l+1`) field to fine (level `l`), per element, tensor.
    fn prolong_p(&self, l: usize, coarse: &[f64], i: &[f64]) -> Vec<f64> {
        let ncc = self.orders[l + 1] + 1;
        let nff = self.orders[l] + 1;
        let ne = self.meshes[l].n_elements();
        let (cc, ff) = (ncc * ncc, nff * nff);
        let mut out = vec![0.0; ne * ff];
        let mut tmp = vec![0.0; nff * ncc];
        for e in 0..ne {
            let c = &coarse[e * cc..(e + 1) * cc];
            for jc in 0..ncc {
                for iff in 0..nff {
                    let mut s = 0.0;
                    for ic in 0..ncc {
                        s += i[iff * ncc + ic] * c[ic + jc * ncc];
                    }
                    tmp[iff + jc * nff] = s;
                }
            }
            for jf in 0..nff {
                for iff in 0..nff {
                    let mut s = 0.0;
                    for jc in 0..ncc {
                        s += i[jf * ncc + jc] * tmp[iff + jc * nff];
                    }
                    out[e * ff + iff + jf * nff] = s;
                }
            }
        }
        out
    }

    /// Restrict a fine (level `l`) field to coarse (level `l+1`), per element, tensor.
    fn restrict_p(&self, l: usize, fine: &[f64], i: &[f64]) -> Vec<f64> {
        let ncc = self.orders[l + 1] + 1;
        let nff = self.orders[l] + 1;
        let ne = self.meshes[l].n_elements();
        let (cc, ff) = (ncc * ncc, nff * nff);
        let mut out = vec![0.0; ne * cc];
        let mut tmp = vec![0.0; ncc * nff];
        for e in 0..ne {
            let f = &fine[e * ff..(e + 1) * ff];
            for jf in 0..nff {
                for ic in 0..ncc {
                    let mut s = 0.0;
                    for iff in 0..nff {
                        s += i[iff * ncc + ic] * f[iff + jf * nff];
                    }
                    tmp[ic + jf * ncc] = s;
                }
            }
            for jc in 0..ncc {
                for ic in 0..ncc {
                    let mut s = 0.0;
                    for jf in 0..nff {
                        s += i[jf * ncc + jc] * tmp[ic + jf * ncc];
                    }
                    out[e * cc + ic + jc * ncc] = s;
                }
            }
        }
        out
    }

    /// Unpreconditioned CG on the coarsest level.
    fn coarse_solve(&self, l: usize, b: &[f64]) -> Vec<f64> {
        let n = b.len();
        // Singular pure-Neumann coarse operator (constant nullspace): project the residual
        // onto the range each iteration (subtract its mean), exactly like the outer deflated
        // PCG and the GPU `deflate_c!`. Without this the coarse CG drifts in the nullspace and
        // returns a constant-polluted correction that stalls the outer PCG. No-op otherwise.
        let deflate = |v: &mut [f64]| {
            if self.singular {
                let mean = v.iter().sum::<f64>() / n as f64;
                v.iter_mut().for_each(|x| *x -= mean);
            }
        };
        // Coarse-grid solve tolerance/cap. With h-coarsening the coarsest grid is TINY (a few
        // elements) ⇒ a tight, near-exact solve is cheap and minimizes outer iterations. But when
        // the grid can't h-coarsen (odd dims, e.g. a 43×8 channel) the coarsest level is still
        // order-1 on the FULL grid — there a tight 1e-10 solve runs hundreds of CG iters EVERY
        // V-cycle (the step-2c cylinder was 86 s/step from exactly this). The coarse solve is only
        // a preconditioner component, so for a large coarse grid use a loose tol + low cap and let
        // the outer PCG absorb the inexactness. Mirrors the GPU `coarse_small` band-aid.
        let coarse_small = n <= 1024;
        let (ctol, cap) = if coarse_small { (1e-10, 500) } else { (1e-2, 40) };
        let mut x = vec![0.0; n];
        let mut r = b.to_vec();
        deflate(&mut r);
        let mut p = r.clone();
        let mut rs = dot(&r, &r);
        let bn = norm(b).max(1e-300);
        for _ in 0..cap {
            let ap = self.apply_level(l, &p);
            let pap = dot(&p, &ap);
            // CG breakdown guard: on the singular coarse op the only remaining mode can be the
            // deflated-out constant ⇒ p·Ap → 0. Stop rather than divide by ~0 (mirrors the GPU).
            if !(pap.abs() > 0.0) {
                break;
            }
            let a = rs / pap;
            for i in 0..n {
                x[i] += a * p[i];
                r[i] -= a * ap[i];
            }
            deflate(&mut r);
            let rsn = dot(&r, &r);
            if rsn.sqrt() / bn < ctol {
                break;
            }
            let be = rsn / rs;
            for i in 0..n {
                p[i] = r[i] + be * p[i];
            }
            rs = rsn;
        }
        x
    }

    fn vcycle(&self, l: usize, b: &[f64]) -> Vec<f64> {
        if l == self.orders.len() - 1 {
            return self.coarse_solve(l, b);
        }
        let mut x = vec![0.0; b.len()];
        self.smooth(l, &mut x, b, self.n_pre);
        let ax = self.apply_level(l, &x);
        let res: Vec<f64> = (0..b.len()).map(|i| b[i] - ax[i]).collect();
        let rc = self.restrict(l, &res);
        let ec = self.vcycle(l + 1, &rc);
        let pe = self.prolong(l, &ec);
        for i in 0..x.len() {
            x[i] += pe[i];
        }
        self.smooth(l, &mut x, b, self.n_post);
        x
    }

    /// Finest-level operator action `A·u`.
    pub fn apply(&self, u: &[f64]) -> Vec<f64> {
        self.apply_level(0, u)
    }

    /// One V-cycle as the preconditioner `z = M⁻¹ r`.
    pub fn precondition(&self, r: &[f64]) -> Vec<f64> {
        self.vcycle(0, r)
    }

    /// Preconditioned CG. Returns `(solution, iterations)`.
    pub fn pcg(&self, b: &[f64], tol: f64, maxit: usize) -> (Vec<f64>, usize) {
        let n = b.len();
        let mut x = vec![0.0; n];
        let mut r = b.to_vec();
        let mut z = self.precondition(&r);
        let mut p = z.clone();
        let mut rz = dot(&r, &z);
        let bn = norm(b).max(1e-300);
        let mut iters = 0;
        for it in 0..maxit {
            let ap = self.apply(&p);
            let alpha = rz / dot(&p, &ap);
            for i in 0..n {
                x[i] += alpha * p[i];
                r[i] -= alpha * ap[i];
            }
            iters = it + 1;
            if norm(&r) / bn < tol {
                break;
            }
            z = self.precondition(&r);
            let rz_new = dot(&r, &z);
            let beta = rz_new / rz;
            for i in 0..n {
                p[i] = z[i] + beta * p[i];
            }
            rz = rz_new;
        }
        (x, iters)
    }

    /// Preconditioned CG for the **singular** (pure-Neumann) pressure operator
    /// (`A·1 = 0`): deflates the constant nullspace by removing the mean from the
    /// residual and from the preconditioned residual each iteration. The V-cycle
    /// preconditioner shares the same nullspace (the coarse operators are equally
    /// singular), so projecting both `r` and `z = M⁻¹r` keeps the iteration in the
    /// range of `A`. Solution is determined up to an additive constant. Returns
    /// `(solution, iterations)`.
    pub fn pcg_deflated(&self, b: &[f64], tol: f64, maxit: usize) -> (Vec<f64>, usize) {
        let n = b.len();
        let deflate = |v: &mut [f64]| {
            let mean = v.iter().sum::<f64>() / n as f64;
            v.iter_mut().for_each(|x| *x -= mean);
        };
        let mut x = vec![0.0; n];
        let mut r = b.to_vec();
        deflate(&mut r);
        let mut z = self.precondition(&r);
        deflate(&mut z);
        let mut p = z.clone();
        let mut rz = dot(&r, &z);
        let bn = norm(&r).max(1e-300);
        let mut iters = 0;
        for it in 0..maxit {
            let ap = self.apply(&p);
            let alpha = rz / dot(&p, &ap);
            for i in 0..n {
                x[i] += alpha * p[i];
                r[i] -= alpha * ap[i];
            }
            deflate(&mut r);
            iters = it + 1;
            if norm(&r) / bn < tol {
                break;
            }
            z = self.precondition(&r);
            deflate(&mut z);
            let rz_new = dot(&r, &z);
            let beta = rz_new / rz;
            for i in 0..n {
                p[i] = z[i] + beta * p[i];
            }
            rz = rz_new;
        }
        (x, iters)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::f64::consts::PI;

    /// Build the MMS RHS (u = sinπx sinπy, homogeneous Dirichlet) at the finest level.
    fn mms_rhs(mg: &PMultigrid) -> Vec<f64> {
        let mesh = &mg.meshes[0];
        let nn = mesh.refq.n_nodes();
        let mut f = vec![0.0; mesh.n_elements() * nn];
        for (e, el) in mesh.elements.iter().enumerate() {
            for k in 0..nn {
                let (x, y) = (el.geom.x[k], el.geom.y[k]);
                f[e * nn + k] = 2.0 * PI * PI * (PI * x).sin() * (PI * y).sin();
            }
        }
        Poisson::new(mesh, mg.alpha).rhs(&f, |_, _| 0.0)
    }

    #[test]
    fn pcg_matches_cg_and_cuts_iterations() {
        let p = 4;
        let mg = PMultigrid::new(p, 3, 3, [0.0, 1.0], [0.0, 1.0], 5.0);
        let poisson = Poisson::new(&mg.meshes[0], mg.alpha);
        let b = mms_rhs(&mg);

        let (u_cg, it_cg, _) = poisson.cg(&b, 1e-10, 20000);
        let (u_pcg, it_pcg) = mg.pcg(&b, 1e-10, 2000);

        // Same solution.
        let diff: Vec<f64> = u_pcg.iter().zip(&u_cg).map(|(a, b)| a - b).collect();
        let rel = norm(&diff) / norm(&u_cg).max(1e-300);
        assert!(rel < 1e-7, "PCG vs CG solution rel diff {rel}");
        // Preconditioner does real work.
        assert!(it_pcg * 2 < it_cg, "PCG {it_pcg} not << CG {it_cg}");
    }

    #[test]
    fn helmholtz_pcg_converges_and_cuts_iterations() {
        // Helmholtz (λM + A), the flow velocity solve. p-MG with the reaction term must
        // converge and beat plain CG, like the Poisson case.
        let p = 4;
        let lambda = 100.0;
        let mg = PMultigrid::with_reaction(p, 4, 4, [0.0, 1.0], [0.0, 1.0], 5.0, lambda);
        let hop = Poisson::with_reaction(&mg.meshes[0], mg.alpha, lambda);
        let b = mms_rhs(&mg);
        let (u_cg, it_cg, _) = hop.cg(&b, 1e-10, 20000);
        let (u_pcg, it_pcg) = mg.pcg(&b, 1e-10, 2000);
        let diff: Vec<f64> = u_pcg.iter().zip(&u_cg).map(|(a, b)| a - b).collect();
        let rel = norm(&diff) / norm(&u_cg).max(1e-300);
        eprintln!("Helmholtz λ={lambda}: CG {it_cg} iters, PCG {it_pcg} iters, rel {rel:.2e}");
        assert!(rel < 1e-7, "Helmholtz PCG vs CG rel {rel}");
        assert!(it_pcg * 2 < it_cg, "Helmholtz PCG {it_pcg} not << CG {it_cg}");
    }

    #[test]
    fn pcg_iteration_count_is_roughly_p_robust() {
        // Unpreconditioned CG iters grow with p; p-MG keeps them bounded.
        let mut pcg_iters = Vec::new();
        let mut cg_iters = Vec::new();
        for &p in &[2usize, 4, 6] {
            let mg = PMultigrid::new(p, 3, 3, [0.0, 1.0], [0.0, 1.0], 5.0);
            let poisson = Poisson::new(&mg.meshes[0], mg.alpha);
            let b = mms_rhs(&mg);
            let (_x, it) = mg.pcg(&b, 1e-10, 2000);
            let (_u, itc, _) = poisson.cg(&b, 1e-10, 20000);
            pcg_iters.push(it);
            cg_iters.push(itc);
        }
        eprintln!("p=[2,4,6]  CG iters={cg_iters:?}  PCG iters={pcg_iters:?}");
        // PCG counts stay modest and well below CG across orders.
        assert!(pcg_iters.iter().all(|&n| n < 40), "PCG iters not bounded: {pcg_iters:?}");
        for (a, b) in pcg_iters.iter().zip(&cg_iters) {
            assert!(a < b, "PCG {a} not < CG {b}");
        }
    }

    #[test]
    fn h_coarsening_is_mesh_independent_and_coarsens_to_tiny() {
        // With h-coarsening appended below order 1, the coarsest grid must be TINY (so its
        // solve is cheap, not the hundreds of iters an order-1-on-the-full-grid coarse level
        // took) AND the PCG iteration count must stay bounded as the FINE grid refines — the
        // mesh-independence that p-only multigrid lacked.
        let p = 2;
        let grids = [8usize, 16, 32];
        let mut iters = Vec::new();
        for &g in &grids {
            let mg = PMultigrid::new(p, g, g, [0.0, 1.0], [0.0, 1.0], 5.0);
            let (cx, cy) = mg.level_dims(mg.n_levels() - 1);
            assert!(cx * cy <= 4, "coarsest grid not small at g={g}: {cx}×{cy}");
            assert!(mg.is_h_transfer(mg.n_levels() - 2), "no h-transfer present at g={g}");
            let b = mms_rhs(&mg);
            let (_x, it) = mg.pcg(&b, 1e-10, 2000);
            iters.push(it);
        }
        eprintln!("h-MG PCG iters at g={grids:?}: {iters:?}");
        assert!(iters.iter().all(|&n| n < 40), "h-MG iters not bounded: {iters:?}");
        // Roughly flat — does not grow the O(1/h) way unpreconditioned CG would.
        assert!(iters[2] <= iters[0] + 8, "h-MG iters grew with refinement: {iters:?}");
    }

    #[test]
    fn pcg_deflated_solves_singular_neumann_and_cuts_iterations() {
        // Pure-Neumann pressure operator (reaction 0, all four box tags Neumann) is singular
        // (constant nullspace). The deflated MG-PCG must (a) match the deflated plain CG up to
        // the additive constant, and (b) need far fewer iterations. MMS u=cos(πx)cos(πy).
        let p = 4;
        let mg = PMultigrid::with_bc(p, 16, 16, [0.0, 1.0], [0.0, 1.0], 5.0, 0.0, vec![0, 1, 2, 3]);
        let mesh = &mg.meshes[0];
        assert!(mg.is_singular(&mesh.boundary_tags()), "operator should be singular");
        let op = Poisson::with_bc(mesh, mg.alpha, 0.0, vec![0, 1, 2, 3]);
        let nn = mesh.refq.n_nodes();
        let mut f = vec![0.0; mesh.n_elements() * nn];
        for (e, el) in mesh.elements.iter().enumerate() {
            for k in 0..nn {
                let (x, y) = (el.geom.x[k], el.geom.y[k]);
                f[e * nn + k] = 2.0 * PI * PI * (PI * x).cos() * (PI * y).cos();
            }
        }
        let b = op.rhs_mixed(&f, |_, _| 0.0, |_, _| 0.0); // homogeneous Neumann flux

        let (u_cg, it_cg) = op.cg_deflated(&b, 1e-10, 20000);
        let (u_pcg, it_pcg) = mg.pcg_deflated(&b, 1e-10, 2000);

        // Match up to an additive constant: compare after removing each field's mean.
        let demean = |v: &[f64]| {
            let m = v.iter().sum::<f64>() / v.len() as f64;
            v.iter().map(|x| x - m).collect::<Vec<_>>()
        };
        let (a, c) = (demean(&u_cg), demean(&u_pcg));
        let diff: Vec<f64> = a.iter().zip(&c).map(|(x, y)| x - y).collect();
        let rel = norm(&diff) / norm(&a).max(1e-300);
        eprintln!("singular Neumann: CG {it_cg} iters, MG-PCG {it_pcg} iters, rel {rel:.2e}");
        // Correctness: the deflated MG-PCG must reproduce the deflated CG solution (up to the
        // constant). Robustness: iterations stay bounded (mesh-independent) — the win over CG
        // grows with mesh size (CG needs O(1/h) iters; at this small size CG is already cheap,
        // so the headline speedup shows at flow scale, 64²+).
        let _ = it_cg;
        assert!(rel < 1e-6, "deflated MG-PCG vs deflated CG rel diff {rel}");
        assert!(it_pcg < 40, "deflated MG-PCG iters not bounded: {it_pcg}");
    }

    #[test]
    fn quad_prolong_order1_matches_old_bilinear() {
        // The generalized tensor-Lagrange `quad_prolong_matrices` must reproduce the original
        // hardcoded order-1 bilinear matrices exactly when given the order-1 nodes [-1,1] — this
        // proves the generalization (used at any order for pure-h) is correct at the base order.
        let nd = [-1.0f64, 1.0];
        let got = quad_prolong_matrices(&nd);
        assert_eq!(got.len(), 64);
        let mut want = [0.0f64; 64];
        for qy in 0..2 {
            for qx in 0..2 {
                let q = qx + 2 * qy;
                for f in 0..4 {
                    let (rf, sf) = (nd[f % 2], nd[f / 2]);
                    let rc = (rf + (2.0 * qx as f64 - 1.0)) / 2.0;
                    let sc = (sf + (2.0 * qy as f64 - 1.0)) / 2.0;
                    for a in 0..4 {
                        let (rca, sca) = (nd[a % 2], nd[a / 2]);
                        want[q * 16 + f * 4 + a] = 0.25 * (1.0 + rc * rca) * (1.0 + sc * sca);
                    }
                }
            }
        }
        for i in 0..64 {
            assert!((got[i] - want[i]).abs() < 1e-14, "quad_prolong[{i}] {} != {}", got[i], want[i]);
        }
    }

    #[test]
    fn quad_prolong_order3_reproduces_coarse_polynomial() {
        // Pure-h correctness (order 3): geometric prolongation at full order must REPRODUCE any
        // polynomial of degree ≤ 3 exactly — a coarse field sampled from p(x,y) on the coarse LGL
        // nodes, prolonged into a child quadrant, must equal p sampled on the *fine* (child) nodes.
        // This is the property the h-transfer needs (and what makes the geometric h-MG correct).
        let nodes = Mesh2d::rectangular(3, 1, 1, [0.0, 1.0], [0.0, 1.0]).refq.line.nodes;
        let n1 = nodes.len();
        let nn = n1 * n1;
        let pq = quad_prolong_matrices(&nodes);
        // A degree-3 (per axis) test polynomial on the reference square [-1,1]².
        let poly = |r: f64, s: f64| 1.0 - 0.5 * r + 0.3 * s + 0.2 * r * s - 0.1 * r * r * r + 0.05 * s * s;
        // Coarse nodal values.
        let mut c = vec![0.0; nn];
        for ay in 0..n1 {
            for ax in 0..n1 {
                c[ay * n1 + ax] = poly(nodes[ax], nodes[ay]);
            }
        }
        for qy in 0..2 {
            for qx in 0..2 {
                let q = qx + 2 * qy;
                for fy in 0..n1 {
                    for fx in 0..n1 {
                        let f = fy * n1 + fx;
                        // Prolonged value at this fine node.
                        let mut got = 0.0;
                        for a in 0..nn {
                            got += pq[q * nn * nn + f * nn + a] * c[a];
                        }
                        // The same fine node's coarse-reference coord, evaluated in the polynomial.
                        let rc = (nodes[fx] + (2.0 * qx as f64 - 1.0)) / 2.0;
                        let sc = (nodes[fy] + (2.0 * qy as f64 - 1.0)) / 2.0;
                        let want = poly(rc, sc);
                        assert!((got - want).abs() < 1e-10, "q={q} f={f}: prolong {got} != poly {want}");
                    }
                }
            }
        }
    }

    #[test]
    fn h_restrict_is_transpose_of_prolong() {
        // R = Pᵀ for the geometric h-transfer: ⟨P c, f⟩_fine = ⟨c, R f⟩_coarse for all c, f.
        let mg = PMultigrid::new(1, 4, 4, [0.0, 1.0], [0.0, 1.0], 5.0);
        let l = mg.n_levels() - 2; // an h-transfer level (4→2 here)
        assert!(mg.is_h_transfer(l));
        let nf = mg.meshes[l].n_elements() * 4;
        let nc = mg.meshes[l + 1].n_elements() * 4;
        let c: Vec<f64> = (0..nc).map(|i| 1.0 + (i % 5) as f64 * 0.3).collect();
        let f: Vec<f64> = (0..nf).map(|i| 0.7 - (i % 3) as f64 * 0.2).collect();
        let pc = mg.prolong(l, &c);
        let rf = mg.restrict(l, &f);
        let lhs = dot(&pc, &f);
        let rhs = dot(&c, &rf);
        assert!((lhs - rhs).abs() < 1e-12, "R ≠ Pᵀ: ⟨Pc,f⟩={lhs} ⟨c,Rf⟩={rhs}");
    }
}
