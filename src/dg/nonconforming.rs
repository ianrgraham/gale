//! Non-conforming (2:1) DG with mortar coupling — the adaptive-solver core. A coarse
//! element's edge may face two refined neighbors; the numerical flux is computed on
//! the fine "mortar" resolution and projected back to the coarse edge (see
//! [`super::amr`]). This module demonstrates a working non-conforming **advection**
//! operator on the minimal real configuration and validates the two properties that
//! matter: free-stream preservation and exactness across the hanging-node interface.
//!
//! Geometry (one 2:1 interface):
//! ```text
//!   ┌─────────┬────────┐  C  = coarse  [0,1]×[0,1]
//!   │         │   R1   │  R1 = fine    [1,2]×[0.5,1]
//!   │    C    ├────────┤  R0 = fine    [1,2]×[0,0.5]
//!   │         │   R0   │  C.East (full) ↔ R0.West (lower half) + R1.West (upper half)
//!   └─────────┴────────┘  R0.North ↔ R1.South is conforming; all else boundary.
//! ```
//! Conservation across the hanging node holds because the coarse-edge Jacobian (½) ×
//! the mortar restriction Jacobian (½) equals the fine-edge Jacobian (¼).

use super::amr::RefineQuad;
use super::face::{quad_faces, Edge, FaceData};
use super::geometry::QuadGeometry;
use super::quad::Reference2dQuad;
use std::collections::HashMap;

/// What lies across one edge of an element in a (possibly refined) mesh.
#[derive(Clone, Debug)]
pub enum NcNeighbor {
    Boundary { tag: u32 },
    /// Same-level neighbor (conforming face).
    Conforming { elem: usize, edge: Edge },
    /// This element is the coarse side of a 2:1 interface, facing two finer elements
    /// (ordered by the edge's tangential coordinate, half 0 then 1).
    CoarseToFine { fine: [(usize, Edge); 2] },
    /// This element is a fine side, covering `half ∈ {0,1}` of a coarser neighbor's
    /// edge (handled from the coarse side; carried for completeness).
    FineToCoarse { coarse: usize, edge: Edge, half: usize },
}

/// A Cartesian quad mesh with single-level `h`-refinement of selected cells, with
/// 2:1-balanced non-conforming connectivity. Elements: unrefined cells contribute one
/// element; refined cells contribute four children. The general adaptive-mesh data
/// structure on which the non-conforming solver runs.
pub struct NcMesh {
    pub refq: Reference2dQuad,
    pub geom: Vec<QuadGeometry>,
    pub faces: Vec<[FaceData; 4]>,
    pub neigh: Vec<[NcNeighbor; 4]>,
}

// Children of a refined cell, indexed c = sx + 2*sy (sx,sy ∈ {0,1}).
#[derive(Clone, Copy)]
enum Cell {
    Single(usize),
    Quad([usize; 4]),
}

const EDGES: [Edge; 4] = [Edge::South, Edge::East, Edge::North, Edge::West];

// (dx, dy) cell offset and the opposing edge for each edge.
fn edge_offset(e: Edge) -> (i64, i64) {
    match e {
        Edge::South => (0, -1),
        Edge::East => (1, 0),
        Edge::North => (0, 1),
        Edge::West => (-1, 0),
    }
}
fn opposite(e: Edge) -> Edge {
    match e {
        Edge::South => Edge::North,
        Edge::East => Edge::West,
        Edge::North => Edge::South,
        Edge::West => Edge::East,
    }
}
fn boundary_tag(e: Edge) -> u32 {
    e as u32
}

impl NcMesh {
    /// Build a Cartesian `nx × ny` mesh, refining the listed cells once each.
    pub fn cartesian_refined(
        order: usize,
        nx: usize,
        ny: usize,
        xr: [f64; 2],
        yr: [f64; 2],
        refine: &[(usize, usize)],
    ) -> Self {
        let refq = Reference2dQuad::new(order);
        let (x0, y0) = (xr[0], yr[0]);
        let dx = (xr[1] - xr[0]) / nx as f64;
        let dy = (yr[1] - yr[0]) / ny as f64;
        let refined: std::collections::HashSet<(usize, usize)> = refine.iter().copied().collect();

        // Assign element indices and build geometry.
        let mut geom = Vec::new();
        let mut cells: HashMap<(usize, usize), Cell> = HashMap::new();
        let quad = |gx0: f64, gy0: f64, w: f64, h: f64| {
            QuadGeometry::from_corners(
                &refq,
                [[gx0, gy0], [gx0 + w, gy0], [gx0 + w, gy0 + h], [gx0, gy0 + h]],
            )
        };
        for cy in 0..ny {
            for cx in 0..nx {
                let (cell_x, cell_y) = (x0 + cx as f64 * dx, y0 + cy as f64 * dy);
                if refined.contains(&(cx, cy)) {
                    let mut ids = [0usize; 4];
                    for sy in 0..2 {
                        for sx in 0..2 {
                            ids[sx + 2 * sy] = geom.len();
                            geom.push(quad(
                                cell_x + sx as f64 * 0.5 * dx,
                                cell_y + sy as f64 * 0.5 * dy,
                                0.5 * dx,
                                0.5 * dy,
                            ));
                        }
                    }
                    cells.insert((cx, cy), Cell::Quad(ids));
                } else {
                    let id = geom.len();
                    geom.push(quad(cell_x, cell_y, dx, dy));
                    cells.insert((cx, cy), Cell::Single(id));
                }
            }
        }
        let faces: Vec<_> = geom.iter().map(|g| quad_faces(&refq, g)).collect();

        // Connectivity.
        let cell_at = |cx: i64, cy: i64| -> Option<Cell> {
            if cx < 0 || cy < 0 || cx as usize >= nx || cy as usize >= ny {
                None
            } else {
                cells.get(&(cx as usize, cy as usize)).copied()
            }
        };
        // The two children of a neighbor cell on its `side` (= the edge facing us),
        // ordered by the shared edge's tangential coordinate.
        let side_children = |ch: [usize; 4], side: Edge| -> [(usize, Edge); 2] {
            let c = |sx: usize, sy: usize| ch[sx + 2 * sy];
            match side {
                Edge::North => [(c(0, 1), Edge::North), (c(1, 1), Edge::North)], // order by x
                Edge::South => [(c(0, 0), Edge::South), (c(1, 0), Edge::South)],
                Edge::West => [(c(0, 0), Edge::West), (c(0, 1), Edge::West)], // order by y
                Edge::East => [(c(1, 0), Edge::East), (c(1, 1), Edge::East)],
            }
        };

        let mut neigh: Vec<[NcNeighbor; 4]> =
            geom.iter().map(|_| std::array::from_fn(|_| NcNeighbor::Boundary { tag: 0 })).collect();

        for cy in 0..ny {
            for cx in 0..ny.max(nx) {
                if cx >= nx {
                    break;
                }
                let cell = cells[&(cx, cy)];
                for &e in &EDGES {
                    let (ox, oy) = edge_offset(e);
                    let nb = cell_at(cx as i64 + ox, cy as i64 + oy);
                    let opp = opposite(e);
                    match cell {
                        Cell::Single(id) => {
                            neigh[id][e as usize] = match nb {
                                None => NcNeighbor::Boundary { tag: boundary_tag(e) },
                                Some(Cell::Single(n)) => NcNeighbor::Conforming { elem: n, edge: opp },
                                Some(Cell::Quad(ch)) => {
                                    NcNeighbor::CoarseToFine { fine: side_children(ch, opp) }
                                }
                            };
                        }
                        Cell::Quad(ch) => {
                            // Each child: internal edges → sibling (conforming);
                            // external edges → the neighbor cell.
                            for sy in 0..2usize {
                                for sx in 0..2usize {
                                    let id = ch[sx + 2 * sy];
                                    // Is edge e internal (toward a sibling) or external?
                                    let (internal, sib) = match e {
                                        Edge::South => (sy == 1, (sx, 0usize)),
                                        Edge::North => (sy == 0, (sx, 1usize)),
                                        Edge::West => (sx == 1, (0usize, sy)),
                                        Edge::East => (sx == 0, (1usize, sy)),
                                    };
                                    if internal {
                                        neigh[id][e as usize] = NcNeighbor::Conforming {
                                            elem: ch[sib.0 + 2 * sib.1],
                                            edge: opp,
                                        };
                                    } else {
                                        neigh[id][e as usize] = match nb {
                                            None => NcNeighbor::Boundary { tag: boundary_tag(e) },
                                            Some(Cell::Quad(nch)) => {
                                                // child-to-child conforming.
                                                let (nsx, nsy) = match e {
                                                    Edge::South => (sx, 1),
                                                    Edge::North => (sx, 0),
                                                    Edge::West => (1, sy),
                                                    Edge::East => (0, sy),
                                                };
                                                NcNeighbor::Conforming {
                                                    elem: nch[nsx + 2 * nsy],
                                                    edge: opp,
                                                }
                                            }
                                            Some(Cell::Single(n)) => {
                                                // fine → coarse: which half of the coarse edge.
                                                let half = match e {
                                                    Edge::South | Edge::North => sx,
                                                    Edge::East | Edge::West => sy,
                                                };
                                                NcNeighbor::FineToCoarse { coarse: n, edge: opp, half }
                                            }
                                        };
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }

        Self { refq, geom, faces, neigh }
    }

    pub fn n_elements(&self) -> usize {
        self.geom.len()
    }

    /// Face-local indices ordered by the edge's tangential coordinate, increasing.
    fn sorted_edge(&self, e: usize, edge: Edge) -> Vec<usize> {
        let f = &self.faces[e][edge as usize];
        let g = &self.geom[e];
        let vertical = matches!(edge, Edge::East | Edge::West);
        let mut idx: Vec<usize> = (0..f.nodes.len()).collect();
        idx.sort_by(|&a, &b| {
            let ca = if vertical { g.y[f.nodes[a]] } else { g.x[f.nodes[a]] };
            let cb = if vertical { g.y[f.nodes[b]] } else { g.x[f.nodes[b]] };
            ca.partial_cmp(&cb).unwrap()
        });
        idx
    }

    /// Linear-advection `∂ₜu` over the whole (possibly refined) mesh, mortar-coupled
    /// at every 2:1 interface. `bc(x,y)` is the exterior state on domain boundaries.
    pub fn advection_rhs(
        &self,
        u: &[Vec<f64>],
        ax: f64,
        ay: f64,
        mortar: &RefineQuad,
        bc: impl Fn(f64, f64) -> f64,
    ) -> Vec<Vec<f64>> {
        let nn = self.refq.n_nodes();
        let n1 = self.refq.line.nodes.len();
        let flux_n = |ul: f64, ur: f64, nx: f64, ny: f64| {
            let an = ax * nx + ay * ny;
            0.5 * an * (ul + ur) - 0.5 * an.abs() * (ur - ul)
        };
        // Volume.
        let mut res: Vec<Vec<f64>> = (0..self.n_elements())
            .map(|e| {
                let g = &self.geom[e];
                let wfx: Vec<f64> = (0..nn).map(|k| g.jw[k] * ax * u[e][k]).collect();
                let wfy: Vec<f64> = (0..nn).map(|k| g.jw[k] * ay * u[e][k]).collect();
                let a = g.gradx_t(&self.refq, &wfx);
                let b = g.grady_t(&self.refq, &wfy);
                (0..nn).map(|k| a[k] + b[k]).collect()
            })
            .collect();

        for e in 0..self.n_elements() {
            for &edge in &EDGES {
                match self.neigh[e][edge as usize].clone() {
                    NcNeighbor::Boundary { .. } => {
                        let f = &self.faces[e][edge as usize];
                        let g = &self.geom[e];
                        for idx in 0..f.nodes.len() {
                            let k = f.nodes[idx];
                            let ur = bc(g.x[k], g.y[k]);
                            res[e][k] -= f.sw[idx] * flux_n(u[e][k], ur, f.nx[idx], f.ny[idx]);
                        }
                    }
                    NcNeighbor::Conforming { elem: nb, edge: nedge } => {
                        let a_idx = self.sorted_edge(e, edge);
                        let b_idx = self.sorted_edge(nb, nedge);
                        let fa = &self.faces[e][edge as usize];
                        let fb = &self.faces[nb][nedge as usize];
                        for m in 0..a_idx.len() {
                            let (ia, ib) = (a_idx[m], b_idx[m]);
                            let ka = fa.nodes[ia];
                            let kb = fb.nodes[ib];
                            res[e][ka] -= fa.sw[ia] * flux_n(u[e][ka], u[nb][kb], fa.nx[ia], fa.ny[ia]);
                        }
                    }
                    NcNeighbor::FineToCoarse { .. } => { /* handled from the coarse side */ }
                    NcNeighbor::CoarseToFine { fine } => {
                        let ce = self.sorted_edge(e, edge);
                        let fce = &self.faces[e][edge as usize];
                        let (ncx, ncy) = (fce.nx[ce[0]], fce.ny[ce[0]]); // axis-aligned ⇒ constant
                        let uc: Vec<f64> = ce.iter().map(|&i| u[e][fce.nodes[i]]).collect();
                        let mut fstar_halves: [Vec<f64>; 2] = [vec![0.0; n1], vec![0.0; n1]];
                        for h in 0..2 {
                            let (re, redge) = fine[h];
                            let rw = self.sorted_edge(re, redge);
                            let frw = &self.faces[re][redge as usize];
                            let uc_h = mortar.mortar_to_fine(&uc, h);
                            let uf: Vec<f64> = rw.iter().map(|&i| u[re][frw.nodes[i]]).collect();
                            let mut fstar = vec![0.0; rw.len()];
                            for m in 0..rw.len() {
                                fstar[m] = flux_n(uc_h[m], uf[m], ncx, ncy);
                                let i = rw[m];
                                let k = frw.nodes[i];
                                res[re][k] -= frw.sw[i] * flux_n(uf[m], uc_h[m], frw.nx[i], frw.ny[i]);
                            }
                            fstar_halves[h] = fstar;
                        }
                        let fstar_c = mortar.mortar_to_coarse(&fstar_halves);
                        for m in 0..ce.len() {
                            let i = ce[m];
                            res[e][fce.nodes[i]] -= fce.sw[i] * fstar_c[m];
                        }
                    }
                }
            }
        }

        (0..self.n_elements())
            .map(|e| {
                let g = &self.geom[e];
                (0..nn).map(|k| res[e][k] / g.jw[k]).collect()
            })
            .collect()
    }
}

/// Linear-advection operator on the 3-element 2:1 mesh (elements `[C, R0, R1]`).
pub struct NcAdvection {
    pub refq: Reference2dQuad,
    pub geom: Vec<QuadGeometry>,
    pub faces: Vec<[FaceData; 4]>,
    pub mortar: RefineQuad,
    pub ax: f64,
    pub ay: f64,
}

impl NcAdvection {
    pub fn new(order: usize, ax: f64, ay: f64) -> Self {
        let refq = Reference2dQuad::new(order);
        let corners = [
            [[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]], // C
            [[1.0, 0.0], [2.0, 0.0], [2.0, 0.5], [1.0, 0.5]], // R0 (lower)
            [[1.0, 0.5], [2.0, 0.5], [2.0, 1.0], [1.0, 1.0]], // R1 (upper)
        ];
        let geom: Vec<_> = corners.iter().map(|c| QuadGeometry::from_corners(&refq, *c)).collect();
        let faces: Vec<_> = geom.iter().map(|g| super::face::quad_faces(&refq, g)).collect();
        Self { refq, geom, faces, mortar: RefineQuad::new(order), ax, ay }
    }

    fn nn(&self) -> usize {
        self.refq.n_nodes()
    }

    /// Face-local indices `idx` ordered by the edge's tangential coordinate
    /// (y for East/West, x for North/South), increasing.
    fn sorted_edge(&self, e: usize, edge: Edge) -> Vec<usize> {
        let f = &self.faces[e][edge as usize];
        let g = &self.geom[e];
        let vertical = matches!(edge, Edge::East | Edge::West);
        let mut idx: Vec<usize> = (0..f.nodes.len()).collect();
        idx.sort_by(|&a, &b| {
            let ca = if vertical { g.y[f.nodes[a]] } else { g.x[f.nodes[a]] };
            let cb = if vertical { g.y[f.nodes[b]] } else { g.x[f.nodes[b]] };
            ca.partial_cmp(&cb).unwrap()
        });
        idx
    }

    /// Upwind normal flux `F*·n` for the element with outward normal `(nx,ny)`.
    fn flux_n(&self, ul: f64, ur: f64, nx: f64, ny: f64) -> f64 {
        let an = self.ax * nx + self.ay * ny;
        0.5 * an * (ul + ur) - 0.5 * an.abs() * (ur - ul)
    }

    /// `∂ₜu` for the 3 elements. `bc(x,y)` gives the exterior state on domain
    /// boundaries.
    pub fn rhs(&self, u: &[Vec<f64>; 3], bc: impl Fn(f64, f64) -> f64) -> [Vec<f64>; 3] {
        let nn = self.nn();
        // Volume: gradxᵀ(W aₓu) + gradyᵀ(W a_yu).
        let mut res: Vec<Vec<f64>> = (0..3)
            .map(|e| {
                let g = &self.geom[e];
                let wfx: Vec<f64> = (0..nn).map(|k| g.jw[k] * self.ax * u[e][k]).collect();
                let wfy: Vec<f64> = (0..nn).map(|k| g.jw[k] * self.ay * u[e][k]).collect();
                let a = g.gradx_t(&self.refq, &wfx);
                let b = g.grady_t(&self.refq, &wfy);
                (0..nn).map(|k| a[k] + b[k]).collect()
            })
            .collect();

        // Boundary faces: (elem, edge) with the exterior state from `bc`.
        let boundary = [
            (0, Edge::South), (0, Edge::North), (0, Edge::West),
            (1, Edge::South), (1, Edge::East),
            (2, Edge::East), (2, Edge::North),
        ];
        for &(e, edge) in &boundary {
            let f = &self.faces[e][edge as usize];
            for idx in 0..f.nodes.len() {
                let k = f.nodes[idx];
                let g = &self.geom[e];
                let ur = bc(g.x[k], g.y[k]);
                res[e][k] -= f.sw[idx] * self.flux_n(u[e][k], ur, f.nx[idx], f.ny[idx]);
            }
        }

        // Conforming interface: R0.North ↔ R1.South (matched by x).
        let r0n = self.sorted_edge(1, Edge::North);
        let r1s = self.sorted_edge(2, Edge::South);
        let (f0, f1) = (&self.faces[1][Edge::North as usize], &self.faces[2][Edge::South as usize]);
        for m in 0..r0n.len() {
            let (i0, i1) = (r0n[m], r1s[m]);
            let (k0, k1) = (f0.nodes[i0], f1.nodes[i1]);
            let (ul, ur) = (u[1][k0], u[2][k1]);
            res[1][k0] -= f0.sw[i0] * self.flux_n(ul, ur, f0.nx[i0], f0.ny[i0]);
            res[2][k1] -= f1.sw[i1] * self.flux_n(ur, ul, f1.nx[i1], f1.ny[i1]);
        }

        // Non-conforming interface: C.East ↔ R0.West (half 0) + R1.West (half 1).
        let n1 = self.refq.line.nodes.len();
        let ce = self.sorted_edge(0, Edge::East);
        let fce = &self.faces[0][Edge::East as usize];
        let uc: Vec<f64> = ce.iter().map(|&i| u[0][fce.nodes[i]]).collect();
        let mut fstar_halves: [Vec<f64>; 2] = [vec![0.0; n1], vec![0.0; n1]];
        for h in 0..2 {
            let re = 1 + h;
            let rw = self.sorted_edge(re, Edge::West);
            let frw = &self.faces[re][Edge::West as usize];
            let uc_h = self.mortar.mortar_to_fine(&uc, h); // coarse projected onto this half
            let uf: Vec<f64> = rw.iter().map(|&i| u[re][frw.nodes[i]]).collect();
            let mut fstar = vec![0.0; rw.len()];
            for m in 0..rw.len() {
                // Flux with the coarse-outward normal (+x); reused (negated) for R.
                fstar[m] = self.flux_n(uc_h[m], uf[m], 1.0, 0.0);
                let i = rw[m];
                let k = frw.nodes[i];
                res[re][k] -= frw.sw[i] * self.flux_n(uf[m], uc_h[m], frw.nx[i], frw.ny[i]);
            }
            fstar_halves[h] = fstar;
        }
        // Coarse edge receives the back-projected mortar flux.
        let fstar_c = self.mortar.mortar_to_coarse(&fstar_halves);
        for m in 0..ce.len() {
            let i = ce[m];
            res[0][fce.nodes[i]] -= fce.sw[i] * fstar_c[m];
        }

        // ∂ₜu = M⁻¹ res.
        std::array::from_fn(|e| {
            let g = &self.geom[e];
            (0..nn).map(|k| res[e][k] / g.jw[k]).collect()
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn free_stream_preserved_across_hanging_node() {
        // A uniform state must give zero residual everywhere, including across the 2:1
        // interface — the decisive consistency test for the mortar coupling + metrics.
        let op = NcAdvection::new(4, 0.8, -0.5);
        let nn = op.nn();
        let u = [vec![1.3; nn], vec![1.3; nn], vec![1.3; nn]];
        let r = op.rhs(&u, |_, _| 1.3);
        let md = r.iter().flat_map(|v| v.iter()).fold(0.0f64, |a, &x| a.max(x.abs()));
        assert!(md < 1e-10, "free-stream not preserved: {md}");
    }

    #[test]
    fn linear_field_advected_exactly_across_hanging_node() {
        // For a globally-linear field u = α x + β y + γ (continuous across the
        // interface), the weak DG reproduces ∂ₜu = −(aₓα + a_yβ) exactly — high-order
        // accuracy is retained through the hanging node.
        let (ax, ay) = (0.8, -0.5);
        let op = NcAdvection::new(4, ax, ay);
        let nn = op.nn();
        let (alpha, beta, gamma) = (0.7, -0.4, 0.2);
        let f = |x: f64, y: f64| alpha * x + beta * y + gamma;
        let u: [Vec<f64>; 3] = std::array::from_fn(|e| {
            let g = &op.geom[e];
            (0..nn).map(|k| f(g.x[k], g.y[k])).collect()
        });
        let r = op.rhs(&u, f);
        let expected = -(ax * alpha + ay * beta);
        let md = r.iter().flat_map(|v| v.iter()).fold(0.0f64, |a, &x| a.max((x - expected).abs()));
        assert!(md < 1e-9, "linear advection not exact across hanging node: {md} (want ∂ₜu={expected})");
    }

    #[test]
    fn general_refined_mesh_free_stream_and_linear_exact() {
        // A 3×3 base mesh with the centre cell refined → that cell's 4 neighbours each
        // see a 2:1 interface, plus 4 child-child conforming faces. Validate the
        // GENERAL solver: free-stream preserved and linear advection exact everywhere,
        // across all the hanging nodes at once.
        let (ax, ay) = (0.8, -0.5);
        let mesh = NcMesh::cartesian_refined(4, 3, 3, [0.0, 3.0], [0.0, 3.0], &[(1, 1)]);
        let mortar = RefineQuad::new(4);
        let nn = mesh.refq.n_nodes();
        // 8 unrefined cells + 4 children = 12 elements.
        assert_eq!(mesh.n_elements(), 12, "element count");

        // (a) free-stream.
        let uc = vec![vec![2.4; nn]; mesh.n_elements()];
        let r0 = mesh.advection_rhs(&uc, ax, ay, &mortar, |_, _| 2.4);
        let md0 = r0.iter().flat_map(|v| v.iter()).fold(0.0f64, |a, &x| a.max(x.abs()));
        assert!(md0 < 1e-9, "free-stream not preserved on refined mesh: {md0}");

        // (b) linear field advected exactly.
        let (al, be, ga) = (0.7, -0.4, 0.2);
        let f = |x: f64, y: f64| al * x + be * y + ga;
        let u: Vec<Vec<f64>> = (0..mesh.n_elements())
            .map(|e| {
                let g = &mesh.geom[e];
                (0..nn).map(|k| f(g.x[k], g.y[k])).collect()
            })
            .collect();
        let r = mesh.advection_rhs(&u, ax, ay, &mortar, f);
        let expected = -(ax * al + ay * be);
        let md = r.iter().flat_map(|v| v.iter()).fold(0.0f64, |a, &x| a.max((x - expected).abs()));
        eprintln!("refined-mesh linear advection max err = {md:.3e}");
        assert!(md < 1e-9, "linear advection not exact on refined mesh: {md}");
    }

    #[test]
    fn nonconforming_interface_conserves_to_boundary_flux() {
        // The discrete volume telescopes (Dᵀ sums to 0) and every interior face — the
        // conforming one AND the 2:1 mortar — cancels, so the total rate equals exactly
        // the analytic outer-boundary flux. With a=(1,0) and bc = the exact field,
        // Σ Jw ∂ₜu = −∮ F·n = −∫₀¹[f(2,y)−f(0,y)]dy = −sin(3.4) (the y²/const parts cancel).
        let op = NcAdvection::new(4, 1.0, 0.0);
        let nn = op.nn();
        let f = |x: f64, y: f64| (1.7 * x).sin() + 0.3 * y * y + 1.0;
        let u: [Vec<f64>; 3] = std::array::from_fn(|e| {
            let g = &op.geom[e];
            (0..nn).map(|k| f(g.x[k], g.y[k])).collect()
        });
        let r = op.rhs(&u, f);
        let total: f64 = (0..3)
            .map(|e| {
                let g = &op.geom[e];
                (0..nn).map(|k| g.jw[k] * r[e][k]).sum::<f64>()
            })
            .sum();
        let expected = -(3.4_f64).sin();
        assert!((total - expected).abs() < 1e-9, "interface not conservative: total={total}, want {expected}");
    }
}
