//! Face-based 2D quad mesh — the connectivity backbone.
//!
//! Stored element-centric and **unstructured-capable from the start** (the
//! abstraction committed to in `docs/mesh-and-adaptivity-strategy.md` §7): one
//! shared [`Reference2dQuad`], and per element its [`QuadGeometry`], its four
//! [`FaceData`] traces, and a [`Neighbor`] per local edge. Interior-face trace
//! nodes are matched across the shared face by **physical-coordinate proximity**,
//! so connectivity is orientation-agnostic (it resolves to identity on a Cartesian
//! grid but handles arbitrary element orientation).
//!
//! The [`Mesh2d::rectangular`] builder makes a Cartesian grid; the structures it
//! fills are general.

use super::face::{quad_faces, Edge, FaceData};
use super::geometry::QuadGeometry;
use super::quad::Reference2dQuad;

/// What lies across one local edge of an element.
#[derive(Clone, Debug)]
pub enum Neighbor {
    /// An interior face: the neighbor element, its local edge, and a permutation
    /// `perm` mapping this edge's trace-node position `a` to the matching trace-node
    /// position on the neighbor's edge.
    Interior {
        elem: usize,
        edge: Edge,
        perm: Vec<usize>,
    },
    /// A domain boundary, with a tag (rectangular builder: 0=bottom,1=right,2=top,3=left).
    Boundary { tag: u32 },
    /// Coarse side of a 2:1 non-conforming interface: two finer neighbours, each
    /// `(elem, edge)`, ordered by the edge's tangential coordinate (half 0 then 1).
    CoarseToFine { fine: [(usize, Edge); 2] },
    /// Fine side of a 2:1 interface: covers `half ∈ {0,1}` of a coarser neighbour's
    /// edge. The flux is computed from the coarse side, so solvers skip this entry.
    FineToCoarse { coarse: usize, edge: Edge, half: usize },
}

/// One physical element: its corners, geometry, 4 face traces, and 4 neighbors
/// (indexed by `Edge as usize`).
#[derive(Clone, Debug)]
pub struct Element {
    pub corners: [[f64; 2]; 4],
    pub geom: QuadGeometry,
    pub faces: [FaceData; 4],
    pub neighbors: [Neighbor; 4],
}

/// A face-based quad mesh with one shared reference element.
#[derive(Clone, Debug)]
pub struct Mesh2d {
    pub order: usize,
    pub refq: Reference2dQuad,
    pub elements: Vec<Element>,
}

impl Mesh2d {
    /// Number of elements.
    pub fn n_elements(&self) -> usize {
        self.elements.len()
    }

    /// Total physical area (∑ of element `detJ·w`).
    pub fn area(&self) -> f64 {
        self.elements.iter().flat_map(|e| e.geom.jw.iter()).sum()
    }

    /// The distinct boundary-face tags present in the mesh, sorted. Used to map a
    /// [`BoundaryConditions`](super::bc::BoundaryConditions) registry onto the
    /// elliptic operators (which tags are Neumann vs Dirichlet).
    pub fn boundary_tags(&self) -> Vec<u32> {
        let mut tags: Vec<u32> = self
            .elements
            .iter()
            .flat_map(|el| el.neighbors.iter())
            .filter_map(|n| match n {
                Neighbor::Boundary { tag } => Some(*tag),
                _ => None,
            })
            .collect();
        tags.sort_unstable();
        tags.dedup();
        tags
    }

    /// The axis of the (assumed axis-aligned) outward normal of boundary `tag`'s faces,
    /// read from the first face carrying that tag: `0` for an x-normal, `1` for a
    /// y-normal. Used to route symmetry/slip BCs per velocity component. Returns `None`
    /// if no face carries the tag.
    pub fn boundary_tag_normal_axis(&self, tag: u32) -> Option<usize> {
        for el in &self.elements {
            for (e, nb) in el.neighbors.iter().enumerate() {
                if matches!(nb, Neighbor::Boundary { tag: t } if *t == tag) {
                    let f = &el.faces[e];
                    return Some(if f.nx[0].abs() >= f.ny[0].abs() { 0 } else { 1 });
                }
            }
        }
        None
    }

    /// Build a Cartesian `nx × ny` quad mesh over `[x0,x1] × [y0,y1]` at order `p`.
    pub fn rectangular(order: usize, nx: usize, ny: usize, xr: [f64; 2], yr: [f64; 2]) -> Self {
        assert!(nx >= 1 && ny >= 1 && order >= 1);
        let refq = Reference2dQuad::new(order);
        let (x0, x1, y0, y1) = (xr[0], xr[1], yr[0], yr[1]);
        let dx = (x1 - x0) / nx as f64;
        let dy = (y1 - y0) / ny as f64;
        let vert = |cx: usize, cy: usize| [x0 + cx as f64 * dx, y0 + cy as f64 * dy];
        let eidx = |cx: usize, cy: usize| cx + cy * nx;

        // Pass 1: geometry + faces for every element.
        let mut corners_all = Vec::with_capacity(nx * ny);
        let mut geoms = Vec::with_capacity(nx * ny);
        let mut faces_all = Vec::with_capacity(nx * ny);
        for cy in 0..ny {
            for cx in 0..nx {
                let corners = [
                    vert(cx, cy),         // (-1,-1) bottom-left
                    vert(cx + 1, cy),     // ( 1,-1) bottom-right
                    vert(cx + 1, cy + 1), // ( 1, 1) top-right
                    vert(cx, cy + 1),     // (-1, 1) top-left
                ];
                let g = QuadGeometry::from_corners(&refq, corners);
                let f = quad_faces(&refq, &g);
                corners_all.push(corners);
                geoms.push(g);
                faces_all.push(f);
            }
        }

        // Pass 2: neighbors (uses immutable views of pass-1 data).
        let mut neighbors_all: Vec<[Neighbor; 4]> = Vec::with_capacity(nx * ny);
        for cy in 0..ny {
            for cx in 0..nx {
                let e = eidx(cx, cy);
                let south = if cy > 0 {
                    interior(e, Edge::South, eidx(cx, cy - 1), Edge::North, &faces_all, &geoms)
                } else {
                    Neighbor::Boundary { tag: 0 }
                };
                let east = if cx + 1 < nx {
                    interior(e, Edge::East, eidx(cx + 1, cy), Edge::West, &faces_all, &geoms)
                } else {
                    Neighbor::Boundary { tag: 1 }
                };
                let north = if cy + 1 < ny {
                    interior(e, Edge::North, eidx(cx, cy + 1), Edge::South, &faces_all, &geoms)
                } else {
                    Neighbor::Boundary { tag: 2 }
                };
                let west = if cx > 0 {
                    interior(e, Edge::West, eidx(cx - 1, cy), Edge::East, &faces_all, &geoms)
                } else {
                    Neighbor::Boundary { tag: 3 }
                };
                neighbors_all.push([south, east, north, west]);
            }
        }

        // Pass 3: assemble (moves pass-1 data into elements).
        let elements = corners_all
            .into_iter()
            .zip(geoms)
            .zip(faces_all)
            .zip(neighbors_all)
            .map(|(((corners, geom), faces), neighbors)| Element { corners, geom, faces, neighbors })
            .collect();

        Self { order, refq, elements }
    }

    /// Cartesian `nx × ny` mesh with **doubly-periodic** wrap-around connectivity
    /// (no boundary faces). Trace nodes are matched by the tangential coordinate.
    pub fn rectangular_periodic(order: usize, nx: usize, ny: usize, xr: [f64; 2], yr: [f64; 2]) -> Self {
        assert!(nx >= 1 && ny >= 1 && order >= 1);
        let refq = Reference2dQuad::new(order);
        let (x0, x1, y0, y1) = (xr[0], xr[1], yr[0], yr[1]);
        let dx = (x1 - x0) / nx as f64;
        let dy = (y1 - y0) / ny as f64;
        let vert = |cx: usize, cy: usize| [x0 + cx as f64 * dx, y0 + cy as f64 * dy];
        let eidx = |cx: usize, cy: usize| cx + cy * nx;

        let mut corners_all = Vec::with_capacity(nx * ny);
        let mut geoms = Vec::with_capacity(nx * ny);
        let mut faces_all = Vec::with_capacity(nx * ny);
        for cy in 0..ny {
            for cx in 0..nx {
                let corners = [vert(cx, cy), vert(cx + 1, cy), vert(cx + 1, cy + 1), vert(cx, cy + 1)];
                let g = QuadGeometry::from_corners(&refq, corners);
                let f = quad_faces(&refq, &g);
                corners_all.push(corners);
                geoms.push(g);
                faces_all.push(f);
            }
        }

        // Every edge is interior (wraps around); match nodes by the tangential coord
        // (axis 0 = x for South/North edges, axis 1 = y for East/West edges).
        let mut neighbors_all: Vec<[Neighbor; 4]> = Vec::with_capacity(nx * ny);
        for cy in 0..ny {
            for cx in 0..nx {
                let e = eidx(cx, cy);
                let south = interior_tangential(e, Edge::South, eidx(cx, (cy + ny - 1) % ny), Edge::North, &faces_all, &geoms, 0);
                let east = interior_tangential(e, Edge::East, eidx((cx + 1) % nx, cy), Edge::West, &faces_all, &geoms, 1);
                let north = interior_tangential(e, Edge::North, eidx(cx, (cy + 1) % ny), Edge::South, &faces_all, &geoms, 0);
                let west = interior_tangential(e, Edge::West, eidx((cx + nx - 1) % nx, cy), Edge::East, &faces_all, &geoms, 1);
                neighbors_all.push([south, east, north, west]);
            }
        }

        let elements = corners_all
            .into_iter()
            .zip(geoms)
            .zip(faces_all)
            .zip(neighbors_all)
            .map(|(((corners, geom), faces), neighbors)| Element { corners, geom, faces, neighbors })
            .collect();
        Self { order, refq, elements }
    }

    /// Cartesian `nx × ny` **channel**: periodic in x (East↔West wrap) with solid
    /// walls in y (South = boundary tag 0, North = boundary tag 2). For a
    /// fully-developed flow this removes the inlet/outlet–wall corner stress
    /// singularity that a fully-Dirichlet box introduces.
    pub fn channel_x(order: usize, nx: usize, ny: usize, xr: [f64; 2], yr: [f64; 2]) -> Self {
        assert!(nx >= 1 && ny >= 1 && order >= 1);
        let refq = Reference2dQuad::new(order);
        let (x0, x1, y0, y1) = (xr[0], xr[1], yr[0], yr[1]);
        let dx = (x1 - x0) / nx as f64;
        let dy = (y1 - y0) / ny as f64;
        let vert = |cx: usize, cy: usize| [x0 + cx as f64 * dx, y0 + cy as f64 * dy];
        let eidx = |cx: usize, cy: usize| cx + cy * nx;

        let mut corners_all = Vec::with_capacity(nx * ny);
        let mut geoms = Vec::with_capacity(nx * ny);
        let mut faces_all = Vec::with_capacity(nx * ny);
        for cy in 0..ny {
            for cx in 0..nx {
                let corners = [vert(cx, cy), vert(cx + 1, cy), vert(cx + 1, cy + 1), vert(cx, cy + 1)];
                let g = QuadGeometry::from_corners(&refq, corners);
                let f = quad_faces(&refq, &g);
                corners_all.push(corners);
                geoms.push(g);
                faces_all.push(f);
            }
        }

        // x: periodic wrap (match by tangential y, axis 1). y: interior or wall.
        let mut neighbors_all: Vec<[Neighbor; 4]> = Vec::with_capacity(nx * ny);
        for cy in 0..ny {
            for cx in 0..nx {
                let e = eidx(cx, cy);
                let south = if cy > 0 {
                    interior(e, Edge::South, eidx(cx, cy - 1), Edge::North, &faces_all, &geoms)
                } else {
                    Neighbor::Boundary { tag: 0 }
                };
                let east = interior_tangential(e, Edge::East, eidx((cx + 1) % nx, cy), Edge::West, &faces_all, &geoms, 1);
                let north = if cy + 1 < ny {
                    interior(e, Edge::North, eidx(cx, cy + 1), Edge::South, &faces_all, &geoms)
                } else {
                    Neighbor::Boundary { tag: 2 }
                };
                let west = interior_tangential(e, Edge::West, eidx((cx + nx - 1) % nx, cy), Edge::East, &faces_all, &geoms, 1);
                neighbors_all.push([south, east, north, west]);
            }
        }

        let elements = corners_all
            .into_iter()
            .zip(geoms)
            .zip(faces_all)
            .zip(neighbors_all)
            .map(|(((corners, geom), faces), neighbors)| Element { corners, geom, faces, neighbors })
            .collect();
        Self { order, refq, elements }
    }

    /// Cartesian `nx × ny` mesh with the listed cells single-level `h`-refined into
    /// four children each (2:1-balanced). Unrefined neighbours of a refined cell get a
    /// [`Neighbor::CoarseToFine`]; the children get [`Neighbor::FineToCoarse`]; all
    /// same-level shared faces (incl. child–child) are conforming. Solvers that
    /// understand the non-conforming variants (the `Hyperbolic` weak operator) run
    /// adaptively on the result.
    pub fn cartesian_refined(
        order: usize,
        nx: usize,
        ny: usize,
        xr: [f64; 2],
        yr: [f64; 2],
        refine: &[(usize, usize)],
    ) -> Self {
        use std::collections::{HashMap, HashSet};
        use Edge::{East, North, South, West};
        let refq = Reference2dQuad::new(order);
        let (x0, y0) = (xr[0], yr[0]);
        let dx = (xr[1] - xr[0]) / nx as f64;
        let dy = (yr[1] - yr[0]) / ny as f64;
        let refined: HashSet<(usize, usize)> = refine.iter().copied().collect();
        let mk = |gx0: f64, gy0: f64, w: f64, h: f64| {
            [[gx0, gy0], [gx0 + w, gy0], [gx0 + w, gy0 + h], [gx0, gy0 + h]]
        };

        #[derive(Clone, Copy)]
        enum Cell {
            Single(usize),
            Quad([usize; 4]),
        }
        let mut corners_all: Vec<[[f64; 2]; 4]> = Vec::new();
        let mut geoms: Vec<QuadGeometry> = Vec::new();
        let mut cells: HashMap<(usize, usize), Cell> = HashMap::new();
        for cy in 0..ny {
            for cx in 0..nx {
                let (px, py) = (x0 + cx as f64 * dx, y0 + cy as f64 * dy);
                if refined.contains(&(cx, cy)) {
                    let mut ids = [0usize; 4];
                    for sy in 0..2 {
                        for sx in 0..2 {
                            let c = mk(px + sx as f64 * 0.5 * dx, py + sy as f64 * 0.5 * dy, 0.5 * dx, 0.5 * dy);
                            ids[sx + 2 * sy] = geoms.len();
                            geoms.push(QuadGeometry::from_corners(&refq, c));
                            corners_all.push(c);
                        }
                    }
                    cells.insert((cx, cy), Cell::Quad(ids));
                } else {
                    let c = mk(px, py, dx, dy);
                    cells.insert((cx, cy), Cell::Single(geoms.len()));
                    geoms.push(QuadGeometry::from_corners(&refq, c));
                    corners_all.push(c);
                }
            }
        }
        let faces_all: Vec<[FaceData; 4]> = geoms.iter().map(|g| quad_faces(&refq, g)).collect();

        let off = |e: Edge| match e {
            South => (0i64, -1i64),
            East => (1, 0),
            North => (0, 1),
            West => (-1, 0),
        };
        let opp = |e: Edge| match e {
            South => North,
            East => West,
            North => South,
            West => East,
        };
        let cell_at = |cx: i64, cy: i64| -> Option<Cell> {
            if cx < 0 || cy < 0 || cx as usize >= nx || cy as usize >= ny {
                None
            } else {
                cells.get(&(cx as usize, cy as usize)).copied()
            }
        };
        // Two children of a neighbour on its `side` (the edge facing us), tangential order.
        let side_children = |ch: [usize; 4], side: Edge| -> [(usize, Edge); 2] {
            let c = |sx: usize, sy: usize| ch[sx + 2 * sy];
            match side {
                North => [(c(0, 1), North), (c(1, 1), North)],
                South => [(c(0, 0), South), (c(1, 0), South)],
                West => [(c(0, 0), West), (c(0, 1), West)],
                East => [(c(1, 0), East), (c(1, 1), East)],
            }
        };

        let mut neighbors_all: Vec<[Neighbor; 4]> =
            (0..geoms.len()).map(|_| std::array::from_fn(|_| Neighbor::Boundary { tag: 0 })).collect();
        for cy in 0..ny {
            for cx in 0..nx {
                let cell = cells[&(cx, cy)];
                for e in [South, East, North, West] {
                    let (ox, oy) = off(e);
                    let nb = cell_at(cx as i64 + ox, cy as i64 + oy);
                    let o = opp(e);
                    match cell {
                        Cell::Single(id) => {
                            neighbors_all[id][e as usize] = match nb {
                                None => Neighbor::Boundary { tag: e as u32 },
                                Some(Cell::Single(n)) => interior(id, e, n, o, &faces_all, &geoms),
                                Some(Cell::Quad(ch)) => Neighbor::CoarseToFine { fine: side_children(ch, o) },
                            };
                        }
                        Cell::Quad(ch) => {
                            for sy in 0..2usize {
                                for sx in 0..2usize {
                                    let id = ch[sx + 2 * sy];
                                    let (internal, sib) = match e {
                                        South => (sy == 1, (sx, 0usize)),
                                        North => (sy == 0, (sx, 1usize)),
                                        West => (sx == 1, (0usize, sy)),
                                        East => (sx == 0, (1usize, sy)),
                                    };
                                    neighbors_all[id][e as usize] = if internal {
                                        interior(id, e, ch[sib.0 + 2 * sib.1], o, &faces_all, &geoms)
                                    } else {
                                        match nb {
                                            None => Neighbor::Boundary { tag: e as u32 },
                                            Some(Cell::Quad(nch)) => {
                                                let (nsx, nsy) = match e {
                                                    South => (sx, 1),
                                                    North => (sx, 0),
                                                    West => (1, sy),
                                                    East => (0, sy),
                                                };
                                                interior(id, e, nch[nsx + 2 * nsy], o, &faces_all, &geoms)
                                            }
                                            Some(Cell::Single(n)) => {
                                                let half = match e {
                                                    South | North => sx,
                                                    East | West => sy,
                                                };
                                                Neighbor::FineToCoarse { coarse: n, edge: o, half }
                                            }
                                        }
                                    };
                                }
                            }
                        }
                    }
                }
            }
        }

        let elements = corners_all
            .into_iter()
            .zip(geoms)
            .zip(faces_all)
            .zip(neighbors_all)
            .map(|(((corners, geom), faces), neighbors)| Element { corners, geom, faces, neighbors })
            .collect();
        Self { order, refq, elements }
    }
}

/// Interior neighbor matched by a single tangential coordinate `axis` (for periodic
/// faces, where the normal-direction coordinate differs by the domain period).
fn interior_tangential(
    le: usize,
    ledge: Edge,
    re: usize,
    redge: Edge,
    faces: &[[FaceData; 4]],
    geoms: &[QuadGeometry],
    axis: usize,
) -> Neighbor {
    let lf = &faces[le][ledge as usize];
    let rf = &faces[re][redge as usize];
    let (lg, rg) = (&geoms[le], &geoms[re]);
    let coord = |g: &QuadGeometry, v: usize| if axis == 0 { g.x[v] } else { g.y[v] };
    let n = lf.nodes.len();
    let mut perm = vec![0usize; n];
    for a in 0..n {
        let lc = coord(lg, lf.nodes[a]);
        let mut best = 0;
        let mut best_d = f64::INFINITY;
        for b in 0..n {
            let d = (coord(rg, rf.nodes[b]) - lc).abs();
            if d < best_d {
                best_d = d;
                best = b;
            }
        }
        perm[a] = best;
    }
    Neighbor::Interior { elem: re, edge: redge, perm }
}

/// Build the interior-neighbor record for element `le`'s edge `ledge`, matching its
/// trace nodes to element `re`'s edge `redge` by physical-coordinate proximity.
fn interior(
    le: usize,
    ledge: Edge,
    re: usize,
    redge: Edge,
    faces: &[[FaceData; 4]],
    geoms: &[QuadGeometry],
) -> Neighbor {
    let lf = &faces[le][ledge as usize];
    let rf = &faces[re][redge as usize];
    let (lg, rg) = (&geoms[le], &geoms[re]);
    let n = lf.nodes.len();
    let mut perm = vec![0usize; n];
    for a in 0..n {
        let lv = lf.nodes[a];
        let (px, py) = (lg.x[lv], lg.y[lv]);
        let mut best = 0;
        let mut best_d = f64::INFINITY;
        for b in 0..n {
            let rv = rf.nodes[b];
            let d = (rg.x[rv] - px).powi(2) + (rg.y[rv] - py).powi(2);
            if d < best_d {
                best_d = d;
                best = b;
            }
        }
        perm[a] = best;
    }
    Neighbor::Interior { elem: re, edge: redge, perm }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn counts_and_area() {
        let m = Mesh2d::rectangular(4, 3, 2, [0.0, 3.0], [0.0, 2.0]);
        assert_eq!(m.n_elements(), 6);
        assert!((m.area() - 6.0).abs() < 1e-10, "area={}", m.area());
    }

    #[test]
    fn boundary_and_interior_edge_counts() {
        let (nx, ny) = (4, 3);
        let m = Mesh2d::rectangular(3, nx, ny, [0.0, 1.0], [0.0, 1.0]);
        let mut n_bnd = 0;
        let mut n_int = 0;
        for e in &m.elements {
            for nb in &e.neighbors {
                match nb {
                    Neighbor::Boundary { .. } => n_bnd += 1,
                    Neighbor::Interior { .. } => n_int += 1,
                    _ => {}
                }
            }
        }
        // Outer boundary edges, and each interior face counted from both sides.
        assert_eq!(n_bnd, 2 * (nx + ny));
        assert_eq!(n_int, 4 * nx * ny - 2 * (nx + ny));
        assert_eq!(n_int % 2, 0);
    }

    #[test]
    fn channel_x_periodic_in_x_walls_in_y() {
        // East/West always interior (periodic wrap); only South of the bottom row
        // and North of the top row are walls (tags 0 and 2).
        let (nx, ny) = (3, 2);
        let m = Mesh2d::channel_x(3, nx, ny, [0.0, 1.0], [0.0, 1.0]);
        let mut n_bnd = 0;
        for (e, el) in m.elements.iter().enumerate() {
            let cy = e / nx;
            assert!(matches!(el.neighbors[Edge::East as usize], Neighbor::Interior { .. }), "E not periodic");
            assert!(matches!(el.neighbors[Edge::West as usize], Neighbor::Interior { .. }), "W not periodic");
            match (&el.neighbors[Edge::South as usize], cy == 0) {
                (Neighbor::Boundary { tag }, true) => {
                    assert_eq!(*tag, 0);
                    n_bnd += 1;
                }
                (Neighbor::Interior { .. }, false) => {}
                _ => panic!("South neighbor wrong at e={e}"),
            }
            match (&el.neighbors[Edge::North as usize], cy == ny - 1) {
                (Neighbor::Boundary { tag }, true) => {
                    assert_eq!(*tag, 2);
                    n_bnd += 1;
                }
                (Neighbor::Interior { .. }, false) => {}
                _ => panic!("North neighbor wrong at e={e}"),
            }
        }
        // Only the y-walls are boundaries: nx top + nx bottom.
        assert_eq!(n_bnd, 2 * nx);
    }

    #[test]
    fn interior_faces_reciprocal_conforming_and_opposite_normals() {
        let m = Mesh2d::rectangular(5, 3, 3, [-1.0, 2.0], [0.0, 4.0]);
        for (le, el) in m.elements.iter().enumerate() {
            for ledge in Edge::ALL {
                if let Neighbor::Interior { elem: re, edge: redge, perm } = &el.neighbors[ledge as usize] {
                    let rel = &m.elements[*re];
                    // Reciprocity: the neighbor's matching edge points back to us.
                    let back = &rel.neighbors[*redge as usize];
                    let Neighbor::Interior { elem: be, edge: bedge, perm: bperm } = back else {
                        panic!("neighbor edge not interior");
                    };
                    assert_eq!(*be, le);
                    assert_eq!(*bedge, ledge);

                    let lf = &el.faces[ledge as usize];
                    let rf = &rel.faces[*redge as usize];
                    for a in 0..lf.nodes.len() {
                        let b = perm[a];
                        // Inverse permutation.
                        assert_eq!(bperm[b], a, "perm not invertible");
                        // Conforming: matched trace nodes are the same physical point.
                        let lv = lf.nodes[a];
                        let rv = rf.nodes[b];
                        let dx = el.geom.x[lv] - rel.geom.x[rv];
                        let dy = el.geom.y[lv] - rel.geom.y[rv];
                        assert!(dx.hypot(dy) < 1e-12, "non-conforming match");
                        // Opposite outward normals across the shared face.
                        assert!((lf.nx[a] + rf.nx[b]).abs() < 1e-10);
                        assert!((lf.ny[a] + rf.ny[b]).abs() < 1e-10);
                    }
                }
            }
        }
    }

    #[test]
    fn boundary_face_nodes_lie_on_domain_boundary() {
        let (x0, x1, y0, y1) = (-1.0, 2.0, 0.5, 3.5);
        let m = Mesh2d::rectangular(4, 3, 4, [x0, x1], [y0, y1]);
        for el in &m.elements {
            for edge in Edge::ALL {
                if let Neighbor::Boundary { .. } = el.neighbors[edge as usize] {
                    let f = &el.faces[edge as usize];
                    for &v in &f.nodes {
                        let (x, y) = (el.geom.x[v], el.geom.y[v]);
                        let on = (x - x0).abs() < 1e-12
                            || (x - x1).abs() < 1e-12
                            || (y - y0).abs() < 1e-12
                            || (y - y1).abs() < 1e-12;
                        assert!(on, "boundary node off-boundary at ({x},{y})");
                    }
                }
            }
        }
    }

    #[test]
    fn cartesian_perm_is_identity() {
        // On an axis-aligned grid, matched edges traverse the same physical
        // direction, so the trace-node permutation is the identity.
        let m = Mesh2d::rectangular(4, 2, 2, [0.0, 1.0], [0.0, 1.0]);
        for el in &m.elements {
            for nb in &el.neighbors {
                if let Neighbor::Interior { perm, .. } = nb {
                    for (a, &b) in perm.iter().enumerate() {
                        assert_eq!(a, b, "expected identity perm on Cartesian mesh");
                    }
                }
            }
        }
    }
}
