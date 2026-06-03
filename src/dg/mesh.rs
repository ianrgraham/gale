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
