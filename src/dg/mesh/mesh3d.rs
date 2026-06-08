//! Face-based 3D hex mesh — the 3D analogue of [`Mesh2d`](super::mesh::Mesh2d).
//!
//! Element-centric and unstructured-capable: one shared [`Reference3dHex`], and per
//! element its [`HexGeometry`], its six [`HexFaceData`] traces, and a [`Neighbor3`]
//! per local face. Interior-face trace nodes are matched across the shared face by
//! **physical-coordinate proximity** on the two in-face tangential axes — the same
//! orientation-agnostic strategy used for 2D edges, which here resolves the eight
//! possible 3D face orientations without any explicit enumeration
//! (`docs/3d-strategy.md` §6).
//!
//! [`Mesh3d::rectangular`] / [`Mesh3d::rectangular_periodic`] build Cartesian box
//! meshes; the structures they fill are general. (Non-conforming/octree AMR is a
//! later step; `Neighbor3` carries only `Interior`/`Boundary` for now.)

use super::face3d::{hex_faces, Face, HexFaceData};
use super::geometry3d::HexGeometry;
use super::hex::Reference3dHex;

/// What lies across one local face of a hex.
#[derive(Clone, Debug)]
pub enum Neighbor3 {
    /// An interior face: neighbour element, its local face, and a permutation `perm`
    /// mapping this face's trace-node position `a` to the matching node on the
    /// neighbour's face.
    Interior { elem: usize, face: Face, perm: Vec<usize> },
    /// A domain boundary, tagged by the local face index (`Face as usize`).
    Boundary { tag: u32 },
}

/// One physical hex element.
#[derive(Clone, Debug)]
pub struct HexElement {
    pub corners: [[f64; 3]; 8],
    pub geom: HexGeometry,
    pub faces: [HexFaceData; 6],
    pub neighbors: [Neighbor3; 6],
}

/// A face-based hex mesh with one shared reference element.
#[derive(Clone, Debug)]
pub struct Mesh3d {
    pub order: usize,
    pub refh: Reference3dHex,
    pub elements: Vec<HexElement>,
}

impl Mesh3d {
    /// Number of elements.
    pub fn n_elements(&self) -> usize {
        self.elements.len()
    }

    /// The distinct boundary-face tags present in the mesh, sorted.
    pub fn boundary_tags(&self) -> Vec<u32> {
        let mut tags: Vec<u32> = self
            .elements
            .iter()
            .flat_map(|el| el.neighbors.iter())
            .filter_map(|n| match n {
                Neighbor3::Boundary { tag } => Some(*tag),
                _ => None,
            })
            .collect();
        tags.sort_unstable();
        tags.dedup();
        tags
    }

    /// The axis of the (assumed axis-aligned) outward normal of boundary `tag`'s faces,
    /// read from the first face carrying that tag: `0` for x-normal, `1` for y-normal,
    /// `2` for z-normal. Used to route symmetry/slip BCs per velocity component.
    pub fn boundary_tag_normal_axis(&self, tag: u32) -> Option<usize> {
        for el in &self.elements {
            for (f, nb) in el.neighbors.iter().enumerate() {
                if matches!(nb, Neighbor3::Boundary { tag: t } if *t == tag) {
                    let fc = &el.faces[f];
                    let (ax, ay, az) = (fc.nx[0].abs(), fc.ny[0].abs(), fc.nz[0].abs());
                    return Some(if ax >= ay && ax >= az {
                        0
                    } else if ay >= az {
                        1
                    } else {
                        2
                    });
                }
            }
        }
        None
    }

    /// Total physical volume (∑ of element `detJ·w`).
    pub fn volume(&self) -> f64 {
        self.elements.iter().flat_map(|e| e.geom.jw.iter()).sum()
    }

    /// Build a Cartesian `nx × ny × nz` hex mesh over `[x0,x1]×[y0,y1]×[z0,z1]`,
    /// with `periodic` selecting wrap-around vs domain-boundary faces.
    fn build(
        order: usize,
        nx: usize,
        ny: usize,
        nz: usize,
        xr: [f64; 2],
        yr: [f64; 2],
        zr: [f64; 2],
        periodic: bool,
    ) -> Self {
        assert!(nx >= 1 && ny >= 1 && nz >= 1 && order >= 1);
        let refh = Reference3dHex::new(order);
        let (hx, hy, hz) =
            ((xr[1] - xr[0]) / nx as f64, (yr[1] - yr[0]) / ny as f64, (zr[1] - zr[0]) / nz as f64);
        let cell = |cx: usize, cy: usize, cz: usize| -> usize { cx + cy * nx + cz * nx * ny };

        // Geometry + faces for every cell.
        let mut geoms = Vec::with_capacity(nx * ny * nz);
        let mut faces_all = Vec::with_capacity(nx * ny * nz);
        let mut corners_all = Vec::with_capacity(nx * ny * nz);
        for cz in 0..nz {
            for cy in 0..ny {
                for cx in 0..nx {
                    let (x0, x1) = (xr[0] + cx as f64 * hx, xr[0] + (cx + 1) as f64 * hx);
                    let (y0, y1) = (yr[0] + cy as f64 * hy, yr[0] + (cy + 1) as f64 * hy);
                    let (z0, z1) = (zr[0] + cz as f64 * hz, zr[0] + (cz + 1) as f64 * hz);
                    let mut corners = [[0.0; 3]; 8];
                    for (c, slot) in corners.iter_mut().enumerate() {
                        *slot = [
                            if c & 1 == 0 { x0 } else { x1 },
                            if c & 2 == 0 { y0 } else { y1 },
                            if c & 4 == 0 { z0 } else { z1 },
                        ];
                    }
                    let g = HexGeometry::from_corners(&refh, corners);
                    let f = hex_faces(&refh, &g);
                    geoms.push(g);
                    faces_all.push(f);
                    corners_all.push(corners);
                }
            }
        }

        // Tangential coordinates of a face's nodes (the two non-normal axes).
        let tang = |g: &HexGeometry, fd: &HexFaceData| -> Vec<[f64; 2]> {
            let ax = fd.face.normal_axis();
            fd.nodes
                .iter()
                .map(|&k| {
                    let xyz = [g.x[k], g.y[k], g.z[k]];
                    match ax {
                        0 => [xyz[1], xyz[2]],
                        1 => [xyz[0], xyz[2]],
                        _ => [xyz[0], xyz[1]],
                    }
                })
                .collect()
        };
        // perm[a] = neighbour node index whose tangential coords match my node a.
        let match_perm = |mine: &[[f64; 2]], theirs: &[[f64; 2]]| -> Vec<usize> {
            mine.iter()
                .map(|m| {
                    let mut best = 0usize;
                    let mut bd = f64::INFINITY;
                    for (b, t) in theirs.iter().enumerate() {
                        let d = (m[0] - t[0]).powi(2) + (m[1] - t[1]).powi(2);
                        if d < bd {
                            bd = d;
                            best = b;
                        }
                    }
                    best
                })
                .collect()
        };

        // The neighbour cell + its matching face across each local face.
        // (local face, Δcx, Δcy, Δcz, neighbour's matching face)
        let across = [
            (Face::Bottom, 0i64, 0, -1, Face::Top),
            (Face::Top, 0, 0, 1, Face::Bottom),
            (Face::South, 0, -1, 0, Face::North),
            (Face::North, 0, 1, 0, Face::South),
            (Face::West, -1, 0, 0, Face::East),
            (Face::East, 1, 0, 0, Face::West),
        ];

        let mut elements = Vec::with_capacity(nx * ny * nz);
        for cz in 0..nz {
            for cy in 0..ny {
                for cx in 0..nx {
                    let e = cell(cx, cy, cz);
                    let neighbors: [Neighbor3; 6] = std::array::from_fn(|lf| {
                        let (face, dcx, dcy, dcz, nbr_face) = across[lf];
                        debug_assert_eq!(face as usize, lf);
                        // Neighbour cell coords (with wrap for periodic).
                        let resolve = |c: usize, d: i64, n: usize| -> Option<usize> {
                            let v = c as i64 + d;
                            if v >= 0 && (v as usize) < n {
                                Some(v as usize)
                            } else if periodic {
                                Some(((v + n as i64) % n as i64) as usize)
                            } else {
                                None
                            }
                        };
                        match (resolve(cx, dcx, nx), resolve(cy, dcy, ny), resolve(cz, dcz, nz)) {
                            (Some(ncx), Some(ncy), Some(ncz)) => {
                                let ne = cell(ncx, ncy, ncz);
                                let mine = tang(&geoms[e], &faces_all[e][lf]);
                                let theirs = tang(&geoms[ne], &faces_all[ne][nbr_face as usize]);
                                Neighbor3::Interior {
                                    elem: ne,
                                    face: nbr_face,
                                    perm: match_perm(&mine, &theirs),
                                }
                            }
                            _ => Neighbor3::Boundary { tag: lf as u32 },
                        }
                    });
                    elements.push(HexElement {
                        corners: corners_all[e],
                        geom: geoms[e].clone(),
                        faces: faces_all[e].clone(),
                        neighbors,
                    });
                }
            }
        }

        Self { order, refh, elements }
    }

    /// Cartesian box mesh with domain-boundary faces.
    pub fn rectangular(
        order: usize,
        nx: usize,
        ny: usize,
        nz: usize,
        xr: [f64; 2],
        yr: [f64; 2],
        zr: [f64; 2],
    ) -> Self {
        Self::build(order, nx, ny, nz, xr, yr, zr, false)
    }

    /// Fully periodic Cartesian box mesh (every face interior).
    pub fn rectangular_periodic(
        order: usize,
        nx: usize,
        ny: usize,
        nz: usize,
        xr: [f64; 2],
        yr: [f64; 2],
        zr: [f64; 2],
    ) -> Self {
        Self::build(order, nx, ny, nz, xr, yr, zr, true)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn element_count_and_volume() {
        let m = Mesh3d::rectangular(3, 3, 4, 2, [0.0, 3.0], [0.0, 2.0], [0.0, 1.0]);
        assert_eq!(m.n_elements(), 3 * 4 * 2);
        assert!((m.volume() - 6.0).abs() < 1e-9, "volume={}", m.volume());
    }

    #[test]
    fn interior_neighbor_traces_share_coordinates() {
        // On a non-periodic box, each Interior face's perm must map my face node to a
        // neighbour node at the *same physical point* (the shared face).
        let m = Mesh3d::rectangular(3, 3, 3, 3, [0.0, 1.0], [0.0, 1.0], [0.0, 1.0]);
        for (e, el) in m.elements.iter().enumerate() {
            for lf in 0..6 {
                if let Neighbor3::Interior { elem: ne, face, perm } = &el.neighbors[lf] {
                    let myf = &el.faces[lf];
                    let nf = &m.elements[*ne].faces[*face as usize];
                    let (gm, gn) = (&el.geom, &m.elements[*ne].geom);
                    for a in 0..myf.nodes.len() {
                        let mk = myf.nodes[a];
                        let nk = nf.nodes[perm[a]];
                        let d = (gm.x[mk] - gn.x[nk]).powi(2)
                            + (gm.y[mk] - gn.y[nk]).powi(2)
                            + (gm.z[mk] - gn.z[nk]).powi(2);
                        assert!(d < 1e-18, "e={e} lf={lf} node {a} not coincident: d²={d}");
                    }
                }
            }
        }
    }

    #[test]
    fn boundary_faces_only_on_domain_edges() {
        let (nx, ny, nz) = (3usize, 3, 3);
        let m = Mesh3d::rectangular(2, nx, ny, nz, [0.0, 1.0], [0.0, 1.0], [0.0, 1.0]);
        // Count boundary faces; a box has 2(nx·ny + ny·nz + nx·nz) exterior faces.
        let mut nb = 0;
        for el in &m.elements {
            for nb_ in &el.neighbors {
                if matches!(nb_, Neighbor3::Boundary { .. }) {
                    nb += 1;
                }
            }
        }
        let expect = 2 * (nx * ny + ny * nz + nx * nz);
        assert_eq!(nb, expect, "boundary face count");
    }

    #[test]
    fn periodic_mesh_has_no_boundary_faces() {
        let m = Mesh3d::rectangular_periodic(3, 2, 2, 2, [0.0, 1.0], [0.0, 1.0], [0.0, 1.0]);
        for el in &m.elements {
            for nb in &el.neighbors {
                assert!(matches!(nb, Neighbor3::Interior { .. }), "periodic mesh has a boundary face");
            }
        }
    }

    #[test]
    fn periodic_traces_share_tangential_coordinates() {
        // Across a periodic face the matched nodes agree on the two tangential
        // coordinates (the normal coordinate differs by the domain period).
        let m = Mesh3d::rectangular_periodic(3, 2, 2, 2, [0.0, 1.0], [0.0, 1.0], [0.0, 1.0]);
        for el in &m.elements {
            for lf in 0..6 {
                if let Neighbor3::Interior { elem: ne, face, perm } = &el.neighbors[lf] {
                    let ax = Face::ALL[lf].normal_axis();
                    let myf = &el.faces[lf];
                    let nf = &m.elements[*ne].faces[*face as usize];
                    let (gm, gn) = (&el.geom, &m.elements[*ne].geom);
                    for a in 0..myf.nodes.len() {
                        let mk = myf.nodes[a];
                        let nk = nf.nodes[perm[a]];
                        let mc = [gm.x[mk], gm.y[mk], gm.z[mk]];
                        let nc = [gn.x[nk], gn.y[nk], gn.z[nk]];
                        for axis in 0..3 {
                            if axis != ax {
                                assert!((mc[axis] - nc[axis]).abs() < 1e-12, "tangential mismatch");
                            }
                        }
                    }
                }
            }
        }
    }
}
