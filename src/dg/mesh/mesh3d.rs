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
    /// **Coarse** side of a 2:1 octree interface: four finer neighbours, each `(elem, face)`,
    /// in quarter order `q = ha + 2·hb` where `(ha, hb) ∈ {0,1}²` is the quarter's position
    /// along the face's two tangential axes (the non-normal axes, in increasing index order).
    CoarseToFine { fine: [(usize, Face); 4] },
    /// **Fine** side of a 2:1 octree interface: this hex covers quarter `quad` (`= ha + 2·hb`)
    /// of a coarser neighbour's face. The flux is assembled from the coarse side (mirroring the
    /// 2D `Neighbor::FineToCoarse`), so operators skip this entry.
    FineToCoarse { coarse: usize, face: Face, quad: usize },
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

    /// Cartesian box mesh with each base cell in `refine` split once into 8 octree children
    /// (single-level 2:1 non-conforming AMR). The 3D analogue of [`Mesh2d::cartesian_refined`]:
    /// a refined cell's six external faces become `CoarseToFine` (seen from a coarse neighbour)
    /// or the neighbour sees `FineToCoarse`; sibling and fine-fine faces stay `Interior` (matched
    /// by physical-coordinate proximity, like [`build`]). Only single-level refinement, so every
    /// non-conforming interface is exactly 2:1.
    pub fn cartesian_refined(
        order: usize,
        nx: usize,
        ny: usize,
        nz: usize,
        xr: [f64; 2],
        yr: [f64; 2],
        zr: [f64; 2],
        refine: &[(usize, usize, usize)],
    ) -> Self {
        use std::collections::{HashMap, HashSet};
        use Face::{Bottom, East, North, South, Top, West};
        let refh = Reference3dHex::new(order);
        let (hx, hy, hz) =
            ((xr[1] - xr[0]) / nx as f64, (yr[1] - yr[0]) / ny as f64, (zr[1] - zr[0]) / nz as f64);
        let refined: HashSet<(usize, usize, usize)> = refine.iter().copied().collect();
        // Corners of an axis-aligned box [x0,x0+wx]×[y0,y0+wy]×[z0,z0+wz] in the canonical order
        // corner c = (c&1 ? +x : ., c&2 ? +y : ., c&4 ? +z : .).
        let box_corners = |x0: f64, y0: f64, z0: f64, wx: f64, wy: f64, wz: f64| -> [[f64; 3]; 8] {
            let mut c = [[0.0; 3]; 8];
            for (i, slot) in c.iter_mut().enumerate() {
                *slot = [
                    if i & 1 == 0 { x0 } else { x0 + wx },
                    if i & 2 == 0 { y0 } else { y0 + wy },
                    if i & 4 == 0 { z0 } else { z0 + wz },
                ];
            }
            c
        };

        #[derive(Clone, Copy)]
        enum Cell {
            Single(usize),
            Oct([usize; 8]), // child (sx,sy,sz) at index sx + 2sy + 4sz
        }
        let mut corners_all: Vec<[[f64; 3]; 8]> = Vec::new();
        let mut geoms: Vec<HexGeometry> = Vec::new();
        let mut cells: HashMap<(usize, usize, usize), Cell> = HashMap::new();
        for cz in 0..nz {
            for cy in 0..ny {
                for cx in 0..nx {
                    let (px, py, pz) = (xr[0] + cx as f64 * hx, yr[0] + cy as f64 * hy, zr[0] + cz as f64 * hz);
                    if refined.contains(&(cx, cy, cz)) {
                        let mut ids = [0usize; 8];
                        for sz in 0..2 {
                            for sy in 0..2 {
                                for sx in 0..2 {
                                    let c = box_corners(
                                        px + sx as f64 * 0.5 * hx,
                                        py + sy as f64 * 0.5 * hy,
                                        pz + sz as f64 * 0.5 * hz,
                                        0.5 * hx,
                                        0.5 * hy,
                                        0.5 * hz,
                                    );
                                    ids[sx + 2 * sy + 4 * sz] = geoms.len();
                                    geoms.push(HexGeometry::from_corners(&refh, c));
                                    corners_all.push(c);
                                }
                            }
                        }
                        cells.insert((cx, cy, cz), Cell::Oct(ids));
                    } else {
                        let c = box_corners(px, py, pz, hx, hy, hz);
                        cells.insert((cx, cy, cz), Cell::Single(geoms.len()));
                        geoms.push(HexGeometry::from_corners(&refh, c));
                        corners_all.push(c);
                    }
                }
            }
        }
        let faces_all: Vec<[HexFaceData; 6]> = geoms.iter().map(|g| hex_faces(&refh, g)).collect();

        // Interior-face perm by tangential-coordinate proximity (orientation-agnostic, as in `build`).
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
        let interior = |e: usize, lf: usize, ne: usize, nf: Face| -> Neighbor3 {
            let mine = tang(&geoms[e], &faces_all[e][lf]);
            let theirs = tang(&geoms[ne], &faces_all[ne][nf as usize]);
            let perm = mine
                .iter()
                .map(|m| {
                    let (mut best, mut bd) = (0usize, f64::INFINITY);
                    for (b, t) in theirs.iter().enumerate() {
                        let d = (m[0] - t[0]).powi(2) + (m[1] - t[1]).powi(2);
                        if d < bd {
                            bd = d;
                            best = b;
                        }
                    }
                    best
                })
                .collect();
            Neighbor3::Interior { elem: ne, face: nf, perm }
        };

        // (local face, Δcx, Δcy, Δcz, neighbour's matching face)
        let across = [
            (Bottom, 0i64, 0, -1, Top),
            (Top, 0, 0, 1, Bottom),
            (South, 0, -1, 0, North),
            (North, 0, 1, 0, South),
            (West, -1, 0, 0, East),
            (East, 1, 0, 0, West),
        ];
        let cell_at = |cx: i64, cy: i64, cz: i64| -> Option<Cell> {
            if cx < 0 || cy < 0 || cz < 0 || cx as usize >= nx || cy as usize >= ny || cz as usize >= nz {
                None
            } else {
                cells.get(&(cx as usize, cy as usize, cz as usize)).copied()
            }
        };
        // The two tangential axes (non-normal), increasing index order.
        let tang_axes = |n: usize| -> (usize, usize) {
            match n {
                0 => (1, 2),
                1 => (0, 2),
                _ => (0, 1),
            }
        };
        let child_id = |oct: &[usize; 8], c: [usize; 3]| oct[c[0] + 2 * c[1] + 4 * c[2]];
        // The 4 children of `oct` on its face `f` (the side facing the coarse neighbour), in
        // quarter order q = ha + 2·hb. `f` is the neighbour-child's own local face.
        let face_children = |oct: &[usize; 8], f: Face| -> [(usize, Face); 4] {
            let n = f.normal_axis();
            let side = if matches!(f, Top | North | East) { 1 } else { 0 };
            let (a, b) = tang_axes(n);
            let mut out = [(0usize, f); 4];
            for hb in 0..2 {
                for ha in 0..2 {
                    let mut c = [0usize; 3];
                    c[n] = side;
                    c[a] = ha;
                    c[b] = hb;
                    out[ha + 2 * hb] = (child_id(oct, c), f);
                }
            }
            out
        };

        let mut neighbors_all: Vec<[Neighbor3; 6]> =
            (0..geoms.len()).map(|_| std::array::from_fn(|_| Neighbor3::Boundary { tag: 0 })).collect();
        for cz in 0..nz {
            for cy in 0..ny {
                for cx in 0..nx {
                    let cell = cells[&(cx, cy, cz)];
                    for (lf, (face, dcx, dcy, dcz, opp)) in across.into_iter().enumerate() {
                        let nb = cell_at(cx as i64 + dcx, cy as i64 + dcy, cz as i64 + dcz);
                        let n = face.normal_axis();
                        let side = if matches!(face, Top | North | East) { 1 } else { 0 };
                        let (a, b) = tang_axes(n);
                        match cell {
                            Cell::Single(id) => {
                                neighbors_all[id][lf] = match nb {
                                    None => Neighbor3::Boundary { tag: lf as u32 },
                                    Some(Cell::Single(ne)) => interior(id, lf, ne, opp),
                                    Some(Cell::Oct(nch)) => {
                                        Neighbor3::CoarseToFine { fine: face_children(&nch, opp) }
                                    }
                                };
                            }
                            Cell::Oct(ch) => {
                                for sz in 0..2 {
                                    for sy in 0..2 {
                                        for sx in 0..2 {
                                            let c = [sx, sy, sz];
                                            let id = child_id(&ch, c);
                                            if c[n] != side {
                                                // internal: sibling across this face within the oct
                                                let mut sib = c;
                                                sib[n] = side;
                                                neighbors_all[id][lf] = interior(id, lf, child_id(&ch, sib), opp);
                                            } else {
                                                neighbors_all[id][lf] = match nb {
                                                    None => Neighbor3::Boundary { tag: lf as u32 },
                                                    Some(Cell::Oct(nch)) => {
                                                        // matching neighbour child: same (a,b), opp side along n
                                                        let mut nc = [0usize; 3];
                                                        nc[n] = 1 - side;
                                                        nc[a] = c[a];
                                                        nc[b] = c[b];
                                                        interior(id, lf, child_id(&nch, nc), opp)
                                                    }
                                                    Some(Cell::Single(ne)) => Neighbor3::FineToCoarse {
                                                        coarse: ne,
                                                        face: opp,
                                                        quad: c[a] + 2 * c[b],
                                                    },
                                                };
                                            }
                                        }
                                    }
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
            .map(|(((corners, geom), faces), neighbors)| HexElement { corners, geom, faces, neighbors })
            .collect();
        Self { order, refh, elements }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn octree_nonconforming_connectivity() {
        // Refine the corner cell of a 2×2×2 box: 7 unrefined + 8 children = 15 elements.
        let m = Mesh3d::cartesian_refined(2, 2, 2, 2, [0.0, 1.0], [0.0, 1.0], [0.0, 1.0], &[(0, 0, 0)]);
        assert_eq!(m.n_elements(), 7 + 8);
        let (mut ctf, mut ftc) = (0, 0);
        for (e, el) in m.elements.iter().enumerate() {
            for (lf, nb) in el.neighbors.iter().enumerate() {
                match nb {
                    Neighbor3::CoarseToFine { fine } => {
                        ctf += 1;
                        let cf = &el.faces[lf];
                        let n = cf.face.normal_axis();
                        let coord = |g: &HexGeometry, k: usize| [g.x[k], g.y[k], g.z[k]][n];
                        let cn = coord(&el.geom, cf.nodes[0]);
                        let mut seen = [false; 4];
                        for &(fe, ff) in fine {
                            // reciprocity: each fine child points back via FineToCoarse to this coarse face
                            match &m.elements[fe].neighbors[ff as usize] {
                                Neighbor3::FineToCoarse { coarse, face, quad } => {
                                    assert_eq!(*coarse, e);
                                    assert_eq!(*face as usize, lf);
                                    seen[*quad] = true;
                                }
                                other => panic!("fine child not FineToCoarse: {other:?}"),
                            }
                            // geometric: every fine-face node lies on the coarse face plane
                            let (ffd, fg) = (&m.elements[fe].faces[ff as usize], &m.elements[fe].geom);
                            for &k in &ffd.nodes {
                                assert!((coord(fg, k) - cn).abs() < 1e-12, "fine face off the coarse plane");
                            }
                        }
                        assert_eq!(seen, [true; 4], "the 4 quarters must be distinct");
                    }
                    Neighbor3::FineToCoarse { .. } => ftc += 1,
                    _ => {}
                }
            }
        }
        // The corner refined cell has 3 interior faces (→ 3 coarse neighbours see CoarseToFine, and
        // its 4 children on each interface see FineToCoarse ⇒ 12) and 3 domain-boundary faces.
        assert_eq!(ctf, 3, "CoarseToFine faces");
        assert_eq!(ftc, 12, "FineToCoarse faces");
    }

    #[test]
    fn fully_refined_is_conforming() {
        // Refining the only cell of a 1×1×1 box gives a uniform 2×2×2 of children: all interior
        // faces conforming, all external faces domain boundary — no non-conforming interfaces.
        let m = Mesh3d::cartesian_refined(2, 1, 1, 1, [0.0, 1.0], [0.0, 1.0], [0.0, 1.0], &[(0, 0, 0)]);
        assert_eq!(m.n_elements(), 8);
        for el in &m.elements {
            for nb in &el.neighbors {
                assert!(
                    matches!(nb, Neighbor3::Interior { .. } | Neighbor3::Boundary { .. }),
                    "fully-refined mesh should be conforming"
                );
            }
        }
    }

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
