//! Face / trace operators for a hex element — the per-element half of 3D DG flux
//! coupling. The 3D analogue of [`face`](super::face): each of the **6 faces** is a
//! quad with `(p+1)²` trace nodes.
//!
//! The scaled outward normal on a reference face is `±J·∇ξ` for the corresponding
//! reference coordinate (`r`, `s`, or `t`), in terms of [`HexGeometry`]'s metrics:
//!
//! | face   | ref coord | scaled outward normal |
//! |--------|-----------|-----------------------|
//! | Bottom | t = −1    | −J·(tx, ty, tz)       |
//! | Top    | t = +1    | +J·(tx, ty, tz)       |
//! | South  | s = −1    | −J·(sx, sy, sz)       |
//! | North  | s = +1    | +J·(sx, sy, sz)       |
//! | West   | r = −1    | −J·(rx, ry, rz)       |
//! | East   | r = +1    | +J·(rx, ry, rz)       |
//!
//! `sJ = |scaled normal|`, `(nx,ny,nz) = scaled/sJ`, surface weight `= w_a·w_b · sJ`
//! (the two in-face 1D quadrature weights times the surface Jacobian).

use super::geometry3d::HexGeometry;
use super::hex::Reference3dHex;

/// Local face identifier of a hex. Discriminants are the local face index used by
/// mesh connectivity.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Face {
    Bottom = 0, // t = -1
    Top = 1,    // t = +1
    South = 2,  // s = -1
    North = 3,  // s = +1
    West = 4,   // r = -1
    East = 5,   // r = +1
}

impl Face {
    /// All six faces in canonical order.
    pub const ALL: [Face; 6] = [Face::Bottom, Face::Top, Face::South, Face::North, Face::West, Face::East];

    /// The axis (0=x, 1=y, 2=z) the face normal points along on an axis-aligned cell
    /// — used to pick the two in-face tangential coordinates for connectivity.
    pub fn normal_axis(self) -> usize {
        match self {
            Face::Bottom | Face::Top => 2,
            Face::South | Face::North => 1,
            Face::West | Face::East => 0,
        }
    }
}

/// Geometric trace data for one face of one hex (length `(p+1)²` per field).
#[derive(Clone, Debug)]
pub struct HexFaceData {
    pub face: Face,
    /// Volume node indices on this face.
    pub nodes: Vec<usize>,
    /// Unit outward normal components at each face node.
    pub nx: Vec<f64>,
    pub ny: Vec<f64>,
    pub nz: Vec<f64>,
    /// Surface Jacobian at each face node.
    pub sj: Vec<f64>,
    /// Surface quadrature weight `w_a·w_b · sJ` at each face node.
    pub sw: Vec<f64>,
}

/// Build the six [`HexFaceData`] of a hex element from its geometry.
pub fn hex_faces(refh: &Reference3dHex, g: &HexGeometry) -> [HexFaceData; 6] {
    let n = refh.n_1d();
    let w = &refh.line.weights;
    let n2 = n * n;

    let sr = |k: usize| [g.jac[k] * g.rx[k], g.jac[k] * g.ry[k], g.jac[k] * g.rz[k]];
    let ss = |k: usize| [g.jac[k] * g.sx[k], g.jac[k] * g.sy[k], g.jac[k] * g.sz[k]];
    let st = |k: usize| [g.jac[k] * g.tx[k], g.jac[k] * g.ty[k], g.jac[k] * g.tz[k]];

    let build = |face: Face, idx: Vec<usize>, wt: Vec<f64>, scaled: &dyn Fn(usize) -> [f64; 3], sign: f64| {
        let m = idx.len();
        let (mut nx, mut ny, mut nz, mut sj, mut sw) =
            (vec![0.0; m], vec![0.0; m], vec![0.0; m], vec![0.0; m], vec![0.0; m]);
        for a in 0..m {
            let v = scaled(idx[a]);
            let s = [sign * v[0], sign * v[1], sign * v[2]];
            let mag = (s[0] * s[0] + s[1] * s[1] + s[2] * s[2]).sqrt();
            nx[a] = s[0] / mag;
            ny[a] = s[1] / mag;
            nz[a] = s[2] / mag;
            sj[a] = mag;
            sw[a] = wt[a] * mag;
        }
        HexFaceData { face, nodes: idx, nx, ny, nz, sj, sw }
    };

    // Per face: the list of volume node indices and the matching in-face weight
    // products. Iteration order is consistent within (nodes, weights); connectivity
    // matches across elements by physical coordinate, so the order need not be
    // canonical across elements.
    let mut bottom_i = Vec::with_capacity(n2);
    let mut bottom_w = Vec::with_capacity(n2);
    let mut top_i = Vec::with_capacity(n2);
    let mut top_w = Vec::with_capacity(n2);
    for j in 0..n {
        for i in 0..n {
            bottom_i.push(i + j * n); // k = 0
            bottom_w.push(w[i] * w[j]);
            top_i.push(i + j * n + (n - 1) * n2); // k = n-1
            top_w.push(w[i] * w[j]);
        }
    }
    let mut south_i = Vec::with_capacity(n2);
    let mut south_w = Vec::with_capacity(n2);
    let mut north_i = Vec::with_capacity(n2);
    let mut north_w = Vec::with_capacity(n2);
    for k in 0..n {
        for i in 0..n {
            south_i.push(i + k * n2); // j = 0
            south_w.push(w[i] * w[k]);
            north_i.push(i + (n - 1) * n + k * n2); // j = n-1
            north_w.push(w[i] * w[k]);
        }
    }
    let mut west_i = Vec::with_capacity(n2);
    let mut west_w = Vec::with_capacity(n2);
    let mut east_i = Vec::with_capacity(n2);
    let mut east_w = Vec::with_capacity(n2);
    for k in 0..n {
        for j in 0..n {
            west_i.push(j * n + k * n2); // i = 0
            west_w.push(w[j] * w[k]);
            east_i.push((n - 1) + j * n + k * n2); // i = n-1
            east_w.push(w[j] * w[k]);
        }
    }

    [
        build(Face::Bottom, bottom_i, bottom_w, &st, -1.0),
        build(Face::Top, top_i, top_w, &st, 1.0),
        build(Face::South, south_i, south_w, &ss, -1.0),
        build(Face::North, north_i, north_w, &ss, 1.0),
        build(Face::West, west_i, west_w, &sr, -1.0),
        build(Face::East, east_i, east_w, &sr, 1.0),
    ]
}

#[cfg(test)]
mod tests {
    use super::*;

    const MAX_P: usize = 4;

    fn cube_corners(xr: [f64; 2], yr: [f64; 2], zr: [f64; 2]) -> [[f64; 3]; 8] {
        let mut c = [[0.0; 3]; 8];
        for (cc, slot) in c.iter_mut().enumerate() {
            *slot = [
                if cc & 1 == 0 { xr[0] } else { xr[1] },
                if cc & 2 == 0 { yr[0] } else { yr[1] },
                if cc & 4 == 0 { zr[0] } else { zr[1] },
            ];
        }
        c
    }

    #[test]
    fn reference_cube_normals() {
        let corners = cube_corners([-1.0, 1.0], [-1.0, 1.0], [-1.0, 1.0]);
        let expect = [
            (Face::Bottom, [0.0, 0.0, -1.0]),
            (Face::Top, [0.0, 0.0, 1.0]),
            (Face::South, [0.0, -1.0, 0.0]),
            (Face::North, [0.0, 1.0, 0.0]),
            (Face::West, [-1.0, 0.0, 0.0]),
            (Face::East, [1.0, 0.0, 0.0]),
        ];
        for p in 1..=MAX_P {
            let refh = Reference3dHex::new(p);
            let g = HexGeometry::from_corners(&refh, corners);
            let faces = hex_faces(&refh, &g);
            for (f, (face, nrm)) in faces.iter().zip(expect) {
                assert_eq!(f.face, face);
                assert_eq!(f.nodes.len(), (p + 1) * (p + 1));
                for a in 0..f.nodes.len() {
                    assert!((f.nx[a] - nrm[0]).abs() < 1e-10);
                    assert!((f.ny[a] - nrm[1]).abs() < 1e-10);
                    assert!((f.nz[a] - nrm[2]).abs() < 1e-10);
                    assert!((f.sj[a] - 1.0).abs() < 1e-10);
                }
            }
        }
    }

    #[test]
    fn box_face_areas() {
        // [0,2]×[0,3]×[0,4]: each face's ∑sw = that face's physical area.
        let corners = cube_corners([0.0, 2.0], [0.0, 3.0], [0.0, 4.0]);
        for p in 1..=MAX_P {
            let refh = Reference3dHex::new(p);
            let g = HexGeometry::from_corners(&refh, corners);
            let faces = hex_faces(&refh, &g);
            let area = |f: &HexFaceData| -> f64 { f.sw.iter().sum() };
            // Bottom/Top: x×y = 6; South/North: x×z = 8; West/East: y×z = 12.
            assert!((area(&faces[Face::Bottom as usize]) - 6.0).abs() < 1e-10);
            assert!((area(&faces[Face::Top as usize]) - 6.0).abs() < 1e-10);
            assert!((area(&faces[Face::South as usize]) - 8.0).abs() < 1e-10);
            assert!((area(&faces[Face::East as usize]) - 12.0).abs() < 1e-10);
        }
    }

    #[test]
    fn divergence_theorem_exact_on_affine_box() {
        // F = (x², y², z²) ⇒ ∇·F = 2x+2y+2z. ∮F·n dS == ∫∇·F dV exactly (affine).
        let corners = cube_corners([0.0, 2.0], [1.0, 4.0], [-1.0, 2.0]);
        for p in 2..=MAX_P {
            let refh = Reference3dHex::new(p);
            let g = HexGeometry::from_corners(&refh, corners);
            let faces = hex_faces(&refh, &g);
            let nn = refh.n_nodes();
            let fx: Vec<f64> = (0..nn).map(|k| g.x[k] * g.x[k]).collect();
            let fy: Vec<f64> = (0..nn).map(|k| g.y[k] * g.y[k]).collect();
            let fz: Vec<f64> = (0..nn).map(|k| g.z[k] * g.z[k]).collect();
            let vol: f64 = {
                let dx = g.grad_x(&refh, &fx);
                let dy = g.grad_y(&refh, &fy);
                let dz = g.grad_z(&refh, &fz);
                (0..nn).map(|k| g.jw[k] * (dx[k] + dy[k] + dz[k])).sum()
            };
            let mut surf = 0.0;
            for f in &faces {
                for (a, &v) in f.nodes.iter().enumerate() {
                    surf += f.sw[a] * (fx[v] * f.nx[a] + fy[v] * f.ny[a] + fz[v] * f.nz[a]);
                }
            }
            assert!((vol - surf).abs() < 1e-8, "p={p} vol={vol} surf={surf}");
        }
    }

    #[test]
    fn closed_surface_normal_integral_is_zero() {
        // Discrete GCL: ∮ n dS = 0 for a closed element (a non-axis-aligned box here).
        let corners = cube_corners([0.0, 2.3], [0.5, 3.1], [-1.0, 1.7]);
        for p in 1..=MAX_P {
            let refh = Reference3dHex::new(p);
            let g = HexGeometry::from_corners(&refh, corners);
            let faces = hex_faces(&refh, &g);
            let (mut ix, mut iy, mut iz) = (0.0, 0.0, 0.0);
            for f in &faces {
                for a in 0..f.nodes.len() {
                    ix += f.sw[a] * f.nx[a];
                    iy += f.sw[a] * f.ny[a];
                    iz += f.sw[a] * f.nz[a];
                }
            }
            assert!(ix.abs() < 1e-9 && iy.abs() < 1e-9 && iz.abs() < 1e-9, "p={p} ∮n=({ix},{iy},{iz})");
        }
    }
}
