//! Face / trace operators for a quad element — the per-element half of DG flux
//! coupling.
//!
//! For each of the four edges this gives: the volume node indices along the edge,
//! the unit outward normal, the surface Jacobian `sJ`, and the surface quadrature
//! weight `w₁d · sJ`. The scaled outward normal on a reference face is `J·∇r` (for
//! `r = ±1` faces) or `J·∇s` (for `s = ±1`), which in terms of [`QuadGeometry`]'s
//! metrics is:
//!
//! | edge  | ref normal | scaled outward normal |
//! |-------|------------|-----------------------|
//! | South | (0,−1)     | −J·(sx, sy)           |
//! | East  | (+1,0)     | +J·(rx, ry)           |
//! | North | (0,+1)     | +J·(sx, sy)           |
//! | West  | (−1,0)     | −J·(rx, ry)           |
//!
//! `sJ = |scaled normal|`, `(nx,ny) = scaled/sJ`, surface weight `= w₁d · sJ`.

use super::geometry::QuadGeometry;
use super::quad::Reference2dQuad;

/// Local edge identifier, CCW from the bottom. Discriminants are the local face
/// index used by mesh connectivity (step 5).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Edge {
    South = 0,
    East = 1,
    North = 2,
    West = 3,
}

impl Edge {
    /// All four edges in canonical (CCW) order.
    pub const ALL: [Edge; 4] = [Edge::South, Edge::East, Edge::North, Edge::West];
}

/// Geometric trace data for one edge of one element (length `p+1` per field).
#[derive(Clone, Debug)]
pub struct FaceData {
    pub edge: Edge,
    /// Volume node indices along the edge, in increasing along-edge parameter order.
    pub nodes: Vec<usize>,
    /// Unit outward normal components at each face node.
    pub nx: Vec<f64>,
    pub ny: Vec<f64>,
    /// Surface Jacobian (physical/reference edge-length ratio) at each face node.
    pub sj: Vec<f64>,
    /// Surface quadrature weight `w₁d · sJ` at each face node.
    pub sw: Vec<f64>,
}

/// Build the four [`FaceData`] of a quad element from its geometry.
pub fn quad_faces(refq: &Reference2dQuad, g: &QuadGeometry) -> [FaceData; 4] {
    let n = refq.n_1d();
    let w = &refq.line.weights;

    // (volume-index list, scaled-outward-normal at a volume index) per edge.
    let south_idx: Vec<usize> = (0..n).collect();
    let east_idx: Vec<usize> = (0..n).map(|j| (n - 1) + j * n).collect();
    let north_idx: Vec<usize> = (0..n).map(|i| i + (n - 1) * n).collect();
    let west_idx: Vec<usize> = (0..n).map(|j| j * n).collect();

    let scaled_r = |k: usize| [g.jac[k] * g.rx[k], g.jac[k] * g.ry[k]]; // J·∇r
    let scaled_s = |k: usize| [g.jac[k] * g.sx[k], g.jac[k] * g.sy[k]]; // J·∇s

    let build = |edge: Edge, idx: Vec<usize>, scaled: &dyn Fn(usize) -> [f64; 2], sign: f64| {
        let m = idx.len();
        let mut nx = vec![0.0; m];
        let mut ny = vec![0.0; m];
        let mut sj = vec![0.0; m];
        let mut sw = vec![0.0; m];
        for a in 0..m {
            let v = scaled(idx[a]);
            let sx = sign * v[0];
            let sy = sign * v[1];
            let s = (sx * sx + sy * sy).sqrt();
            nx[a] = sx / s;
            ny[a] = sy / s;
            sj[a] = s;
            sw[a] = w[a] * s;
        }
        FaceData { edge, nodes: idx, nx, ny, sj, sw }
    };

    [
        build(Edge::South, south_idx, &scaled_s, -1.0),
        build(Edge::East, east_idx, &scaled_r, 1.0),
        build(Edge::North, north_idx, &scaled_s, 1.0),
        build(Edge::West, west_idx, &scaled_r, -1.0),
    ]
}

#[cfg(test)]
mod tests {
    use super::*;

    const MAX_P: usize = 6;

    fn approx(a: f64, b: f64, tol: f64) -> bool {
        (a - b).abs() < tol
    }

    #[test]
    fn reference_square_normals() {
        let corners = [[-1.0, -1.0], [1.0, -1.0], [1.0, 1.0], [-1.0, 1.0]];
        let expect = [
            (Edge::South, [0.0, -1.0]),
            (Edge::East, [1.0, 0.0]),
            (Edge::North, [0.0, 1.0]),
            (Edge::West, [-1.0, 0.0]),
        ];
        for p in 1..=MAX_P {
            let refq = Reference2dQuad::new(p);
            let g = QuadGeometry::from_corners(&refq, corners);
            let faces = quad_faces(&refq, &g);
            for (f, (edge, nrm)) in faces.iter().zip(expect) {
                assert_eq!(f.edge, edge);
                assert_eq!(f.nodes.len(), p + 1);
                for a in 0..f.nodes.len() {
                    assert!(approx(f.nx[a], nrm[0], 1e-10) && approx(f.ny[a], nrm[1], 1e-10));
                    assert!(approx(f.sj[a], 1.0, 1e-10), "ref sJ");
                    assert!(approx(f.sw[a], refq.line.weights[a], 1e-12));
                }
            }
        }
    }

    #[test]
    fn rectangle_surface_jacobian_and_perimeter() {
        // [0,3] × [0,5]: east sJ = Ly/2 = 2.5, north sJ = Lx/2 = 1.5; perimeter = 16.
        let (lx, ly) = (3.0, 5.0);
        let corners = [[0.0, 0.0], [lx, 0.0], [lx, ly], [0.0, ly]];
        for p in 1..=MAX_P {
            let refq = Reference2dQuad::new(p);
            let g = QuadGeometry::from_corners(&refq, corners);
            let faces = quad_faces(&refq, &g);
            for s in &faces[Edge::East as usize].sj {
                assert!(approx(*s, ly / 2.0, 1e-10));
            }
            for s in &faces[Edge::North as usize].sj {
                assert!(approx(*s, lx / 2.0, 1e-10));
            }
            let perim: f64 = faces.iter().flat_map(|f| f.sw.iter()).sum();
            assert!(approx(perim, 2.0 * (lx + ly), 1e-10), "p={p} perim={perim}");
        }
    }

    #[test]
    fn divergence_theorem_exact_on_affine_parallelogram() {
        // Sheared (affine) cell ⇒ constant metrics; for polynomial F the discrete
        // ∮ F·n ds must equal ∫ ∇·F dA exactly.  F = (x², y²) ⇒ ∇·F = 2x + 2y.
        let corners = [[0.0, 0.0], [2.0, 0.5], [2.5, 2.0], [0.5, 1.5]];
        for p in 2..=MAX_P {
            let refq = Reference2dQuad::new(p);
            let g = QuadGeometry::from_corners(&refq, corners);
            let faces = quad_faces(&refq, &g);
            let nn = refq.n_nodes();
            let fx: Vec<f64> = (0..nn).map(|k| g.x[k] * g.x[k]).collect();
            let fy: Vec<f64> = (0..nn).map(|k| g.y[k] * g.y[k]).collect();

            let vol: f64 = {
                let dfx = g.grad_x(&refq, &fx);
                let dfy = g.grad_y(&refq, &fy);
                (0..nn).map(|k| g.jw[k] * (dfx[k] + dfy[k])).sum()
            };
            let mut surf = 0.0;
            for f in &faces {
                for (a, &v) in f.nodes.iter().enumerate() {
                    surf += f.sw[a] * (fx[v] * f.nx[a] + fy[v] * f.ny[a]);
                }
            }
            assert!((vol - surf).abs() < 1e-9, "p={p} vol={vol} surf={surf}");
        }
    }

    #[test]
    fn closed_surface_normal_integral_is_zero_on_bilinear_quad() {
        // Discrete GCL: ∮ n ds = 0 for any closed element, even a non-affine quad.
        let corners = [[0.0, 0.0], [2.0, 0.3], [2.4, 2.1], [0.2, 1.8]];
        for p in 1..=MAX_P {
            let refq = Reference2dQuad::new(p);
            let g = QuadGeometry::from_corners(&refq, corners);
            let faces = quad_faces(&refq, &g);
            let (mut ix, mut iy) = (0.0, 0.0);
            for f in &faces {
                for a in 0..f.nodes.len() {
                    ix += f.sw[a] * f.nx[a];
                    iy += f.sw[a] * f.ny[a];
                }
            }
            assert!(ix.abs() < 1e-9 && iy.abs() < 1e-9, "p={p} ∮n=({ix},{iy})");
        }
    }

    #[test]
    fn surface_weights_and_jacobian_positive() {
        let corners = [[0.0, 0.0], [2.0, 0.3], [2.4, 2.1], [0.2, 1.8]];
        for p in 1..=MAX_P {
            let refq = Reference2dQuad::new(p);
            let g = QuadGeometry::from_corners(&refq, corners);
            for f in quad_faces(&refq, &g) {
                assert!(f.sj.iter().all(|&s| s > 0.0));
                assert!(f.sw.iter().all(|&s| s > 0.0));
            }
        }
    }
}
