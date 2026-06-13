//! Shifted Boundary Method (SBM) geometry — a **sharp** embedded boundary built on whole
//! mesh elements, with NO cut-cell quadrature. See `docs/research-sharp-interface.md`.
//!
//! The true solid is a signed-distance **level set** (`φ < 0` inside the solid, `> 0` in
//! the fluid). The **surrogate** fluid domain is the union of mesh elements lying ENTIRELY
//! in the fluid; the **surrogate boundary** is the set of faces between an active (fluid)
//! element and an inactive (cut/solid) one. The no-slip BC is later imposed weakly
//! (Nitsche) on the surrogate boundary, *shifted* to the true boundary by the distance
//! vector `d = (closest true-boundary point) − (surrogate point)`, with a Taylor correction
//! recovering high-order accuracy. This module builds that geometry — the foundation step of
//! the SBM build; the element classification + closest-point machinery is also what a future
//! cut-cell DG backend will reuse.

use super::{Edge, Mesh2d, Neighbor};

/// A signed-distance-like **level set**: `phi < 0` inside the solid, `> 0` in the fluid,
/// `= 0` on the true boundary.
pub trait LevelSet {
    /// Signed level-set value at a point (negative inside the solid).
    fn phi(&self, x: f64, y: f64) -> f64;
    /// Closest point on the true boundary `{phi = 0}` to `(x, y)`, and the fluid-side
    /// outward unit normal there (pointing from the solid into the fluid).
    fn closest(&self, x: f64, y: f64) -> ([f64; 2], [f64; 2]);
}

/// Solid **disk** of radius `r` centred at `(cx, cy)` — the fluid is the exterior.
/// `phi(x) = |x − c| − r` (an exact signed distance).
#[derive(Clone, Copy, Debug)]
pub struct CircleLevelSet {
    pub cx: f64,
    pub cy: f64,
    pub r: f64,
}

impl CircleLevelSet {
    pub fn new(cx: f64, cy: f64, r: f64) -> Self {
        Self { cx, cy, r }
    }
}

impl LevelSet for CircleLevelSet {
    fn phi(&self, x: f64, y: f64) -> f64 {
        ((x - self.cx).powi(2) + (y - self.cy).powi(2)).sqrt() - self.r
    }
    fn closest(&self, x: f64, y: f64) -> ([f64; 2], [f64; 2]) {
        let (dx, dy) = (x - self.cx, y - self.cy);
        let d = (dx * dx + dy * dy).sqrt().max(1e-300);
        let (nx, ny) = (dx / d, dy / d); // outward (away from the disk centre) = into fluid
        ([self.cx + self.r * nx, self.cy + self.r * ny], [nx, ny])
    }
}

/// One surrogate-boundary face node: the surrogate point (a mesh face quadrature node), the
/// shift vector `d` to the true boundary, the true-boundary outward (fluid-side) normal, the
/// surface quadrature weight, and the owning element's volume-node index (for the trace).
#[derive(Clone, Copy, Debug)]
pub struct SurrogateNode {
    pub x: f64,
    pub y: f64,
    /// Shift `d = closest_true_point − surrogate_point` (the Taylor-correction offset).
    pub dx: f64,
    pub dy: f64,
    /// True-boundary outward unit normal (from solid into fluid).
    pub tnx: f64,
    pub tny: f64,
    /// Surface quadrature weight at this node.
    pub sw: f64,
    /// Volume-node index within the owning element (the trace dof).
    pub node: usize,
}

/// A surrogate-boundary face: an edge of an active element that faces an inactive one.
#[derive(Clone, Debug)]
pub struct SurrogateFace {
    pub elem: usize,
    pub edge: Edge,
    pub nodes: Vec<SurrogateNode>,
}

/// SBM geometry for a level set on a mesh: which elements are active (the surrogate fluid
/// domain) and the surrogate boundary (where the embedded BC will be imposed).
#[derive(Clone, Debug)]
pub struct ShiftedBoundary {
    /// `active[e]` ⇒ element `e` lies entirely in the fluid (part of the surrogate domain).
    pub active: Vec<bool>,
    /// The surrogate-boundary faces (active-element edges facing inactive elements).
    pub faces: Vec<SurrogateFace>,
    /// Per element: indices into `faces` owned by that element (for parallel, per-element
    /// operator assembly). `faces_by_elem[e]` lists the surrogate faces of element `e`.
    pub faces_by_elem: Vec<Vec<usize>>,
}

impl ShiftedBoundary {
    /// Classify elements and build the surrogate boundary for level set `ls` on `mesh`. An
    /// element is **active** iff every one of its volume nodes lies in the fluid
    /// (`phi ≥ 0`); the surrogate boundary is each active element's edge whose **interior**
    /// neighbour is inactive (domain-wall edges are ordinary BCs, not surrogate faces).
    pub fn new(mesh: &Mesh2d, ls: &impl LevelSet) -> Self {
        let nn = mesh.refq.n_nodes();
        let active: Vec<bool> = mesh
            .elements
            .iter()
            .map(|el| (0..nn).all(|k| ls.phi(el.geom.x[k], el.geom.y[k]) >= 0.0))
            .collect();

        let mut faces = Vec::new();
        for (e, el) in mesh.elements.iter().enumerate() {
            if !active[e] {
                continue;
            }
            for (le, nb) in el.neighbors.iter().enumerate() {
                let inactive_neighbor = match nb {
                    Neighbor::Interior { elem, .. } => !active[*elem],
                    Neighbor::Boundary { .. } => false,
                    // 2:1 AMR (CoarseToFine/FineToCoarse) surrogate faces are not yet handled
                    // — SBM is built on conforming meshes first (the benchmark case).
                    _ => false,
                };
                if !inactive_neighbor {
                    continue;
                }
                let face = &el.faces[le];
                let nodes = face
                    .nodes
                    .iter()
                    .enumerate()
                    .map(|(a, &vn)| {
                        let (px, py) = (el.geom.x[vn], el.geom.y[vn]);
                        let (c, n) = ls.closest(px, py);
                        SurrogateNode {
                            x: px,
                            y: py,
                            dx: c[0] - px,
                            dy: c[1] - py,
                            tnx: n[0],
                            tny: n[1],
                            sw: face.sw[a],
                            node: vn,
                        }
                    })
                    .collect();
                faces.push(SurrogateFace { elem: e, edge: Edge::ALL[le], nodes });
            }
        }
        let mut faces_by_elem = vec![Vec::new(); mesh.n_elements()];
        for (i, sf) in faces.iter().enumerate() {
            faces_by_elem[sf.elem].push(i);
        }
        Self { active, faces, faces_by_elem }
    }

    /// Number of active (surrogate-fluid) elements.
    pub fn n_active(&self) -> usize {
        self.active.iter().filter(|&&a| a).count()
    }

    /// Active surrogate-domain area (∑ `jw` over active elements).
    pub fn active_area(&self, mesh: &Mesh2d) -> f64 {
        let nn = mesh.refq.n_nodes();
        self.active
            .iter()
            .enumerate()
            .filter(|(_, a)| **a)
            .map(|(e, _)| mesh.elements[e].geom.jw[..nn].iter().sum::<f64>())
            .sum()
    }

    /// Total surrogate-boundary length (∑ surface weights over all surrogate nodes).
    pub fn surrogate_length(&self) -> f64 {
        self.faces.iter().flat_map(|f| f.nodes.iter().map(|n| n.sw)).sum()
    }

    /// Largest shift magnitude `|d|` over all surrogate nodes (the Taylor-correction
    /// distance; should be on the order of one element size).
    pub fn max_shift(&self) -> f64 {
        self.faces
            .iter()
            .flat_map(|f| f.nodes.iter().map(|n| n.dx.hypot(n.dy)))
            .fold(0.0, f64::max)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dg::Mesh2d;
    use std::f64::consts::PI;

    #[test]
    fn circle_level_set_sign_and_closest_point() {
        let ls = CircleLevelSet::new(0.5, 0.5, 0.2);
        assert!(ls.phi(0.5, 0.5) < 0.0, "centre is inside the solid");
        assert!(ls.phi(0.9, 0.5) > 0.0, "far point is in the fluid");
        assert!((ls.phi(0.5, 0.8) - 0.1).abs() < 1e-12, "phi = dist − r");
        // Closest point on the circle from an exterior point lands on the circle, with the
        // outward (fluid-side) normal pointing away from the centre.
        let (c, n) = ls.closest(0.9, 0.5);
        assert!((c[0] - 0.7).abs() < 1e-12 && (c[1] - 0.5).abs() < 1e-12);
        assert!((n[0] - 1.0).abs() < 1e-12 && n[1].abs() < 1e-12);
    }

    #[test]
    fn surrogate_nodes_shift_onto_the_true_boundary() {
        // Every surrogate node + its shift d must land exactly on the true circle, and the
        // shift must point along the true normal (d ∥ n). This is the geometric core SBM's
        // Taylor correction relies on.
        let mesh = Mesh2d::rectangular(4, 24, 24, [0.0, 1.0], [0.0, 1.0]);
        let ls = CircleLevelSet::new(0.5, 0.5, 0.2);
        let sb = ShiftedBoundary::new(&mesh, &ls);
        assert!(!sb.faces.is_empty(), "expected surrogate faces around the disk");
        for f in &sb.faces {
            for n in &f.nodes {
                let onb = ls.phi(n.x + n.dx, n.y + n.dy); // φ at the shifted (true) point
                assert!(onb.abs() < 1e-10, "shifted point not on the boundary: φ={onb}");
                // d is along the true normal (cross product ≈ 0).
                let cross = n.dx * n.tny - n.dy * n.tnx;
                assert!(cross.abs() < 1e-10, "shift not normal to the boundary: {cross}");
            }
        }
    }

    #[test]
    fn active_region_excludes_the_disk_and_is_inside_the_true_fluid() {
        // The surrogate (active) domain must (a) exclude the disk interior, and (b) be a
        // subset of the true fluid (active area ≤ box − disk), with the deficit no larger
        // than a one-element-thick layer around the surrogate boundary.
        let (p, n) = (4, 32);
        let mesh = Mesh2d::rectangular(p, n, n, [0.0, 1.0], [0.0, 1.0]);
        let ls = CircleLevelSet::new(0.5, 0.5, 0.2);
        let sb = ShiftedBoundary::new(&mesh, &ls);
        let box_area = 1.0;
        let disk_area = PI * 0.2 * 0.2;
        let true_fluid = box_area - disk_area;
        let a = sb.active_area(&mesh);
        assert!(a < true_fluid + 1e-9, "active area {a} exceeds the true fluid {true_fluid}");
        let h = 1.0 / n as f64;
        let deficit = true_fluid - a;
        // The excluded cut layer is ~ (circumference)·h.
        assert!(deficit > 0.0 && deficit < 3.0 * 2.0 * PI * 0.2 * h, "deficit {deficit} too large (h={h})");
        // No active element has its centre inside the disk.
        let nn = mesh.refq.n_nodes();
        for (e, act) in sb.active.iter().enumerate() {
            if *act {
                let cx: f64 = mesh.elements[e].geom.x[..nn].iter().sum::<f64>() / nn as f64;
                let cy: f64 = mesh.elements[e].geom.y[..nn].iter().sum::<f64>() / nn as f64;
                assert!(ls.phi(cx, cy) > -h, "active element centre deep inside the disk");
            }
        }
    }

    #[test]
    fn surrogate_shift_vanishes_with_refinement() {
        // The defining SBM consistency property: as the mesh refines, the max shift |d|
        // from the surrogate boundary to the true boundary → 0 as O(h), so the surrogate
        // boundary approaches the true interface and the (Taylor-corrected) BC becomes
        // exact. NOTE the surrogate length does NOT converge to the smooth circumference
        // 2πr — a Cartesian surrogate boundary is a STAIRCASE whose length tends to the
        // taxicab perimeter 8r (staircase paradox); SBM integrates over that staircase but
        // uses the TRUE normal + shift, so its length is not meant to equal 2πr. We only
        // require the length to stay bounded between ≈2πr and ≈ the taxicab perimeter.
        let ls = CircleLevelSet::new(0.5, 0.5, 0.2);
        let (circ, taxicab) = (2.0 * PI * 0.2, 8.0 * 0.2);
        let mut prev_shift = f64::INFINITY;
        for &n in &[32usize, 64, 128] {
            let mesh = Mesh2d::rectangular(2, n, n, [0.0, 1.0], [0.0, 1.0]);
            let sb = ShiftedBoundary::new(&mesh, &ls);
            let shift = sb.max_shift();
            let len = sb.surrogate_length();
            let h = 1.0 / n as f64;
            eprintln!("n={n}: max|d|={shift:.4} (h={h:.4}, ratio {:.2})  len={len:.4} (2πr={circ:.3}, taxicab={taxicab:.3})", shift / h);
            assert!(shift < 2.5 * h, "max shift {shift} should be O(h)={h}");
            assert!(shift < prev_shift, "shift must shrink with refinement ({shift} !< {prev_shift})");
            assert!(len > 0.9 * circ && len < 1.3 * taxicab, "surrogate length {len} outside [2πr, ~taxicab]");
            prev_shift = shift;
        }
    }
}
