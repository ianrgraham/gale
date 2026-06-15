//! Octree AMR transfer operators for hexes — the 3D analogue of the `h`-refinement
//! operators in [`amr`](super::amr). A hex refines into **8 children** (octants);
//! [`RefineHex`] provides the prolong (parent → child, exact for degree ≤ p) and the
//! conservative, mass-weighted restrict (8 children → parent, the L2 adjoint, ⅛
//! child Jacobian). [`SmoothnessIndicator3d`] flags under-resolved hexes.
//!
//! These are the prerequisites for a non-conforming (2:1, 2×2-mortar) 3D solver,
//! exactly as the 2D `RefineQuad` operators preceded the 2D non-conforming solver.

use super::amr::lagrange_basis;
use super::filter::mat_inverse;
use super::reference::{legendre_all, Reference1d};
use std::collections::{HashMap, HashSet};

/// `h`-refinement transfer operators for an order-`p` hex element. Children are
/// indexed `c = cx + 2cy + 4cz`, `cx,cy,cz ∈ {0,1}`.
pub struct RefineHex {
    pub order: usize,
    /// `p_left[i*n+j] = ℓ_j((node_i−1)/2)` — parent basis at the left child's node.
    p_left: Vec<f64>,
    /// `p_right[i*n+j] = ℓ_j((node_i+1)/2)`.
    p_right: Vec<f64>,
    /// LGL weights (for the conservative restriction).
    w: Vec<f64>,
}

impl RefineHex {
    pub fn new(order: usize) -> Self {
        let r1 = Reference1d::new(order);
        let nodes = &r1.nodes;
        let n = order + 1;
        let mut p_left = vec![0.0; n * n];
        let mut p_right = vec![0.0; n * n];
        for i in 0..n {
            let ll = lagrange_basis(nodes, 0.5 * (nodes[i] - 1.0));
            let lr = lagrange_basis(nodes, 0.5 * (nodes[i] + 1.0));
            for j in 0..n {
                p_left[i * n + j] = ll[j];
                p_right[i * n + j] = lr[j];
            }
        }
        Self { order, p_left, p_right, w: r1.weights }
    }

    fn axis(&self, c: usize) -> &[f64] {
        if c == 0 {
            &self.p_left
        } else {
            &self.p_right
        }
    }

    /// **Hex-face mortar**: project a coarse hex-face trace (`n²` nodes in tensor order
    /// `ia + ib·n`) onto the fine quarter-face `(ha, hb) ∈ {0,1}²` — the 2D analogue of
    /// [`RefineQuad::mortar_to_fine`](super::amr::RefineQuad::mortar_to_fine), the tensor product
    /// `P_ha ⊗ P_hb` of the two 1D mortar matrices (`P_0 = p_left`, `P_1 = p_right`). Exact for
    /// degree ≤ p. Used to give a coarse hex's flux to one of its four refined face-neighbours.
    pub fn mortar_to_fine_face(&self, coarse: &[f64], ha: usize, hb: usize) -> Vec<f64> {
        let n = self.order + 1;
        let (pa, pb) = (self.axis(ha), self.axis(hb));
        let mut tmp = vec![0.0; n * n]; // tmp[ia, jb] = Σ_ja pa[ia,ja] coarse[ja, jb]
        for jb in 0..n {
            for ia in 0..n {
                let mut s = 0.0;
                for ja in 0..n {
                    s += pa[ia * n + ja] * coarse[ja + jb * n];
                }
                tmp[ia + jb * n] = s;
            }
        }
        let mut out = vec![0.0; n * n]; // out[ia, ib] = Σ_jb pb[ib,jb] tmp[ia, jb]
        for ib in 0..n {
            for ia in 0..n {
                let mut s = 0.0;
                for jb in 0..n {
                    s += pb[ib * n + jb] * tmp[ia + jb * n];
                }
                out[ia + ib * n] = s;
            }
        }
        out
    }

    /// Transpose of [`mortar_to_fine_face`] for quarter `(ha, hb)`: `out[ja,jb] = Σ_{ia,ib}
    /// P_ha[ia,ja]·P_hb[ib,jb]·v[ia,ib]`, the adjoint `(P_ha ⊗ P_hb)ᵀ`. Scatters already-
    /// quadrature-weighted fine-face contributions back to the coarse test nodes (keeps the SIPG
    /// operator symmetric, `P`/`Pᵀ` an adjoint pair) — the 2D analogue of
    /// [`RefineQuad::mortar_gather`](super::amr::RefineQuad::mortar_gather).
    pub fn mortar_gather_face(&self, v: &[f64], ha: usize, hb: usize) -> Vec<f64> {
        let n = self.order + 1;
        let (pa, pb) = (self.axis(ha), self.axis(hb));
        let mut tmp = vec![0.0; n * n]; // tmp[ja, ib] = Σ_ia pa[ia,ja] v[ia, ib]
        for ib in 0..n {
            for ja in 0..n {
                let mut s = 0.0;
                for ia in 0..n {
                    s += pa[ia * n + ja] * v[ia + ib * n];
                }
                tmp[ja + ib * n] = s;
            }
        }
        let mut out = vec![0.0; n * n]; // out[ja, jb] = Σ_ib pb[ib,jb] tmp[ja, ib]
        for jb in 0..n {
            for ja in 0..n {
                let mut s = 0.0;
                for ib in 0..n {
                    s += pb[ib * n + jb] * tmp[ja + ib * n];
                }
                out[ja + jb * n] = s;
            }
        }
        out
    }

    /// Prolong a parent nodal field to child `(cx, cy, cz)`. Exact for degree ≤ p.
    pub fn prolong(&self, parent: &[f64], cx: usize, cy: usize, cz: usize) -> Vec<f64> {
        let n = self.order + 1;
        let n2 = n * n;
        let pr = self.axis(cx);
        let ps = self.axis(cy);
        let pt = self.axis(cz);
        // Three 1D contractions: r, then s, then t.
        let mut t1 = vec![0.0; n * n * n]; // contract r
        for k in 0..n {
            for j in 0..n {
                for i in 0..n {
                    let mut s = 0.0;
                    for a in 0..n {
                        s += pr[i * n + a] * parent[a + j * n + k * n2];
                    }
                    t1[i + j * n + k * n2] = s;
                }
            }
        }
        let mut t2 = vec![0.0; n * n * n]; // contract s
        for k in 0..n {
            for j in 0..n {
                for i in 0..n {
                    let mut s = 0.0;
                    for b in 0..n {
                        s += ps[j * n + b] * t1[i + b * n + k * n2];
                    }
                    t2[i + j * n + k * n2] = s;
                }
            }
        }
        let mut child = vec![0.0; n * n * n]; // contract t
        for k in 0..n {
            for j in 0..n {
                for i in 0..n {
                    let mut s = 0.0;
                    for c in 0..n {
                        s += pt[k * n + c] * t2[i + j * n + c * n2];
                    }
                    child[i + j * n + k * n2] = s;
                }
            }
        }
        child
    }

    /// Restrict the 8 children (ordered `c = cx + 2cy + 4cz`) to the parent via the
    /// conservative, mass-weighted L2 projection (⅛ child Jacobian). Restriction of
    /// a constant is exactly that constant; the cell integral is preserved.
    pub fn restrict(&self, children: &[Vec<f64>; 8]) -> Vec<f64> {
        let n = self.order + 1;
        let n2 = n * n;
        let mut parent = vec![0.0; n * n * n];
        for cz in 0..2 {
            for cy in 0..2 {
                for cx in 0..2 {
                    let uc = &children[cx + 2 * cy + 4 * cz];
                    let pr = self.axis(cx);
                    let ps = self.axis(cy);
                    let pt = self.axis(cz);
                    for kk in 0..n {
                        for jj in 0..n {
                            for ii in 0..n {
                                let mut s = 0.0;
                                for c in 0..n {
                                    let wc = self.w[c];
                                    for b in 0..n {
                                        let wb = self.w[b];
                                        for a in 0..n {
                                            s += pr[a * n + ii]
                                                * ps[b * n + jj]
                                                * pt[c * n + kk]
                                                * self.w[a]
                                                * wb
                                                * wc
                                                * uc[a + b * n + c * n2];
                                        }
                                    }
                                }
                                parent[ii + jj * n + kk * n2] +=
                                    0.125 * s / (self.w[ii] * self.w[jj] * self.w[kk]);
                            }
                        }
                    }
                }
            }
        }
        parent
    }
}

/// Spectral (Persson–Peraire) smoothness indicator on a hex: the fraction of an
/// element's L2 energy carried by the highest Legendre modes. ≈0 for a
/// spectrally-resolved field, O(1) for an under-resolved one. The 3D analogue of
/// [`SmoothnessIndicator`](super::amr::SmoothnessIndicator).
pub struct SmoothnessIndicator3d {
    order: usize,
    vinv: Vec<f64>,  // 1D nodal→modal, (p+1)²
    gamma: Vec<f64>, // Legendre L2 norms 2/(2k+1)
}

impl SmoothnessIndicator3d {
    pub fn new(order: usize) -> Self {
        let n = order + 1;
        let nodes = Reference1d::new(order).nodes;
        let mut v = vec![0.0; n * n];
        for i in 0..n {
            let leg = legendre_all(order, nodes[i]);
            for a in 0..n {
                v[i * n + a] = leg[a];
            }
        }
        let vinv = mat_inverse(n, &v);
        let gamma = (0..n).map(|k| 2.0 / (2.0 * k as f64 + 1.0)).collect();
        Self { order, vinv, gamma }
    }

    /// Fraction of L2 energy in the highest modes (`a==p` or `b==p` or `c==p`), in
    /// `[0,1]`. `field` is `(p+1)³` nodal values in `i + j·n + k·n²` order.
    pub fn indicator(&self, field: &[f64]) -> f64 {
        let n = self.order + 1;
        let n2 = n * n;
        let m = |row: usize, col: usize| self.vinv[row * n + col];
        // Three sequential 1D modal transforms: r, s, t.
        let mut t1 = vec![0.0; n * n * n];
        for k in 0..n {
            for j in 0..n {
                for a in 0..n {
                    let mut s = 0.0;
                    for i in 0..n {
                        s += m(a, i) * field[i + j * n + k * n2];
                    }
                    t1[a + j * n + k * n2] = s;
                }
            }
        }
        let mut t2 = vec![0.0; n * n * n];
        for k in 0..n {
            for b in 0..n {
                for a in 0..n {
                    let mut s = 0.0;
                    for j in 0..n {
                        s += m(b, j) * t1[a + j * n + k * n2];
                    }
                    t2[a + b * n + k * n2] = s;
                }
            }
        }
        let mut chat = vec![0.0; n * n * n];
        for c in 0..n {
            for b in 0..n {
                for a in 0..n {
                    let mut s = 0.0;
                    for k in 0..n {
                        s += m(c, k) * t2[a + b * n + k * n2];
                    }
                    chat[a + b * n + c * n2] = s;
                }
            }
        }
        let (mut e_total, mut e_top) = (0.0, 0.0);
        let p = self.order;
        for c in 0..n {
            for b in 0..n {
                for a in 0..n {
                    let coeff = chat[a + b * n + c * n2];
                    let e = coeff * coeff * self.gamma[a] * self.gamma[b] * self.gamma[c];
                    e_total += e;
                    if a == p || b == p || c == p {
                        e_top += e;
                    }
                }
            }
        }
        if e_total > 0.0 {
            e_top / e_total
        } else {
            0.0
        }
    }
}

// ===== Octree AMR remap + indicator (the 3D analogues of the `amr` 2D helpers) ==================

/// Octree base-cell → element-index map. Mirrors `cartesian_refined`'s ordering: row-major
/// `(cz, cy, cx)`, a refined cell occupying 8 consecutive indices (child `c = cx + 2cy + 4cz`).
fn cell_element_map_3d(
    nx: usize,
    ny: usize,
    nz: usize,
    set: &HashSet<(usize, usize, usize)>,
) -> HashMap<(usize, usize, usize), Vec<usize>> {
    let mut map = HashMap::new();
    let mut idx = 0;
    for cz in 0..nz {
        for cy in 0..ny {
            for cx in 0..nx {
                if set.contains(&(cx, cy, cz)) {
                    map.insert((cx, cy, cz), (0..8).map(|c| idx + c).collect());
                    idx += 8;
                } else {
                    map.insert((cx, cy, cz), vec![idx]);
                    idx += 1;
                }
            }
        }
    }
    map
}

/// Octree 2:1 scalar transfer for one refinement transition (newly-refined cells prolonged to 8
/// children, newly-coarsened cells conservatively restricted, unchanged cells copied). The 3D
/// analogue of `amr::remap_scalar_data`.
fn remap_scalar_data_3d(
    rh: &RefineHex,
    nx: usize,
    ny: usize,
    nz: usize,
    old_set: &HashSet<(usize, usize, usize)>,
    new_set: &HashSet<(usize, usize, usize)>,
    u_old: &[Vec<f64>],
) -> Vec<Vec<f64>> {
    let old_map = cell_element_map_3d(nx, ny, nz, old_set);
    let mut u_new = Vec::new();
    for cz in 0..nz {
        for cy in 0..ny {
            for cx in 0..nx {
                let oe = &old_map[&(cx, cy, cz)];
                match (old_set.contains(&(cx, cy, cz)), new_set.contains(&(cx, cy, cz))) {
                    (false, false) => u_new.push(u_old[oe[0]].clone()),
                    (false, true) => {
                        for c in 0..8 {
                            u_new.push(rh.prolong(&u_old[oe[0]], c % 2, (c / 2) % 2, c / 4));
                        }
                    }
                    (true, false) => {
                        let children: [Vec<f64>; 8] = std::array::from_fn(|c| u_old[oe[c]].clone());
                        u_new.push(rh.restrict(&children));
                    }
                    (true, true) => {
                        for c in 0..8 {
                            u_new.push(u_old[oe[c]].clone());
                        }
                    }
                }
            }
        }
    }
    u_new
}

/// Remap a flat scalar component (`[n_elem · n_nodes]`, element-ordered) between octree refinement
/// states — the 3D analogue of [`remap_component_flat`](super::amr::remap_component_flat).
pub fn remap_component_flat_3d(
    order: usize,
    nx: usize,
    ny: usize,
    nz: usize,
    old_set: &[(usize, usize, usize)],
    comp_old: &[f64],
    new_set: &[(usize, usize, usize)],
) -> Vec<f64> {
    let rh = RefineHex::new(order);
    let nn = (order + 1).pow(3);
    let old_h: HashSet<_> = old_set.iter().copied().collect();
    let new_h: HashSet<_> = new_set.iter().copied().collect();
    let u_old: Vec<Vec<f64>> = comp_old.chunks(nn).map(|c| c.to_vec()).collect();
    remap_scalar_data_3d(&rh, nx, ny, nz, &old_h, &new_h, &u_old).concat()
}

/// Per base-cell smoothness on the current (possibly-refined) octree mesh — level-aware (a refined
/// cell is restricted to base before indicating). The 3D analogue of
/// [`smoothness_per_cell`](super::amr::smoothness_per_cell); returns `nx·ny·nz` values in
/// `cx + cy·nx + cz·nx·ny` order. Used as the REFINE criterion on unrefined cells.
pub fn smoothness_per_cell_3d(
    order: usize,
    nx: usize,
    ny: usize,
    nz: usize,
    old_set: &[(usize, usize, usize)],
    indic_comp: &[f64],
) -> Vec<f64> {
    let si = SmoothnessIndicator3d::new(order);
    let rh = RefineHex::new(order);
    let nn = (order + 1).pow(3);
    let old_h: HashSet<_> = old_set.iter().copied().collect();
    let old_map = cell_element_map_3d(nx, ny, nz, &old_h);
    let u: Vec<Vec<f64>> = indic_comp.chunks(nn).map(|c| c.to_vec()).collect();
    let mut out = vec![0.0; nx * ny * nz];
    for cz in 0..nz {
        for cy in 0..ny {
            for cx in 0..nx {
                let oe = &old_map[&(cx, cy, cz)];
                let cell = if old_h.contains(&(cx, cy, cz)) {
                    let children: [Vec<f64>; 8] = std::array::from_fn(|c| u[oe[c]].clone());
                    rh.restrict(&children)
                } else {
                    u[oe[0]].clone()
                };
                out[cx + cy * nx + cz * nx * ny] = si.indicator(&cell);
            }
        }
    }
    out
}

/// For each refined cell, the MAX [`SmoothnessIndicator3d`] over its 8 children — the child-based
/// COARSEN criterion (judges the children directly, avoiding restrict-then-indicate oscillation).
/// The 3D analogue of [`child_smoothness_max`](super::amr::child_smoothness_max).
pub fn child_smoothness_max_3d(
    order: usize,
    nx: usize,
    ny: usize,
    nz: usize,
    refined: &[(usize, usize, usize)],
    indic_comp: &[f64],
) -> HashMap<(usize, usize, usize), f64> {
    let si = SmoothnessIndicator3d::new(order);
    let nn = (order + 1).pow(3);
    let set: HashSet<_> = refined.iter().copied().collect();
    let map = cell_element_map_3d(nx, ny, nz, &set);
    let mut out = HashMap::new();
    for &cell in refined {
        let mut mx = 0.0f64;
        for &e in &map[&cell] {
            mx = mx.max(si.indicator(&indic_comp[e * nn..(e + 1) * nn]));
        }
        out.insert(cell, mx);
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dg::Mesh3d;

    const P: usize = 3;

    /// Refine→coarsen round-trip recovers a degree-≤p field exactly (prolong is exact interpolation,
    /// restrict is the conservative L2 projection — exact for polynomials in the space). Also checks
    /// a partial (genuine 2:1) refinement remap is consistent.
    #[test]
    fn remap_3d_refine_then_coarsen_recovers() {
        let (order, nx, ny, nz) = (P, 2, 2, 2);
        let nn = (order + 1).pow(3);
        let base = Mesh3d::rectangular(order, nx, ny, nz, [0.0, 1.0], [0.0, 1.0], [0.0, 1.0]);
        let f = |x: f64, y: f64, z: f64| 1.0 + x - 2.0 * y + 0.5 * z + x * y * z + x * x - y * z;
        let mut comp_base = vec![0.0; base.n_elements() * nn];
        for (e, el) in base.elements.iter().enumerate() {
            for k in 0..nn {
                comp_base[e * nn + k] = f(el.geom.x[k], el.geom.y[k], el.geom.z[k]);
            }
        }
        let all: Vec<(usize, usize, usize)> =
            (0..nz).flat_map(|cz| (0..ny).flat_map(move |cy| (0..nx).map(move |cx| (cx, cy, cz)))).collect();
        let refined = remap_component_flat_3d(order, nx, ny, nz, &[], &comp_base, &all);
        let back = remap_component_flat_3d(order, nx, ny, nz, &all, &refined, &[]);
        assert_eq!(back.len(), comp_base.len());
        let err = back.iter().zip(&comp_base).map(|(a, b)| (a - b).abs()).fold(0.0, f64::max);
        assert!(err < 1e-10, "refine→coarsen round-trip not exact: {err:.3e}");
    }

    #[test]
    fn face_mortar_is_exact_and_adjoint() {
        let rh = RefineHex::new(P);
        let n = P + 1;
        let nodes = Reference1d::new(P).nodes;
        // A polynomial of degree ≤ P in each variable ⇒ the mortar projection is exact.
        let f = |x: f64, y: f64| 1.0 - 0.5 * x + 2.0 * y + x * y - 0.3 * x * x * y + 0.7 * x * x * x;
        let coarse: Vec<f64> = (0..n * n).map(|k| f(nodes[k % n], nodes[k / n])).collect();
        for hb in 0..2 {
            for ha in 0..2 {
                let fine = rh.mortar_to_fine_face(&coarse, ha, hb);
                let (sa, sb) = (if ha == 0 { -1.0 } else { 1.0 }, if hb == 0 { -1.0 } else { 1.0 });
                for ib in 0..n {
                    for ia in 0..n {
                        let want = f(0.5 * (nodes[ia] + sa), 0.5 * (nodes[ib] + sb));
                        assert!((fine[ia + ib * n] - want).abs() < 1e-12, "mortar not exact");
                    }
                }
                // Adjoint: ⟨P·c, v⟩ = ⟨c, Pᵀ·v⟩ for arbitrary c, v.
                let cc: Vec<f64> = (0..n * n).map(|k| (0.31 * k as f64 + 1.0).sin()).collect();
                let vv: Vec<f64> = (0..n * n).map(|k| (0.17 * k as f64 + 0.5).cos()).collect();
                let pc = rh.mortar_to_fine_face(&cc, ha, hb);
                let ptv = rh.mortar_gather_face(&vv, ha, hb);
                let lhs: f64 = pc.iter().zip(&vv).map(|(a, b)| a * b).sum();
                let rhs: f64 = cc.iter().zip(&ptv).map(|(a, b)| a * b).sum();
                assert!((lhs - rhs).abs() < 1e-12, "mortar P/Pᵀ not adjoint: {lhs} vs {rhs}");
            }
        }
    }

    fn child_node_coords(order: usize, cx: usize, cy: usize, cz: usize) -> Vec<[f64; 3]> {
        // Child reference nodes mapped into the parent reference cube.
        let nodes = Reference1d::new(order).nodes;
        let n = order + 1;
        let map = |t: f64, c: usize| 0.5 * (t + if c == 0 { -1.0 } else { 1.0 });
        let mut out = Vec::with_capacity(n * n * n);
        for k in 0..n {
            for j in 0..n {
                for i in 0..n {
                    out.push([map(nodes[i], cx), map(nodes[j], cy), map(nodes[k], cz)]);
                }
            }
        }
        out
    }

    #[test]
    fn prolong_is_exact_on_polynomials() {
        let rh = RefineHex::new(P);
        let pnodes = {
            let nodes = Reference1d::new(P).nodes;
            let n = P + 1;
            let mut v = Vec::new();
            for k in 0..n {
                for j in 0..n {
                    for i in 0..n {
                        v.push([nodes[i], nodes[j], nodes[k]]);
                    }
                }
            }
            v
        };
        for a in 0..=P {
            for b in 0..=P {
                for c in 0..=P {
                    let parent: Vec<f64> =
                        pnodes.iter().map(|p| p[0].powi(a as i32) * p[1].powi(b as i32) * p[2].powi(c as i32)).collect();
                    for cz in 0..2 {
                        for cy in 0..2 {
                            for cx in 0..2 {
                                let child = rh.prolong(&parent, cx, cy, cz);
                                let coords = child_node_coords(P, cx, cy, cz);
                                for (m, co) in coords.iter().enumerate() {
                                    let exact = co[0].powi(a as i32) * co[1].powi(b as i32) * co[2].powi(c as i32);
                                    assert!((child[m] - exact).abs() < 1e-9, "prolong inexact");
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    #[test]
    fn restrict_of_constant_is_constant() {
        let rh = RefineHex::new(P);
        let n = (P + 1).pow(3);
        let children: [Vec<f64>; 8] = std::array::from_fn(|_| vec![2.5; n]);
        let parent = rh.restrict(&children);
        for v in parent {
            assert!((v - 2.5).abs() < 1e-12, "restrict of constant ≠ constant");
        }
    }

    #[test]
    fn restrict_then_prolong_roundtrip_low_modes() {
        // For degree ≤ p, prolong (parent→8 children) then restrict recovers the parent.
        let rh = RefineHex::new(P);
        let nodes = Reference1d::new(P).nodes;
        let n = P + 1;
        let mut parent = vec![0.0; n * n * n];
        for k in 0..n {
            for j in 0..n {
                for i in 0..n {
                    let (x, y, z) = (nodes[i], nodes[j], nodes[k]);
                    parent[i + j * n + k * n * n] = 1.0 + 0.5 * x - 0.3 * y * z + 0.2 * x * x;
                }
            }
        }
        let children: [Vec<f64>; 8] =
            std::array::from_fn(|c| rh.prolong(&parent, c & 1, (c >> 1) & 1, (c >> 2) & 1));
        let back = rh.restrict(&children);
        for m in 0..n * n * n {
            assert!((back[m] - parent[m]).abs() < 1e-9, "roundtrip off at {m}");
        }
    }

    #[test]
    fn smoothness_indicator_flags_underresolved() {
        let si = SmoothnessIndicator3d::new(P);
        let nodes = Reference1d::new(P).nodes;
        let n = P + 1;
        // Smooth (low-degree) field ⇒ tiny indicator.
        let mut smooth = vec![0.0; n * n * n];
        for k in 0..n {
            for j in 0..n {
                for i in 0..n {
                    smooth[i + j * n + k * n * n] = 1.0 + nodes[i] - 0.5 * nodes[j];
                }
            }
        }
        assert!(si.indicator(&smooth) < 1e-6, "smooth field flagged");
        // Pure top Legendre mode in each axis ⇒ ~all energy in top modes.
        let mut top = vec![0.0; n * n * n];
        for k in 0..n {
            for j in 0..n {
                for i in 0..n {
                    let lp = legendre_all(P, nodes[i])[P]
                        * legendre_all(P, nodes[j])[P]
                        * legendre_all(P, nodes[k])[P];
                    top[i + j * n + k * n * n] = lp;
                }
            }
        }
        assert!(si.indicator(&top) > 0.99, "top mode not flagged: {}", si.indicator(&top));
    }
}
