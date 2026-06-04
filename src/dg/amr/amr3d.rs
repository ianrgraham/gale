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

#[cfg(test)]
mod tests {
    use super::*;

    const P: usize = 3;

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
