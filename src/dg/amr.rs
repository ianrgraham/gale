//! Adaptive mesh refinement (AMR) — foundational transfer operators for `h`-adaptive
//! DG-SEM. A quad element refines into 4 children (quadrants). Two operators move the
//! solution between levels:
//! - **prolong**: parent → child — evaluate the parent polynomial at the child's
//!   nodes. Exact for polynomials of degree ≤ p (the child sub-element exactly
//!   represents the parent field), so refinement loses nothing.
//! - **restrict**: 4 children → parent — the conservative, mass-weighted L2 adjoint,
//!   so the cell-integral (mass/momentum) is preserved on coarsening.
//!
//! Children are indexed `c = cx + 2·cy`, `cx,cy ∈ {0,1}` (0 = lower/left half of the
//! reference element in that axis, 1 = upper/right). The non-conforming *interface*
//! coupling (mortar / hanging nodes) that makes the adaptive solver run is the next
//! piece; these transfer operators are its prerequisite.

use super::filter::mat_inverse;
use super::reference::{legendre_all, Reference1d};

/// 1D Lagrange basis `ℓ_j(x)` at the reference nodes, evaluated at `x`.
fn lagrange_basis(nodes: &[f64], x: f64) -> Vec<f64> {
    let n = nodes.len();
    (0..n)
        .map(|i| {
            let mut li = 1.0;
            for j in 0..n {
                if j != i {
                    li *= (x - nodes[j]) / (nodes[i] - nodes[j]);
                }
            }
            li
        })
        .collect()
}

/// `h`-refinement transfer operators for an order-`p` quad element.
pub struct RefineQuad {
    pub order: usize,
    /// `p_left[i*n+j] = ℓ_j((node_i−1)/2)` — parent basis at the left-child's node `i`.
    p_left: Vec<f64>,
    /// `p_right[i*n+j] = ℓ_j((node_i+1)/2)` — parent basis at the right-child's node.
    p_right: Vec<f64>,
    /// LGL weights (for the conservative restriction).
    w: Vec<f64>,
}

impl RefineQuad {
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

    /// Prolong a parent nodal field to child `(cx, cy)`. Exact for degree ≤ p.
    pub fn prolong(&self, parent: &[f64], cx: usize, cy: usize) -> Vec<f64> {
        let n = self.order + 1;
        let pr = self.axis(cx);
        let ps = self.axis(cy);
        // child[i,j] = Σ_a Σ_b pr[i,a] ps[j,b] parent[a,b] (two 1D passes).
        let mut tmp = vec![0.0; n * n];
        for b in 0..n {
            for i in 0..n {
                let mut s = 0.0;
                for a in 0..n {
                    s += pr[i * n + a] * parent[a + b * n];
                }
                tmp[i + b * n] = s;
            }
        }
        let mut child = vec![0.0; n * n];
        for j in 0..n {
            for i in 0..n {
                let mut s = 0.0;
                for b in 0..n {
                    s += ps[j * n + b] * tmp[i + b * n];
                }
                child[i + j * n] = s;
            }
        }
        child
    }

    /// **Mortar** projection of a non-conforming face: coarse edge trace → fine
    /// half-edge trace (`half ∈ {0,1}`). Exact for degree ≤ p — the 1D analog of
    /// [`prolong`](Self::prolong), used to give a coarse element's flux to its two
    /// refined neighbors.
    pub fn mortar_to_fine(&self, coarse: &[f64], half: usize) -> Vec<f64> {
        let n = self.order + 1;
        let p = self.axis(half);
        (0..n)
            .map(|i| (0..n).map(|j| p[i * n + j] * coarse[j]).sum())
            .collect()
    }

    /// **Mortar** projection: two fine half-edge traces → the coarse edge trace, the
    /// conservative L2 adjoint (½ Jacobian). Preserves the edge integral.
    pub fn mortar_to_coarse(&self, fine: &[Vec<f64>; 2]) -> Vec<f64> {
        let n = self.order + 1;
        let mut c = vec![0.0; n];
        for half in 0..2 {
            let p = self.axis(half);
            let uf = &fine[half];
            for i in 0..n {
                let mut s = 0.0;
                for a in 0..n {
                    s += p[a * n + i] * self.w[a] * uf[a];
                }
                c[i] += 0.5 * s / self.w[i];
            }
        }
        c
    }

    /// Restrict the 4 children (ordered `c = cx + 2cy`) to the parent via the
    /// conservative, mass-weighted L2 projection:
    /// `u_p[i,j] = (¼ / (w_i w_j)) Σ_c Σ_{a,b} P_cx[a,i] P_cy[b,j] w_a w_b u_c[a,b]`.
    /// The ¼ is the 2D child Jacobian; restriction of a constant is exactly that
    /// constant and the cell integral is preserved.
    pub fn restrict(&self, children: &[Vec<f64>; 4]) -> Vec<f64> {
        let n = self.order + 1;
        let mut parent = vec![0.0; n * n];
        for cy in 0..2 {
            for cx in 0..2 {
                let uc = &children[cx + 2 * cy];
                let pr = self.axis(cx);
                let ps = self.axis(cy);
                for j in 0..n {
                    for i in 0..n {
                        let mut s = 0.0;
                        for b in 0..n {
                            let wb = self.w[b];
                            for a in 0..n {
                                s += pr[a * n + i] * ps[b * n + j] * self.w[a] * wb * uc[a + b * n];
                            }
                        }
                        parent[i + j * n] += 0.25 * s / (self.w[i] * self.w[j]);
                    }
                }
            }
        }
        parent
    }
}

/// Spectral **smoothness indicator** (Persson–Peraire) for "where to refine": the
/// fraction of an element's L2 energy carried by the highest Legendre modes. A
/// well-resolved (spectrally-decaying) field gives ≈0; an under-resolved or sharp
/// field gives O(1) — flag those elements for `h`-refinement (e.g. around the
/// high-stress strands at particle near-contacts).
pub struct SmoothnessIndicator {
    order: usize,
    /// Nodal→modal transform `V⁻¹` (`(p+1)²` row-major), `V[i,a] = P_a(node_i)`.
    vinv: Vec<f64>,
    /// Legendre L2 norms `γ_k = 2/(2k+1)`.
    gamma: Vec<f64>,
}

impl SmoothnessIndicator {
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

    /// Fraction of L2 energy in the highest modes (`a==p` or `b==p`), in `[0,1]`.
    pub fn indicator(&self, field: &[f64]) -> f64 {
        let n = self.order + 1;
        // Nodal → modal: ĉ[a,b] = Σ_ij V⁻¹[a,i] V⁻¹[b,j] u[i,j].
        let mut tmp = vec![0.0; n * n]; // tmp[a,j] = Σ_i V⁻¹[a,i] u[i,j]
        for j in 0..n {
            for a in 0..n {
                let mut s = 0.0;
                for i in 0..n {
                    s += self.vinv[a * n + i] * field[i + j * n];
                }
                tmp[a + j * n] = s;
            }
        }
        let (mut e_total, mut e_top) = (0.0, 0.0);
        for b in 0..n {
            for a in 0..n {
                let mut c = 0.0;
                for j in 0..n {
                    c += self.vinv[b * n + j] * tmp[a + j * n];
                }
                let e = c * c * self.gamma[a] * self.gamma[b]; // Parseval energy
                e_total += e;
                if a == self.order || b == self.order {
                    e_top += e;
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
    use crate::dg::quad::Reference2dQuad;

    #[test]
    fn prolong_is_exact_on_polynomials() {
        // Refining loses nothing: prolong reproduces a degree-≤p polynomial exactly at
        // the child nodes (whose parent-reference coords are the half-mapped LGL nodes).
        let p = 4;
        let refq = Reference2dQuad::new(p);
        let nodes = &refq.line.nodes;
        let n = p + 1;
        let rq = RefineQuad::new(p);
        let f = |r: f64, s: f64| 1.0 - 2.0 * r + 0.5 * s + r * r - 0.7 * r * s * s + 0.3 * s.powi(4);
        let parent: Vec<f64> = (0..n * n).map(|k| f(nodes[k % n], nodes[k / n])).collect();
        for cy in 0..2 {
            for cx in 0..2 {
                let child = rq.prolong(&parent, cx, cy);
                for j in 0..n {
                    for i in 0..n {
                        // child node (i,j) → parent reference coordinate.
                        let rp = 0.5 * (nodes[i] + (2.0 * cx as f64 - 1.0));
                        let sp = 0.5 * (nodes[j] + (2.0 * cy as f64 - 1.0));
                        let got = child[i + j * n];
                        assert!((got - f(rp, sp)).abs() < 1e-10, "prolong c=({cx},{cy}) ({i},{j})");
                    }
                }
            }
        }
    }

    #[test]
    fn smoothness_indicator_flags_underresolved_fields() {
        // Resolved (low-degree) fields → ~0; fields with strong top-mode content → O(1).
        let p = 4;
        let refq = Reference2dQuad::new(p);
        let nodes = &refq.line.nodes;
        let n = p + 1;
        let si = SmoothnessIndicator::new(p);

        // (a) Low-order polynomial (degree ≤ p−1): no top mode ⇒ ≈ 0.
        let smooth: Vec<f64> =
            (0..n * n).map(|k| 1.0 - 0.5 * nodes[k % n] + 0.3 * nodes[k / n].powi(2)).collect();
        let s_smooth = si.indicator(&smooth);
        assert!(s_smooth < 1e-12, "smooth field flagged: {s_smooth}");

        // (b) The top Legendre mode P_p(x): all energy in the highest mode ⇒ ≈ 1.
        let top: Vec<f64> = (0..n * n).map(|k| legendre_all(p, nodes[k % n])[p]).collect();
        let s_top = si.indicator(&top);
        assert!(s_top > 0.99, "pure top mode not flagged: {s_top}");

        // (c) An under-resolved oscillation has appreciable top-mode energy.
        let osc: Vec<f64> =
            (0..n * n).map(|k| (6.0 * nodes[k % n]).sin() * (5.0 * nodes[k / n]).cos()).collect();
        let s_osc = si.indicator(&osc);
        eprintln!("indicator: smooth={s_smooth:.2e}, top={s_top:.3}, osc={s_osc:.3}");
        assert!(s_osc > 0.05, "under-resolved oscillation not flagged: {s_osc}");
    }

    #[test]
    fn mortar_face_projection_exact_and_conservative() {
        // 1D non-conforming face transfer: coarse trace → fine halves is exact for
        // degree ≤ p, fine halves → coarse conserves the edge integral and inverts
        // an exactly-represented trace.
        let p = 4;
        let nodes = Reference1d::new(p).nodes;
        let w = Reference1d::new(p).weights;
        let n = p + 1;
        let rq = RefineQuad::new(p);
        let f = |x: f64| 0.5 - 0.3 * x + 0.4 * x * x - 0.1 * x.powi(3) + 0.05 * x.powi(4);
        let coarse: Vec<f64> = nodes.iter().map(|&x| f(x)).collect();
        // to-fine exactness.
        for half in 0..2 {
            let fine = rq.mortar_to_fine(&coarse, half);
            for i in 0..n {
                let xp = 0.5 * (nodes[i] + (2.0 * half as f64 - 1.0));
                assert!((fine[i] - f(xp)).abs() < 1e-10, "mortar→fine half={half} i={i}");
            }
        }
        // to-coarse conservation and round-trip.
        let fine: [Vec<f64>; 2] = std::array::from_fn(|h| rq.mortar_to_fine(&coarse, h));
        let back = rq.mortar_to_coarse(&fine);
        let int_c: f64 = (0..n).map(|i| w[i] * coarse[i]).sum();
        let int_f: f64 = fine.iter().map(|u| 0.5 * (0..n).map(|i| w[i] * u[i]).sum::<f64>()).sum();
        let int_b: f64 = (0..n).map(|i| w[i] * back[i]).sum();
        assert!((int_c - int_f).abs() < 1e-12 && (int_c - int_b).abs() < 1e-12, "edge integral not conserved");
        let err = back.iter().zip(&coarse).fold(0.0f64, |a, (x, y)| a.max((x - y).abs()));
        assert!(err < 1e-2, "mortar round-trip too lossy: {err}");
    }

    #[test]
    fn nonconforming_mortar_flux_is_conservative_and_consistent() {
        // The heart of AMR-DG: a numerical flux across a coarse edge facing TWO fine
        // half-edges. Compute the upwind advection flux on each fine half (at the fine
        // resolution, using the coarse trace projected onto it), then project the half
        // fluxes back to the coarse edge. Conservation: the coarse-edge flux integral
        // equals the sum of the two fine half-edge flux integrals. Consistency: a
        // uniform field gives a uniform flux with no spurious interface jump.
        let p = 4;
        let nodes = Reference1d::new(p).nodes;
        let w = Reference1d::new(p).weights;
        let n = p + 1;
        let rq = RefineQuad::new(p);
        let an = 0.8; // advection speed along the (coarse-outward) normal

        let upwind = |ul: f64, ur: f64| 0.5 * an * (ul + ur) - 0.5 * an.abs() * (ur - ul);

        // Some traces: coarse (full edge) and the two fine half-edges.
        let coarse: Vec<f64> = nodes.iter().map(|&y| 1.0 + (1.3 * y).sin()).collect();
        let fine: [Vec<f64>; 2] =
            std::array::from_fn(|h| nodes.iter().map(|&y| {
                let yp = 0.5 * (y + (2.0 * h as f64 - 1.0));
                0.9 + 0.5 * yp - 0.2 * yp * yp
            }).collect());

        // Flux on each fine half = upwind(coarse projected onto half, fine trace).
        let half_flux: [Vec<f64>; 2] = std::array::from_fn(|h| {
            let cproj = rq.mortar_to_fine(&coarse, h);
            (0..n).map(|i| upwind(cproj[i], fine[h][i])).collect()
        });
        // Coarse edge receives the back-projected mortar flux.
        let coarse_flux = rq.mortar_to_coarse(&half_flux);

        // Conservation: ∫_coarse-edge = Σ_halves ½ ∫_half.
        let int_coarse: f64 = (0..n).map(|i| w[i] * coarse_flux[i]).sum();
        let int_fine: f64 = half_flux
            .iter()
            .map(|f| 0.5 * (0..n).map(|i| w[i] * f[i]).sum::<f64>())
            .sum();
        assert!((int_coarse - int_fine).abs() < 1e-12, "mortar flux not conservative: {int_coarse} vs {int_fine}");

        // Consistency: a uniform field C ⇒ flux = an·C everywhere, no interface jump.
        let cc = 1.7;
        let const_coarse = vec![cc; n];
        let const_half: [Vec<f64>; 2] = std::array::from_fn(|h| {
            let cp = rq.mortar_to_fine(&const_coarse, h);
            (0..n).map(|i| upwind(cp[i], cc)).collect()
        });
        let cf = rq.mortar_to_coarse(&const_half);
        let err = cf.iter().fold(0.0f64, |a, &v| a.max((v - an * cc).abs()));
        assert!(err < 1e-12, "mortar flux not consistent on a uniform field: {err}");
    }

    #[test]
    fn restrict_of_constant_is_constant() {
        let p = 4;
        let rq = RefineQuad::new(p);
        let n = (p + 1) * (p + 1);
        let children = [vec![2.5; n], vec![2.5; n], vec![2.5; n], vec![2.5; n]];
        let parent = rq.restrict(&children);
        let err = parent.iter().fold(0.0f64, |a, &v| a.max((v - 2.5).abs()));
        assert!(err < 1e-12, "restrict(const) not constant: {err}");
    }

    #[test]
    fn restrict_conserves_cell_integral() {
        // Σ_ij w_i w_j u_p = ¼ Σ_c Σ_ab w_a w_b u_c  (mass/momentum conserved on coarsen).
        let p = 4;
        let refq = Reference2dQuad::new(p);
        let nodes = &refq.line.nodes;
        let n = p + 1;
        let w = &Reference1d::new(p).weights;
        let rq = RefineQuad::new(p);
        // Distinct smooth data in each child.
        let mk = |o: f64| -> Vec<f64> {
            (0..n * n).map(|k| o + (nodes[k % n]).sin() + 0.3 * nodes[k / n]).collect()
        };
        let children = [mk(0.0), mk(1.0), mk(-0.5), mk(0.2)];
        let parent = rq.restrict(&children);
        let int_p: f64 = (0..n * n).map(|k| w[k % n] * w[k / n] * parent[k]).sum();
        let int_c: f64 = children
            .iter()
            .map(|uc| 0.25 * (0..n * n).map(|k| w[k % n] * w[k / n] * uc[k]).sum::<f64>())
            .sum();
        assert!((int_p - int_c).abs() < 1e-12, "not conservative: {int_p} vs {int_c}");
    }

    #[test]
    fn restrict_then_prolong_round_trip_for_low_modes() {
        // A field already representable on the parent survives prolong→restrict to
        // mass-lumping accuracy (LGL integrates the degree-2p mass only to 2p−1).
        let p = 4;
        let refq = Reference2dQuad::new(p);
        let nodes = &refq.line.nodes;
        let n = p + 1;
        let rq = RefineQuad::new(p);
        let f = |r: f64, s: f64| 0.4 + 0.6 * r - 0.2 * s + 0.1 * r * s;
        let parent: Vec<f64> = (0..n * n).map(|k| f(nodes[k % n], nodes[k / n])).collect();
        let children: [Vec<f64>; 4] =
            std::array::from_fn(|c| rq.prolong(&parent, c % 2, c / 2));
        let back = rq.restrict(&children);
        let err = back.iter().zip(&parent).fold(0.0f64, |a, (x, y)| a.max((x - y).abs()));
        eprintln!("restrict∘prolong max err = {err:.3e}");
        assert!(err < 1e-2, "round-trip too lossy: {err}");
    }
}
