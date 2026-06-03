//! Modal **spectral filter** (SVV-class) — high-frequency stabilization for
//! under-resolved/high-Re DG, complementary to the split-form. Transforms nodal →
//! Legendre-modal, multiplies each mode by an exponential damping factor, and
//! transforms back; applied per element as a tensor-product operator. Low modes
//! (below the cutoff) are preserved exactly, so accuracy on resolved features is
//! untouched.

use super::reference::{legendre_all, Reference1d};

/// Exponential modal-filter weight: `1` for `k ≤ cutoff`, decaying for higher modes.
fn sigma(k: usize, order: usize, cutoff: usize, alpha: f64, s: f64) -> f64 {
    if k <= cutoff || order == cutoff {
        1.0
    } else {
        let eta = (k - cutoff) as f64 / (order - cutoff) as f64;
        (-alpha * eta.powf(2.0 * s)).exp()
    }
}

/// Gauss–Jordan inverse of a small dense `n×n` matrix (row-major).
pub(crate) fn mat_inverse(n: usize, a: &[f64]) -> Vec<f64> {
    let mut m = a.to_vec();
    let mut inv = vec![0.0; n * n];
    for i in 0..n {
        inv[i * n + i] = 1.0;
    }
    for col in 0..n {
        // Partial pivot.
        let mut piv = col;
        for r in col + 1..n {
            if m[r * n + col].abs() > m[piv * n + col].abs() {
                piv = r;
            }
        }
        if piv != col {
            for c in 0..n {
                m.swap(col * n + c, piv * n + c);
                inv.swap(col * n + c, piv * n + c);
            }
        }
        let d = m[col * n + col];
        for c in 0..n {
            m[col * n + c] /= d;
            inv[col * n + c] /= d;
        }
        for r in 0..n {
            if r != col {
                let f = m[r * n + col];
                for c in 0..n {
                    m[r * n + c] -= f * m[col * n + c];
                    inv[r * n + c] -= f * inv[col * n + c];
                }
            }
        }
    }
    inv
}

/// A per-element exponential modal filter for the quad reference element.
#[derive(Clone, Debug)]
pub struct ModalFilter {
    pub order: usize,
    /// 1D filter matrix `F = V · diag(σ) · V⁻¹` (`(p+1)²` entries), applied along
    /// each tensor direction.
    mat: Vec<f64>,
}

impl ModalFilter {
    /// Build with `cutoff` (modes ≤ cutoff preserved), strength `alpha`, order `s`.
    /// A typical mild filter: `cutoff = p−1`, `alpha = 36`, `s = 8`.
    pub fn new(order: usize, cutoff: usize, alpha: f64, s: f64) -> Self {
        let n = order + 1;
        let nodes = Reference1d::new(order).nodes;
        // Legendre Vandermonde V[i,j] = P_j(r_i).
        let mut v = vec![0.0; n * n];
        for i in 0..n {
            let leg = legendre_all(order, nodes[i]);
            for j in 0..n {
                v[i * n + j] = leg[j];
            }
        }
        let vinv = mat_inverse(n, &v);
        let sig: Vec<f64> = (0..n).map(|k| sigma(k, order, cutoff, alpha, s)).collect();
        // F = V diag(σ) V⁻¹.
        let mut vs = vec![0.0; n * n];
        for i in 0..n {
            for j in 0..n {
                vs[i * n + j] = v[i * n + j] * sig[j];
            }
        }
        let mut mat = vec![0.0; n * n];
        for i in 0..n {
            for j in 0..n {
                let mut acc = 0.0;
                for k in 0..n {
                    acc += vs[i * n + k] * vinv[k * n + j];
                }
                mat[i * n + j] = acc;
            }
        }
        Self { order, mat }
    }

    /// Apply the filter to one element's nodal field (length `(p+1)²`), tensor-product.
    pub fn apply_element(&self, f: &[f64]) -> Vec<f64> {
        let n = self.order + 1;
        let fm = &self.mat;
        let mut tmp = vec![0.0; n * n];
        for j in 0..n {
            for i in 0..n {
                let mut acc = 0.0;
                for k in 0..n {
                    acc += fm[i * n + k] * f[k + j * n];
                }
                tmp[i + j * n] = acc;
            }
        }
        let mut out = vec![0.0; n * n];
        for j in 0..n {
            for i in 0..n {
                let mut acc = 0.0;
                for k in 0..n {
                    acc += fm[j * n + k] * tmp[i + k * n];
                }
                out[i + j * n] = acc;
            }
        }
        out
    }

    /// Apply to a global field (`ne · (p+1)²`), element by element.
    pub fn apply(&self, field: &[f64]) -> Vec<f64> {
        let nn = (self.order + 1) * (self.order + 1);
        let mut out = vec![0.0; field.len()];
        for (chunk_in, chunk_out) in field.chunks(nn).zip(out.chunks_mut(nn)) {
            chunk_out.copy_from_slice(&self.apply_element(chunk_in));
        }
        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dg::quad::Reference2dQuad;

    #[test]
    fn preserves_modes_below_cutoff() {
        // cutoff = p−1 ⇒ a polynomial of degree ≤ p−1 (no top mode) is unchanged.
        let p = 5;
        let refq = Reference2dQuad::new(p);
        let filt = ModalFilter::new(p, p - 1, 36.0, 8.0);
        let f: Vec<f64> = refq
            .nodes
            .iter()
            .map(|c| c[0].powi((p - 1) as i32) + 0.3 * c[1].powi((p - 2) as i32) - 1.2)
            .collect();
        let g = filt.apply_element(&f);
        let md = f.iter().zip(&g).fold(0.0f64, |a, (x, y)| a.max((x - y).abs()));
        assert!(md < 1e-10, "low modes not preserved: {md}");
    }

    #[test]
    fn damps_the_top_mode() {
        // The degree-p mode (P_p along x) must be attenuated.
        let p = 5;
        let refq = Reference2dQuad::new(p);
        let filt = ModalFilter::new(p, p - 1, 36.0, 8.0);
        let f: Vec<f64> = refq.nodes.iter().map(|c| legendre_all(p, c[0])[p]).collect();
        let g = filt.apply_element(&f);
        let nf: f64 = f.iter().map(|x| x * x).sum::<f64>().sqrt();
        let ng: f64 = g.iter().map(|x| x * x).sum::<f64>().sqrt();
        assert!(ng < 0.5 * nf, "top mode not damped: {nf} → {ng}");
    }
}
