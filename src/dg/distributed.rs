//! Multi-GPU foundation: domain decomposition + halo exchange. Each rank (GPU) owns
//! a subset of elements and only its own state; the trace data needed across a
//! partition boundary is obtained by an explicit **halo exchange** (the CPU model of
//! a GPU peer-to-peer / NCCL transfer). Because a DG operator is element-local +
//! face-coupled, partition + exchange + local compute reproduces the monolithic
//! result exactly — validated here against `Hyperbolic`. Swapping the halo copy for a
//! `cuda` P2P `memcpy_dtod` is the GPU step; the decomposition/exchange logic is the
//! correctness-critical part and is proven on CPU.

use super::face::Edge;
use super::mesh::{Mesh2d, Neighbor};
use std::collections::HashMap;

/// Assign elements to `nparts` ranks in contiguous blocks (a simple decomposition;
/// a graph/space-filling partitioner is a drop-in replacement).
pub fn partition_blocks(ne: usize, nparts: usize) -> Vec<usize> {
    (0..ne).map(|e| (e * nparts / ne).min(nparts - 1)).collect()
}

/// Per-rank halo: for every interior face whose neighbour is on another rank, the
/// neighbour's trace at *this* element's face-node order (what a P2P transfer would
/// deliver). Keyed by `(element, edge as usize)`.
pub fn halo_exchange(mesh: &Mesh2d, parts: &[usize], state: &[f64]) -> HashMap<(usize, usize), Vec<f64>> {
    let nn = mesh.refq.n_nodes();
    let mut ghost = HashMap::new();
    for (e, el) in mesh.elements.iter().enumerate() {
        for edge in Edge::ALL {
            if let Neighbor::Interior { elem: re, edge: redge, perm } = &el.neighbors[edge as usize] {
                if parts[*re] != parts[e] {
                    let rf = &mesh.elements[*re].faces[*redge as usize];
                    let tr: Vec<f64> = perm.iter().map(|&b| state[*re * nn + rf.nodes[b]]).collect();
                    ghost.insert((e, edge as usize), tr);
                }
            }
        }
    }
    ghost
}

/// Distributed linear-advection RHS: computed element-by-element using each element's
/// own state plus, at cross-rank faces, the halo (never a direct remote read). Equals
/// the monolithic operator. `bc(x,y)` is the exterior boundary state.
pub fn distributed_advection_rhs(
    mesh: &Mesh2d,
    parts: &[usize],
    state: &[f64],
    ax: f64,
    ay: f64,
    bc: impl Fn(f64, f64) -> f64,
) -> Vec<f64> {
    let refq = &mesh.refq;
    let nn = refq.n_nodes();
    let ghost = halo_exchange(mesh, parts, state);
    let flux_n = |ul: f64, ur: f64, nx: f64, ny: f64| {
        let an = ax * nx + ay * ny;
        0.5 * an * (ul + ur) - 0.5 * an.abs() * (ur - ul)
    };
    let mut out = vec![0.0; mesh.n_elements() * nn];
    for (e, el) in mesh.elements.iter().enumerate() {
        let g = &el.geom;
        let sl = e * nn..(e + 1) * nn;
        let wfx: Vec<f64> = (0..nn).map(|k| g.jw[k] * ax * state[e * nn + k]).collect();
        let wfy: Vec<f64> = (0..nn).map(|k| g.jw[k] * ay * state[e * nn + k]).collect();
        let a = g.gradx_t(refq, &wfx);
        let b = g.grady_t(refq, &wfy);
        let mut res: Vec<f64> = (0..nn).map(|k| a[k] + b[k]).collect();
        for edge in Edge::ALL {
            let face = &el.faces[edge as usize];
            match &el.neighbors[edge as usize] {
                Neighbor::Boundary { .. } => {
                    for ai in 0..face.nodes.len() {
                        let vl = face.nodes[ai];
                        let ur = bc(g.x[vl], g.y[vl]);
                        res[vl] -= face.sw[ai] * flux_n(state[e * nn + vl], ur, face.nx[ai], face.ny[ai]);
                    }
                }
                Neighbor::Interior { elem: re, edge: redge, perm } => {
                    let cross = parts[*re] != parts[e];
                    let halo = ghost.get(&(e, edge as usize));
                    let rf = &mesh.elements[*re].faces[*redge as usize];
                    for ai in 0..face.nodes.len() {
                        let vl = face.nodes[ai];
                        let up = if cross {
                            halo.unwrap()[ai] // from the exchanged halo, not a remote read
                        } else {
                            state[*re * nn + rf.nodes[perm[ai]]]
                        };
                        res[vl] -= face.sw[ai] * flux_n(state[e * nn + vl], up, face.nx[ai], face.ny[ai]);
                    }
                }
                _ => panic!("distributed advection: non-conforming faces not supported"),
            }
        }
        for (k, r) in res.iter().enumerate() {
            out[sl.start + k] = r / g.jw[k];
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dg::{Hyperbolic, LinearAdvection};

    #[test]
    fn partition_blocks_balanced() {
        let p = partition_blocks(10, 2);
        assert_eq!(p.iter().filter(|&&r| r == 0).count(), 5);
        assert_eq!(p.iter().filter(|&&r| r == 1).count(), 5);
        assert!(p.iter().all(|&r| r < 2));
    }

    #[test]
    fn distributed_matches_monolithic() {
        // The decisive multi-GPU correctness property: domain decomposition + halo
        // exchange reproduces the single-domain operator bit-for-bit, for several
        // partition counts (so cross-rank faces are genuinely exercised).
        let p = 4;
        let (ax, ay) = (0.8, -0.5);
        let mesh = Mesh2d::rectangular_periodic(p, 4, 4, [0.0, 1.0], [0.0, 1.0]);
        let nn = mesh.refq.n_nodes();
        let ndof = mesh.n_elements() * nn;
        let mut state = vec![0.0; ndof];
        for (e, el) in mesh.elements.iter().enumerate() {
            for k in 0..nn {
                state[e * nn + k] = (2.0 * el.geom.x[k]).sin() * (3.0 * el.geom.y[k]).cos() + 0.2 * el.geom.x[k];
            }
        }
        // Monolithic reference (periodic ⇒ bc never used).
        let op = Hyperbolic::new(&mesh, LinearAdvection { ax, ay });
        let mono = op.rhs(&[state.clone()], 0.0, &|_, _, _, _: &mut [f64]| {});

        for nparts in [2usize, 3, 4, 8] {
            let parts = partition_blocks(mesh.n_elements(), nparts);
            let dist = distributed_advection_rhs(&mesh, &parts, &state, ax, ay, |_, _| 0.0);
            let md = dist.iter().zip(&mono[0]).fold(0.0f64, |m, (a, b)| m.max((a - b).abs()));
            assert!(md < 1e-13, "distributed != monolithic for {nparts} ranks: {md}");
        }
    }

    #[test]
    fn halo_only_covers_cross_rank_faces() {
        // The halo contains exactly the cross-rank interior faces — nothing more.
        let p = 3;
        let mesh = Mesh2d::rectangular(p, 4, 1, [0.0, 4.0], [0.0, 1.0]); // 4 cells in a row
        let nn = mesh.refq.n_nodes();
        let state = vec![1.0; mesh.n_elements() * nn];
        let parts = partition_blocks(mesh.n_elements(), 2); // [0,0,1,1]
        let ghost = halo_exchange(&mesh, &parts, &state);
        // Only the 0|1 interface (element 1 East ↔ element 2 West) is cross-rank:
        // two ghost entries (one from each side).
        assert_eq!(ghost.len(), 2, "expected exactly the one shared interface (2 sides)");
    }
}
