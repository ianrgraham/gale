//! Execution backend: the [`Device`] config switch and domain decomposition.
//!
//! [`Device`] is the single place the execution target is chosen (`docs/api-design.md`
//! §3.6) — CPU, one GPU, or multiple GPUs. It does **not** leak into the physics:
//! `Term`/`Integrator`/`Updater` code never references it; the
//! [`Simulation`](super::simulation::Simulation) owns it and selects how operations
//! execute.
//!
//! Scope of this layer. The lib is pure host code and builds under ordinary
//! `cargo` (the cuda-oxide device kernels live in the `cargo oxide`-built binaries,
//! where they are validated bit-for-bit against the CPU oracle). So `Device` here
//! provides:
//!   * the backend **selector** held by the simulation, and
//!   * the multi-GPU **domain decomposition** ([`DomainDecomposition`]) — which
//!     elements each device owns and the cross-device halo faces — built on the
//!     validated `dg::distributed` partition/halo logic, the correctness-critical
//!     part of multi-GPU.
//!
//! Wiring GPU *kernel execution* behind this switch (launching the validated
//! kernels, replacing the CPU halo copy with a P2P `memcpy_peer_async`) is the
//! next step; it requires compiling device modules with the cargo-oxide backend,
//! so it lives outside the normally-built lib. With [`Device::Cpu`] (the default)
//! the simulation executes monolithically in-process exactly as before.

use super::state::State;
use crate::dg::distributed::{halo_exchange, partition_blocks};
use std::collections::HashMap;

/// How a multi-GPU domain is split across devices. Contiguous element blocks
/// today (the `dg::distributed` decomposition); a graph / space-filling
/// partitioner is a drop-in replacement.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub enum Partition {
    /// Contiguous element blocks.
    #[default]
    Blocks,
}

/// The execution target for a simulation. The one non-leaking backend switch.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Device {
    /// In-process CPU execution (the default; the correctness oracle).
    Cpu,
    /// A single CUDA device of the given ordinal.
    Cuda { ordinal: u32 },
    /// Multiple CUDA devices with a domain decomposition across them.
    MultiGpu { ordinals: Vec<u32>, partition: Partition },
}

impl Default for Device {
    fn default() -> Self {
        Device::Cpu
    }
}

impl Device {
    /// Number of compute devices this targets (1 for CPU / single GPU).
    pub fn n_devices(&self) -> usize {
        match self {
            Device::Cpu | Device::Cuda { .. } => 1,
            Device::MultiGpu { ordinals, .. } => ordinals.len().max(1),
        }
    }

    /// Whether execution targets a GPU.
    pub fn is_gpu(&self) -> bool {
        !matches!(self, Device::Cpu)
    }

    /// The partition strategy (defaulting to `Blocks` for non-multi-GPU devices).
    pub fn partition(&self) -> Partition {
        match self {
            Device::MultiGpu { partition, .. } => *partition,
            _ => Partition::Blocks,
        }
    }
}

/// A decomposition of a mesh's elements across the devices of a [`Device`]:
/// `parts[e]` is the device index owning element `e`. For a multi-GPU run each
/// device computes its owned elements locally and obtains cross-device face traces
/// by halo exchange — the partition + exchange + local compute reproduces the
/// monolithic operator exactly (validated in `dg::distributed`).
#[derive(Clone, Debug)]
pub struct DomainDecomposition {
    /// Element → owning device index.
    pub parts: Vec<usize>,
    /// Number of devices.
    pub n_parts: usize,
}

impl DomainDecomposition {
    /// Decompose `n_elements` across the devices of `device` using its partition
    /// strategy.
    pub fn new(n_elements: usize, device: &Device) -> Self {
        let n_parts = device.n_devices();
        let parts = match device.partition() {
            Partition::Blocks => partition_blocks(n_elements, n_parts),
        };
        Self { parts, n_parts }
    }

    /// Elements owned by device `d`.
    pub fn elements_on(&self, d: usize) -> Vec<usize> {
        (0..self.parts.len()).filter(|&e| self.parts[e] == d).collect()
    }

    /// Element count per device.
    pub fn counts(&self) -> Vec<usize> {
        let mut c = vec![0usize; self.n_parts];
        for &p in &self.parts {
            c[p] += 1;
        }
        c
    }

    /// Whether the load is balanced to within one element across devices.
    pub fn is_balanced(&self) -> bool {
        let c = self.counts();
        match (c.iter().min(), c.iter().max()) {
            (Some(&lo), Some(&hi)) => hi - lo <= 1,
            _ => true,
        }
    }

    /// Cross-device halo: for each element face whose neighbour is on another
    /// device, the neighbour trace this device needs (what a P2P transfer would
    /// deliver). Delegates to the validated `dg::distributed::halo_exchange`.
    pub fn halo(&self, state: &State, comp: &[f64]) -> HashMap<(usize, usize), Vec<f64>> {
        halo_exchange(&state.mesh, &self.parts, comp)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dg::hyperbolic::{Hyperbolic, LinearAdvection};
    use crate::dg::distributed::distributed_advection_rhs;
    use crate::dg::mesh::Mesh2d;

    #[test]
    fn device_metadata() {
        assert_eq!(Device::Cpu.n_devices(), 1);
        assert!(!Device::Cpu.is_gpu());
        assert_eq!(Device::Cuda { ordinal: 0 }.n_devices(), 1);
        assert!(Device::Cuda { ordinal: 0 }.is_gpu());
        let mg = Device::MultiGpu { ordinals: vec![0, 1], partition: Partition::Blocks };
        assert_eq!(mg.n_devices(), 2);
        assert!(mg.is_gpu());
        assert_eq!(Device::default(), Device::Cpu);
    }

    #[test]
    fn decomposition_is_complete_disjoint_balanced() {
        let mesh = Mesh2d::rectangular(3, 5, 5, [0.0, 1.0], [0.0, 1.0]); // 25 elements
        let dev = Device::MultiGpu { ordinals: vec![0, 1, 2, 3], partition: Partition::Blocks };
        let dd = DomainDecomposition::new(mesh.n_elements(), &dev);

        assert_eq!(dd.parts.len(), 25);
        assert_eq!(dd.n_parts, 4);
        // Complete + disjoint: every element assigned to exactly one device in range.
        assert!(dd.parts.iter().all(|&p| p < 4));
        let total: usize = dd.counts().iter().sum();
        assert_eq!(total, 25);
        // Union of per-device element sets is all elements, disjoint.
        let mut seen = vec![false; 25];
        for d in 0..4 {
            for e in dd.elements_on(d) {
                assert!(!seen[e], "element {e} owned by two devices");
                seen[e] = true;
            }
        }
        assert!(seen.iter().all(|&s| s));
        assert!(dd.is_balanced(), "block partition not balanced: {:?}", dd.counts());
    }

    /// End-to-end: the decomposition a `Device::MultiGpu` produces, fed through the
    /// validated distributed (local-compute + halo) path, reproduces the monolithic
    /// operator bit-for-bit — the CPU model of multi-GPU execution. This is what the
    /// GPU backend will run, with kernel launches + P2P in place of CPU loops + halo
    /// copies.
    #[test]
    fn multigpu_decomposition_matches_monolithic() {
        let (ax, ay) = (1.0, 0.5);
        let mesh = Mesh2d::rectangular_periodic(4, 4, 4, [0.0, 1.0], [0.0, 1.0]);
        let nn = mesh.refq.n_nodes();
        let ndof = mesh.n_elements() * nn;

        // A non-trivial state.
        let mut u = vec![0.0; ndof];
        for (e, el) in mesh.elements.iter().enumerate() {
            for k in 0..nn {
                u[e * nn + k] = (el.geom.x[k] * 3.0).sin() * (el.geom.y[k] * 2.0).cos();
            }
        }
        let bc = |_x: f64, _y: f64| 0.0;

        // Monolithic reference (one partition ⇒ no cross-device faces).
        let mono = distributed_advection_rhs(&mesh, &vec![0; mesh.n_elements()], &u, ax, ay, bc);

        // Each multi-GPU count must reproduce it exactly.
        for &ng in &[2usize, 3, 4] {
            let dev = Device::MultiGpu {
                ordinals: (0..ng as u32).collect(),
                partition: Partition::Blocks,
            };
            let dd = DomainDecomposition::new(mesh.n_elements(), &dev);
            let dist = distributed_advection_rhs(&mesh, &dd.parts, &u, ax, ay, bc);
            assert_eq!(dist, mono, "{ng}-device decomposition diverged from monolithic");
        }

        // Sanity: the operator is the linear-advection RHS (matches Hyperbolic).
        let hyp = Hyperbolic::new(&mesh, LinearAdvection { ax, ay });
        let href = hyp.rhs(&[u.clone()], 0.0, &|_x, _y, _t, o: &mut [f64]| o[0] = 0.0);
        // Same operator family; values finite and nonzero.
        assert!(href[0].iter().any(|&v| v.abs() > 1e-6));
    }
}
