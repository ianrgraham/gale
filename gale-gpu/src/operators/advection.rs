//! GPU linear-advection (weak-form DG) operator — reusable library component.
//!
//! Carries the `#[cuda_module]` device kernel plus a host launch wrapper
//! ([`advection_rhs`]) and a framework-level semidiscretization ([`GpuAdvection`])
//! that implements [`gale::sim::StateSemi`], so it can drive a `gale::sim::Simulation`
//! entirely on the GPU. Validated bit-for-bit against `gale::dg::Hyperbolic`.

use cuda_core::{CudaContext, DeviceBuffer, LaunchConfig};
use cuda_device::{kernel, thread, DisjointSlice, SharedArray};
use cuda_host::cuda_module;
use gale::dg::{Edge, Mesh2d, Neighbor};

const NN_MAX: usize = 81; // (p+1)² up to p=8

#[cuda_module]
mod kernels {
    use super::*;

    /// Weak-form linear-advection RHS, Rusanov flux. `u` is the global state;
    /// `face_nbr` indexes into it. `e = blockIdx.x`, node `m = threadIdx.x`.
    ///
    /// Named `advect2d_rhs` (not `advect_rhs`) because kernel export names share a
    /// single crate-wide device bundle in cuda-oxide, so they must be unique across
    /// gale-gpu (cf. the 3D / multi-GPU advection kernels).
    #[kernel]
    #[allow(clippy::too_many_arguments)]
    pub fn advect2d_rhs(
        d: &[f64], u: &[f64], rx: &[f64], ry: &[f64], sx: &[f64], sy: &[f64], jw: &[f64],
        ax: f64, ay: f64, n1: u32,
        face_vl: &[u32], face_nx: &[f64], face_ny: &[f64], face_sw: &[f64], face_nbr: &[u32],
        mut out: DisjointSlice<f64>,
    ) {
        static mut DS: SharedArray<f64, NN_MAX> = SharedArray::UNINIT;
        static mut PR: SharedArray<f64, NN_MAX> = SharedArray::UNINIT;
        static mut PS: SharedArray<f64, NN_MAX> = SharedArray::UNINIT;
        static mut RFACE: SharedArray<f64, NN_MAX> = SharedArray::UNINIT;
        let n1 = n1 as usize;
        let nn = n1 * n1;
        let e = thread::blockIdx_x() as usize;
        let m = thread::threadIdx_x() as usize;
        let b = e * nn + m;
        unsafe {
            DS[m] = d[m];
            RFACE[m] = 0.0;
            let um = u[b];
            let wfx = jw[b] * ax * um;
            let wfy = jw[b] * ay * um;
            PR[m] = rx[b] * wfx + ry[b] * wfy;
            PS[m] = sx[b] * wfx + sy[b] * wfy;
        }
        thread::sync_threads();
        if m == 0 {
            let mut t = 0usize;
            while t < 4 {
                let mut a = 0usize;
                while a < n1 {
                    let idx = (e * 4 + t) * n1 + a;
                    let vl = face_vl[idx] as usize;
                    let nx = face_nx[idx];
                    let ny = face_ny[idx];
                    let sw = face_sw[idx];
                    let um = u[e * nn + vl];
                    let up = u[face_nbr[idx] as usize];
                    let an = ax * nx + ay * ny;
                    let fstar = 0.5 * an * (um + up) - 0.5 * an.abs() * (up - um);
                    unsafe {
                        RFACE[vl] += sw * fstar;
                    }
                    a += 1;
                }
                t += 1;
            }
        }
        thread::sync_threads();
        let i = m % n1;
        let j = m / n1;
        let mut vol = 0.0f64;
        let mut k = 0usize;
        while k < n1 {
            unsafe {
                vol += DS[k * n1 + i] * PR[k + j * n1] + DS[k * n1 + j] * PS[i + k * n1];
            }
            k += 1;
        }
        let rf = unsafe { RFACE[m] };
        if let Some(o) = out.get_mut(thread::index_1d()) {
            *o = (vol - rf) / jw[b];
        }
    }
}

/// Compute the linear-advection weak-form RHS `∂ₜu = M⁻¹L(u)` on the GPU for a
/// **periodic** quad mesh, returning the nodal time-derivative. Reusable host
/// wrapper around the [`kernels::advect2d_rhs`] device kernel: flattens the mesh
/// metrics + face connectivity, uploads, launches one block per element, and
/// gathers the result. Bit-for-bit equal to `gale::dg::Hyperbolic` advection.
///
/// Panics if the mesh has boundary faces (this PoC kernel handles interior faces
/// only — use a periodic mesh).
pub fn advection_rhs(
    mesh: &Mesh2d,
    u: &[f64],
    ax: f64,
    ay: f64,
) -> Result<Vec<f64>, Box<dyn std::error::Error>> {
    let mut backend = AdvectionBackend::new(mesh)?;
    backend.rhs(u, ax, ay)
}

/// Flattened, GPU-ready mesh metrics + interior-face connectivity for the advection kernel.
/// Built once from the mesh and shared by the one-shot [`advection_rhs`] and the persistent
/// [`AdvectionBackend`], so both feed the kernel byte-identical inputs.
struct AdvectionGeom {
    ne: usize,
    nn: usize,
    n1: usize,
    ndof: usize,
    rx: Vec<f64>,
    ry: Vec<f64>,
    sx: Vec<f64>,
    sy: Vec<f64>,
    jw: Vec<f64>,
    fvl: Vec<u32>,
    fnx: Vec<f64>,
    fny: Vec<f64>,
    fsw: Vec<f64>,
    fnbr: Vec<u32>,
}

/// Flatten a periodic quad mesh's per-node metrics and interior-face connectivity into the dense
/// arrays the advection kernel reads. Panics on a boundary face (PoC kernel handles interior only).
fn flatten_advection_mesh(mesh: &Mesh2d) -> AdvectionGeom {
    let nn = mesh.refq.n_nodes();
    let ne = mesh.n_elements();
    let ndof = ne * nn;
    let n1 = mesh.order + 1;
    assert!(nn <= NN_MAX, "p too large for NN_MAX={NN_MAX}");

    // Flatten per-node metrics.
    let (mut rx, mut ry, mut sx, mut sy, mut jw) =
        (vec![0.0; ndof], vec![0.0; ndof], vec![0.0; ndof], vec![0.0; ndof], vec![0.0; ndof]);
    for (e, el) in mesh.elements.iter().enumerate() {
        for k in 0..nn {
            rx[e * nn + k] = el.geom.rx[k];
            ry[e * nn + k] = el.geom.ry[k];
            sx[e * nn + k] = el.geom.sx[k];
            sy[e * nn + k] = el.geom.sy[k];
            jw[e * nn + k] = el.geom.jw[k];
        }
    }

    // Flatten face connectivity (interior faces only).
    let nfc = ne * 4 * n1;
    let (mut fvl, mut fnx, mut fny, mut fsw, mut fnbr) =
        (vec![0u32; nfc], vec![0.0; nfc], vec![0.0; nfc], vec![0.0; nfc], vec![0u32; nfc]);
    for (e, el) in mesh.elements.iter().enumerate() {
        for (t, edge) in Edge::ALL.iter().enumerate() {
            let face = &el.faces[*edge as usize];
            let Neighbor::Interior { elem: re, edge: redge, perm } = &el.neighbors[*edge as usize]
            else {
                panic!("advection_rhs: mesh has a boundary face (use a periodic mesh)");
            };
            let rf = &mesh.elements[*re].faces[*redge as usize];
            for a in 0..n1 {
                let idx = (e * 4 + t) * n1 + a;
                fvl[idx] = face.nodes[a] as u32;
                fnx[idx] = face.nx[a];
                fny[idx] = face.ny[a];
                fsw[idx] = face.sw[a];
                fnbr[idx] = (*re * nn + rf.nodes[perm[a]]) as u32;
            }
        }
    }
    AdvectionGeom { ne, nn, n1, ndof, rx, ry, sx, sy, jw, fvl, fnx, fny, fsw, fnbr }
}

/// **Persistent** GPU advection backend: loads the device module and uploads the static mesh
/// geometry + face connectivity **once**, then each [`rhs`](Self::rhs) only uploads the changed
/// state `u`, launches, and downloads the result. This is what makes [`GpuAdvection`] usable in a
/// real time-stepping loop — the one-shot [`advection_rhs`] reloads + recompiles the whole module
/// (NVVM→nvJitLink→cubin, ~hundreds of ms) and re-uploads all geometry on every call, which is
/// catastrophic when driven per rhs evaluation by an integrator.
struct AdvectionBackend {
    _ctx: std::sync::Arc<CudaContext>,
    stream: std::sync::Arc<cuda_core::CudaStream>,
    module: kernels::LoadedModule,
    d_dev: DeviceBuffer<f64>,
    rx_dev: DeviceBuffer<f64>,
    ry_dev: DeviceBuffer<f64>,
    sx_dev: DeviceBuffer<f64>,
    sy_dev: DeviceBuffer<f64>,
    jw_dev: DeviceBuffer<f64>,
    fvl_dev: DeviceBuffer<u32>,
    fnx_dev: DeviceBuffer<f64>,
    fny_dev: DeviceBuffer<f64>,
    fsw_dev: DeviceBuffer<f64>,
    fnbr_dev: DeviceBuffer<u32>,
    u_dev: DeviceBuffer<f64>,
    out_dev: DeviceBuffer<f64>,
    cfg: LaunchConfig,
    n1: u32,
    ndof: usize,
}

impl AdvectionBackend {
    /// Build the backend for `mesh`: flatten geometry, retain the context, load the module, and
    /// upload all static arrays once. The per-step `u`/`out` buffers are allocated here and reused.
    fn new(mesh: &Mesh2d) -> Result<Self, Box<dyn std::error::Error>> {
        let g = flatten_advection_mesh(mesh);
        let ctx = CudaContext::new(0)?;
        let stream = ctx.default_stream();
        let up = |v: &[f64]| DeviceBuffer::from_host(&stream, v);
        let upu = |v: &[u32]| DeviceBuffer::from_host(&stream, v);
        let d_dev = up(&mesh.refq.line.diff)?;
        let rx_dev = up(&g.rx)?;
        let ry_dev = up(&g.ry)?;
        let sx_dev = up(&g.sx)?;
        let sy_dev = up(&g.sy)?;
        let jw_dev = up(&g.jw)?;
        let fvl_dev = upu(&g.fvl)?;
        let fnx_dev = up(&g.fnx)?;
        let fny_dev = up(&g.fny)?;
        let fsw_dev = up(&g.fsw)?;
        let fnbr_dev = upu(&g.fnbr)?;
        let u_dev = DeviceBuffer::<f64>::zeroed(&stream, g.ndof)?;
        let out_dev = DeviceBuffer::<f64>::zeroed(&stream, g.ndof)?;
        let module = kernels::load(&ctx)?;
        let cfg = LaunchConfig {
            grid_dim: (g.ne as u32, 1, 1),
            block_dim: (g.nn as u32, 1, 1),
            shared_mem_bytes: 0,
        };
        Ok(Self {
            _ctx: ctx,
            stream,
            module,
            d_dev, rx_dev, ry_dev, sx_dev, sy_dev, jw_dev,
            fvl_dev, fnx_dev, fny_dev, fsw_dev, fnbr_dev,
            u_dev, out_dev, cfg,
            n1: g.n1 as u32,
            ndof: g.ndof,
        })
    }

    /// One advection rhs on the persistent backend: upload `u` into the resident buffer, launch the
    /// kernel, download the result. No module load, no geometry re-upload — microseconds, not the
    /// ~hundreds of ms a full module recompile costs.
    fn rhs(&mut self, u: &[f64], ax: f64, ay: f64) -> Result<Vec<f64>, Box<dyn std::error::Error>> {
        assert_eq!(u.len(), self.ndof, "state length must be n_elements·n_nodes");
        unsafe {
            cuda_core::memory::memcpy_htod_async(
                self.u_dev.cu_deviceptr(),
                u.as_ptr(),
                std::mem::size_of_val(u),
                self.stream.cu_stream(),
            )?;
        }
        self.module.advect2d_rhs(
            &self.stream, self.cfg, &self.d_dev, &self.u_dev, &self.rx_dev, &self.ry_dev,
            &self.sx_dev, &self.sy_dev, &self.jw_dev, ax, ay, self.n1,
            &self.fvl_dev, &self.fnx_dev, &self.fny_dev, &self.fsw_dev, &self.fnbr_dev,
            &mut self.out_dev,
        )?;
        Ok(self.out_dev.to_host_vec(&self.stream)?)
    }
}

/// A GPU-backed semidiscretization of scalar linear advection, usable directly by
/// gale's framework: it implements [`gale::sim::StateSemi`], so a
/// [`gale::sim::Mol`] + [`gale::sim::SspRk3State`] integrator drives it through
/// [`gale::sim::Simulation::run`] with **each rhs evaluation on the GPU**.
pub struct GpuAdvection {
    field: gale::sim::FieldId,
    ax: f64,
    ay: f64,
    /// Persistent GPU backend, built lazily on the first rhs from `state.mesh` (the mesh isn't
    /// available at construction) and reused across every subsequent rhs evaluation. `RefCell`
    /// because `StateSemi::rhs` takes `&self`; the sim driver is single-threaded so this is sound.
    backend: std::cell::RefCell<Option<AdvectionBackend>>,
}

impl GpuAdvection {
    /// Advect the scalar `field` (1 component) with velocity `(ax, ay)`.
    pub fn new(field: gale::sim::FieldId, ax: f64, ay: f64) -> Self {
        Self { field, ax, ay, backend: std::cell::RefCell::new(None) }
    }
}

impl gale::sim::StateSemi<Mesh2d> for GpuAdvection {
    fn evolving(&self) -> Vec<gale::sim::FieldId> {
        vec![self.field]
    }
    fn rhs(&self, state: &gale::sim::State<Mesh2d>, _t: f64, dot: &mut gale::sim::FieldVec) {
        let u = state.fields.by_id(self.field).component(0).to_vec();
        // Build the persistent backend once (module + geometry uploaded), then reuse it every step —
        // each rhs is just upload-u / launch / download-r. (The old path called the one-shot
        // `advection_rhs` here, which reloaded + recompiled the whole module per evaluation.)
        let mut slot = self.backend.borrow_mut();
        if slot.is_none() {
            *slot = Some(
                AdvectionBackend::new(&state.mesh).expect("gale-gpu: advection backend build failed"),
            );
        }
        let r = slot
            .as_mut()
            .unwrap()
            .rhs(&u, self.ax, self.ay)
            .expect("gale-gpu: advection_rhs launch failed");
        dot.set(self.field, vec![r]);
    }
}
