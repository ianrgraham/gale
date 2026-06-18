//! **GPU p-multigrid preconditioner for the non-conforming (AMR) SIPG operator** — the device port of
//! the CPU [`PMultigridNc`](gale::dg::PMultigridNc). The plain-CG NC solve is `operator_nc`-matvec-bound
//! (~6000 iters/solve at scale; see the `nc-solve-cg-bound` finding); this collapses it to the
//! mesh-independent ~27 iters of a p-multigrid-preconditioned CG, **fully device-resident** — the
//! V-cycle (smooth → restrict → recurse → prolong → smooth) runs as kernel launches with the fields
//! never leaving the GPU; only the CG/deflation scalars read back (a host V-cycle would HtoD/DtoH
//! every PCG iteration — a RULE ZERO violation).
//!
//! Setup (one-time, on the host) reuses the validated CPU `PMultigridNc`: order sequence, the colored
//! diagonal `D⁻¹`, the Jacobi weights, and the 1D p-transfer matrices. The per-step V-cycle is all GPU.
//! Per-level operators are `GpuPoissonNc` at each order on the same refined mesh; they share the
//! device primary context (so cross-level buffers interoperate).

use crate::operators::poisson::GpuPoissonMg;
use crate::operators::poisson_nc::GpuPoissonNc;
use cuda_core::DeviceBuffer;
use gale::dg::{Mesh2d, PMultigrid, PMultigridNc};
use std::cell::RefCell;

/// Per-level resident scratch (sized to that level's `ndof`).
struct Scratch {
    x: RefCell<DeviceBuffer<f64>>,
    b: RefCell<DeviceBuffer<f64>>,
    r: RefCell<DeviceBuffer<f64>>,
    ax: RefCell<DeviceBuffer<f64>>,
    gx: RefCell<DeviceBuffer<f64>>,
    gy: RefCell<DeviceBuffer<f64>>,
    tmp: RefCell<DeviceBuffer<f64>>,
}

/// GPU p-multigrid-preconditioned CG for the non-conforming SIPG operator.
pub struct GpuPMultigridNc {
    levels: Vec<GpuPoissonNc>,
    /// `D⁻¹` per level (device).
    inv_diag: Vec<DeviceBuffer<f64>>,
    /// 1D Lagrange p-transfer per transition `l→l+1` (fine×coarse, device).
    interp: Vec<DeviceBuffer<f64>>,
    /// boundary marker per level (rebuilt once for the run's neumann_tags).
    fnbr: Vec<DeviceBuffer<u32>>,
    scratch: Vec<Scratch>,
    orders: Vec<usize>,
    omega: Vec<f64>,
    reaction: f64,
    singular: bool,
    pre: usize,
    post: usize,
    ndof0: usize,
    ones: DeviceBuffer<f64>,
    // ---- h-coarsening tail: collapse order-1-refined → order-1-uniform-base, solve with the
    // conforming GPU MG (which h-coarsens the base to a tiny grid). Replaces the O(N) Jacobi coarse. ----
    base_mg: GpuPoissonMg,
    base_kind: DeviceBuffer<u8>,
    base_ids: DeviceBuffer<u32>,
    quad_pq: DeviceBuffer<f64>,
    nbc: usize,
    base_resid: RefCell<DeviceBuffer<f64>>,
    base_corr: RefCell<DeviceBuffer<f64>>,
    coarse_tol: f64,
}

impl GpuPMultigridNc {
    /// Build the GPU hierarchy for `(reaction·M + A)` on the order-`p` refined mesh defined by
    /// `(nx,ny,xr,yr,refine)`. `neumann_tags` empty ⇒ all-Dirichlet; all boundary tags + `singular`
    /// ⇒ the deflated pure-Neumann pressure operator.
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        p: usize, nx: usize, ny: usize, xr: [f64; 2], yr: [f64; 2], refine: &[(usize, usize)], alpha: f64,
        reaction: f64, neumann_tags: Vec<u32>, singular: bool,
    ) -> Result<Self, Box<dyn std::error::Error>> {
        // CPU setup (orders, colored diagonal, Jacobi weights, transfer matrices) — validated reference.
        let cpu = PMultigridNc::new(p, nx, ny, xr, yr, refine, alpha, reaction, neumann_tags.clone(), singular);
        let nl = cpu.n_levels();
        let orders: Vec<usize> = (0..nl).map(|l| cpu.order(l)).collect();

        let mut levels = Vec::with_capacity(nl);
        let mut inv_diag = Vec::with_capacity(nl);
        let mut fnbr = Vec::with_capacity(nl);
        let mut scratch = Vec::with_capacity(nl);
        for l in 0..nl {
            let mesh = Mesh2d::cartesian_refined(orders[l], nx, ny, xr, yr, refine);
            let op = GpuPoissonNc::new(&mesh, alpha)?;
            inv_diag.push(op.upload(cpu.inv_diag(l))?);
            fnbr.push(op.build_fnbr_dev(&neumann_tags)?);
            scratch.push(Scratch {
                x: RefCell::new(op.alloc()?),
                b: RefCell::new(op.alloc()?),
                r: RefCell::new(op.alloc()?),
                ax: RefCell::new(op.alloc()?),
                gx: RefCell::new(op.alloc()?),
                gy: RefCell::new(op.alloc()?),
                tmp: RefCell::new(op.alloc()?),
            });
            levels.push(op);
        }
        let interp: Vec<DeviceBuffer<f64>> =
            (0..nl - 1).map(|l| levels[l].upload(cpu.interp(l))).collect::<Result<_, _>>()?;
        let omega: Vec<f64> = (0..nl).map(|l| cpu.jacobi_omega(l)).collect();
        let (pre, post) = cpu.smoothing();
        let ndof0 = levels[0].ndof();
        let ones = levels[0].upload(&vec![1.0f64; ndof0])?;

        // h-coarsening tail: conforming GPU MG on the order-1 UNIFORM base (h-coarsens to a tiny grid),
        // plus the refined→base transfer metadata (reused from the CPU setup so the maps are identical).
        let base_mg = GpuPoissonMg::new(PMultigrid::with_bc(1, nx, ny, xr, yr, alpha, reaction, neumann_tags.clone()))?;
        let (kind, ids, pq, bnx, bny) = cpu.base_transfer_data();
        let nbc = bnx * bny;
        let base_kind = levels[0].upload_u8(&kind)?;
        let base_ids = levels[0].upload_u32(&ids)?;
        let quad_pq = levels[0].upload(&pq)?;
        let base_resid = RefCell::new(base_mg.alloc_field()?);
        let base_corr = RefCell::new(base_mg.alloc_field()?);
        let coarse_tol = std::env::var("NC_MG_COARSE_TOL").ok().and_then(|s| s.parse().ok()).unwrap_or(1e-2);
        let _ = neumann_tags; // consumed at construction (build_fnbr_dev); not retained
        Ok(Self {
            levels, inv_diag, interp, fnbr, scratch, orders, omega, reaction, singular, pre, post, ndof0, ones,
            base_mg, base_kind, base_ids, quad_pq, nbc, base_resid, base_corr, coarse_tol,
        })
    }

    pub fn ndof(&self) -> usize {
        self.ndof0
    }
    pub fn upload(&self, v: &[f64]) -> Result<DeviceBuffer<f64>, Box<dyn std::error::Error>> {
        self.levels[0].upload(v)
    }
    pub fn download(&self, v: &DeviceBuffer<f64>) -> Result<Vec<f64>, Box<dyn std::error::Error>> {
        self.levels[0].download(v)
    }

    fn deflate(&self, v: &mut DeviceBuffer<f64>) -> Result<(), Box<dyn std::error::Error>> {
        if self.singular {
            let mean = self.levels[0].dot_dev(v, &self.ones)? / self.ndof0 as f64;
            self.levels[0].axpy_dev(v, &self.ones, -mean)?;
        }
        Ok(())
    }

    /// Recursive V-cycle solving `A·x_l = scratch[l].b` into `scratch[l].x` (x reset to 0 on entry).
    fn v_cycle(&self, l: usize) -> Result<(), Box<dyn std::error::Error>> {
        let op = &self.levels[l];
        if l == self.orders.len() - 1 {
            // Coarsest p-level = order-1 REFINED. Smooth + H-COARSE-CORRECT: collapse the residual onto
            // the order-1 UNIFORM base and solve it with the conforming GPU MG (which h-coarsens to a
            // tiny grid ⇒ cheap + mesh-independent), then prolong the correction back. Replaces the
            // O(N) Jacobi coarse that made large meshes take tens of seconds/step.
            {
                let mut x = self.scratch[l].x.borrow_mut();
                op.scal_dev(&mut x, 0.0)?;
            }
            for _ in 0..self.pre {
                let b = self.scratch[l].b.borrow();
                let mut x = self.scratch[l].x.borrow_mut();
                let (mut gx, mut gy, mut ax) =
                    (self.scratch[l].gx.borrow_mut(), self.scratch[l].gy.borrow_mut(), self.scratch[l].ax.borrow_mut());
                op.jacobi_dev(&mut x, &b, &self.inv_diag[l], self.omega[l], self.reaction, &self.fnbr[l], &mut gx, &mut gy, &mut ax)?;
            }
            // residual r = b − A·x  → restrict to base → conforming MG solve → prolong → correct.
            {
                let x = self.scratch[l].x.borrow();
                let b = self.scratch[l].b.borrow();
                let (mut gx, mut gy, mut ax, mut r) =
                    (self.scratch[l].gx.borrow_mut(), self.scratch[l].gy.borrow_mut(), self.scratch[l].ax.borrow_mut(), self.scratch[l].r.borrow_mut());
                op.apply_dev(&x, self.reaction, &self.fnbr[l], &mut gx, &mut gy, &mut ax)?;
                op.copy_dev(&mut r, &b)?;
                op.axpy_dev(&mut r, &ax, -1.0)?;
                let mut bresid = self.base_resid.borrow_mut();
                op.restrict_to_base_dev(&r, &self.base_kind, &self.base_ids, &self.quad_pq, self.nbc, &mut bresid)?;
            }
            {
                let bresid = self.base_resid.borrow();
                let mut bcorr = self.base_corr.borrow_mut();
                self.base_mg.solve_dev(&bresid, None, &mut bcorr, self.coarse_tol, 200)?;
            }
            {
                let bcorr = self.base_corr.borrow();
                let mut tmp = self.scratch[l].tmp.borrow_mut();
                op.prolong_from_base_dev(&bcorr, &self.base_kind, &self.base_ids, &self.quad_pq, self.nbc, &mut tmp)?;
                let mut x = self.scratch[l].x.borrow_mut();
                op.axpy_dev(&mut x, &tmp, 1.0)?;
            }
            for _ in 0..self.post {
                let b = self.scratch[l].b.borrow();
                let mut x = self.scratch[l].x.borrow_mut();
                let (mut gx, mut gy, mut ax) =
                    (self.scratch[l].gx.borrow_mut(), self.scratch[l].gy.borrow_mut(), self.scratch[l].ax.borrow_mut());
                op.jacobi_dev(&mut x, &b, &self.inv_diag[l], self.omega[l], self.reaction, &self.fnbr[l], &mut gx, &mut gy, &mut ax)?;
            }
            return Ok(());
        }
        // x_l ← 0, pre-smooth.
        {
            let mut x = self.scratch[l].x.borrow_mut();
            op.scal_dev(&mut x, 0.0)?;
        }
        for _ in 0..self.pre {
            let b = self.scratch[l].b.borrow();
            let mut x = self.scratch[l].x.borrow_mut();
            let (mut gx, mut gy, mut ax) = (self.scratch[l].gx.borrow_mut(), self.scratch[l].gy.borrow_mut(), self.scratch[l].ax.borrow_mut());
            op.jacobi_dev(&mut x, &b, &self.inv_diag[l], self.omega[l], self.reaction, &self.fnbr[l], &mut gx, &mut gy, &mut ax)?;
        }
        // r_l = b_l − A·x_l ; restrict r_l → b_{l+1}.
        {
            let x = self.scratch[l].x.borrow();
            let b = self.scratch[l].b.borrow();
            let (mut gx, mut gy, mut ax, mut r) =
                (self.scratch[l].gx.borrow_mut(), self.scratch[l].gy.borrow_mut(), self.scratch[l].ax.borrow_mut(), self.scratch[l].r.borrow_mut());
            op.apply_dev(&x, self.reaction, &self.fnbr[l], &mut gx, &mut gy, &mut ax)?;
            op.copy_dev(&mut r, &b)?;
            op.axpy_dev(&mut r, &ax, -1.0)?;
        }
        {
            let r = self.scratch[l].r.borrow();
            let mut bc = self.scratch[l + 1].b.borrow_mut();
            // restrict on the COARSE level (its n1 sizes the output); pass the fine block size.
            self.levels[l + 1].restrict_p_dev(&r, &self.interp[l], op.n1_pub(), &mut bc)?;
        }
        // recurse
        self.v_cycle(l + 1)?;
        // prolong x_{l+1} → tmp_l ; x_l += tmp_l ; post-smooth.
        {
            let ec = self.scratch[l + 1].x.borrow();
            let mut tmp = self.scratch[l].tmp.borrow_mut();
            op.prolong_p_dev(&ec, &self.interp[l], self.levels[l + 1].n1_pub(), &mut tmp)?;
        }
        {
            let tmp = self.scratch[l].tmp.borrow();
            let mut x = self.scratch[l].x.borrow_mut();
            op.axpy_dev(&mut x, &tmp, 1.0)?;
        }
        for _ in 0..self.post {
            let b = self.scratch[l].b.borrow();
            let mut x = self.scratch[l].x.borrow_mut();
            let (mut gx, mut gy, mut ax) = (self.scratch[l].gx.borrow_mut(), self.scratch[l].gy.borrow_mut(), self.scratch[l].ax.borrow_mut());
            op.jacobi_dev(&mut x, &b, &self.inv_diag[l], self.omega[l], self.reaction, &self.fnbr[l], &mut gx, &mut gy, &mut ax)?;
        }
        Ok(())
    }

    /// `z ← M⁻¹·r` (one V-cycle, finest level, zero initial guess); writes into `scratch[0].x`,
    /// copying `r` into `scratch[0].b` first.
    fn precondition(&self, r: &DeviceBuffer<f64>, z: &mut DeviceBuffer<f64>) -> Result<(), Box<dyn std::error::Error>> {
        {
            let mut b0 = self.scratch[0].b.borrow_mut();
            self.levels[0].copy_dev(&mut b0, r)?;
        }
        self.v_cycle(0)?;
        let x0 = self.scratch[0].x.borrow();
        self.levels[0].copy_dev(z, &x0)?;
        Ok(())
    }

    /// Device-resident p-MG-PCG: solves `(reaction·M + A)·x = rhs`, writing `out`. Fields stay on the
    /// GPU; only the CG/deflation scalars read back. Returns iterations.
    /// Host-convenience solve: upload `b`, run the device-resident p-MG-PCG, download `x`. Returns
    /// `(x, iters)` — a drop-in for the host-orchestrated flow integrator's `GpuPoissonNc::solve`
    /// (the VE step is host-orchestrated, so a per-solve up/download is the existing pattern).
    pub fn solve(&self, b: &[f64], tol: f64, maxit: usize) -> Result<(Vec<f64>, usize), Box<dyn std::error::Error>> {
        let bd = self.upload(b)?;
        let mut xd = DeviceBuffer::<f64>::zeroed(self.levels[0].stream(), self.ndof0)?;
        let it = self.solve_dev(&bd, &mut xd, tol, maxit)?;
        Ok((self.download(&xd)?, it))
    }

    pub fn solve_dev(
        &self, rhs: &DeviceBuffer<f64>, out: &mut DeviceBuffer<f64>, tol: f64, maxit: usize,
    ) -> Result<usize, Box<dyn std::error::Error>> {
        let op = &self.levels[0];
        let mut r = op.alloc()?;
        let mut z = op.alloc()?;
        let mut p = op.alloc()?;
        let mut ap = op.alloc()?;
        let (mut gx, mut gy) = (op.alloc()?, op.alloc()?);

        // x = 0 ; r = b (deflated)
        op.scal_dev(out, 0.0)?;
        op.copy_dev(&mut r, rhs)?;
        self.deflate(&mut r)?;
        let bn = self.levels[0].dot_dev(&r, &r)?.sqrt().max(1e-300);

        self.precondition(&r, &mut z)?;
        self.deflate(&mut z)?;
        op.copy_dev(&mut p, &z)?;
        let mut rz = op.dot_dev(&r, &z)?;
        let mut iters = maxit;
        for it in 0..maxit {
            if (op.dot_dev(&r, &r)?).sqrt() / bn < tol {
                iters = it;
                break;
            }
            op.apply_dev(&p, self.reaction, &self.fnbr[0], &mut gx, &mut gy, &mut ap)?;
            self.deflate(&mut ap)?;
            let pap = op.dot_dev(&p, &ap)?;
            if !(pap > 0.0) {
                iters = it;
                break;
            }
            let alpha = rz / pap;
            op.axpy_dev(out, &p, alpha)?;
            op.axpy_dev(&mut r, &ap, -alpha)?;
            self.deflate(&mut r)?;
            self.precondition(&r, &mut z)?;
            self.deflate(&mut z)?;
            let rz_new = op.dot_dev(&r, &z)?;
            let beta = rz_new / rz;
            // p = z + beta·p
            op.scal_dev(&mut p, beta)?;
            op.axpy_dev(&mut p, &z, 1.0)?;
            rz = rz_new;
            iters = it + 1;
        }
        let _ = (gx, gy);
        Ok(iters)
    }
}
