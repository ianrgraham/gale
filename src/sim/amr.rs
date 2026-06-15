//! Adaptive mesh refinement as a framework [`Updater`].
//!
//! [`AmrUpdater`] is the dynamic `h`-adaptation operation: when triggered it
//! evaluates a spectral smoothness indicator on a designated field, decides which
//! base cells should be refined, and — if the refinement pattern changed — remaps
//! **every** `State` field to the new mesh and swaps in that mesh. It builds on the
//! generalized `dg::amr` layer (`remap_component_flat`, `smoothness_per_cell`),
//! which handles multi-component flat fields and level-aware (restrict-then-judge)
//! indication.
//!
//! Scope: single-level 2:1 `h`-refinement over a Cartesian base grid — the refinement the mesh
//! (`Neighbor::CoarseToFine`/`FineToCoarse`) supports. Refinement is indicator-driven; de-refinement
//! (coarsening) is opt-in via [`AmrUpdater::with_coarsening`] and uses a **child-based** criterion
//! (a refined cell coarsens when the max indicator over its children falls below the coarsen
//! threshold) — judging the children directly rather than restricting the parent and re-indicating
//! (which low-passes the field and oscillates), with the refine/coarsen threshold gap providing
//! hysteresis. The updater stores the base-grid parameters (the mesh does not carry them) and the
//! current refined set, so successive fires adapt the mesh incrementally in both directions.

use super::simulation::Updater;
use super::state::State;
use crate::dg::amr::{child_smoothness_max, remap_component_flat, smoothness_per_cell};
use crate::dg::{Mesh2d, Mesh3d};

/// Indicator-driven `h`-AMR updater over a Cartesian base grid.
pub struct AmrUpdater {
    order: usize,
    nx: usize,
    ny: usize,
    xr: [f64; 2],
    yr: [f64; 2],
    /// Field whose smoothness drives refinement.
    indicator_field: String,
    /// Which component of that field to use.
    indicator_comp: usize,
    /// Refine a base cell when its indicator exceeds this.
    threshold: f64,
    /// Coarsen a refined cell when the MAX indicator over its children falls below this
    /// (`None` ⇒ refine-only). For stable hysteresis use a value well below `threshold` so a
    /// just-refined cell doesn't immediately coarsen; the child-based criterion (vs restricting
    /// the parent) is what prevents oscillation.
    coarsen_threshold: Option<f64>,
    /// Current set of refined base cells (starts empty: the base grid).
    refined: Vec<(usize, usize)>,
}

impl AmrUpdater {
    /// New updater for an `nx × ny` order-`p` Cartesian base grid over
    /// `[xr] × [yr]`, refining where `indicator_field`'s component 0 exceeds
    /// `threshold`. Assumes the `State` starts on the unrefined base grid.
    pub fn new(
        order: usize,
        nx: usize,
        ny: usize,
        xr: [f64; 2],
        yr: [f64; 2],
        indicator_field: impl Into<String>,
        threshold: f64,
    ) -> Self {
        Self {
            order,
            nx,
            ny,
            xr,
            yr,
            indicator_field: indicator_field.into(),
            indicator_comp: 0,
            threshold,
            coarsen_threshold: None,
            refined: Vec::new(),
        }
    }

    /// Use a specific component of the indicator field.
    pub fn indicator_component(mut self, comp: usize) -> Self {
        self.indicator_comp = comp;
        self
    }

    /// Enable de-refinement: a refined cell whose MAX child indicator drops below
    /// `coarsen_threshold` is merged back (restrict 4 children → parent). Pick it well below the
    /// refine `threshold` (hysteresis) so cells don't thrash refine↔coarsen. Refine-only by default.
    pub fn with_coarsening(mut self, coarsen_threshold: f64) -> Self {
        self.coarsen_threshold = Some(coarsen_threshold);
        self
    }

    /// Seed the current refined set — use when the `State` starts on an ALREADY-refined mesh
    /// (`Mesh2d::cartesian_refined(.., set)`) rather than the base grid, so the updater's bookkeeping
    /// matches the actual mesh (otherwise it would misread the field layout and never coarsen).
    pub fn with_initial_refined(mut self, refined: Vec<(usize, usize)>) -> Self {
        self.refined = refined;
        self
    }

    /// The current set of refined base cells.
    pub fn refined_cells(&self) -> &[(usize, usize)] {
        &self.refined
    }
}

impl Updater for AmrUpdater {
    fn update(&mut self, state: &mut State, _step: u64) {
        let old_set = self.refined.clone();
        let old_h: std::collections::HashSet<(usize, usize)> = old_set.iter().copied().collect();

        // REFINE: an unrefined base cell whose indicator exceeds `threshold`. (`smoothness_per_cell`
        // is level-aware — it restricts a refined cell to base before indicating — but here we only
        // act on its UNREFINED entries; refined cells are judged for coarsening separately below.)
        let comp = self.indicator_comp;
        let cell_ind = {
            let f = state.field(&self.indicator_field);
            smoothness_per_cell(self.order, self.nx, self.ny, &old_set, f.component(comp))
        };
        let mut new_h = old_h.clone();
        for cy in 0..self.ny {
            for cx in 0..self.nx {
                if !old_h.contains(&(cx, cy)) && cell_ind[cx + cy * self.nx] > self.threshold {
                    new_h.insert((cx, cy));
                }
            }
        }
        // COARSEN (optional): a refined cell whose MAX child indicator falls below the coarsen
        // threshold is merged back. Judging the children directly (not the restricted parent) is
        // what keeps refine↔coarsen from oscillating; the threshold gap adds hysteresis.
        if let Some(ct) = self.coarsen_threshold {
            let child_max = {
                let f = state.field(&self.indicator_field);
                child_smoothness_max(self.order, self.nx, self.ny, &old_set, f.component(comp))
            };
            for (cell, mx) in child_max {
                if mx < ct {
                    new_h.remove(&cell);
                }
            }
        }

        // Compare as SETS (coarsen + refine can leave the count unchanged while the pattern moves).
        let mut new_set: Vec<(usize, usize)> = new_h.iter().copied().collect();
        new_set.sort_unstable();
        let mut old_sorted = old_set.clone();
        old_sorted.sort_unstable();
        if new_set == old_sorted {
            return;
        }

        // Remap every field to the new mesh (read all old data first, then swap).
        let names: Vec<(String, usize)> =
            state.fields.iter().map(|f| (f.name.clone(), f.n_comp)).collect();
        let mut remapped: Vec<(String, Vec<Vec<f64>>)> = Vec::with_capacity(names.len());
        for (name, n_comp) in &names {
            let f = state.field(name);
            let comps: Vec<Vec<f64>> = (0..*n_comp)
                .map(|c| {
                    remap_component_flat(
                        self.order,
                        self.nx,
                        self.ny,
                        &old_set,
                        f.component(c),
                        &new_set,
                    )
                })
                .collect();
            remapped.push((name.clone(), comps));
        }

        // Swap in the refined mesh, then install the remapped fields (new ndof).
        state.mesh = Mesh2d::cartesian_refined(self.order, self.nx, self.ny, self.xr, self.yr, &new_set);
        for (name, comps) in remapped {
            state.field_mut(&name).replace_components(comps);
        }
        self.refined = new_set;
    }
}

/// The 3D (octree) analogue of [`AmrUpdater`]: indicator-driven 2:1 `h`-adaptation over a Cartesian
/// `nx×ny×nz` hex base grid, as an [`Updater<Mesh3d>`]. Refine + optional coarsen, same child-based
/// criterion and hysteresis; remaps every field through the octree transfer operators.
pub struct AmrUpdater3d {
    order: usize,
    nx: usize,
    ny: usize,
    nz: usize,
    xr: [f64; 2],
    yr: [f64; 2],
    zr: [f64; 2],
    indicator_field: String,
    indicator_comp: usize,
    threshold: f64,
    coarsen_threshold: Option<f64>,
    refined: Vec<(usize, usize, usize)>,
}

impl AmrUpdater3d {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        order: usize,
        nx: usize,
        ny: usize,
        nz: usize,
        xr: [f64; 2],
        yr: [f64; 2],
        zr: [f64; 2],
        indicator_field: impl Into<String>,
        threshold: f64,
    ) -> Self {
        Self {
            order,
            nx,
            ny,
            nz,
            xr,
            yr,
            zr,
            indicator_field: indicator_field.into(),
            indicator_comp: 0,
            threshold,
            coarsen_threshold: None,
            refined: Vec::new(),
        }
    }

    pub fn indicator_component(mut self, comp: usize) -> Self {
        self.indicator_comp = comp;
        self
    }

    /// Enable de-refinement (see [`AmrUpdater::with_coarsening`]).
    pub fn with_coarsening(mut self, coarsen_threshold: f64) -> Self {
        self.coarsen_threshold = Some(coarsen_threshold);
        self
    }

    /// Seed the refined set when the `State` starts on an already-refined mesh.
    pub fn with_initial_refined(mut self, refined: Vec<(usize, usize, usize)>) -> Self {
        self.refined = refined;
        self
    }

    pub fn refined_cells(&self) -> &[(usize, usize, usize)] {
        &self.refined
    }
}

impl Updater<Mesh3d> for AmrUpdater3d {
    fn update(&mut self, state: &mut State<Mesh3d>, _step: u64) {
        use crate::dg::amr3d::{child_smoothness_max_3d, remap_component_flat_3d, smoothness_per_cell_3d};
        let old_set = self.refined.clone();
        let old_h: std::collections::HashSet<(usize, usize, usize)> = old_set.iter().copied().collect();
        let comp = self.indicator_comp;

        // REFINE: unrefined cells whose indicator exceeds threshold.
        let cell_ind = {
            let f = state.field(&self.indicator_field);
            smoothness_per_cell_3d(self.order, self.nx, self.ny, self.nz, &old_set, f.component(comp))
        };
        let mut new_h = old_h.clone();
        for cz in 0..self.nz {
            for cy in 0..self.ny {
                for cx in 0..self.nx {
                    let i = cx + cy * self.nx + cz * self.nx * self.ny;
                    if !old_h.contains(&(cx, cy, cz)) && cell_ind[i] > self.threshold {
                        new_h.insert((cx, cy, cz));
                    }
                }
            }
        }
        // COARSEN (optional): refined cells whose max child indicator falls below the threshold.
        if let Some(ct) = self.coarsen_threshold {
            let child_max = {
                let f = state.field(&self.indicator_field);
                child_smoothness_max_3d(self.order, self.nx, self.ny, self.nz, &old_set, f.component(comp))
            };
            for (cell, mx) in child_max {
                if mx < ct {
                    new_h.remove(&cell);
                }
            }
        }

        let mut new_set: Vec<(usize, usize, usize)> = new_h.iter().copied().collect();
        new_set.sort_unstable();
        let mut old_sorted = old_set.clone();
        old_sorted.sort_unstable();
        if new_set == old_sorted {
            return;
        }

        // Remap every field (read all old data first), then swap in the new mesh.
        let names: Vec<(String, usize)> = state.fields.iter().map(|f| (f.name.clone(), f.n_comp)).collect();
        let mut remapped: Vec<(String, Vec<Vec<f64>>)> = Vec::with_capacity(names.len());
        for (name, n_comp) in &names {
            let f = state.field(name);
            let comps: Vec<Vec<f64>> = (0..*n_comp)
                .map(|c| {
                    remap_component_flat_3d(
                        self.order, self.nx, self.ny, self.nz, &old_set, f.component(c), &new_set,
                    )
                })
                .collect();
            remapped.push((name.clone(), comps));
        }
        state.mesh =
            Mesh3d::cartesian_refined(self.order, self.nx, self.ny, self.nz, self.xr, self.yr, self.zr, &new_set);
        for (name, comps) in remapped {
            state.field_mut(&name).replace_components(comps);
        }
        self.refined = new_set;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::sim::Periodic;
    use crate::sim::Simulation;
    use std::f64::consts::PI;

    /// On a smooth (well-resolved) indicator field, the updater refines nothing:
    /// the mesh and fields are left untouched.
    #[test]
    fn no_refinement_when_field_is_smooth() {
        let (order, nx, ny) = (4, 4, 4);
        let mut st = State::new(Mesh2d::rectangular(order, nx, ny, [0.0, 1.0], [0.0, 1.0]));
        // Linear field — spectrally exact, indicator ≈ 0.
        st.add_field_from("u", &[|x: f64, y: f64| x + 2.0 * y]);
        let n0 = st.mesh.n_elements();

        let mut amr =
            AmrUpdater::new(order, nx, ny, [0.0, 1.0], [0.0, 1.0], "u", 1e-6);
        amr.update(&mut st, 0);

        assert_eq!(st.mesh.n_elements(), n0, "smooth field should not refine");
        assert!(amr.refined_cells().is_empty());
    }

    /// On an under-resolved indicator field the updater refines, growing the mesh,
    /// and a co-located low-degree field is transferred **exactly** (prolong is
    /// exact for degree ≤ p) — verified at the new mesh nodes.
    #[test]
    fn refines_and_transfers_fields_exactly() {
        let (order, nx, ny) = (4, 4, 4);
        let xr = [0.0, 1.0];
        let yr = [0.0, 1.0];
        let mut st = State::new(Mesh2d::rectangular(order, nx, ny, xr, yr));
        // Indicator field: 1 wavelength per cell ⇒ under-resolved ⇒ high indicator.
        st.add_field_from("wiggle", &[|x: f64, y: f64| {
            (20.0 * PI * x).sin() * (20.0 * PI * y).sin()
        }]);
        // Transported field: degree-1, prolong-exact. Checked after remap.
        st.add_field_from("u", &[|x: f64, y: f64| x + 2.0 * y]);
        let n0 = st.mesh.n_elements();

        let mut amr =
            AmrUpdater::new(order, nx, ny, xr, yr, "wiggle", 1e-2);
        amr.update(&mut st, 0);

        assert!(st.mesh.n_elements() > n0, "under-resolved field should refine");
        assert!(!amr.refined_cells().is_empty());

        // The degree-1 field equals x + 2y at every node of the refined mesh.
        let nn = st.mesh.refq.n_nodes();
        let u = st.field("u");
        assert_eq!(u.component(0).len(), st.mesh.n_elements() * nn);
        let mut maxerr = 0.0f64;
        for (e, el) in st.mesh.elements.iter().enumerate() {
            for k in 0..nn {
                let exact = el.geom.x[k] + 2.0 * el.geom.y[k];
                maxerr = maxerr.max((u.component(0)[e * nn + k] - exact).abs());
            }
        }
        assert!(maxerr < 1e-10, "prolong not exact through AMR: {maxerr:e}");
    }

    /// Re-running the updater on the same (now-refined) state is idempotent:
    /// refinement is monotone (refine-only), so already-refined cells are kept and
    /// no new cells are added on the second pass.
    #[test]
    fn re_adaptation_is_idempotent() {
        let (order, nx, ny) = (4, 4, 4);
        let xr = [0.0, 1.0];
        let yr = [0.0, 1.0];
        let mut st = State::new(Mesh2d::rectangular(order, nx, ny, xr, yr));
        st.add_field_from("wiggle", &[|x: f64, y: f64| {
            (20.0 * PI * x).sin() * (20.0 * PI * y).sin()
        }]);

        let mut amr = AmrUpdater::new(order, nx, ny, xr, yr, "wiggle", 1e-2);
        amr.update(&mut st, 0);
        let after_first = st.mesh.n_elements();
        let set_first = amr.refined_cells().to_vec();
        amr.update(&mut st, 1);
        assert_eq!(st.mesh.n_elements(), after_first, "second adaptation changed the mesh");
        assert_eq!(amr.refined_cells(), set_first.as_slice());
    }

    /// With coarsening enabled, a refined mesh de-refines once the field becomes smooth: refine on
    /// a wiggly field, then overwrite with a (linear) smooth field ⇒ the next fire coarsens it back.
    #[test]
    fn coarsening_de_refines_when_smooth() {
        let (order, nx, ny) = (4, 4, 4);
        let xr = [0.0, 1.0];
        let yr = [0.0, 1.0];
        let mut st = State::new(Mesh2d::rectangular(order, nx, ny, xr, yr));
        st.add_field_from("u", &[|x: f64, y: f64| (20.0 * PI * x).sin() * (20.0 * PI * y).sin()]);

        let mut amr = AmrUpdater::new(order, nx, ny, xr, yr, "u", 1e-2).with_coarsening(1e-6);
        amr.update(&mut st, 0);
        let refined = amr.refined_cells().len();
        assert!(refined > 0, "wiggly field should have refined some cells");

        // Overwrite with a smooth (linear) field on the current refined mesh ⇒ near-zero indicator.
        let nn = st.mesh.refq.n_nodes();
        let mut smooth = vec![0.0; st.mesh.n_elements() * nn];
        for (e, el) in st.mesh.elements.iter().enumerate() {
            for k in 0..nn {
                smooth[e * nn + k] = el.geom.x[k] + 2.0 * el.geom.y[k];
            }
        }
        st.field_mut("u").replace_components(vec![smooth]);

        amr.update(&mut st, 1);
        assert!(
            amr.refined_cells().is_empty(),
            "smooth field should have coarsened everything back, {} cells left",
            amr.refined_cells().len()
        );
        // The mesh and field are consistent after coarsening (ndof matches the base grid).
        assert_eq!(st.mesh.n_elements(), nx * ny);
        assert_eq!(st.field("u").component(0).len(), nx * ny * nn);
    }

    /// End-to-end CPU **3D adaptive flow**: a uniform free stream `(1,0,0)` on a pre-refined hex
    /// mesh, advanced by `DualSplitting3d` while `AmrUpdater3d` coarsens it mid-run. The free stream
    /// must survive both the NC dual-splitting step (the 3D mortar operator preserves a constant) and
    /// the coarsening remesh (restrict of a constant is exact) — the decisive 3D-adaptive-flow check.
    /// Ignored by default: 3D dual-splitting with CG solves is very slow in a debug build (~24 min);
    /// run explicitly, ideally release: `cargo test -p gale --release cpu_3d_adaptive_flow -- --ignored`.
    #[test]
    #[ignore = "slow in debug (3D NS CG solves); run with --release --ignored"]
    fn cpu_3d_adaptive_flow_preserves_free_stream() {
        use crate::dg::Mesh3d;
        use crate::sim::{DualSplitting3d, OnStep, Simulation};
        let (order, nx, ny, nz) = (2usize, 2usize, 2usize, 2usize);
        let (xr, yr, zr) = ([0.0, 1.0], [0.0, 1.0], [0.0, 1.0]);
        let pre = vec![(0usize, 0usize, 0usize), (1, 1, 1)];
        let mut st = State::new(Mesh3d::cartesian_refined(order, nx, ny, nz, xr, yr, zr, &pre));
        let vid = st.add_field_from(
            "velocity",
            &[
                Box::new(|_: f64, _: f64, _: f64| 1.0) as Box<dyn Fn(f64, f64, f64) -> f64>,
                Box::new(|_: f64, _: f64, _: f64| 0.0),
                Box::new(|_: f64, _: f64, _: f64| 0.0),
            ],
        );
        let pre_ne = st.mesh.n_elements();
        let mut sim = Simulation::new(st);
        sim.set_integrator(DualSplitting3d::new(vid, 1e-2, 0.05, 5.0).boundary(
            |_, _, _, _| 1.0,
            |_, _, _, _| 0.0,
            |_, _, _, _| 0.0,
        ));
        sim.add_updater(
            AmrUpdater3d::new(order, nx, ny, nz, xr, yr, zr, "velocity", 1e9)
                .with_coarsening(1e-1)
                .with_initial_refined(pre.clone()),
            OnStep { step: 2 },
        );
        sim.run(5);
        let u = sim.state.field("velocity");
        let eu = u.component(0).iter().map(|&v| (v - 1.0).abs()).fold(0.0, f64::max);
        let ev = u.component(1).iter().map(|&v| v.abs()).fold(0.0, f64::max);
        let ew = u.component(2).iter().map(|&v| v.abs()).fold(0.0, f64::max);
        assert!(eu < 1e-7 && ev < 1e-7 && ew < 1e-7, "free stream not preserved: {eu:.2e} {ev:.2e} {ew:.2e}");
        assert!(sim.state.mesh.n_elements() < pre_ne, "should have coarsened ({} → {})", pre_ne, sim.state.mesh.n_elements());
    }

    /// The updater works as a triggered operation inside a Simulation (no
    /// integrator needed for this structural check — just the AMR cadence).
    #[test]
    fn runs_as_triggered_updater() {
        let (order, nx, ny) = (4, 3, 3);
        let xr = [0.0, 1.0];
        let yr = [0.0, 1.0];
        let mut st = State::new(Mesh2d::rectangular(order, nx, ny, xr, yr));
        st.add_field_from("wiggle", &[|x: f64, _y: f64| (20.0 * PI * x).sin()]);
        let n0 = st.mesh.n_elements();

        let mut sim = Simulation::new(st);
        sim.add_updater(
            AmrUpdater::new(order, nx, ny, xr, yr, "wiggle", 1e-2),
            Periodic::new(1),
        );
        // No integrator: drive only the updater schedule by stepping updaters
        // manually through the operations (run requires an integrator), so here we
        // just confirm the updater is registered and fires via a direct call.
        for u in sim.operations.updaters.iter_mut() {
            if u.trigger.fires(0) {
                u.op.update(&mut sim.state, 0);
            }
        }
        assert!(sim.state.mesh.n_elements() > n0, "AMR updater did not refine via the op");
    }
}
