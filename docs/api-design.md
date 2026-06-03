# gale user-facing API design (RFC)

Status: **proposal for review** — not yet implemented.
Date: 2026-06-03.
Scope: the ergonomic, composable interface for *assembling* and *running* DG-SEM
simulations in gale — Rust-first, with a later Python (pyo3) entry point that is a
thin binding over the **same** Rust objects (zero simulation logic in Python).

This document synthesizes a verified deep-research pass (HOOMD-blue and Trixi.jl
primary docs; see `## Sources & evidence`) with the abstractions gale has already
validated (`ConservationLaw`, `Hyperbolic<L>`, `ConstitutiveModel`,
`Mesh2d`/`Neighbor`, `Reference1d`). It proposes a concrete component model and a
migration path from today's bespoke per-regime structs to that model.

---

## 1. Design goals (from the maintainer)

1. **HOOMD-blue-style configuration and running.** Build a `Simulation`, attach
   composable operations, call `run(n_steps)`.
2. **Ergonomic, flexible, composable** assembly of DG simulations — swap
   equations, integrators, immersed bodies, AMR cadence, and output without
   rewriting the loop.
3. **Rust-first.** The full object model is usable and idiomatic from Rust alone.
4. **Python later, with zero divergence.** pyo3 wraps the *same* Rust objects 1:1.
   No simulation logic lives in Python — it is a faithful mirror.
5. Must accommodate the headline application: **viscoelastic particle-laden
   microfluidic suspensions** (incompressible NS + log-conformation + IBM + AMR,
   multi-GPU).

## 2. The two templates, and why we fuse them

The research confirmed (24/25 verified claims) that two existing designs map
almost directly onto gale, and that they are complementary:

- **HOOMD-blue** gives the *orchestration* model: one `Simulation` owns the
  `State`, the `Operations`, and the `Device`; operations are a small fixed
  taxonomy (Compute / Updater / Integrator / Writer / Tuner) run in deterministic
  per-step order and gated by **Triggers**; physics contributions are **additive**
  (the integrator sums a user-supplied list of forces into a net quantity);
  the Python API is a pybind11 1:1 mirror of the C++ core. This is the proven
  template for "assemble and run."

- **Trixi.jl** gives the *PDE/DG seam*: a `SemidiscretizationHyperbolic` bundles
  `{mesh, equations, solver, initial/boundary conditions, source_terms}` into a
  spatial operator that becomes an ODE `rhs` (method-of-lines), advanced by a
  separate time integrator, with cross-cutting concerns expressed as an **ordered
  `CallbackSet`** at two granularities — *step* callbacks (AMR, analysis, I/O, CFL
  stepsize) and *stage* callbacks (limiters/positivity, run between RK stages).

gale already lives at this seam: `ConservationLaw` ≈ Trixi `equations`,
`Hyperbolic<'m, L>` ≈ the semidiscretization, and each regime carries its own
`step_ssp_rk3`. The framework's job is to **lift the hardcoded stepper out** of
each regime and replace it with a HOOMD-style orchestrator that drives any
semidiscretization through any integrator.

> Caveat carried from the research: only the additive/explicit path
> (HOOMD methods+forces, Trixi semidiscretize+SSP-RK) is externally evidenced.
> Structured schemes (incompressible projection / dual-splitting / IMEX), the Rust
> dispatch strategy in hot loops, the device abstraction, and the exact pyo3 sync
> layer are **gale design decisions** resolved below on first principles, flagged
> as such, and to be validated by our own benchmarks rather than by citation.

---

## 3. Component architecture

```
Simulation
├── State          // single source of truth: fields + topology + bodies + particles
├── Device         // CPU | Cuda(ordinals) | MultiGpu(partition)  — execution backend
└── Operations
    ├── computes:   Vec<Box<dyn Compute>>     // read-only derived quantities
    ├── updaters:   Vec<Triggered<dyn Updater>> // mutate state (AMR, body advect, …)
    ├── integrator: Box<dyn Integrator>       // exactly one; advances State in time
    ├── writers:    Vec<Triggered<dyn Writer>>  // read + emit output, no mutation
    └── tuners:     Vec<Triggered<dyn Tuner>>  // adjust other operations' params
```

Per-step schedule (HOOMD order, verified 3-0): **tuners → updaters → integrator →
writers**. Writers run last so they capture the post-step state. Computes are
pulled on demand (and may cache within a timestep).

### 3.1 `State` — the single source of truth

gale's State is *field-based* (DG modal/nodal coefficients per element), not
particle-based like HOOMD — the orchestration transfers, the data model does not.
State owns everything an operation might read or mutate:

```rust
pub struct State {
    pub mesh: Mesh,                  // topology + geometry + AMR tree (Neighbor incl. CoarseToFine/FineToCoarse)
    pub fields: FieldSet,            // named DG fields: velocity, pressure, conformation/Ψ, …
    pub bodies: Vec<ImmersedBody>,   // rigid + deformable membranes (volume penalization / front tracking)
    pub particles: ParticleData,     // optional: suspended particle centroids/orientations
    pub time: Time,                  // { t, step } — step read-only to operations except the integrator
    device_mirror: DeviceFields,     // GPU-resident copies; sync managed by the backend, not the user
}
```

`FieldSet` is a typed, named registry rather than a fixed struct, so the *same*
State type spans incompressible / compressible / viscoelastic regimes without a
per-regime State type (resolves research open-question "state typing across
regimes"). A field is `{ name, n_components, layout, host: Vec<f64>, dev: Option<DeviceBuffer> }`.

Access contract (mirrors HOOMD's snapshot get/set, but zero-copy is the hot path):

```rust
impl State {
    pub fn field(&self, name: &str) -> &Field;
    pub fn field_mut(&mut self, name: &str) -> &mut Field;   // borrow, not copy
    pub fn snapshot(&self) -> Snapshot;                      // owned copy: checkpoint / Python boundary
}
```

### 3.2 The central seam: `Semidiscretization`, `Term`, `Integrator`

We keep gale's existing seam and formalize it.

**Equations** stay as today's trait (rename for clarity; `ConservationLaw` is an
acceptable alias):

```rust
pub trait Equations {
    fn n_vars(&self) -> usize;
    fn flux(&self, u: &[f64], dir: usize, out: &mut [f64]);
    fn numerical_flux(&self, ul: &[f64], ur: &[f64], n: [f64; 2], out: &mut [f64]);
    // entropy variables / EC flux hooks as already implemented for Euler
}
```

**Semidiscretization** bundles the spatial operator (Trixi pattern, 3-0):

```rust
pub struct Semidiscretization<E: Equations> {
    pub mesh: Mesh,
    pub equations: E,
    pub solver: DgSem,                  // order, reference element, volume form, dissipation
    pub boundary: BoundaryConditions,
    pub source_terms: Terms,            // additive — see below
}

impl<E: Equations> Semidiscretization<E> {
    /// Method-of-lines rhs: du/dt = rhs(u, t). This is gale's existing Hyperbolic::rhs,
    /// plus a sum over additive source/coupling Terms.
    pub fn rhs(&self, u: &State, t: f64, dudt: &mut State);
}
```

**Terms** are the additive composability primitive (HOOMD net-force pattern, 3-0).
Convection, diffusion, polymer-stress divergence, IBM forcing, gravity/buoyancy,
and stabilization are all interchangeable `Term`s that *accumulate* into the rhs:

```rust
pub trait Term {
    /// Accumulate this term's contribution into dudt (+=). Never overwrites.
    fn accumulate(&self, state: &State, t: f64, dudt: &mut State);
}
```

`Terms` is the ordered collection. **Dispatch decision** (resolves the highest-risk
open question): gale uses **coarse-grained dispatch**. A `Box<dyn Term>` boundary
is crossed *once per term per rhs evaluation* — i.e. a handful of virtual calls per
timestep — and each call then runs a monomorphized, statically-dispatched
element-loop / GPU kernel over *all* DOFs. The dynamic-dispatch cost is therefore
O(#terms), not O(#DOFs), and is negligible. We do **not** put `dyn` inside the hot
element loop. For the few performance-critical fixed compositions we can still
offer a generic tuple `(T1, T2, …): Terms` for full monomorphization, but the
default user-facing path is `Vec<Box<dyn Term>>` for ergonomics and Python parity.

**Integrator** is the time-advance strategy — one trait, multiple families
(resolves the additive-vs-structured open question):

```rust
pub trait Integrator {
    fn dt(&self) -> f64;
    /// Advance state by one step. May call semi.rhs(...) (method-of-lines)
    /// OR orchestrate its own stages (projection / IMEX), drawing on the same Terms.
    fn step(&mut self, semi: &mut dyn SemiDyn, state: &mut State, stage_hook: &mut dyn StageHook);
}
```

- **Explicit method-of-lines** integrators (`SspRk3`, `SspRk2`, `Rk4`) consume
  `semi.rhs(...)` — they sum the additive Terms and advance. This is gale's current
  `step_ssp_rk3`, lifted out of each regime into a reusable integrator.
- **Structured schemes** (`DualSplitting` for incompressible NS, `Imex` for stiff
  viscoelastic relaxation) are *also* `Integrator` implementors. They internally
  orchestrate stages — pressure Poisson solve, Helmholtz viscous solve,
  log-conformation update — and draw on the *same* `Term` objects for the explicit
  pieces. The orchestration lives inside the integrator, not in user code.

This is the key unification: **one `Integrator` trait, honored identically by every
family**, so Terms / callbacks written once work across all of them (Trixi's
"shared callback interface" lesson, 3-0). Exactly one integrator per Simulation
keeps the schedule unambiguous (HOOMD, 3-0).

**Stage hook** (Trixi stage-callback granularity, 3-0): limiters, positivity /
entropy enforcement, the SVV filter, and the IBM volume-penalization projection
run *between* RK stages, not per step. The `Integrator::step` receives a
`StageHook` it invokes after each internal stage:

```rust
pub trait StageHook { fn after_stage(&mut self, state: &mut State, stage: usize); }
```

### 3.3 Operations taxonomy (HOOMD five roles, verified 3-0; 7-role refuted 0-3)

```rust
pub trait Compute  { fn compute(&self, state: &State) -> ComputeValue; }      // read-only
pub trait Updater  { fn update(&mut self, state: &mut State, step: u64); }    // mutate
pub trait Writer   { fn write(&mut self, state: &State, step: u64); }         // emit, no mutate
pub trait Tuner    { fn tune(&mut self, ops: &mut Operations, step: u64); }   // adjust params
// Integrator: see §3.2 (exactly one)
```

gale's concrete operations:

| Role | Examples |
|------|----------|
| Compute | kinetic energy, enstrophy, drag/lift on a body, max Wi, CFL number |
| Updater | `AmrUpdater` (refine/coarsen via smoothness indicator), `BodyAdvect` (move membranes/particles), `ReBalance` (multi-GPU repartition) |
| Writer  | `VtkWriter`, `Hdf5Writer`, `CheckpointWriter`, `ConsoleProgress` |
| Tuner   | adaptive `dt` controller, penalization-parameter tuner |

### 3.4 Where physics lives: the four homes

Every piece of physics lands in exactly one of four places. "Is this a `Term`?" is
a question about the **numerical treatment**, not the physics — the same force is a
`Term` when discretized explicitly and a `StageHook` when discretized implicitly.

Decision rule (ask in order):

1. **Additive — `du/dt += f(u)`?** No → **Integrator** (if it's a solve / constraint
   / stiff split) or **StageHook** (if it's `u ← g(u)`: filter, limiter, projection).
2. **Self-contained, i.e. doesn't need joint flux design with the base hyperbolic
   operator?** No → it belongs in **Equations** (entropy-stable split-form needs the
   volume + surface flux designed together; an independent additive surface term
   would break that).
3. **Genuinely optional / swappable?** No (regime-defining, always present) → fold
   into **Equations**; Term-ifying adds indirection and lets users build nonsense
   (NS with no convection) for zero flexibility gain.
4. **Would fine-graining force extra full-data passes?** Yes → merge into a coarser
   **Term** (the `dyn` call is free; the un-fused sweep is not).

Survives all four → it's a `Term`.

| Physics | Home | Why |
|---|---|---|
| Convection (hyperbolic flux) | Equations | Regime-defining; needs joint EC-flux coupling (fails #2, #3) |
| Viscous diffusion | Equations (LDG/BR1) or Integrator (incompressible Helmholtz) | Own interface treatment, or implicit (fails #1/#2) |
| Polymer stress ∇·τ_p | **Term** | Additive momentum source; optional; swappable model; volume-local |
| Buoyancy / body force | **Term** | Pure local additive source; optional |
| IBM forcing | **Term** (explicit χ/η·(u−uₛ)) or StageHook (implicit Brinkman) | Depends on discretization — the form we chose is implicit → StageHook |
| Pressure projection | Integrator (DualSplitting) | Constraint solve, not additive (fails #1) |
| SVV filter | StageHook | `u ← Fu` between stages (fails #1) |
| Conformation transport | Term, or own sub-system w/ IMEX | Judgment call: treat stiff relaxation explicitly (Term) or implicitly (Integrator) |

**Trade-offs of the `Term` boundary.** Making something a `Term` buys swappability,
zero operator-splitting error (all terms summed and advanced simultaneously),
integrator portability, and free Python exposure. It costs the ability to express
anything non-additive (those go to Integrator/StageHook), risks breaking
jointly-designed flux stability if you over-split the hyperbolic operator, and adds
one un-fused data pass per term — so prefer coarse terms.

### 3.5 Triggers (HOOMD TriggeredOperation, verified 3-0)

Updaters, Writers, and Tuners are wrapped in `Triggered<_>` — *what to do* is
decoupled from *when to do it*:

```rust
pub trait Trigger { fn fires(&self, step: u64) -> bool; }
pub struct Periodic { pub period: u64, pub phase: u64 }
pub struct OnStep   { pub step: u64 }
pub struct When<F>  { pub cond: F }   // condition-based, e.g. when max-Wi exceeds threshold

pub struct Triggered<T: ?Sized> { pub trigger: Box<dyn Trigger>, pub op: Box<T> }
```

AMR cadence, checkpoint interval, and diagnostics output all become triggers
rather than hardcoded `if step % n == 0` branches.

### 3.6 Device / backend (resolves open question; not externally evidenced)

`Device` is the single execution switch, owned by `Simulation`, never referenced by
`Term`/`Equations`/`Updater` code (no backend leakage into physics):

```rust
pub enum Device {
    Cpu,
    Cuda { ordinal: u32 },
    MultiGpu { ordinals: Vec<u32>, partition: Partition },
}
```

Mechanism: every `Term`/`Integrator` operation has a CPU element-loop and a
cuda-oxide kernel that are validated bit-for-bit (gale's existing discipline). The
`Device` selects which executes and owns the field mirrors + halo exchange. The
physics traits are written once; the backend dispatch happens at the `accumulate` /
`step` boundary, below the user-facing API. Multi-GPU halo exchange
(`memcpy_peer_async`, already in our cuda-oxide fork) is driven by the `Device`,
invisible to Term authors. CPU↔GPU parity remains the correctness oracle.

---

## 4. What assembling a simulation looks like

### 4.1 Rust — viscoelastic particle-laden channel

```rust
use gale::prelude::*;

// 1. Mesh + state
let mesh = Mesh::channel_x(/*order*/ 4, /*nx*/ 64, /*ny*/ 16, [0.0, 8.0], [-1.0, 1.0]);
let mut state = State::new(mesh);
state.add_field("velocity", 2);
state.add_field("pressure", 1);
state.add_field("psi", 3);                       // log-conformation tensor (symmetric 2x2)

// 2. Physics: incompressible NS with additive polymer-stress coupling + IBM forcing
let semi = Semidiscretization::builder()
    .equations(IncompressibleNs { reynolds: 1.0 })
    .solver(DgSem::order(4).with_svv())
    .boundary(BoundaryConditions::no_slip_walls().inflow_x(parabolic))
    .term(PolymerStress::log_conf(OldroydB { lambda: 2.0, eta_p: 0.5 }))  // ∇·τ_p coupling
    .term(ImmersedForcing::volume_penalization(eta_b))
    .build();

// 3. Integrator: structured dual-splitting (incompressible), with per-stage IBM projection
let integrator = DualSplitting::new(dt).with_stage_hook(VolumePenalizationProjection);

// 4. Assemble simulation on two GPUs
let mut sim = Simulation::new(state, Device::MultiGpu {
    ordinals: vec![0, 1],
    partition: Partition::stripes_x(),
});
sim.set_integrator(integrator);

// 5. Suspended deformable particles as bodies + an advecting updater
for c in seed_centroids() { sim.state.bodies.push(ImmersedBody::capsule(c, a, b)); }
sim.add_updater(BodyAdvect::default(), Periodic { period: 1, phase: 0 });

// 6. AMR on a smoothness indicator, every 10 steps
sim.add_updater(
    AmrUpdater::new(SmoothnessIndicator::persson_peraire()).max_level(3),
    Periodic { period: 10, phase: 0 },
);

// 7. Diagnostics + output
sim.add_compute("drag", DragOnBodies);
sim.add_writer(Hdf5Writer::new("out.h5"), Periodic { period: 50, phase: 0 });
sim.add_writer(ConsoleProgress::default(), Periodic { period: 100, phase: 0 });

// 8. Run
sim.run(10_000)?;
```

### 4.2 Python — the *same* objects, mechanically mirrored

```python
import gale

mesh = gale.Mesh.channel_x(order=4, nx=64, ny=16, xr=(0.0, 8.0), yr=(-1.0, 1.0))
state = gale.State(mesh)
state.add_field("velocity", 2); state.add_field("pressure", 1); state.add_field("psi", 3)

semi = (gale.Semidiscretization.builder()
    .equations(gale.IncompressibleNs(reynolds=1.0))
    .solver(gale.DgSem(order=4, svv=True))
    .boundary(gale.BoundaryConditions.no_slip_walls().inflow_x(parabolic))
    .term(gale.PolymerStress.log_conf(gale.OldroydB(lambda_=2.0, eta_p=0.5)))
    .term(gale.ImmersedForcing.volume_penalization(eta_b))
    .build())

sim = gale.Simulation(state, device=gale.Device.multi_gpu([0, 1], gale.Partition.stripes_x()))
sim.set_integrator(gale.DualSplitting(dt, stage_hook=gale.VolumePenalizationProjection()))
sim.add_updater(gale.AmrUpdater(gale.SmoothnessIndicator.persson_peraire(), max_level=3),
                trigger=gale.Periodic(period=10))
sim.add_writer(gale.Hdf5Writer("out.h5"), trigger=gale.Periodic(period=50))
sim.run(10_000)
```

The Python script contains **no simulation logic** — every call constructs or wires
a Rust object. The loop, the rhs, the kernels, the schedule all execute in Rust.

---

## 5. Rust ↔ Python parity strategy

Mirror HOOMD's pybind11 layering (verified 3-0), in pyo3 terms:

- **One `#[pyclass]` per core type** (`Simulation`, `State`, `Semidiscretization`,
  every `Term`/`Integrator`/`Updater`/`Writer`/`Trigger`/`Device`). The wrapper
  holds the Rust object; methods delegate directly. No behavior is added in Python.
- **Builders cross the boundary as methods.** `Semidiscretization.builder().term(...)`
  works identically; each builder method takes already-wrapped Rust objects.
- **Polymorphic collections.** Python appends `#[pyclass]` trait-object wrappers
  (e.g. a `PyTerm` holding `Box<dyn Term>`) into the same `Vec<Box<dyn Term>>` the
  Rust API uses — HOOMD's `SyncedList` analog.
- **Zero-copy fields.** `State.field(...)` exposes DG field arrays as `numpy`
  views over the host `ndarray` (via `rust-numpy`), no copy on the hot path;
  `snapshot()` is the explicit owned-copy path for checkpointing.
- **Custom operations from Python** follow HOOMD's Action pattern (verified 3-0): a
  Python object implementing `update(state, step)` is wrapped as a `Box<dyn Updater>`
  that calls back into Python. Accepts the GIL cost because custom Python operations
  fire on a trigger (not every DOF), consistent with the "no logic in the hot path"
  rule. Native operations never touch the GIL.
- **GIL during `run()`.** The long native `run(n_steps)` releases the GIL
  (`Python::allow_threads`) except when invoking a Python-defined callback.

**Parity discipline:** the binding crate (`gale-py`) contains *only* `#[pyclass]`
wrappers and signature glue. A test asserts every public Rust constructor/builder
method has a Python counterpart, so the two cannot silently drift.

---

## 6. Migration path (bespoke kernels → framework)

The validated kernels stay the source of truth; we wrap, not rewrite.

1. **`State` + `FieldSet`.** Introduce the named-field State. Provide adapters that
   view today's `[Vec<f64>; 3]` / `Vec<Vec<f64>>` layouts as fields — no kernel
   changes.
2. **`Integrator` trait + `SspRk3`.** Extract the existing `step_ssp_rk3` bodies
   (hyperbolic, viscoelastic) into one reusable `SspRk3: Integrator`. Validate it
   reproduces current results bit-for-bit on the existing test problems.
3. **`Semidiscretization` + `Term`.** Wrap `Hyperbolic::rhs` as the base rhs; wrap
   `ViscoelasticFlow` stress divergence and `VolumePenalization::apply` as `Term`s.
   Each wrapped term re-runs its existing validation test through the new path.
4. **`DualSplitting: Integrator`.** Move the incompressible dual-splitting
   orchestration behind the integrator trait, with the IBM projection as a
   `StageHook`. Validate against the existing channel / Stokes tests.
5. **Operations + Triggers.** Wrap AMR `adapt_scalar`/`remap_scalar` as
   `AmrUpdater`, membrane/particle motion as `BodyAdvect`, and add `VtkWriter`.
6. **`Device` + multi-GPU.** Route the existing CPU/GPU/`memcpy_peer_async` paths
   through `Device`, preserving the bit-for-bit CPU oracle at each step.
7. **`gale-py`.** Add the pyo3 crate last, once the Rust object model is stable.

Each step is one validated increment (gale's existing discipline): the framework is
correct iff every migrated piece still passes the test that validated the bespoke
version.

---

## 7. Key risks & how the design addresses them

| Risk | Mitigation |
|------|------------|
| **Dynamic dispatch in hot loops** | Dispatch only at the `Term::accumulate` / `Integrator::step` boundary — O(#terms) virtual calls per step, each fanning into a monomorphized loop/kernel over all DOFs. Generic-tuple `Terms` available for fixed hot compositions. |
| **State typing across regimes** | One `State` with a named, typed `FieldSet` registry instead of per-regime State structs; regimes differ by which fields/terms are registered. |
| **Rust/Python drift** | `gale-py` is wrappers-only; an automated test asserts API surface parity; no logic in Python by construction. |
| **Structured vs additive schemes in one API** | Both are `Integrator` implementors; structured schemes orchestrate stages internally while reusing the same `Term`s; `StageHook` handles per-stage limiters/projection. |
| **Backend leakage into physics** | `Device` owned by `Simulation`; physics traits are backend-agnostic; CPU↔GPU bit-for-bit parity is the correctness oracle. |
| **Unverified design areas** (dispatch cost, device mechanics, pyo3 sync) | Flagged explicitly; resolved on first principles here and to be confirmed by gale's own benchmarks/tests, not assumed. |

---

## 8. Open questions for review

1. **Term granularity for incompressible NS.** Should pressure/viscous solves be
   `Term`s or live entirely inside `DualSplitting`? Proposal: keep them *inside* the
   integrator (they are not additive rhs contributions); only explicit
   contributions (convection, polymer stress, IBM forcing, buoyancy) are `Term`s.
2. **`Equations` vs `Term` boundary.** Today `IncompressibleConvection` is a
   `ConservationLaw`. Should convection be the `Equations` flux or a `Term`?
   Proposal: the hyperbolic flux stays in `Equations`; `Term`s are the *additional*
   physics summed onto the base rhs.
3. **Generic vs trait-object default.** Default to `Vec<Box<dyn Term>>` for
   ergonomics/Python, or default to generic tuples and offer boxing as opt-in?
   Proposal: trait-object default; benchmark before reconsidering.
4. **Field naming vs typed accessors.** String-named fields (flexible, Python-easy)
   vs typed field handles (compile-time safe). Proposal: string names at the API
   surface with cached integer handles internally.

---

## 9. Relationship to existing frameworks (prior art)

Honest positioning: **the ingredients are all borrowed on purpose; the combination
and its packaging are what's novel — and the most novel parts are the riskiest
ones.** Novelty is not the goal — reusing proven patterns where they exist is a
strength. This section records where gale sits so the design's originality (and its
risk) is legible.

### What is *not* novel (deliberately)

| Abstraction | Borrowed from |
|---|---|
| Simulation / State / Operations / Triggers | HOOMD-blue (also LAMMPS, ESPResSo in spirit) |
| Semidiscretization + method-of-lines + step/stage callbacks | Trixi.jl / SciML |
| Additive terms summed into an rhs | Trixi `source_terms`, OpenFOAM `fvm`/`fvc` algebra, deal.II |
| Backend/device as a config switch | PyFR backends, libParanumal/OCCA, HOOMD `Device` |
| Rust core + thin pyo3 mirror | polars, tokenizers, pydantic-core (data tooling, not simulation) |

If any of these were novel it would be a warning sign — it would mean inventing
where a battle-tested pattern already exists.

### Mildly novel: the synthesis

Putting HOOMD's role-structured Operations model (Compute/Updater/Writer/Tuner +
Triggers) **on top of** a Trixi-style DG semidiscretization. The two come from
disjoint worlds — HOOMD is particle MD with no PDE/DG; Trixi is PDE/DG with
callbacks but not HOOMD's operation-role taxonomy. The marriage is sensible but
uncommon. This is synthesis, not a research contribution.

### Moderately distinctive: one Integrator trait across paradigms

Most CFD codes pick one lane:

- **Trixi / SciML** — method-of-lines + explicit/IMEX; incompressible projection
  isn't the idiom.
- **OpenFOAM** — structured PIMPLE/SIMPLE solvers hand-coded per application; no
  unified integrator abstraction with composable terms feeding both explicit and
  implicit paths.
- **PyFR** — compressible FR, explicit + dual-time; incompressible via artificial
  compressibility.
- **Dedalus** — spectral IMEX driven by symbolic equation strings; a different
  paradigm.

gale unifies "explicit SSP-RK consumes the additive rhs" **and**
"dual-splitting / IMEX orchestrates its own stages while reusing the same `Term`s"
under one `Integrator` trait, with the four-homes discipline (§3.4) surfaced as an
explicit API contract. The novelty is the *uniformity and discipline*, not a deep
idea — "structured schemes are just Integrator implementors" is natural once stated.

### Genuinely novel: the point in the design space

> A **Rust-first** DG-SEM CFD framework, with native-Rust GPU kernels (cuda-oxide),
> a HOOMD-style composable API, and a **zero-logic-divergence pyo3 Python mirror**,
> targeting viscoelastic particle-laden suspensions with IBM + AMR + multi-GPU.

No code occupies that point. The DG/CFD field is C++ (deal.II, MFEM, Nektar++,
SU2), Python+codegen (PyFR), or Julia (Trixi); there is no mature Rust DG fluid
solver. The "Rust core / thin pyo3 binding / zero logic in Python" pattern is proven
only in data tooling, never (to our knowledge) for a PDE simulation framework. And
cuda-oxide native device kernels for DG are essentially unexplored, the toolchain
being new.

Codes that *are* already strongly composable — so we are not claiming composability
itself is new — include **AMReX** (block-structured AMR + composable operators,
C++), **deal.II / MFEM** (composable FEM assembly), and **FEniCS / Firedrake**
(UFL symbolic forms, an arguably more general composition model). What is new is
*this* composition model, in *this* language, with *that* Python-parity guarantee.

### Novelty and risk are correlated

The borrowed parts are safe because they are proven. The novel parts — the unified
Integrator, coarse-grained `dyn Term` dispatch in GPU hot loops, the cuda-oxide
substrate, and the zero-divergence pyo3 layer — are exactly the four areas the
research pass found **no external evidence for** (see caveats below). For a
research-grade proof of concept that is acceptable and expected; it is why those
four are the parts to validate by our own benchmarks early rather than assume.

## Sources & evidence

Deep-research pass (2026-06; 6 angles, 28 sources fetched, 130 claims extracted,
25 adversarially verified at 3-vote, 24 confirmed / 1 refuted). All surviving
claims are from **primary** HOOMD-blue and Trixi.jl documentation.

Confirmed and load-bearing here:

- Simulation owns State + Operations + Device (3-0) — HOOMD `simulation.html`,
  `module-hoomd-operation.html`.
- Five operation roles; per-step order tuners→updaters→integrator→writers (3-0) —
  HOOMD `ARCHITECTURE.md`. Seven-role taxonomy **refuted** (0-3).
- Triggers as a separate firing-condition abstraction (3-0) — HOOMD operation module.
- Additive net-force composition (3-0) — HOOMD `md/integrator.html`, `ARCHITECTURE.md`.
- Methods (equations of motion) separated from forces/terms (3-0) — HOOMD integrator.
- `SemidiscretizationHyperbolic` bundling mesh+equations+solver+conditions+sources
  → ODE rhs (3-0) — Trixi `overview`, `callbacks`.
- Ordered `CallbackSet`; step vs stage granularity; shared callback interface
  across integrators (3-0) — Trixi `callbacks.md`.
- State as single mutable source of truth via snapshot get/set (2-1 / 3-0) — HOOMD.
- Action extension pattern; pybind11 1:1 Python mirror (3-0) — HOOMD tutorials,
  `ARCHITECTURE.md`.

**Scope gap (validate before relying):** PyFR, Dedalus, deal.II/MFEM, SU2/OpenFOAM,
and the Rust+Python interop exemplars (polars, tokenizers, tch-rs, rust-numpy)
produced **no surviving verified claims** in this pass — the dispatch-cost, device
abstraction, type-state builder, and zero-copy/GIL recommendations above are gale
design decisions, not externally evidenced findings. Confirm by our own benchmarks.
