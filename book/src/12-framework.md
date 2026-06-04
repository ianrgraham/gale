# The Simulation Framework and Validation

> 🎓 **Reviewer — chapter verdict:** The validation half of this chapter is excellent and the "CPU is the oracle" discipline is conveyed as the genuinely smart move it is — the "silently, with a plausible-looking wrong number" paragraph is the emotional core and lands hard. My one structural worry: the chapter front-loads a lot of framework taxonomy (integrator families, four-homes rule, hooks vs computes) before it gets to the validation payoff, and a blog reader's attention is finite. The "two integrator families" section is the most interesting framework idea and is well-argued (the "forcing a projection into method-of-lines would be a lie" line is great); the "four-homes rule" is the most at-risk of reading as bookkeeping. Consider whether the chapter is really *two* posts — "how you assemble a sim" and "how we know it's right" — because the second is strong enough to stand alone and is currently buried below the fold.

The last chapter explained how gale's operators run on the GPU. This one is about the
layer *above* the operators — the part a user actually touches. How do you take the
pieces this book has built — a hyperbolic operator, an elliptic solver, a projection
scheme, a polymer model, an immersed body — and *assemble* them into a running
simulation? And, given that some of those pieces are hand-written GPU kernels on a
young toolchain, how do you convince yourself the whole thing is **correct**?

Those two questions — composition and correctness — are the subject here. The
correctness half is the engineering counterpart to the numerical-stability throughline
of the rest of the book: where earlier chapters asked "what could blow up?", this one
asks "how do we know this kernel computes the right thing?"

## The user-facing design: assemble, then run

gale's framework borrows its shape from two mature codes in adjacent fields:
**HOOMD-blue** (molecular dynamics) for the *orchestration* model, and **Trixi.jl**
(PDE/DG in Julia) for the *PDE seam*. The synthesis is gale's own; the design
rationale lives in `docs/api-design.md`.

The central object is a `Simulation`. It owns:

- a **`State`** — the single source of truth: the mesh (topology + geometry, and the
  AMR tree of Chapter 10) plus a named, multi-component **`FieldSet`**. A field is
  just `{ name, n_components, values }` in the element-major layout the operators
  consume — `velocity` (2 or 3 components), `pressure` (1), `conformation`/`Ψ`
  (3 in 2D). The same `State` type spans every regime; an incompressible run and a
  viscoelastic run differ only by which fields are registered, not by a different
  `State` struct.
- exactly one **integrator** — the sole authority that advances time.
- optional **stage hooks** — per-stage corrections (filters, projections).
- optional **computes** (read-only diagnostics like drag) and **updaters/writers**
  gated by **triggers** (fire every *n* steps, etc.).

The goal of the whole design is captured in one sentence: **assemble a simulation
declaratively, and run the *same* assembly on CPU or GPU.** You build a `State`, add
fields, set an integrator, attach a hook and a diagnostic, and call `sim.run(n)`. The
per-step schedule is deterministic (updaters → integrator → writers), and a GPU run
is wired with the identical API as a CPU run — the device is a switch, not a
rewrite.

## Two integrator families, and why both are needed

Here is the most interesting design decision in the framework, and it is forced by
the physics this book covers.

The obvious way to advance a PDE in time is the **method of lines**: write the
spatial discretization as an ODE \\\( \partial_t u = \mathrm{rhs}(u) \\\), then march it
with a standard time-stepper. This is exactly right for the hyperbolic/transport
operators of Chapter 4. In gale a `StateSemi` (semidiscretization) defines that
right-hand side — a base operator per evolving field, plus additive cross-field
**`StateTerm`s** that *accumulate* into it (the polymer-stress coupling reading the
conformation field and forcing momentum is one such term). A generic SSP-RK3
integrator, `Mol`, consumes that rhs and marches it with the strong-stability-
preserving Runge–Kutta scheme of Chapter 4. Additive, composable, and the natural
fit for transport.

But the incompressible solver of Chapter 6 **does not fit the additive mold.** Its
dual-splitting projection scheme is not "sum up some right-hand-side contributions
and step." It is a structured sequence of *stages* that depend on each other: an
explicit convection/body-force substep, then a pressure-Poisson *solve* to enforce
incompressibility, then an implicit viscous Helmholtz *solve*. A pressure projection
is a constraint solve, not a term you can add to \\\( \partial_t u \\\). Trying to force
it into the method-of-lines shape would be a lie.

> 🎓 **Reviewer:** This is the right call and well-argued — "a pressure projection is a constraint solve, not a term you can add to ∂ₜu" is the sentence a reader will remember, and refusing to fake an additive interface for it is exactly the kind of honest design choice worth dwelling on. Praise noted; don't touch it. (If you want one more drop of intuition: the deeper reason is that projection methods are *fractional-step* — they deliberately split the operator and accept a splitting error in exchange for turning one hard saddle-point system into a sequence of nice SPD solves. Method-of-lines integrators assume the rhs *is* the dynamics; a fractional step is a sequence that is only the dynamics *after* all the stages compose. That's why it can't be a `StateSemi`.)

So gale uses **one trait, multiple families** (the "one `Integrator` trait" idea from
`api-design.md`). Both `Mol` and the structured `DualSplitting` implement the same
`StateIntegrator` trait — `dt()` and `step(state, hook)` — but they honor it
differently:

- `Mol` holds a `StateSemi` and consumes its additive rhs.
- `DualSplitting` holds its own configuration and **orchestrates its own stages**
  internally, calling the validated `Stokes` operator. There is no additive
  semidiscretization; the scheme *is* the integrator. (`ViscoelasticDualSplitting`
  is a third family: a tightly-coupled velocity+conformation advance in the correct
  split order.)

The payoff of unifying them under one trait is that everything *built on top* of the
trait — the `Simulation` loop, the stage hooks, the diagnostics — is written once and
works for either family. The `Simulation` drives a transport problem and a
projection-based incompressible problem with the same `run()` loop, because both
integrators look identical from the outside.

## Hooks and computes: the "four homes" rule

Where does a given piece of physics *live*? `api-design.md` answers this with the
**four-homes rule**: every contribution lands in exactly one of {Equations, Term,
Integrator, StageHook}, and which one is a question about the *numerical treatment*,
not the physics.

The decision rule, simplified: if a contribution is additive
(\\\( \partial_t u \mathrel{+}= f(u) \\\)) it is a **Term**. If it is a solve or a stiff
split, it belongs to the **Integrator**. And if it is a relaxation of the form
\\\( u \leftarrow g(u) \\\) — applied *to* the solution rather than added to its
derivative — it is a **StageHook**, run between RK stages.

The implicit **volume-penalization** projection of Chapter 9 is the textbook
StageHook. Brinkman penalization drives the velocity inside an immersed solid toward
the body's velocity by the relaxation \\\( u \leftarrow (u + \beta\,u_s)/(1+\beta) \\\).
That is `u ← g(u)`, not an additive rhs term — so by the four-homes rule it is a
`StateStageHook`, applied after each integrator stage, **not** a `Term`. gale's
`PenalizationHook` is exactly that, wrapping the validated `VolumePenalization`
operator unchanged. (The spectral-vanishing-viscosity filter of Chapter 4 is the
other canonical hook: a linear `u ← Fu` that is mass-conserving, hence a hook and not
a term.)

A **`Compute`** is the read-only counterpart: a diagnostic pulled from the state on
demand, never mutating it — the hydrodynamic-drag report `F = ∫ (χ/η_b)(u − u_s)`
that goes with the penalization, kinetic energy, max-Weissenberg, and so on.

> 🎓 **Reviewer (rewrite):** The section *says* "this taxonomy is not bureaucracy" — which is a tell that you're worried it reads like bureaucracy, and it slightly does. The fix isn't to argue harder, it's to lead with the payoff instead of the rule. Right now the structure is: (1) here is a four-way classification, (2) here is the decision rule, (3) here are two examples, (4) trust me it's not bureaucracy. Invert it: open with the concrete tension — "a filter and an immersed-solid projection are completely different physics, yet in gale they are *the same kind of object*. Here's the rule that tells you why." Then the four homes arrive as the explanation for a surprise the reader already feels, rather than a taxonomy they have to hold in their head before they know what it buys. Same content, momentum restored.

This taxonomy is not bureaucracy. It is what lets the framework stay composable: a
filter and a penalization projection are *different physics* but the *same kind of
operation* (a per-stage `u ← g(u)`), so they plug into the same `StageHook` seam, and
any integrator picks them up uniformly.

## How the GPU plugs into the same seams

This is the payoff of the whole design, and it is worth stating sharply.

**A GPU operator implements the same traits as its CPU counterpart.** Not a parallel
GPU-specific API — the *identical* trait:

- `GpuAdvection` is a `StateSemi` — it produces the same rhs the CPU semidiscretization
  does, just computed on the device.
- `GpuDualSplitting` (and its 3D sibling) is a `StateIntegrator` — it orchestrates the
  projection stages on the GPU but presents the same `step(state, hook)` interface.
- `GpuPenalizationHook` is a `StateStageHook` — the IBM relaxation, on the device.

Because the seams are trait objects, a GPU run is assembled with the **identical
`Simulation` API** as a CPU run. You set a GPU integrator instead of a CPU one and
attach a GPU hook instead of a CPU one; the `run()` loop, the schedule, the triggers,
the computes are all unchanged. The device is genuinely a switch below the
user-facing API — exactly the "same assembly on CPU or GPU" goal stated at the top.
gale's tests assemble flow-past-a-sphere in 3D through this API with a structured GPU
integrator and a GPU IBM hook, the same way the CPU version is assembled.

## Validation philosophy: the CPU is the oracle

Now the correctness question. gale is hand-written GPU kernels on a v0.1 toolchain.
How is that trustworthy at all?

The answer is a discipline, stated as a principle in Chapter 1 and enforced
everywhere: **the pure-host `gale` library is the oracle.** Every operator exists
first as a clean, `std`-only CPU implementation whose correctness is established by
ordinary means — unit tests, manufactured solutions, analytic cases. The GPU kernel
is then validated *against that oracle*, not against the physics directly.

The mechanism is a fleet of small `*-check` binaries in `gale-gpu/src/bin/` — one per
operator (`advection_check`, `poisson_cg_check`, `euler_check`, `logconf_check`,
`penalize_check`, `ns_check`, `ve_check`, and so on). Each one builds the same problem
two ways, runs the CPU oracle and the GPU kernel, and compares them **bit-for-bit**,
demanding agreement to roughly \\\( 10^{-14} \\\) relative — machine precision, not a
loose tolerance. The advection check is representative: it constructs a mesh, runs
`gale::dg::Hyperbolic` on the CPU and `gale_gpu::advection_rhs` on the device, and
asserts `max|gpu − cpu| / |op| < 1e-10`, typically landing near \\\( 4\times10^{-16} \\\).

> 🎓 **Reviewer (flag):** Be careful with "bit-for-bit" as the headline phrase when the actual assertion is `< 1e-10` and the achieved number is ~4e-16. Those are three different claims and the prose uses all three interchangeably. *Bit-for-bit* means identical bit patterns — zero difference — and you do say "several exact (zero difference)" later, so the term is earned for *some* kernels but not the advection one you hold up as representative, which agrees to round-off, not to the bit. This matters because a GPU FMA (fused multiply-add) contracts `a*b+c` into one rounding where the CPU may do two, so a *correct* port routinely differs in the last bit or two — that's the 4e-16, and it is not a bug, it's the FP standard. I'd state this explicitly: most checks are "agree to floating-point round-off (≈1e-15, the FMA-level difference you'd expect from a correct port)," and a *subset* that avoid fused/reordered arithmetic are genuinely bit-identical. Right now the chapter slightly over-claims "bit-for-bit" across the board, and a hardware-literate reader will catch it — better to own the FMA story, which is more impressive anyway because it shows you know exactly *why* the last bit moves.
This is why the *deterministic, race-free* gather formulation of Chapter 11 matters
so much: a racy kernel could never be pinned to an oracle this tightly. Across gale's
operator set these checks come in at \\\( 10^{-14} \\\) to \\\( 10^{-16} \\\), with several
*exact* (zero difference).

Three complementary layers sit around the bit-for-bit checks:

- **Manufactured solutions (MMS)** check the *physics*, not just CPU↔GPU agreement.
  You pick an exact solution, derive the source term that makes it satisfy the
  equations, and confirm the solver recovers it to discretization accuracy — gale's
  cross-field-coupling and source-term tests do exactly this, including a control run
  *without* the term to prove the term is both correct and necessary.
- **Analytic cases** check whole assembled solvers: the decaying Taylor–Green vortex
  (an exact Navier–Stokes solution) for the incompressible path, and the Poiseuille
  channel for viscoelastic flow. The framework integrators are additionally checked
  **bit-for-bit against a direct loop of the underlying validated operator**, so
  wrapping an operator in the `Simulation` API provably does not perturb its
  dynamics.
- **`probe-*` binaries** pin down *toolchain* capabilities rather than physics —
  `probe_fp64_math`, `probe_nested_write`, `probe_sm70`. These are the regression
  gates for the cuda-oxide bugs of Chapter 11: each reproduces a specific toolchain
  hazard so that a fork update which silently reintroduces it is caught immediately.

Why this much discipline? Because a young GPU stack fails in the worst possible way:
**silently, with a plausible-looking wrong number.** A dropped write, a pointer
bitcast omitted, an off-by-one in a flux index — none of these throw an error; they
just shift the answer. The only defense is an independent, trusted reference and an
unforgiving comparison. The CPU oracle is that reference, and the bit-for-bit check is
> 🎓 **Reviewer (deepen):** This paragraph — "fails silently, with a plausible-looking wrong number" — is the heart of the whole chapter and the single best justification for the oracle discipline. Keep it. Worth adding the practitioner punchline that makes it bite even harder: the reason the CPU oracle is *trustworthy* as a reference isn't that it's correct in some absolute sense — it's that the CPU and GPU implementations fail *independently*. An off-by-one in a flux index would have to be made identically in two separately-written codebases to escape detection, and that essentially never happens. That's the real engineering content of "independent reference": not that the oracle is right, but that two implementations are unlikely to be wrong the *same way*. Say that and the discipline stops sounding like belt-and-suspenders and starts sounding like the only thing that actually works.

that comparison. This is the engineering analogue of the numerical-stability care
elsewhere in the book: there, the worry was that an accurate-on-paper scheme produces
`NaN` on a real problem; here, the worry is that a fast GPU kernel produces a number
that is wrong by \\\( 10^{-3} \\\) and nobody notices. Both are defeated by asking, at
every step, *what could be wrong here, and what are we doing to catch it?*

## How gale does it

- **`src/sim/`** holds the framework, mirroring this chapter: `simulation.rs` (the
  `Simulation` orchestrator, operations, triggers), `state.rs` and `field.rs` (the
  `State` and named `FieldSet`), `integrate.rs` and `dynamics.rs` (the `Semi`/
  `StateSemi` seam, the `Integrator`/`StateIntegrator` trait, `Mol`/SSP-RK3, and the
  structured `DualSplitting` family), `stagehook.rs` (the SVV filter hook), `ibm.rs`
  (the penalization `StateStageHook` and drag `Compute`), and `term.rs` (additive
  terms).
- **The GPU integrators** in `gale-gpu` implement the *same* traits: `GpuAdvection`
  (`StateSemi`), `GpuDualSplitting` / `GpuDualSplitting3d` (`StateIntegrator`),
  `GpuPenalizationHook` (`StateStageHook`) — so a GPU simulation is assembled with the
  identical `Simulation` API.
- **The `*-check` binaries** in `gale-gpu/src/bin/` are the validation fleet, one per
  operator, each comparing the kernel to the CPU oracle bit-for-bit; the `probe-*`
  binaries are the toolchain regression gates.

With the architecture and the framework in place, the final chapter steps back to the
score: what is built and validated today, what is planned, and the open problems on
the road to the headline application.
