# Embedded-boundary methods — landscape & fallback plan

**Purpose.** One place that (1) catalogs every immersed/embedded-boundary method family we'd
consider, and (2) records the **fallbacks to reach for if SBM hits trouble** as we push it into
the uncharted work — *freely-moving bodies, many-body suspensions, and viscoelastic surface
stresses*. We're committed to SBM for now (validated, fast, sharp); this is the backup map so a
switch is a deliberate, pre-scoped decision rather than a scramble.

Detail lives in the deep-dive docs; this is the index + decision matrix:
- `docs/research-sharp-interface.md` — verified deep research: cut-cell DG vs SBM, risk register.
- `docs/immersed-boundary-strategy.md` — broad IBM taxonomy + viscoelastic-IB specifics (§A–C).
- `docs/research-moving-particle-coupling.md` — moving/coupled-particle research.
- `docs/sbm-status.md` — SBM implementation status (CPU + GPU; cylinder beats penalization).

---

## Method catalog

Honest framing (see also the discussion that prompted this doc): the sharp-embedded-boundary
frontier has **several** active members — SBM is one of them, not the sole apex. CutFEM and
cut-cell DG are at least as prominent. SBM's edge for us is *computational* (no cut-cell
quadrature, reuses standard machinery → high-order + GPU friendly).

| # | Method | What it does | Order | Strengths | Weaknesses | Fit for gale (DG-SEM + GPU + moving + VE) | Status in gale |
|---|--------|--------------|-------|-----------|------------|-------------------------------------------|----------------|
| 1 | **Volume penalization** (Brinkman) | Add `−(χ/η)(u−u_b)` body force in the solid | ~1st | trivial, robust, GPU-trivial, handles arbitrary/moving/many bodies with zero geometry work | diffuse interface; accuracy floor (~2% fixed cylinder, ~13% small moving disk — flow sees radius `r+√η`) | excellent robustness, poor accuracy | **DONE** (M1–M4: fixed→moving→strong-coupling→many-body, CPU+GPU). The always-available robust floor. |
| 2 | **Classical IBM** (Peskin direct-forcing, IB-LBM) | Regularized δ-function force spreading to a Lagrangian marker set | 1st–2nd | mature, great for thin/deformable membranes & biological FSI | interface smeared over δ-support; low order; spurious forces | low-order; we already have penalization for the diffuse niche | not used (penalization covers this niche) |
| 3 | **SBM** (Shifted Boundary Method) — **CHOSEN** | BC on a *surrogate* of whole uncut elements + Taylor correction (`S_h u = u+∇u·d`) via Nitsche | high (with Taylor) | no cut-cell quadrature, immune to small-cut-cell problem, reuses element quadrature ⇒ matrix-free + GPU friendly | surrogate "pops" as elements cross in/out under motion; needs ≥1 whole element between surfaces; force recovery needs care; single-source-ish for our workflow | best fit for our matrix-free DG-SEM/GPU stack | **DONE** CPU+GPU: cylinder C_D +1.0% (beats penalization 2.1%), full GPU solve ~40–75×/step |
| 4 | **Cut-cell DG + agglomeration** (XDG; Saye/Algoim or HMF quadrature, BoSSS) | True cut cells with level-set quadrature; merge tiny cells | optimal `h^{k+1}` | sharp surface quadrature (exact stress/force), conservative, **agglomeration freezes per-step topology under motion + builds MG** | cut-cell quadrature generation (root-find + linear solves) **not verified GPU-friendly** → likely host-side + upload/step | the **primary sharp backup**; heavier machinery, fights the CUDA-graph/matrix-free design | not built — the deferred high-accuracy target (`research-sharp-interface.md` §A) |
| 5 | **CutFEM / ghost-penalty unfitted FEM** (Burman–Hansbo–Massing) | Unfitted FEM + *ghost-penalty* term stabilizing cut cells | high | most mathematically mature: rigorous conditioning & conservation theory; the ghost-penalty idea is portable | rooted in continuous-Galerkin FEM (different philosophy from our DG-SEM); still needs cut quadrature | not a drop-in, **but its ghost-penalty stabilization is the borrowable fix** for cut-cell-DG conditioning | not built — borrow the *idea* if cut-cell conditioning bites |
| 6 | **Immersed-interface (IIM) / Nitsche-XFEM / ghost-fluid** | Modify stencils/enrich near the interface for jump conditions | varies | strong for sharp *coefficient/flux jumps* (two-phase) | not competitive with 3/4 for our single-domain no-slip + VE workflow | niche | noted only |

---

## Fallback decision matrix (the point of this doc)

For each upcoming challenge: the SBM plan, the **likely SBM failure mode** to watch for, and the
**backup to reach for + why**. Order of escalation is always: *SBM → a local SBM fix → cut-cell
DG*, with **penalization as the robust low-order floor that already works** for sanity/bring-up.

### A. Freely-moving / rotating single body
- **SBM plan:** recompute the surrogate set + shift vectors `d` each step (cheap, host); reuse
  vs rebuild the GPU MG handle (open question — `research-sharp-interface.md` risk #3).
- **Likely failure mode:** *surrogate popping* — as the body moves, elements flip active↔inactive,
  so the surrogate boundary jumps by a whole element ⇒ force/torque time series shows steps/noise;
  also MG-handle rebuild cost per step if topology changes.
- **Primary backup:** **cut-cell DG + agglomeration** — agglomeration was *designed* for this: it
  freezes per-timestep topology changes and absorbs new/vanished cells smoothly (§A of the
  research doc). **Secondary:** penalization (M1–M4 already handle freely-moving) as the robust
  low-order reference to bracket the answer.
- **Cheaper SBM-side mitigations to try first:** sub-element-accurate force recovery (already
  high-order via Hessian extrapolation), and temporal filtering / smaller `Δt` near pops.

### B. Many-body suspensions (near contact, lubrication)
- **SBM plan:** many surrogates; the existing active-mask/`ShiftedBoundary` is already per-element.
- **Likely failure mode:** **thin gaps** — when two particles approach within ~one element, there
  is *no whole element* left between them to host a surrogate ⇒ SBM breaks down in the gap; also
  unresolved lubrication forces.
- **Primary backup:** **cut-cell DG** (resolves arbitrarily thin cut cells in the gap) **+ local
  AMR** to refine the gap. **Secondary / complementary:** an explicit **lubrication-correction
  model** (sub-grid pairwise force) layered on *any* method — standard practice for dense
  suspensions and method-agnostic; pair with penalization for contact mechanics.
- **Mitigation first:** AMR around near-contact pairs so a whole element survives in the gap (keeps
  SBM valid longer) before resorting to cut cells.

### C. Viscoelastic surface stress (conformation-tensor BC on the embedded surface)
- **SBM plan:** impose the polymer-stress / conformation BC weakly on the surrogate with the same
  Taylor shift; this is the project's headline goal and the **least de-risked** combination
  (risk register: *no source shows projection + Nitsche + VE + moving together*).
- **Likely failure mode:** high-Wi boundary-layer steepness at the surface makes the Taylor
  extrapolation of the polymer stress inaccurate/ill-conditioned; the conformation tensor can lose
  SPD near the shifted boundary; the singular-pressure + Nitsche + VE composition is untested.
- **Primary backup:** **cut-cell DG** for *exact surface quadrature* of the stress BC (no
  extrapolation error at the true surface). **Complementary (do regardless of method):** the
  conformation-tensor-BC recipes + **log-conformation + AMR + bound-preserving limiter** from
  `immersed-boundary-strategy.md` §A–C — these are surface-treatment tools that help *any*
  embedded method at high Wi, and we already have log-conf, the SPD/bound-preserving limiter, and
  IMEX relaxation built.
- **Mitigation first:** AMR + log-conf at the surface; a higher-order Neumann/stress surrogate
  (the current pressure surrogate is only 1st order — `sbm-status.md`).

### D. General conditioning / robustness / cost
- **Likely failure mode:** the SBM-MG smoother is mediocre (~5× a clean p-MG's iters); Nitsche
  penalty `γ` tuning; ill-conditioning as surrogate distance `d` grows.
- **Backups / fixes (cheap → heavy):** Chebyshev (or block-Jacobi) smoother for the SBM-MG;
  CUDA-graph the SBM V-cycle (it's launch-bound); cap `‖d‖` (refine where the shift is large);
  borrow **ghost-penalty** stabilization (method #5) if we ever go cut-cell.

---

## Decision protocol
1. **Don't switch on first difficulty.** Try the SBM-side mitigation listed above, then AMR.
2. **Bracket with penalization.** It already works for moving/many-body (M1–M4); use it to confirm
   the *physics* while debugging the sharp method — divergence between them localizes the problem.
3. **Cut-cell DG is the one real "switch."** It's the high-accuracy sharp backup but a genuine
   investment (host-side quadrature, agglomeration, CUDA-graph re-evaluation). Commit to it only
   when an SBM failure mode above is *confirmed* (not anticipated) and AMR/mitigation didn't close
   it — and even then, agglomeration (#4) is what makes it viable for moving/many-body.
4. **Ghost-penalty (CutFEM) is an idea to borrow, not a port** — its stabilization term is the fix
   for cut-cell conditioning, not a separate codebase for us.
