# Appendix: Notation and Dimensionless Numbers

> 🎓 **Reviewer — chapter verdict:** Accurate reference overall — all four dimensionless groups check out dimensionally (including \\( \mathrm{El}=\lambda\nu/L^2 \\)) and the constitutive symbols are right. Two glossary one-liners are imprecise enough to mislead a newcomer (Deborah-vs-Weissenberg, and the HWNP "loss of positive-definiteness" framing), flagged below.

A reference for the symbols, dimensionless groups, and acronyms used throughout the book.

## Symbols

| Symbol | Meaning |
| --- | --- |
| \\( \mathbf{u}=(u,v,w) \\) | velocity field |
| \\( p \\) | pressure (also, in DG context, polynomial degree — disambiguated by context) |
| \\( t,\ \Delta t \\) | time, time step |
| \\( \rho \\) | density |
| \\( \mu,\ \eta \\) | dynamic viscosity (\\( \eta_s \\) solvent, \\( \eta_p \\) polymer) |
| \\( \nu = \mu/\rho \\) | kinematic viscosity |
| \\( \nabla,\ \nabla\cdot,\ \nabla^2=\Delta \\) | gradient, divergence, Laplacian |
| \\( \mathbf{C} \\) | conformation tensor (SPD), equilibrium \\( \mathbf{C}=\mathbf{I} \\) |
| \\( \Psi=\log\mathbf{C} \\) | log-conformation variable |
| \\( \tau_p \\) | polymer stress, \\( \tau_p=\frac{\eta_p}{\lambda}(\mathbf{C}-\mathbf{I}) \\) |
| \\( \lambda \\) | polymer relaxation time |
| \\( \overset{\triangledown}{\mathbf{C}} \\) | upper-convected derivative of \\( \mathbf{C} \\) |
| \\( h \\) | representative element size |
| \\( p \\) (degree) | polynomial degree per element (gale typically 4–8) |
| \\( \tau \\) | SIPG interior-penalty parameter, \\( \tau=\alpha(p+1)^2/h \\) |
| \\( \alpha \\) | penalty coefficient (user constant in \\( \tau \\)) |
| \\( M,\ A \\) | mass matrix (diagonal for GLL), stiffness/SIPG operator |
| \\( \mathrm{jw} \\) | nodal quadrature weight × Jacobian (the diagonal mass entries) |
| \\( \chi \\) | immersed-body mask (1 inside solid, 0 in fluid) |
| \\( \eta_b \\) | volume-penalization (Brinkman) parameter; small ⇒ strong no-slip |
| \\( [\![\cdot]\!],\ \{\!\!\{\cdot\}\!\!\} \\) | jump and average across an element face |
| \\( P,\ P^T \\) | mortar projection (coarse→fine) and its transpose (fine→coarse) |

## Dimensionless numbers

These set gale's regime and explain its method choices (Chapter 2).

- **Reynolds number** \\( \mathrm{Re}=\dfrac{UL}{\nu} \\): inertia vs. viscous forces.
  *Low* in microfluidics — inertia is weak, so convection can be treated explicitly and
  the flow is smooth (Chapter 6).

- **Weissenberg number** \\( \mathrm{Wi}=\lambda\dot{\gamma} \\): elastic relaxation time
  vs. flow deformation rate (\\( \dot\gamma \\) a shear rate). *Can be large* in gale's
  regime — strong elasticity, steep polymer-stress layers, and the High-Weissenberg
  problem (Chapter 8).

- **Deborah number** \\( \mathrm{De}=\lambda/T \\): relaxation time vs. a flow/observation
  timescale \\( T \\). Closely related to \\( \mathrm{Wi} \\); large \\( \mathrm{De} \\)
  means the fluid "remembers."

> 🎓 **Reviewer (deepen):** "Closely related to Wi" undersells a distinction worth one clause, because conflating them is the single most common viscoelasticity misconception. They coincide only in steady, homogeneous flows. The honest split: **Wi** measures the strength of the elastic response in a flow with a well-defined deformation rate (\\( \lambda\dot\gamma \\) — how stretched the polymers get), while **De** measures whether the flow is *transient relative to relaxation* (\\( \lambda/T \\) — whether the fluid has time to forget). A steady shear flow can have large Wi but \\( \mathrm{De}=0 \\) (nothing is changing, so there's no observation timescale to be fast); a small-amplitude oscillation can have large De but vanishing Wi. Recommend stating they coincide only when the flow timescale *is* \\( 1/\dot\gamma \\), so the reader doesn't treat them as synonyms.

- **Elasticity number** \\( \mathrm{El}=\mathrm{Wi}/\mathrm{Re}=\lambda\nu/L^2 \\):
  elasticity vs. inertia, independent of flow speed. *Large* in microfluidics — the
  signature of the elasticity-dominated, inertia-negligible corner gale targets.

## Acronyms and terms

- **DG** — discontinuous Galerkin: a finite-element method where the solution is a
  separate polynomial per element, coupled only through interface fluxes (Chapter 3).
- **SEM / DG-SEM** — spectral element method: high-order elements with a nodal basis on
  Gauss–Lobatto–Legendre points; DG-SEM is the discontinuous variant gale uses.
- **GLL** — Gauss–Lobatto–Legendre quadrature/nodes; collocating the basis on them makes
  the mass matrix diagonal (Chapter 5).
- **SIPG** — symmetric interior-penalty (discontinuous) Galerkin: gale's way of building
  a stable, symmetric second-order (Laplacian/Helmholtz) operator (Chapter 5).
- **Numerical flux** — the single-valued interface flux that couples DG elements;
  upwinding/Rusanov supplies the dissipation that keeps hyperbolic problems stable (Ch. 4).
- **Split form / KEP / entropy-stable** — flux-differencing volume formulations that
  discretely conserve a secondary quantity (kinetic energy / entropy), giving high-order
  robustness against aliasing (Chapter 4).
- **CG / p-multigrid / deflation** — conjugate gradient (for SPD systems), its
  degree-coarsening multigrid preconditioner, and the nullspace-removal needed for the
  singular pure-Neumann pressure solve (Chapter 7).
- **Dual-splitting / projection / fractional-step** — the Chorin–Karniadakis scheme that
  turns incompressible Navier–Stokes into a sequence of SPD elliptic solves (Chapter 6).
- **Helmholtz–Hodge projection** — the orthogonal projection of a vector field onto its
  divergence-free part; the pressure step enforces incompressibility this way (Chapter 6).
- **Conformation tensor / Oldroyd-B / UCD** — the SPD tensor tracking polymer
  microstructure, the canonical viscoelastic model, and the objective (upper-convected)
  time derivative it evolves under (Chapter 8).
- **HWNP** — High-Weissenberg-Number Problem: numerical loss of positive-definiteness of
  \\( \mathbf{C} \\) at strong elasticity, cured by the log-conformation representation
  (Chapter 8).

> 🎓 **Reviewer (flag):** This one-liner names the *symptom* as if it were the *cause*, which is the framing log-conformation's own authors pushed back on. Loss of SPD-ness of \\( \mathbf{C} \\) is what you *observe*; the underlying problem is that the exponential stress profiles in stress boundary/birefringent-strand layers (e.g. near stagnation points) are badly under-resolved by polynomial interpolation, and the resulting Gibbs-type oscillations drive \\( \mathbf{C} \\) negative, after which the constitutive equation is locally Hadamard-unstable and blows up. Log-conformation helps for two reasons, and the entry credits only the first: (1) evolving \\( \Psi=\log\mathbf{C} \\) and exponentiating *guarantees* \\( \mathbf{C}=e^\Psi \\) stays SPD by construction, and (2) the log variable turns those exponential layers into roughly linear ones that polynomials *can* resolve. Recommend: "numerical breakdown at high Wi — under-resolved exponential stress layers drive the (analytically SPD) conformation tensor to lose positive-definiteness, after which it blows up; log-conformation both guarantees SPD-ness by construction and linearizes the layers." Don't let the reference reduce HWNP to a positivity bug.
- **IBM / volume penalization (Brinkman)** — immersed-boundary method; representing a
  solid by penalizing the fluid toward the solid velocity inside a mask, applied
  implicitly for stability (Chapter 9).
- **AMR / hanging node / mortar** — adaptive mesh refinement; the non-matching face it
  creates; and the conservative, symmetry-preserving projection that couples across it
  (Chapter 10).
- **MMS** — method of manufactured solutions: pick an exact solution, derive the forcing
  that produces it, and measure the solver's error against it (Chapter 12).
- **Oracle / bit-for-bit validation** — the pure-host CPU implementation each GPU kernel
  is checked against to ~\\( 10^{-14} \\) relative error (Chapter 12).
- **PTX / libdevice / cuda-oxide** — NVIDIA's GPU assembly, its math library, and the
  Rust→PTX toolchain gale compiles its device kernels with (Chapter 11).
