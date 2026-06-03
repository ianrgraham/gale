# gale 3D strategy (RFC)

Status: **proposal for review** — not yet implemented.
Date: 2026-06-03.
Scope: extend gale from its current 2D (quad) DG-SEM to full 3D (hexahedra),
covering the dimension abstraction for the framework, the hex operator stack, the
genuinely-harder 3D-specific pieces (face orientation, 3×3 log-conformation eigen,
octree AMR), the increment order, and the risks.

This is a large effort — a near-complete rebuild of the `dg` operator layer,
comparable in size to the original 2D buildout. The `sim` framework orchestration
(`Simulation`/`Operations`/`Triggers`/`Device`) is largely dimension-agnostic and
mostly carries over. The plan mirrors how the 2D core was built: foundation →
operators → physics → framework genericity, one validated increment at a time, CPU
oracle + GPU bit-for-bit.

---

## 1. Current state (what is 2D today)

Everything is 2D quads:

- `Reference1d` (`reference.rs`) — LGL nodes/weights + the 1D differentiation
  matrix `diff`. **Dimension-independent; reused as-is** (3D is a tensor product of
  three 1D operators, exactly as 2D is a tensor product of two).
- `Reference2dQuad` (`quad.rs`) — `line: Reference1d`, `nodes: Vec<[f64; 2]>`,
  `n_1d`/`n_nodes` (= `n_1d²`), tensor-product differentiation.
- `QuadGeometry` (`geometry.rs`) — `x, y`, metric terms `rx, ry, sx, sy`,
  `jac = x_r y_s − x_s y_r`, `jw = detJ·w`. Built from 4 corners by bilinear map.
- `Mesh2d` / `Element` / `Neighbor` (`mesh.rs`) — element-centric, **4 faces**,
  interior faces matched by **physical-coordinate proximity** (orientation-agnostic),
  `Neighbor::{Interior{perm}, Boundary, CoarseToFine, FineToCoarse}` for 2:1 AMR.
- Operators: `Hyperbolic<L>` (weak + split-form, Rusanov), `Poisson` (SIPG),
  `Stokes` (dual-splitting), `ViscoelasticFlow<M>` (conformation = symmetric 2×2 =
  **3 components** `[xx, xy, yy]`; log-conf via the closed-form 2×2 `sym_eig`).
- AMR: `RefineQuad` (prolong/restrict via 1D operators, **4 children**), mortar on
  1D faces. `dg::distributed` (partition + halo) — dimension-agnostic.
- `sim` framework: `State{mesh: Mesh2d, fields, time}`, `Integrator`/`Term`/
  `StageHook`, `Simulation`/`Operations`/`Triggers`, `Device`/`DomainDecomposition`.

## 2. Goal & motivation

The headline application — viscoelastic particle-laden microfluidic suspensions — is
intrinsically 3D (channel flows, particle wakes, shear-induced migration). 2D is a
legitimate research/validation setting and the right place to have built the method,
but the end goal needs 3D.

## 3. What is reused vs rebuilt

| Layer | 3D status |
|-------|-----------|
| `Reference1d` | **Reused unchanged** (tensor-product building block) |
| `dg::distributed` (partition/halo) | **Reused** — face-neighbor based, dimension-agnostic |
| `sim` orchestration (Simulation/Operations/Triggers) | **Reused** via a dimension abstraction (§4) |
| `Device`/`DomainDecomposition` | **Reused** — operates on element count + face neighbors |
| `FieldSet`/`Field`/`FieldVec` | **Reused** — flat `[ndof]`, dimension-blind |
| Reference element, geometry, mesh | **Rebuilt**: `Reference3dHex`, `HexGeometry`, `Mesh3d` |
| All DG operators (hyperbolic, Poisson, Stokes, viscoelastic) | **Rebuilt** on hexes |
| AMR transfer + mortar | **Rebuilt**: octree (8 children), 2D-face mortar |

The orchestration being dimension-agnostic is the key leverage: the HOOMD-style API,
the three integrator families' *structure*, and the multi-GPU machinery carry over;
what's rebuilt is the element-local math.

## 4. The dimension abstraction (framework genericity)

The `sim` framework's generic parts need only a small interface from the mesh:
element count, nodes-per-element (→ `ndof`), total volume, and face-neighbor
iteration (for `Device` partitioning/halo). The dimension-specific *operators* are
constructed inside the integrators' base-rhs closures (already the pattern:
`Hyperbolic::new(&state.mesh, …)` is built transiently per step).

**Proposal: a `DgMesh` trait, with `State` generic over it.**

```rust
pub trait DgMesh {
    fn n_elements(&self) -> usize;
    fn n_nodes(&self) -> usize;          // nodes per element (n_1d^dim)
    fn ndof(&self) -> usize { self.n_elements() * self.n_nodes() }
    fn volume(&self) -> f64;             // ∑ jw  (area in 2D)
    // face-neighbor view for Device decomposition / halo (dimension-agnostic):
    fn neighbors(&self, elem: usize) -> &[Neighbor];
}
impl DgMesh for Mesh2d { … }
impl DgMesh for Mesh3d { … }

pub struct State<M: DgMesh = Mesh2d> { pub mesh: M, pub fields: FieldSet, pub time: Time }
```

- **Static dispatch / monomorphization** (consistent with the coarse-dispatch
  decision in `api-design.md` §3.4): `Simulation<M>`, integrators, terms become
  generic over `M`. No `dyn` in hot paths. The default `M = Mesh2d` keeps all
  existing 2D code/tests source-compatible.
- `Device`/`DomainDecomposition` already only use element count + `Neighbor`; they
  become generic trivially.
- Alternative considered: a `Mesh` *enum* `{ D2(Mesh2d), D3(Mesh3d) }`. Rejected:
  operators need the concrete type, and an enum forces match-dispatch at the
  operator boundary with no benefit over generics. Trait + generic is cleaner and
  monomorphizes.

This refactor touches `sim` broadly but mechanically (add `<M: DgMesh>`); it is one
of the later increments (§9 step 6), after the 3D operators exist to drive.

## 5. Hex reference element + metrics

- `Reference3dHex { line: Reference1d, nodes: Vec<[f64;3]>, n_1d, n_nodes = n_1d³ }`,
  tensor-product differentiation in r/s/t (three 1D passes — sum factorization).
- `HexGeometry { x, y, z, rx..tz (9 metric terms), jac, jw }`, built from 8 corners
  (trilinear) or curved (high-order) maps.
- **Free-stream / GCL (the classic 3D pitfall).** In curvilinear 3D the discrete
  metric terms must satisfy the geometric conservation law or a uniform state
  develops spurious residual. The fix is the **curl-form metric identities**
  (Kopriva 2006): compute metrics as `Ja^i = ∇×(...)` so `∑ ∂(Ja^i)/∂ξ = 0`
  discretely. For straight-sided (affine) hexes this is automatic; we adopt the
  curl form from the start so curved elements work later.
  *Validation:* free-stream preserved to round-off on a curved hex mesh.

## 6. 3D face connectivity — 8 orientations, handled by proximity matching

A quad face shared by two hexes can be matched in **8 orientations** (the dihedral
group of the square: 4 rotations × 2 flips). The good news: gale's 2D mesh already
matches interior faces by **physical-coordinate proximity** rather than orientation
bookkeeping (`mesh.rs`). That strategy generalizes directly — match the `n_1d²`
face-trace nodes of the two hexes by nearest physical coordinate, producing a 2D
permutation `perm`. **No explicit orientation enumeration is needed**; the proximity
matcher that resolves 2D edges resolves 3D faces the same way.

- `Mesh3d`/`Element` with **6 faces**, `Face` enum `{ Bottom, Top, South, North,
  West, East }`, `Neighbor` reused (the `perm` is now over `n_1d²` nodes).
- *Validation:* on a hex mesh, the matched neighbor trace equals the neighbor's face
  values to round-off (consistency); free-stream preserved across arbitrarily
  oriented element pairs.

## 7. Operators on hexes

- **Volume gradient/divergence:** three tensor-contraction directions (sum
  factorization over r/s/t); cost `O(n_1d⁴)` per element vs `O(n_1d³)` in 2D.
- **Surface flux:** integrate over 6 quad faces (each `n_1d²` nodes).
- **Hyperbolic:** `Hyperbolic3d<L>` weak + split-form. 3D **Euler is 5 variables**
  `(ρ, ρu, ρv, ρw, E)`; the Chandrashekar EC two-point flux and Rusanov extend
  component-wise. *Validation:* free-stream, entropy-conservation rate (split form),
  MMS convergence; GPU bit-for-bit.
- **Poisson (SIPG) 3D, Stokes / dual-splitting 3D:** structurally identical, faces
  are 2D. *Validation:* MMS Poisson; 3D Taylor–Green / Beltrami flow for NS.

## 8. Viscoelastic in 3D

- Conformation is **symmetric 3×3 = 6 components** `[Ψxx, Ψxy, Ψxz, Ψyy, Ψyz, Ψzz]`
  (was 3). Stress divergence `∇·τ_p` over 3 directions.
- **Log-conformation needs a 3×3 symmetric eigensolver** — the closed-form 2×2
  `sym_eig` (a `0.5·atan2`) does not extend.
  - *CPU:* analytic symmetric-3×3 eigendecomposition (Kopp 2008 / Smith trig method),
    with a guard for near-degenerate eigenvalues (high-Wi alignment).
  - *GPU:* a **fixed-sweep cyclic Jacobi** (e.g. 6–10 sweeps) — deterministic,
    branch-light, and avoids the device-`atan2`/libdevice trig gap that already bit
    the 2D log-conf port (documented in `cuda-oxide-codegen-notes.md`). Validate the
    Jacobi result bit-near the CPU analytic eig.
  - *Validation:* `exp(log(C)) = C` round-trip, SPD preserved, 3D channel recovers
    the total-viscosity profile; high-Wi steady shear matches analytic.

## 9. AMR in 3D — octree

- `RefineHex`: hex → **8 children**; prolong/restrict via the same 1D operators
  applied on 3 axes (the 2D `RefineQuad` tensor structure extends).
- **Mortar on 2D faces:** a coarse face meets **4 fine faces** (2×2), vs 2 fine
  edges in 2D. The mortar projection `P`/`Pᵀ` adjoint pair and the 2:1-balance
  bookkeeping extend but with more cases (`CoarseToFine{ fine: [_; 4] }`).
  Conservation/free-stream constraints (coarse-Jacobian = ¼ fine in 2D → ⅛ in 3D)
  re-derived. *Validation:* conservative remap round-trip, non-conforming free-stream.
- IBM in 3D: sphere/ellipsoid volume penalization; rigid projection → Jeffery orbits
  for 3D ellipsoids.

## 10. Multi-GPU in 3D

Mostly reused: `DomainDecomposition` partitions elements and `halo_exchange` works
on face neighbors regardless of dimension. The halo trace is now `n_1d²` values per
cross-device face (was `n_1d`). The validated `gpu-multigpu` pattern (combined
`[local | halo]` buffer + P2P `memcpy_peer_async`) carries over with the larger face
traces. *Validation:* 2-GPU 3D operator bit-for-bit vs monolithic CPU.

## 11. Increment order

Each step is one or more validated increments (CPU oracle first, GPU bit-for-bit):

1. **Foundation** — `Reference3dHex`, `HexGeometry` (curl-form metrics), `Mesh3d`
   (6 faces, proximity matching). Validate: metric identities/GCL, gradient
   exactness on polynomials, free-stream.
2. **Hyperbolic** — volume+surface operators, `Hyperbolic3d` (advection → Euler).
   Validate: free-stream, EC entropy rate, MMS; then GPU port bit-for-bit.
3. **Elliptic/incompressible** — SIPG Poisson 3D, Stokes/dual-splitting 3D.
   Validate: MMS, 3D Taylor–Green/Beltrami.
4. **Viscoelastic** — 6-component conformation, 3×3 log-conf (CPU analytic + GPU
   Jacobi). Validate: channel total-viscosity, SPD, high-Wi shear.
5. **IBM + AMR** — sphere penalization, octree `RefineHex` + 2D-face mortar.
   Validate: penalization no-slip, conservative remap, non-conforming free-stream.
6. **Framework genericity** — `DgMesh` trait, `State<M>`, generic `Simulation<M>`;
   wire 3D through the integrators/operations/Device. Validate: existing 2D tests
   unchanged (default `M = Mesh2d`); a 3D sim assembled through the HOOMD API.
7. **3D multi-GPU** — decomposition + P2P halo with `n_1d²` traces; bit-for-bit.

## 12. Risks

- **Cost / memory.** 3D DOF per element is `n_1d³` (p=4 → 125, p=6 → 343). Sum
  factorization is mandatory. Titan V (12 GB) bounds problem size; realistic 3D runs
  need both GPUs and modest `p`/element counts.
- **GPU shared memory.** The 2D kernels stage `n_nodes` arrays in shared memory
  (`NN_MAX = 81`). 3D `n_nodes` (125–343) raises shared-mem pressure and may force a
  different kernel blocking (per-slice rather than per-element-in-shared). A kernel
  redesign, not just a constant bump.
- **Curvilinear free-stream/GCL** — the classic 3D DG correctness trap; mitigated by
  adopting curl-form metrics from step 1 and testing free-stream early.
- **3×3 eig robustness** — degenerate eigenvalues at high-Wi alignment; needs the
  guarded analytic form and a sufficient Jacobi sweep count; GPU determinism.
- **Octree 2:1 balance + 3D mortar** — more non-conforming cases than 2D; the
  conservation constants change (⅛). Most error-prone AMR piece.
- **Framework refactor scope** — `<M: DgMesh>` touches all of `sim`, but mechanically;
  the default type parameter keeps 2D source-compatible and the orchestration logic
  is unchanged.

## 13. Relationship to the API design

3D does not change the `docs/api-design.md` architecture — it instantiates it for a
second mesh dimension. The four-homes rule, the integrator families, the operations
taxonomy, and the Device abstraction are all dimension-agnostic; §4 here is the one
addition (the `DgMesh` trait + `State<M>` genericity) that lets the existing
framework drive both 2D and 3D from the same HOOMD-style API.
