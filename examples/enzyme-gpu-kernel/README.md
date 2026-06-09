# Differentiating gale's REAL GPU kernel (Enzyme → sm_70 → Titan V)

This is the milestone the differentiable-gale research pass was building toward: **Enzyme
automatically differentiates gale's actual `implicit_relax` GPU kernel — the per-node implicit
viscoelastic relaxation solve the solver runs every IMEX substep — with respect to a rheological
model parameter (1/λ), and the gradient is validated against finite differences on the Titan V.**

Unlike [`../enzyme-rheology`](../enzyme-rheology), which differentiates a *C transcription* of the
constitutive math, this differentiates **gale-gpu's own emitted device code** — the same
`#[kernel] implicit_relax` in `gale-gpu/src/operators/logconf.rs` that ships in the solver — with no
edits to the Rust source. It operates entirely on the opaque LLVM IR cuda-oxide emits.

## Result

```
device: NVIDIA TITAN V
node 0: Psi_xx=0.430558  d(Psi_xx)/d(1/lambda): enzyme=-0.030774  fd=-0.030774
...
max|diff_primal - primal| = 0.00e+00 ; max rel |enzyme - FD| = 7.231e-08
PASS: gale's REAL implicit_relax kernel, Enzyme-differentiated, parameter gradient correct on the Titan V.
```

Enzyme's forward-mode tangent matches the finite-difference gradient of the real kernel to
~7e-8, and the Enzyme primal reproduces the unperturbed kernel output exactly.

## Build / run

```sh
./build.sh
```

It is self-contained: it rebuilds the gale-gpu IR dump, differentiates it, lowers to sm_70, and
runs the on-device check. Requires an LLVM-21 toolchain, an `LLVMEnzyme-21` plugin (Enzyme's
supported ceiling), and CUDA `ptxas` + `libdevice` (see `../../docs/probe-enzyme-opt.md` for how
the plugin was built). Override `LLVM_BIN` / `LLVMENZYME` / `LIBDEVICE` / `PTXAS` for a different
toolchain.

## How it works

The build operates on **emitted IR**, never on gale-gpu's Rust source (cuda-oxide's `pliron`
text-export backend has no in-memory `llvm::Module`, so the rustc `std::autodiff` path doesn't
apply — Enzyme is inserted as an `opt` pass on the `.ll` instead). Steps (`build.sh`):

1. **Dump** the gale-gpu device bundle as **opaque-pointer** LLVM IR via
   `CUDA_OXIDE_DUMP_LLVM=1 CUDA_OXIDE_TARGET=sm_120 cargo oxide build` (arch major ≥ 10 selects the
   opaque export path; opaque IR is Enzyme's tested regime, vs. the fragile typed-pointer libNVVM
   shim used for sm_70 lowering).
2. **De-kernelize** `implicit_relax`: drop its `!nvvm.annotations ... !"kernel"` entry so it
   becomes a plain `define void` **`__device__`** function. *This is the load-bearing trick* —
   Enzyme differentiates device functions, not `__global__` kernels; without it the differentiated
   body comes out empty. (The de-kernelization is robust: it locates the annotation node by name,
   not by line number.)
3. **Link** the de-kernelized bundle with [`enzyme_driver.ll`](enzyme_driver.ll), which adds two
   thin `ptx_kernel` wrappers: `primal_relax` (calls the kernel as-is) and `d_implicit_dinvlam`
   (calls `__enzyme_fwddiff` on it, seeding the `inv_lambda` tangent = 1, outputs `dup`).
4. **Enzyme** (`opt -passes='enzyme,default<O2>'`, LLVMEnzyme-21) emits the tangent kernel.
5. **Link libdevice** (for the `__nv_*` math the kernel calls), then `internalize`+`globaldce` to
   drop unused libdevice (notably `tanh.approx.f32`, which the LLVM-21 NVPTX backend won't lower).
6. **`llc -mcpu=sm_70` → `ptxas -arch=sm_70`** → cubin.
7. [`launch.c`](launch.c) (CUDA driver API) runs `primal_relax` at `1/λ` and `1/λ+h` for the FD
   reference, runs `d_implicit_dinvlam` for the Enzyme tangent, and asserts they agree.

**Forward mode** is deliberate: parameter inference has a small parameter count, and forward mode
sidesteps the reverse-mode register-spill cost the probe measured on fully-unrolled DG kernels
(`../../docs/probe-enzyme-opt.md`). The remaining gale-native step is to wire `LLVMEnzyme` as an
`opt` pass directly into `cargo oxide` (after the IR dump, before the typed-pointer shim) with a
Rust-level differentiation annotation — at which point this manual pipeline becomes a build flag.
