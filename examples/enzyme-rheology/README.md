# Differentiable inverse rheology (Enzyme)

A working demonstration of the headline **differentiable-gale** capability: **fitting constitutive-model
parameters by differentiating through the solver** — "inverse rheology".

`giesekus_fit.c` integrates the **Giesekus** conformation tensor under steady simple shear to steady state
(the same relaxation + upper-convected-stretching ODE gale's `LogConfOldroydB`/Giesekus uses), computes the
rheological material functions (first normal-stress difference `N1`, shear stress `τ_xy`), and
**differentiates the entire time-integration w.r.t. the model parameters `(λ, α)`** with
[Enzyme](https://enzyme.mit.edu) forward-mode AD. A gradient-descent loop then recovers `(λ, α)` from
synthetic "measured" data.

## Result

```
measured material functions: N1 = 2.208019, τ_xy = 1.371832  (true λ = 0.500, α = 0.300)
gradient check @ (λ=1.00,α=0.60): dL/dλ enzyme=1.011693 fd=1.011693 | dL/dα enzyme=1.465122 fd=1.465122
  iter    0: λ = 0.9899, α = 0.5853, loss = 5.306e-01
  iter  800: λ = 0.5000, α = 0.3000, loss = 2.613e-30
recovered: λ = 0.5000 (true 0.500), α = 0.3000 (true 0.300)
PASS
```

Enzyme's gradients match central finite differences to machine precision, and gradient descent through the
differentiated 20 000-step solver recovers the true parameters exactly.

## Build / run

```sh
./build.sh && ./giesekus_fit
```

Needs `clang`/`opt` for LLVM 21 and an **LLVMEnzyme** plugin built against LLVM 21 (Enzyme's supported
ceiling — see `../../docs/probe-enzyme-opt.md` for how the plugin was built and the pipeline validated).
Override `CLANG` / `OPT` / `LLVMENZYME` env vars for a different toolchain.

## Why this is a C standalone (and the path to gale-native)

Enzyme is not yet wired into gale's `cargo oxide` build (its `pliron` text-export backend means the rustc
`std::autodiff` path doesn't apply — see `docs/research-differentiable-solver.md` §9). The probe
(`docs/probe-enzyme-opt.md`) proved the full mechanism on real hardware: forward **and** reverse Enzyme AD
through gale kernels *and through cuda-oxide's own emitted IR* → sm_70 PTX, executed on the Titan V. This
example is the **end-to-end use-case** layer on top of that validated mechanism — the math mirrors gale's
constitutive relaxation, so porting it to differentiate gale's actual GPU conformation kernels is the
remaining integration (wire `LLVMEnzyme` as an `opt` pass on the opaque `.ll` cuda-oxide emits, before the
typed-pointer libNVVM shim). Forward-mode is used deliberately: the parameter count is small, and it
sidesteps the reverse-mode register-spill cost the probe measured on DG kernels.
