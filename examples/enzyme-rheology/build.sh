#!/usr/bin/env bash
# Build the differentiable inverse-rheology demo via the Enzyme / LLVM pipeline.
#
# Enzyme is not (yet) wired into gale's cuda-oxide build, so this uses the standalone
# `opt`-on-IR pipeline the probe validated (see ../../docs/probe-enzyme-opt.md):
#   clang -emit-llvm  →  opt -passes=enzyme (LLVMEnzyme plugin)  →  clang link.
#
# Requires LLVMEnzyme built against LLVM 21 (Enzyme's supported ceiling; see the probe doc).
# Override the paths via env vars if your toolchain differs.
set -euo pipefail
CL=${CLANG:-/usr/lib/llvm-21/bin/clang}
OPT=${OPT:-/usr/lib/llvm-21/bin/opt}
PLUGIN=${LLVMENZYME:-/tmp/enzyme-probe/Enzyme/enzyme/build/Enzyme/LLVMEnzyme-21.so}

"$CL" -O2 -ffast-math -S -emit-llvm giesekus_fit.c -o gf.ll
"$OPT" -load-pass-plugin="$PLUGIN" -passes='enzyme,default<O2>' gf.ll -S -o gf_ad.ll
"$CL" -O2 gf_ad.ll -o giesekus_fit -lm
echo "built ./giesekus_fit  —  run it to fit Giesekus (λ, α) by differentiating through the solver"
