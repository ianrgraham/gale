#!/usr/bin/env bash
# Differentiate gale's REAL GPU `implicit_relax` kernel with Enzyme, end-to-end to sm_70,
# and validate the rheological parameter gradient on hardware.
#
# Pipeline (no source edits to gale-gpu — operates on its EMITTED opaque LLVM IR):
#   1. cargo oxide build (CUDA_OXIDE_DUMP_LLVM, sm_120) -> dump the gale-gpu device bundle
#      as OPAQUE-pointer LLVM IR  (sm_120 picks opaque export; see docs/probe-enzyme-opt.md).
#   2. de-kernelize `implicit_relax`: drop its `!nvvm.annotations ... !"kernel"` entry so it
#      becomes a plain `define void` __device__ function (Enzyme differentiates device fns,
#      not __global__ kernels — the core finding of the probe).
#   3. llvm-link the de-kernelized bundle with enzyme_driver.ll (primal + fwddiff wrappers).
#   4. opt -passes='enzyme,default<O2>'  (LLVMEnzyme-21 plugin) -> emit the tangent kernel.
#   5. llvm-link libdevice (for __nv_* math the kernel calls), then internalize+globaldce so
#      unused libdevice (e.g. the un-lowerable tanh.approx.f32) is dropped before codegen.
#   6. llc -mcpu=sm_70  -> PTX ; ptxas -arch=sm_70 -> cubin.
#   7. compile & run launch.c -> on-device FD check on the Titan V.
#
# Toolchain (override via env): LLVM 21 (Enzyme's supported ceiling) + CUDA ptxas/libdevice.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
GALE_GPU="$(cd "$HERE/../../gale-gpu" && pwd)"
LLVM=${LLVM_BIN:-/usr/lib/llvm-21/bin}
OPT="$LLVM/opt"; LINK="$LLVM/llvm-link"; LLC="$LLVM/llc"; CLANG="$LLVM/clang"
PLUGIN=${LLVMENZYME:-/tmp/enzyme-probe/Enzyme/enzyme/build/Enzyme/LLVMEnzyme-21.so}
LIBDEV=${LIBDEVICE:-/usr/local/cuda/nvvm/libdevice/libdevice.10.bc}
PTXAS=${PTXAS:-/usr/local/cuda/bin/ptxas}
CUDA_INC=${CUDA_INC:-/usr/local/cuda/include}
CUDA_STUBS=${CUDA_STUBS:-/usr/local/cuda/lib64/stubs}
B="$HERE/build"; mkdir -p "$B"

echo "[1/7] dump gale-gpu opaque LLVM IR (the real device bundle)"
( cd "$GALE_GPU" && CUDA_OXIDE_DUMP_LLVM=1 CUDA_OXIDE_TARGET=sm_120 cargo oxide build vecadd >/dev/null 2>&1 )
GG="$GALE_GPU/../target/cuda-artifacts/gale_gpu.ll"
test -f "$GG" || { echo "ERROR: expected dump at $GG"; exit 1; }
cp "$GG" "$B/gale_gpu.ll"

echo "[2/7] de-kernelize implicit_relax (drop its nvvm.annotations kernel entry)"
NODE=$(grep -oE '^![0-9]+ = !\{ptr @implicit_relax, !"kernel", i32 1\}' "$B/gale_gpu.ll" | grep -oE '^![0-9]+')
test -n "$NODE" || { echo "ERROR: could not find implicit_relax kernel annotation node"; exit 1; }
python3 - "$NODE" "$B/gale_gpu.ll" "$B/gale_device.ll" <<'PY'
import re,sys
node,src_path,dst=sys.argv[1],sys.argv[2],sys.argv[3]
src=open(src_path).read()
def fix(m):
    parts=[p.strip() for p in m.group(1).split(',')]
    parts=[p for p in parts if p!=node]
    return '!nvvm.annotations = !{'+', '.join(parts)+'}'
src=re.sub(r'!nvvm\.annotations = !\{([^}]*)\}', fix, src)
open(dst,'w').write(src)
print(f"  removed {node} from !nvvm.annotations -> implicit_relax is now a __device__ fn")
PY

echo "[3/7] link de-kernelized bundle + Enzyme driver"
"$LINK" "$B/gale_device.ll" "$HERE/enzyme_driver.ll" -S -o "$B/combined.ll" 2>/dev/null

echo "[4/7] run Enzyme forward-mode + O2"
"$OPT" -load-pass-plugin="$PLUGIN" -passes='enzyme,default<O2>' "$B/combined.ll" -S -o "$B/ad.ll"

echo "[5/7] link libdevice, internalize + DCE unused math"
"$LINK" "$B/ad.ll" "$LIBDEV" -S -o "$B/linked.ll" 2>/dev/null
"$OPT" -internalize-public-api-list=primal_relax,d_implicit_dinvlam \
       -passes='internalize,globaldce,default<O2>' "$B/linked.ll" -S -o "$B/stripped.ll"

echo "[6/7] lower to sm_70 PTX + assemble cubin"
"$LLC" -mcpu=sm_70 "$B/stripped.ll" -o "$B/gale_kernel.ptx"
"$PTXAS" -arch=sm_70 -O3 "$B/gale_kernel.ptx" -o "$B/gale_kernel.cubin"

echo "[7/7] compile & run the on-device FD check"
"$CLANG" "$HERE/launch.c" -o "$B/launch" -I"$CUDA_INC" -L"$CUDA_STUBS" -lcuda -lm
cd "$HERE" && exec "$B/launch"
