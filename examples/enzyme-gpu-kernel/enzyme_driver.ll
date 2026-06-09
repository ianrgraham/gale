; Enzyme forward-mode differentiation driver for gale's REAL `implicit_relax` GPU kernel.
;
; `implicit_relax` is gale-gpu's per-node implicit viscoelastic relaxation solve
; (Oldroyd-B / Giesekus Newton + FENE-P bisection branch) — the exact device function
; the solver runs each IMEX substep. In gale-gpu's emitted bundle it is a `__global__`
; kernel (tagged via `!nvvm.annotations ... !"kernel"`). Enzyme differentiates
; `__device__` functions, NOT kernels, so build.sh first DE-KERNELIZES it (drops its
; nvvm.annotations entry → plain `define void`), then links this driver, which provides:
;
;   primal_relax        — a thin ptx_kernel wrapper that just calls @implicit_relax
;                         (the unperturbed primal, used for the finite-difference check)
;   d_implicit_dinvlam  — a ptx_kernel that calls __enzyme_fwddiff on @implicit_relax,
;                         seeding the inv_lambda (1/λ) argument's tangent = 1.0 and marking
;                         the three output buffers `dup` (primal + tangent shadow). The
;                         tangent shadow receives d(Psi_out)/d(1/λ) — the rheological
;                         parameter gradient, computed on-device.
;
; All other arguments are `enzyme_const`. This is the parameter-inference use case:
; small parameter count → forward mode, which also sidesteps the reverse-mode DG
; register-spill cost measured in docs/probe-enzyme-opt.md.

target triple = "nvptx64-nvidia-cuda"

@enzyme_const = external global i32
@enzyme_dup   = external global i32

; The de-kernelized gale device function (signature must match gale-gpu's emitted IR):
;   (bxx,n, bxy,n, byy,n, gamma, inv_lambda, alpha, ext, n1, oxx,n, oxy,n, oyy,n)
declare void @implicit_relax(ptr, i64, ptr, i64, ptr, i64, double, double, double, double, i32, ptr, i64, ptr, i64, ptr, i64)
declare void @__enzyme_fwddiff(...)

; Primal wrapper — calls the kernel unmodified (for the FD reference & primal-equality check).
define ptx_kernel void @primal_relax(ptr %bxx, i64 %a, ptr %bxy, i64 %b, ptr %byy, i64 %cc, double %g, double %il, double %al, double %ex, i32 %n1, ptr %oxx, i64 %d, ptr %oxy, i64 %e, ptr %oyy, i64 %f) {
  call void @implicit_relax(ptr %bxx, i64 %a, ptr %bxy, i64 %b, ptr %byy, i64 %cc, double %g, double %il, double %al, double %ex, i32 %n1, ptr %oxx, i64 %d, ptr %oxy, i64 %e, ptr %oyy, i64 %f)
  ret void
}

; Forward-diff wrapper — d(Psi_out)/d(1/lambda). Output buffers are `dup` so the shadow
; (doxx/doxy/doyy) carries the tangent; inv_lambda is `dup` with a constant 1.0 tangent.
define ptx_kernel void @d_implicit_dinvlam(ptr %bxx, ptr %bxy, ptr %byy, i64 %n, double %gamma, double %invlam, double %alpha, double %ext, i32 %n1, ptr %oxx, ptr %doxx, ptr %oxy, ptr %doxy, ptr %oyy, ptr %doyy) {
  %ec = load i32, ptr @enzyme_const
  %ed = load i32, ptr @enzyme_dup
  call void (...) @__enzyme_fwddiff(ptr @implicit_relax,
     i32 %ec, ptr %bxx, i32 %ec, i64 %n, i32 %ec, ptr %bxy, i32 %ec, i64 %n, i32 %ec, ptr %byy, i32 %ec, i64 %n,
     i32 %ec, double %gamma, i32 %ed, double %invlam, double 1.0, i32 %ec, double %alpha, i32 %ec, double %ext, i32 %ec, i32 %n1,
     i32 %ed, ptr %oxx, ptr %doxx, i32 %ec, i64 %n, i32 %ed, ptr %oxy, ptr %doxy, i32 %ec, i64 %n, i32 %ed, ptr %oyy, ptr %doyy, i32 %ec, i64 %n)
  ret void
}
