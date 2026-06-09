// Differentiable inverse rheology via Enzyme — the headline differentiable-gale use case.
//
// Integrates the Giesekus conformation tensor under steady simple shear to steady state (the
// SAME relaxation + upper-convected-stretching ODE gale's `LogConfOldroydB`/Giesekus uses), computes
// the rheological material functions (first normal-stress difference N1, shear stress τ_xy), and
// **differentiates the whole time-integration w.r.t. the model parameters (λ, α)** with Enzyme
// forward-mode AD. A gradient-descent loop then recovers (λ, α) from synthetic "measured" data —
// inverse rheology by differentiating through the solver.
//
// Build (Enzyme isn't yet wired into gale's cuda-oxide build — this uses the standalone LLVM pipeline
// the probe validated, docs/probe-enzyme-opt.md): see build.sh.

#include <math.h>
#include <stdio.h>

#define ETA_P 1.0   // polymer viscosity
#define GDOT 2.0    // shear rate γ̇
#define NSTEPS 20000
#define DT 0.001    // explicit Euler to steady state (t = 20 ≫ λ; the fixed point is integrator-independent)

// Giesekus relaxation + UCM stretching under L = [[0,γ̇],[0,0]], integrated to steady state.
//   dC/dt = L·C + C·Lᵀ − (1/λ)[(C−I) + α(C−I)²]
static void steady(double lambda, double alpha, double *n1, double *txy) {
    double cxx = 1.0, cxy = 0.0, cyy = 1.0;
    double il = 1.0 / lambda;
    for (int n = 0; n < NSTEPS; n++) {
        double a = cxx - 1.0, b = cxy, d = cyy - 1.0; // (C − I) components
        double rxx = 2.0 * GDOT * cxy - il * (a + alpha * (a * a + b * b));
        double rxy = GDOT * cyy - il * (b + alpha * b * (a + d));
        double ryy = -il * (d + alpha * (b * b + d * d));
        cxx += DT * rxx;
        cxy += DT * rxy;
        cyy += DT * ryy;
    }
    *n1 = ETA_P * il * (cxx - cyy);   // first normal-stress difference
    *txy = ETA_P * il * cxy;          // shear stress
}

// Least-squares loss vs the measured material functions (file-scope ⇒ a clean R² → R for Enzyme).
static double N1_TARGET, TXY_TARGET;
static double loss(double lambda, double alpha) {
    double n1, txy;
    steady(lambda, alpha, &n1, &txy);
    double e1 = n1 - N1_TARGET, e2 = txy - TXY_TARGET;
    return e1 * e1 + e2 * e2;
}

extern double __enzyme_fwddiff(void *, ...);

int main(void) {
    // Synthetic "measured" data from the true parameters.
    double lam_true = 0.5, alp_true = 0.3;
    steady(lam_true, alp_true, &N1_TARGET, &TXY_TARGET);
    printf("measured material functions: N1 = %.6f, τ_xy = %.6f  (true λ = %.3f, α = %.3f)\n",
           N1_TARGET, TXY_TARGET, lam_true, alp_true);

    // Gradient check: Enzyme forward-mode vs central finite differences at the initial guess.
    double lam = 1.0, alp = 0.6;
    double gl = __enzyme_fwddiff((void *)loss, lam, 1.0, alp, 0.0); // seed λ
    double ga = __enzyme_fwddiff((void *)loss, lam, 0.0, alp, 1.0); // seed α
    double h = 1e-6;
    double fl = (loss(lam + h, alp) - loss(lam - h, alp)) / (2 * h);
    double fa = (loss(lam, alp + h) - loss(lam, alp - h)) / (2 * h);
    printf("gradient check @ (λ=%.2f,α=%.2f): dL/dλ enzyme=%.6f fd=%.6f | dL/dα enzyme=%.6f fd=%.6f\n",
           lam, alp, gl, fl, ga, fa);
    int grad_ok = fabs(gl - fl) < 1e-4 * (fabs(fl) + 1.0) && fabs(ga - fa) < 1e-4 * (fabs(fa) + 1.0);

    // Inverse rheology: gradient descent on (λ, α) through the differentiated solver.
    double rate = 0.01;
    for (int it = 0; it < 4000; it++) {
        double dl = __enzyme_fwddiff((void *)loss, lam, 1.0, alp, 0.0);
        double da = __enzyme_fwddiff((void *)loss, lam, 0.0, alp, 1.0);
        lam -= rate * dl;
        alp -= rate * da;
        if (lam < 0.05) lam = 0.05; // keep parameters physical
        if (alp < 0.0) alp = 0.0;
        if (alp > 1.0) alp = 1.0;
        if (it % 800 == 0)
            printf("  iter %4d: λ = %.4f, α = %.4f, loss = %.3e\n", it, lam, alp, loss(lam, alp));
    }
    printf("recovered: λ = %.4f (true %.3f), α = %.4f (true %.3f)\n", lam, lam_true, alp, alp_true);

    int fit_ok = fabs(lam - lam_true) < 1e-2 && fabs(alp - alp_true) < 1e-2;
    if (grad_ok && fit_ok) {
        printf("\nPASS: Enzyme gradients match FD, and inverse rheology recovered (λ, α) through the solver.\n");
        return 0;
    }
    fprintf(stderr, "\nFAIL: grad_ok=%d fit_ok=%d\n", grad_ok, fit_ok);
    return 1;
}
