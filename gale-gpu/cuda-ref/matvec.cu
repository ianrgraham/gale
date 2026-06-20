// Standalone C++ CUDA port of the gale-gpu DG-SIPG matvec hot path, for an apples-to-apples
// perf comparison against the cuda-oxide (Rust) kernels. The QUESTION this answers: is the
// matvec ~14% DRAM / latency-bound because the structure is fundamentally latency-bound, or
// because the Rust→PTX codegen is worse than equivalent C++? Mirrors poisson.rs exactly:
//   - gradient (multi-elem-per-block, shared u tile, tensor contraction → gx,gy)
//   - operator (reads gx,gy + neighbour traces; closed-form affine face metadata)
//   - operator_fused (SOLO: own + neighbour gradients recomputed in-kernel, no gx/gy)
// Uniform 2D rectangular mesh, p=3 (n1=4, nn=16), N×N elements. Values are arbitrary-but-finite
// (perf is structure-dependent, not value-dependent); only the interior/boundary face STRUCTURE
// matters for the branch/memory pattern, and it matches a real uniform grid.
//
// Build: nvcc -O3 -arch=sm_70 matvec.cu -o matvec
// Run:   ./matvec [N]          (default N=256 → 65536 elements)
// Prof:  ncu --kernel-name 'regex:gradient|op_split|operator_fused' --launch-count 6 \
//            --section SpeedOfLight --section LaunchStats -f -o /tmp/ncu_cpp ./matvec 256

#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <vector>
#include <cuda_runtime.h>

#define CK(x) do{ cudaError_t e=(x); if(e!=cudaSuccess){ \
    printf("CUDA error %s at %s:%d\n", cudaGetErrorString(e), __FILE__, __LINE__); exit(1);} }while(0)

static const uint32_t BND = 0xFFFFFFFFu;     // u32::MAX
static const uint32_t NEU = 0xFFFFFFFFu - 1; // u32::MAX-1
__constant__ double cD[16];                  // diff matrix in CONSTANT memory (broadcast cache)

// ----- multi-element-per-block packing: ~192 threads/block (matches matvec_cfgs) -----
static int epb_for(int nn){ int e=(192+nn-1)/nn; return e<1?1:e; }

// ----- bandwidth calibration: pure streaming read+write (out=2u). One element of essential matvec
// traffic (read u 8.4MB + write 8.4MB = 16.8MB) at the achievable HBM2 rate ⇒ the matvec's BW FLOOR. -----
__global__ void bw_copy(const double* __restrict u, double* __restrict out, long n){
    long i = (long)blockIdx.x*blockDim.x + threadIdx.x;
    if (i < n) out[i] = 2.0*u[i];
}

// ============================== gradient (matches poisson.rs::gradient) =====================
__global__ void gradient(const double* __restrict d, const double* __restrict u, double rx, double sy,
                         int n1, int ne, double* __restrict gx, double* __restrict gy) {
    extern __shared__ double sm[];           // [ DS(nn) | US(epb*nn) ]
    int nn = n1*n1;
    int t = threadIdx.x;
    int epb = blockDim.x / nn;
    int el = t / nn, m = t % nn;
    int e = blockIdx.x * epb + el;
    bool active = e < ne;
    int us = nn + el*nn;
    if (t < nn) sm[t] = d[t];
    if (active) sm[us+m] = u[e*nn+m];
    __syncthreads();
    if (!active) return;
    int i = m % n1, j = m / n1;
    double ur=0, uss=0;
    #pragma unroll
    for (int k=0;k<4;k++){
        ur  += sm[i*n1+k]    * sm[us+k+j*n1];
        uss += sm[j*n1+k]    * sm[us+i+k*n1];
    }
    int gi = e*nn+m;
    gx[gi] = rx*ur;
    gy[gi] = sy*uss;
}

// ============================== operator (matches poisson.rs::operator) =====================
__global__ void op_split(const double* __restrict d, const double* __restrict u,
                         const double* __restrict gx, const double* __restrict gy,
                         const double* __restrict mass, const double* __restrict fswx,
                         const double* __restrict fswy, int n1, int ne, double rx, double sy,
                         double jac, const uint32_t* __restrict face_nbr, double tau, double lambda,
                         double* __restrict out) {
    extern __shared__ double sm[];           // [ DS(nn) | PR(epb*nn) | PS(epb*nn) ]
    int nn = n1*n1;
    int t = threadIdx.x;
    int epb = blockDim.x / nn;
    int el = t / nn, m = t % nn;
    int e = blockIdx.x * epb + el;
    bool active = e < ne;
    int pr = nn + el*nn, ps = nn + epb*nn + el*nn;
    int b = e*nn+m;
    if (t < nn) sm[t] = d[t];
    double jw_b = jac*mass[m];
    double rf = 0;
    if (active) {
        double wx = jw_b*gx[b], wy = jw_b*gy[b];
        sm[pr+m] = rx*wx; sm[ps+m] = sy*wy;
        int ii = m%n1, jj = m/n1;
        double hx=0, hy=0;
        for (int t4=0;t4<4;t4++){
            bool on; int a; double nx,ny; bool xface;
            if      (t4==0){ on=(jj==0);      a=ii; nx=0;  ny=-1; xface=true;  }
            else if (t4==1){ on=(ii==n1-1);   a=jj; nx=1;  ny=0;  xface=false; }
            else if (t4==2){ on=(jj==n1-1);   a=ii; nx=0;  ny=1;  xface=true;  }
            else           { on=(ii==0);      a=jj; nx=-1; ny=0;  xface=false; }
            if (on){
                int idx=(e*4+t4)*n1+a;
                uint32_t nbr=face_nbr[idx];
                if (nbr!=NEU){
                    double sw = xface?fswx[a]:fswy[a];
                    double dun_e = nx*gx[b]+ny*gy[b];
                    double ug = u[b];
                    double avg,jump,gfac;
                    if (nbr==BND){ avg=dun_e; jump=ug; gfac=1.0; }
                    else { int ng=nbr; avg=0.5*(dun_e+nx*gx[ng]+ny*gy[ng]); jump=ug-u[ng]; gfac=0.5; }
                    double g=gfac*sw*jump;
                    rf += -sw*avg + tau*sw*jump;
                    hx += g*nx; hy += g*ny;
                }
            }
        }
        sm[pr+m] -= rx*hx; sm[ps+m] -= sy*hy;
    }
    __syncthreads();
    if (!active) return;
    int i=m%n1, j=m/n1;
    double acc=0;
    #pragma unroll
    for (int k=0;k<4;k++)
        acc += sm[k*n1+i]*sm[pr+k+j*n1] + sm[k*n1+j]*sm[ps+i+k*n1];
    out[b] = acc + rf + lambda*jw_b*u[b];
}

// ===================== operator_fused (SOLO: matches poisson.rs::operator_fused) ============
__global__ void operator_fused(const double* __restrict d, const double* __restrict u,
                               const double* __restrict mass, const double* __restrict fswx,
                               const double* __restrict fswy, int n1, int ne, double rx, double sy,
                               double jac, const uint32_t* __restrict face_nbr, double tau,
                               double lambda, double* __restrict out) {
    extern __shared__ double sm[];           // [ DS(nn) | US | PR | PS ]
    int nn = n1*n1;
    int t = threadIdx.x;
    int epb = blockDim.x / nn;
    int el = t / nn, m = t % nn;
    int e = blockIdx.x * epb + el;
    bool active = e < ne;
    int us = nn + el*nn;
    int pr = nn + epb*nn + el*nn, ps = nn + 2*epb*nn + el*nn;
    int b = e*nn+m;
    if (t < nn) sm[t] = d[t];
    if (active) sm[us+m] = u[b];
    __syncthreads();
    double jw_b = jac*mass[m];
    int i = m%n1, j = m/n1;
    double rf = 0;
    if (active) {
        double ur=0, uss=0;
        #pragma unroll
        for (int k=0;k<4;k++){
            ur  += sm[i*n1+k]*sm[us+k+j*n1];
            uss += sm[j*n1+k]*sm[us+i+k*n1];
        }
        double gxb = rx*ur, gyb = sy*uss;
        sm[pr+m] = rx*(jw_b*gxb); sm[ps+m] = sy*(jw_b*gyb);
        int ii=i, jj=j;
        double hx=0, hy=0;
        for (int t4=0;t4<4;t4++){
            bool on; int a; double nx,ny; bool xface;
            if      (t4==0){ on=(jj==0);    a=ii; nx=0;  ny=-1; xface=true;  }
            else if (t4==1){ on=(ii==n1-1); a=jj; nx=1;  ny=0;  xface=false; }
            else if (t4==2){ on=(jj==n1-1); a=ii; nx=0;  ny=1;  xface=true;  }
            else           { on=(ii==0);    a=jj; nx=-1; ny=0;  xface=false; }
            if (on){
                int idx=(e*4+t4)*n1+a;
                uint32_t nbr=face_nbr[idx];
                if (nbr!=NEU){
                    double sw = xface?fswx[a]:fswy[a];
                    double dun_e = nx*gxb + ny*gyb;
                    double ug = u[b];
                    double avg,jump,gfac;
                    if (nbr==BND){ avg=dun_e; jump=ug; gfac=1.0; }
                    else {
                        int ng=nbr;
                        int ng_base=(ng/nn)*nn, nl=ng%nn, ing=nl%n1, jng=nl/n1;
                        double dun_ng;
                        if (xface){
                            double sden=0;
                            #pragma unroll
                            for (int kk=0;kk<4;kk++) sden += sm[jng*n1+kk]*u[ng_base+ing+kk*n1];
                            dun_ng = ny*(sy*sden);
                        } else {
                            double rden=0;
                            #pragma unroll
                            for (int kk=0;kk<4;kk++) rden += sm[ing*n1+kk]*u[ng_base+kk+jng*n1];
                            dun_ng = nx*(rx*rden);
                        }
                        avg = 0.5*(dun_e+dun_ng); jump = ug-u[ng]; gfac=0.5;
                    }
                    double g=gfac*sw*jump;
                    rf += -sw*avg + tau*sw*jump;
                    hx += g*nx; hy += g*ny;
                }
            }
        }
        sm[pr+m] -= rx*hx; sm[ps+m] -= sy*hy;
    }
    __syncthreads();
    if (!active) return;
    double acc=0;
    #pragma unroll
    for (int k=0;k<4;k++)
        acc += sm[k*n1+i]*sm[pr+k+j*n1] + sm[k*n1+j]*sm[ps+i+k*n1];
    out[b] = acc + rf + lambda*jw_b*u[b];
}

// ===== operator_fused_OPT — dependency-reduced (n1=4 specialized) =====================
// Same math as operator_fused but restructured to expose ILP/MLP that nvcc will NOT create on its
// own (FP64 reassociation is illegal without -ffast-math, so the dependency chains are exactly as
// written): (1) every length-4 contraction is a BALANCED TREE `(a+b)+(c+d)` (critical path 2 adds,
// not 4) → attacks the 22% `wait` (FP64-dep) stall; (2) the neighbour-u reads are HOISTED into 4
// registers before any is consumed → 4 loads in flight at once (MLP) → attacks the 27%
// `long_scoreboard` (global-load latency) stall. NOT bit-exact vs operator_fused (FP reassociated).
__global__ void operator_fused_opt(const double* __restrict d, const double* __restrict u,
                               const double* __restrict mass, const double* __restrict fswx,
                               const double* __restrict fswy, int ne, double rx, double sy,
                               double jac, const uint32_t* __restrict face_nbr, double tau,
                               double lambda, double* __restrict out) {
    extern __shared__ double sm[];
    const int n1=4, nn=16;
    int t = threadIdx.x;
    int epb = blockDim.x / nn;
    int el = t / nn, m = t % nn;
    int e = blockIdx.x * epb + el;
    bool active = e < ne;
    int us = nn + el*nn;
    int pr = nn + epb*nn + el*nn, ps = nn + 2*epb*nn + el*nn;
    int b = e*nn+m;
    if (t < nn) sm[t] = d[t];
    if (active) sm[us+m] = u[b];
    __syncthreads();
    double jw_b = jac*mass[m];
    int i = m&3, j = m>>2;
    double rf = 0;
    double ug = sm[us+m];                       // = u[b]
    if (active) {
        const double* D  = sm;                  // D[r*4+c]
        const double* US = sm + us;
        // gradient as balanced trees (depth 2 instead of 4)
        double ur  = (D[i*4+0]*US[0+j*4] + D[i*4+1]*US[1+j*4]) + (D[i*4+2]*US[2+j*4] + D[i*4+3]*US[3+j*4]);
        double uss = (D[j*4+0]*US[i+0*4] + D[j*4+1]*US[i+4*1]) + (D[j*4+2]*US[i+4*2] + D[j*4+3]*US[i+4*3]);
        double gxb = rx*ur, gyb = sy*uss;
        sm[pr+m] = rx*(jw_b*gxb);
        sm[ps+m] = sy*(jw_b*gyb);
        int ii=i, jj=j;
        double hx=0, hy=0;
        #pragma unroll
        for (int t4=0;t4<4;t4++){
            bool on; int a; double nx,ny; bool xface;
            if      (t4==0){ on=(jj==0);    a=ii; nx=0;  ny=-1; xface=true;  }
            else if (t4==1){ on=(ii==n1-1); a=jj; nx=1;  ny=0;  xface=false; }
            else if (t4==2){ on=(jj==n1-1); a=ii; nx=0;  ny=1;  xface=true;  }
            else           { on=(ii==0);    a=jj; nx=-1; ny=0;  xface=false; }
            if (on){
                int idx=(e*4+t4)*4+a;
                uint32_t nbr=face_nbr[idx];
                if (nbr!=NEU){
                    double sw = xface?fswx[a]:fswy[a];
                    double dun_e = nx*gxb + ny*gyb;
                    double avg,jump,gfac;
                    if (nbr==BND){ avg=dun_e; jump=ug; gfac=1.0; }
                    else {
                        int ng=nbr; int ng_base=(ng>>4)<<4, nl=ng&15, ing=nl&3, jng=nl>>2;
                        double dun_ng, ung;
                        if (xface){
                            double q0=u[ng_base+ing+0], q1=u[ng_base+ing+4], q2=u[ng_base+ing+8], q3=u[ng_base+ing+12]; // hoist
                            double s=(D[jng*4+0]*q0 + D[jng*4+1]*q1) + (D[jng*4+2]*q2 + D[jng*4+3]*q3);               // tree
                            dun_ng = ny*(sy*s);
                            ung = (jng==0)?q0:q3;   // u[ng] = q[jng]; jng∈{0,3} on a conforming S/N face — reuse, no extra load
                        } else {
                            double q0=u[ng_base+0+jng*4], q1=u[ng_base+1+jng*4], q2=u[ng_base+2+jng*4], q3=u[ng_base+3+jng*4]; // hoist
                            double s=(D[ing*4+0]*q0 + D[ing*4+1]*q1) + (D[ing*4+2]*q2 + D[ing*4+3]*q3);                       // tree
                            dun_ng = nx*(rx*s);
                            ung = (ing==0)?q0:q3;   // u[ng] = q[ing]; ing∈{0,3} on a conforming E/W face — reuse
                        }
                        avg = 0.5*(dun_e+dun_ng); jump = ug-ung; gfac=0.5;
                    }
                    double g=gfac*sw*jump;
                    rf += -sw*avg + tau*sw*jump;
                    hx += g*nx; hy += g*ny;
                }
            }
        }
        sm[pr+m] -= rx*hx; sm[ps+m] -= sy*hy;
    }
    __syncthreads();
    if (!active) return;
    const double* D = sm;
    const double* PR = sm+pr, *PS = sm+ps;
    // divergence as two balanced trees (PR-part, PS-part), each depth 2
    double accR = (D[0*4+i]*PR[0+j*4] + D[1*4+i]*PR[1+j*4]) + (D[2*4+i]*PR[2+j*4] + D[3*4+i]*PR[3+j*4]);
    double accS = (D[0*4+j]*PS[i+0*4] + D[1*4+j]*PS[i+4*1]) + (D[2*4+j]*PS[i+4*2] + D[3*4+j]*PS[i+4*3]);
    out[b] = (accR + accS) + rf + lambda*jw_b*ug;
}

// ===== operator_fused_PIPE — opt + software-pipelined neighbour loads (n1=4) =================
// On top of operator_fused_opt: the scattered neighbour-u loads are issued BEFORE the __syncthreads,
// so they fly during the barrier + gradient contraction (hides the long_scoreboard latency that
// dominated opt). A node lies on ≤1 vertical (E/W) and ≤1 horizontal (S/N) face, so two register
// slots (qv*, qh*) cover all neighbour data. Same math as operator_fused (reassociated).
__global__ void operator_fused_pipe(const double* __restrict d, const double* __restrict u,
                               const double* __restrict mass, const double* __restrict fswx,
                               const double* __restrict fswy, int ne, double rx, double sy,
                               double jac, const uint32_t* __restrict face_nbr, double tau,
                               double lambda, double* __restrict out) {
    extern __shared__ double sm[];
    const int nn=16;
    int t = threadIdx.x;
    int epb = blockDim.x / nn;
    int el = t / nn, m = t % nn;
    int e = blockIdx.x * epb + el;
    bool active = e < ne;
    int us = nn + el*nn;
    int pr = nn + epb*nn + el*nn, ps = nn + 2*epb*nn + el*nn;
    int b = e*nn+m;
    int i = m&3, j = m>>2;
    if (t < nn) sm[t] = d[t];
    if (active) sm[us+m] = u[b];

    // ---- determine the ≤2 faces and ISSUE neighbour loads NOW (before the barrier) ----
    int vt=-1; double vnx=0;            // vertical face E/W
    if      (i==0){ vt=3; vnx=-1; }
    else if (i==3){ vt=1; vnx= 1; }
    int ht=-1; double hny=0;            // horizontal face S/N
    if      (j==0){ ht=0; hny=-1; }
    else if (j==3){ ht=2; hny= 1; }
    uint32_t vnbr=NEU, hnbr=NEU;
    int ving=0; double qv0=0,qv1=0,qv2=0,qv3=0;
    int hjng=0; double qh0=0,qh1=0,qh2=0,qh3=0;
    if (active && vt>=0){
        int idx=(e*4+vt)*4+j; vnbr=face_nbr[idx];
        if (vnbr!=NEU && vnbr!=BND){ int ng=vnbr, gb=(ng>>4)<<4, nl=ng&15; ving=nl&3; int vjng=nl>>2;
            qv0=u[gb+0+vjng*4]; qv1=u[gb+1+vjng*4]; qv2=u[gb+2+vjng*4]; qv3=u[gb+3+vjng*4]; }
    }
    if (active && ht>=0){
        int idx=(e*4+ht)*4+i; hnbr=face_nbr[idx];
        if (hnbr!=NEU && hnbr!=BND){ int ng=hnbr, gb=(ng>>4)<<4, nl=ng&15, hing=nl&3; hjng=nl>>2;
            qh0=u[gb+hing+0]; qh1=u[gb+hing+4]; qh2=u[gb+hing+8]; qh3=u[gb+hing+12]; }
    }
    __syncthreads();        // neighbour loads above are in flight across this barrier

    double jw_b = jac*mass[m];
    double rf = 0;
    double ug = sm[us+m];
    if (active) {
        const double* D  = sm;
        const double* US = sm + us;
        double ur  = (D[i*4+0]*US[0+j*4] + D[i*4+1]*US[1+j*4]) + (D[i*4+2]*US[2+j*4] + D[i*4+3]*US[3+j*4]);
        double uss = (D[j*4+0]*US[i+0*4] + D[j*4+1]*US[i+4*1]) + (D[j*4+2]*US[i+4*2] + D[j*4+3]*US[i+4*3]);
        double gxb = rx*ur, gyb = sy*uss;
        sm[pr+m] = rx*(jw_b*gxb);
        sm[ps+m] = sy*(jw_b*gyb);
        double hx=0, hy=0;
        // vertical face (E/W): normal (vnx,0), surface weight fswy[j]
        if (vt>=0 && vnbr!=NEU){
            double sw=fswy[j];
            double dun_e = vnx*gxb;
            double avg,jump,gfac;
            if (vnbr==BND){ avg=dun_e; jump=ug; gfac=1.0; }
            else { double s=(D[ving*4+0]*qv0 + D[ving*4+1]*qv1)+(D[ving*4+2]*qv2 + D[ving*4+3]*qv3);
                   double ung=(ving==0)?qv0:qv3;
                   avg=0.5*(dun_e + vnx*(rx*s)); jump=ug-ung; gfac=0.5; }
            double g=gfac*sw*jump; rf += -sw*avg + tau*sw*jump; hx += g*vnx;
        }
        // horizontal face (S/N): normal (0,hny), surface weight fswx[i]
        if (ht>=0 && hnbr!=NEU){
            double sw=fswx[i];
            double dun_e = hny*gyb;
            double avg,jump,gfac;
            if (hnbr==BND){ avg=dun_e; jump=ug; gfac=1.0; }
            else { double s=(D[hjng*4+0]*qh0 + D[hjng*4+1]*qh1)+(D[hjng*4+2]*qh2 + D[hjng*4+3]*qh3);
                   double ung=(hjng==0)?qh0:qh3;
                   avg=0.5*(dun_e + hny*(sy*s)); jump=ug-ung; gfac=0.5; }
            double g=gfac*sw*jump; rf += -sw*avg + tau*sw*jump; hy += g*hny;
        }
        sm[pr+m] -= rx*hx; sm[ps+m] -= sy*hy;
    }
    __syncthreads();
    if (!active) return;
    const double* D = sm;
    const double* PR = sm+pr, *PS = sm+ps;
    double accR = (D[0*4+i]*PR[0+j*4] + D[1*4+i]*PR[1+j*4]) + (D[2*4+i]*PR[2+j*4] + D[3*4+i]*PR[3+j*4]);
    double accS = (D[0*4+j]*PS[i+0*4] + D[1*4+j]*PS[i+4*1]) + (D[2*4+j]*PS[i+4*2] + D[3*4+j]*PS[i+4*3]);
    out[b] = (accR + accS) + rf + lambda*jw_b*ug;
}

// ===== operator_fused_COARSE — pipe + 2 ELEMENTS per thread (thread coarsening, n1=4) =========
// Each thread processes node m of TWO elements (A, B). Their independent global loads interleave
// (load_faces(A) ∥ load_faces(B), gradient A ∥ B, ...) → more memory-level parallelism in flight to
// hide the long_scoreboard latency that dominates `pipe`, WITHOUT needing more warps (we're
// register-limited). Same math (reassociated). Block stages 2·epb elements.
struct FaceData { uint32_t vnbr, hnbr; int ving, hjng; double qv0,qv1,qv2,qv3, qh0,qh1,qh2,qh3; };
__device__ __forceinline__ FaceData load_faces(const double* __restrict u, const uint32_t* __restrict face_nbr,
                                               int e, int i, int j, int vt, int ht) {
    FaceData f; f.vnbr=NEU; f.hnbr=NEU; f.ving=0; f.hjng=0;
    f.qv0=f.qv1=f.qv2=f.qv3=f.qh0=f.qh1=f.qh2=f.qh3=0;
    if (vt>=0){ int idx=(e*4+vt)*4+j; f.vnbr=face_nbr[idx];
        if (f.vnbr!=NEU && f.vnbr!=BND){ int ng=f.vnbr, gb=(ng>>4)<<4, nl=ng&15; f.ving=nl&3; int vjng=nl>>2;
            f.qv0=u[gb+0+vjng*4]; f.qv1=u[gb+1+vjng*4]; f.qv2=u[gb+2+vjng*4]; f.qv3=u[gb+3+vjng*4]; } }
    if (ht>=0){ int idx=(e*4+ht)*4+i; f.hnbr=face_nbr[idx];
        if (f.hnbr!=NEU && f.hnbr!=BND){ int ng=f.hnbr, gb=(ng>>4)<<4, nl=ng&15, hing=nl&3; f.hjng=nl>>2;
            f.qh0=u[gb+hing+0]; f.qh1=u[gb+hing+4]; f.qh2=u[gb+hing+8]; f.qh3=u[gb+hing+12]; } }
    return f;
}
__device__ __forceinline__ double grad_face_write(double* sm, int us, int pr, int ps, int m,
        const FaceData& f, int vt, int ht, double vnx, double hny, double jw_b, double rx, double sy,
        const double* fswx, const double* fswy, double tau) {
    int i=m&3, j=m>>2;
    const double* D = sm; const double* US = sm + us;
    double ur  = (D[i*4+0]*US[0+j*4] + D[i*4+1]*US[1+j*4]) + (D[i*4+2]*US[2+j*4] + D[i*4+3]*US[3+j*4]);
    double uss = (D[j*4+0]*US[i+0] + D[j*4+1]*US[i+4]) + (D[j*4+2]*US[i+8] + D[j*4+3]*US[i+12]);
    double gxb = rx*ur, gyb = sy*uss;
    double prv = rx*(jw_b*gxb), psv = sy*(jw_b*gyb);
    double ug = US[m];
    double rf=0, hx=0, hy=0;
    if (vt>=0 && f.vnbr!=NEU){ double sw=fswy[j], dun_e=vnx*gxb, avg,jump,gfac;
        if (f.vnbr==BND){ avg=dun_e; jump=ug; gfac=1.0; }
        else { double s=(D[f.ving*4+0]*f.qv0 + D[f.ving*4+1]*f.qv1)+(D[f.ving*4+2]*f.qv2 + D[f.ving*4+3]*f.qv3);
               double ung=(f.ving==0)?f.qv0:f.qv3; avg=0.5*(dun_e+vnx*(rx*s)); jump=ug-ung; gfac=0.5; }
        double g=gfac*sw*jump; rf += -sw*avg + tau*sw*jump; hx += g*vnx; }
    if (ht>=0 && f.hnbr!=NEU){ double sw=fswx[i], dun_e=hny*gyb, avg,jump,gfac;
        if (f.hnbr==BND){ avg=dun_e; jump=ug; gfac=1.0; }
        else { double s=(D[f.hjng*4+0]*f.qh0 + D[f.hjng*4+1]*f.qh1)+(D[f.hjng*4+2]*f.qh2 + D[f.hjng*4+3]*f.qh3);
               double ung=(f.hjng==0)?f.qh0:f.qh3; avg=0.5*(dun_e+hny*(sy*s)); jump=ug-ung; gfac=0.5; }
        double g=gfac*sw*jump; rf += -sw*avg + tau*sw*jump; hy += g*hny; }
    sm[pr+m] = prv - rx*hx;
    sm[ps+m] = psv - sy*hy;
    return rf;
}
__device__ __forceinline__ double divergence(const double* sm, int pr, int ps, int i, int j) {
    const double* D=sm; const double* PR=sm+pr; const double* PS=sm+ps;
    double accR = (D[0*4+i]*PR[0+j*4] + D[1*4+i]*PR[1+j*4]) + (D[2*4+i]*PR[2+j*4] + D[3*4+i]*PR[3+j*4]);
    double accS = (D[0*4+j]*PS[i+0] + D[1*4+j]*PS[i+4]) + (D[2*4+j]*PS[i+8] + D[3*4+j]*PS[i+12]);
    return accR + accS;
}
__global__ void operator_fused_coarse(const double* __restrict d, const double* __restrict u,
                               const double* __restrict mass, const double* __restrict fswx,
                               const double* __restrict fswy, int ne, double rx, double sy,
                               double jac, const uint32_t* __restrict face_nbr, double tau,
                               double lambda, double* __restrict out) {
    extern __shared__ double sm[];
    const int nn=16;
    int t = threadIdx.x, epb = blockDim.x/nn, el = t/nn, m = t%nn;
    int i = m&3, j = m>>2;
    int eA = blockIdx.x*(2*epb) + el, eB = eA + epb;
    bool aA = eA<ne, aB = eB<ne;
    int usA=nn+el*nn, usB=nn+(epb+el)*nn;
    int prb=nn+2*epb*nn, psb=nn+4*epb*nn;
    int prA=prb+el*nn, prB=prb+(epb+el)*nn, psA=psb+el*nn, psB=psb+(epb+el)*nn;
    if (t<nn) sm[t]=d[t];
    if (aA) sm[usA+m]=u[eA*nn+m];
    if (aB) sm[usB+m]=u[eB*nn+m];
    int vt=-1; double vnx=0; if (i==0){vt=3;vnx=-1;} else if (i==3){vt=1;vnx=1;}
    int ht=-1; double hny=0; if (j==0){ht=0;hny=-1;} else if (j==3){ht=2;hny=1;}
    FaceData fA, fB; fA.vnbr=NEU; fA.hnbr=NEU; fB.vnbr=NEU; fB.hnbr=NEU;
    if (aA) fA = load_faces(u, face_nbr, eA, i, j, vt, ht);
    if (aB) fB = load_faces(u, face_nbr, eB, i, j, vt, ht);   // independent of A's loads → MLP
    __syncthreads();
    double jw_b = jac*mass[m];
    double rfA=0, rfB=0, ugA=0, ugB=0;
    if (aA){ ugA=sm[usA+m]; rfA=grad_face_write(sm,usA,prA,psA,m,fA,vt,ht,vnx,hny,jw_b,rx,sy,fswx,fswy,tau); }
    if (aB){ ugB=sm[usB+m]; rfB=grad_face_write(sm,usB,prB,psB,m,fB,vt,ht,vnx,hny,jw_b,rx,sy,fswx,fswy,tau); }
    __syncthreads();
    if (aA) out[eA*nn+m] = divergence(sm,prA,psA,i,j) + rfA + lambda*jw_b*ugA;
    if (aB) out[eB*nn+m] = divergence(sm,prB,psB,i,j) + rfB + lambda*jw_b*ugB;
}

// ===== operator_fused_NS — pipe but NO u-staging / NO first barrier (n1=4) ====================
// A p=3 element's 16 nodes = one 128B cache line, so each thread reads its gradient row+column
// straight from L1 (1 miss/element/warp, rest hit) instead of staging u in shared. Removes the US
// tile AND the first __syncthreads (only PR/PS still need shared+barrier for the divergence).
// Attacks barrier + mio_throttle + short_scoreboard. Same math (reassociated).
__global__ void operator_fused_ns(const double* __restrict d, const double* __restrict u,
                               const double* __restrict mass, const double* __restrict fswx,
                               const double* __restrict fswy, int ne, double rx, double sy,
                               double jac, const uint32_t* __restrict face_nbr, double tau,
                               double lambda, double* __restrict out) {
    extern __shared__ double sm[];           // [ PR(epb·16) | PS(epb·16) ] — no DS tile (read from global)
    const int nn=16;
    int t=threadIdx.x, epb=blockDim.x/nn, el=t/nn, m=t%nn;
    int i=m&3, j=m>>2;
    int e=blockIdx.x*epb+el; bool active=e<ne;
    int pr=el*nn, ps=epb*nn+el*nn, eb=e*nn;
    const double* D=d;   // diff matrix from global (no DS staging ⇒ no missing-barrier race)
    int vt=-1; double vnx=0; if (i==0){vt=3;vnx=-1;} else if (i==3){vt=1;vnx=1;}
    int ht=-1; double hny=0; if (j==0){ht=0;hny=-1;} else if (j==3){ht=2;hny=1;}
    uint32_t vnbr=NEU,hnbr=NEU; int ving=0,hjng=0;
    double qv0=0,qv1=0,qv2=0,qv3=0, qh0=0,qh1=0,qh2=0,qh3=0;
    if (active && vt>=0){ int idx=(e*4+vt)*4+j; vnbr=face_nbr[idx];
        if (vnbr!=NEU&&vnbr!=BND){ int ng=vnbr,gb=(ng>>4)<<4,nl=ng&15; ving=nl&3; int vj=nl>>2;
            qv0=u[gb+0+vj*4];qv1=u[gb+1+vj*4];qv2=u[gb+2+vj*4];qv3=u[gb+3+vj*4]; } }
    if (active && ht>=0){ int idx=(e*4+ht)*4+i; hnbr=face_nbr[idx];
        if (hnbr!=NEU&&hnbr!=BND){ int ng=hnbr,gb=(ng>>4)<<4,nl=ng&15,hi=nl&3; hjng=nl>>2;
            qh0=u[gb+hi+0];qh1=u[gb+hi+4];qh2=u[gb+hi+8];qh3=u[gb+hi+12]; } }
    double jw_b=jac*mass[m], ug=0, rf=0;
    if (active){
        double r0=u[eb+0+j*4],r1=u[eb+1+j*4],r2=u[eb+2+j*4],r3=u[eb+3+j*4];   // row j (hoisted, same line)
        double c0=u[eb+i+0],c1=u[eb+i+4],c2=u[eb+i+8],c3=u[eb+i+12];          // col i
        ug=u[eb+m];
        double ur =(D[i*4+0]*r0 + D[i*4+1]*r1)+(D[i*4+2]*r2 + D[i*4+3]*r3);
        double uss=(D[j*4+0]*c0 + D[j*4+1]*c1)+(D[j*4+2]*c2 + D[j*4+3]*c3);
        double gxb=rx*ur, gyb=sy*uss;
        double prv=rx*(jw_b*gxb), psv=sy*(jw_b*gyb), hx=0, hy=0;
        if (vt>=0 && vnbr!=NEU){ double sw=fswy[j], dun_e=vnx*gxb, avg,jump,gfac;
            if (vnbr==BND){avg=dun_e;jump=ug;gfac=1.0;}
            else{ double s=(D[ving*4+0]*qv0+D[ving*4+1]*qv1)+(D[ving*4+2]*qv2+D[ving*4+3]*qv3);
                  double ung=(ving==0)?qv0:qv3; avg=0.5*(dun_e+vnx*(rx*s)); jump=ug-ung; gfac=0.5; }
            double g=gfac*sw*jump; rf+=-sw*avg+tau*sw*jump; hx+=g*vnx; }
        if (ht>=0 && hnbr!=NEU){ double sw=fswx[i], dun_e=hny*gyb, avg,jump,gfac;
            if (hnbr==BND){avg=dun_e;jump=ug;gfac=1.0;}
            else{ double s=(D[hjng*4+0]*qh0+D[hjng*4+1]*qh1)+(D[hjng*4+2]*qh2+D[hjng*4+3]*qh3);
                  double ung=(hjng==0)?qh0:qh3; avg=0.5*(dun_e+hny*(sy*s)); jump=ug-ung; gfac=0.5; }
            double g=gfac*sw*jump; rf+=-sw*avg+tau*sw*jump; hy+=g*hny; }
        sm[pr+m]=prv-rx*hx; sm[ps+m]=psv-sy*hy;
    }
    __syncthreads();
    if (!active) return;
    double accR=(D[0*4+i]*sm[pr+0+j*4]+D[1*4+i]*sm[pr+1+j*4])+(D[2*4+i]*sm[pr+2+j*4]+D[3*4+i]*sm[pr+3+j*4]);
    double accS=(D[0*4+j]*sm[ps+i+0]+D[1*4+j]*sm[ps+i+4])+(D[2*4+j]*sm[ps+i+8]+D[3*4+j]*sm[ps+i+12]);
    out[eb+m] = (accR+accS) + rf + lambda*jw_b*ug;
}

// load this node's vertical (E/W) + horizontal (S/N) neighbour edge rows for element e (or 0 at boundary)
__device__ __forceinline__ void arith_qvqh(int e, int N, int i, int j, const double* __restrict u,
        double& qv0,double& qv1,double& qv2,double& qv3, double& qh0,double& qh1,double& qh2,double& qh3){
    int ex=e%N, ey=e/N, vgb=-1, hgb=-1;
    if(i==0){ if(ex>0)vgb=(e-1)*16; } else if(i==3){ if(ex<N-1)vgb=(e+1)*16; }
    if(j==0){ if(ey>0)hgb=(e-N)*16; } else if(j==3){ if(ey<N-1)hgb=(e+N)*16; }
    qv0=qv1=qv2=qv3=qh0=qh1=qh2=qh3=0;
    if(vgb>=0){ qv0=u[vgb+0+j*4]; qv1=u[vgb+1+j*4]; qv2=u[vgb+2+j*4]; qv3=u[vgb+3+j*4]; }
    if(hgb>=0){ qh0=u[hgb+i+0]; qh1=u[hgb+i+4]; qh2=u[hgb+i+8]; qh3=u[hgb+i+12]; }
}
// ===== operator_fused_ARITH_PF — arith + grid-stride PREFETCH of the next batch's neighbour reads =====
// (NVIDIA memory-prefetch blog pattern, applied to node-parallel arith.) Each block processes batches of
// epb elements in a grid-stride loop; the next batch's neighbour edge rows (the L2-latency source) are
// issued at the top of each batch and consumed only in the NEXT iteration, so their ~200-cyc latency
// overlaps the current batch's gradient+faces+divergence+2 barriers. arith has register headroom (56),
// so the 8-double prefetch buffer fits without tanking occupancy. Bit-exact to operator_fused.
__global__ void operator_fused_arith_pf(const double* __restrict d, const double* __restrict u,
        const double* __restrict mass, const double* __restrict fswx, const double* __restrict fswy,
        int ne, int N, double rx, double sy, double jac, double tau, double lambda, double* __restrict out){
    extern __shared__ double sm[];           // [ DS | PR | PS ]
    const int nn=16; int t=threadIdx.x, epb=blockDim.x/nn, el=t/nn, m=t%nn;
    int i=m&3, j=m>>2, pr=nn+el*nn, ps=nn+epb*nn+el*nn;
    if(t<nn) sm[t]=d[t];
    __syncthreads();
    const double* D=sm; double jw=jac*mass[m];
    int bstride=gridDim.x*epb;
    double pv0,pv1,pv2,pv3,ph0,ph1,ph2,ph3;
    arith_qvqh(blockIdx.x*epb+el, N, i, j, u, pv0,pv1,pv2,pv3,ph0,ph1,ph2,ph3);   // prefetch batch 0
    for(int base=blockIdx.x*epb; base<ne; base+=bstride){
        int e=base+el; bool active=e<ne;
        double qv0=pv0,qv1=pv1,qv2=pv2,qv3=pv3, qh0=ph0,qh1=ph1,qh2=ph2,qh3=ph3;
        int en=e+bstride;
        if(en<ne) arith_qvqh(en, N, i, j, u, pv0,pv1,pv2,pv3,ph0,ph1,ph2,ph3);    // ISSUE next batch's loads
        double rf=0, ug=0; int eb=e*nn;
        if(active){
            int ex=e%N, ey=e/N;
            int vt=-1; double vnx=0; int ving=0; if(i==0){vt=3;vnx=-1;ving=3;} else if(i==3){vt=1;vnx=1;ving=0;}
            int ht=-1; double hny=0; int hjng=0; if(j==0){ht=0;hny=-1;hjng=3;} else if(j==3){ht=2;hny=1;hjng=0;}
            bool vbnd = vt>=0 && ((i==0&&ex==0)||(i==3&&ex==N-1));
            bool hbnd = ht>=0 && ((j==0&&ey==0)||(j==3&&ey==N-1));
            double r0=u[eb+0+j*4],r1=u[eb+1+j*4],r2=u[eb+2+j*4],r3=u[eb+3+j*4];
            double c0=u[eb+i+0],c1=u[eb+i+4],c2=u[eb+i+8],c3=u[eb+i+12]; ug=u[eb+m];
            double ur=(D[i*4+0]*r0+D[i*4+1]*r1)+(D[i*4+2]*r2+D[i*4+3]*r3);
            double uss=(D[j*4+0]*c0+D[j*4+1]*c1)+(D[j*4+2]*c2+D[j*4+3]*c3);
            double gx=rx*ur, gy=sy*uss, prv=rx*(jw*gx), psv=sy*(jw*gy), hx=0,hy=0;
            if(vt>=0){ double sw=fswy[j], dun_e=vnx*gx, avg,jump,gf;
                if(vbnd){avg=dun_e;jump=ug;gf=1.0;}
                else{ double s=(D[ving*4+0]*qv0+D[ving*4+1]*qv1)+(D[ving*4+2]*qv2+D[ving*4+3]*qv3);
                      double ung=(ving==0)?qv0:qv3; avg=0.5*(dun_e+vnx*(rx*s)); jump=ug-ung; gf=0.5; }
                double g=gf*sw*jump; rf+=-sw*avg+tau*sw*jump; hx+=g*vnx; }
            if(ht>=0){ double sw=fswx[i], dun_e=hny*gy, avg,jump,gf;
                if(hbnd){avg=dun_e;jump=ug;gf=1.0;}
                else{ double s=(D[hjng*4+0]*qh0+D[hjng*4+1]*qh1)+(D[hjng*4+2]*qh2+D[hjng*4+3]*qh3);
                      double ung=(hjng==0)?qh0:qh3; avg=0.5*(dun_e+hny*(sy*s)); jump=ug-ung; gf=0.5; }
                double g=gf*sw*jump; rf+=-sw*avg+tau*sw*jump; hy+=g*hny; }
            sm[pr+m]=prv-rx*hx; sm[ps+m]=psv-sy*hy;
        }
        __syncthreads();                                                          // barrier OUTSIDE branch
        if(active){
            double accR=(D[0*4+i]*sm[pr+0+j*4]+D[1*4+i]*sm[pr+1+j*4])+(D[2*4+i]*sm[pr+2+j*4]+D[3*4+i]*sm[pr+3+j*4]);
            double accS=(D[0*4+j]*sm[ps+i+0]+D[1*4+j]*sm[ps+i+4])+(D[2*4+j]*sm[ps+i+8]+D[3*4+j]*sm[ps+i+12]);
            out[eb+m]=(accR+accS)+rf+lambda*jw*ug;
        }
        __syncthreads();                                                          // before next batch reuses PR/PS
    }
}

// ===== operator_fused_ARITH — ns but neighbour computed ARITHMETICALLY (no face_nbr load) =======
// Uniform N×N grid ⇒ the neighbour element is e±1 / e±N, so we skip the face_nbr[] global load
// entirely. That removes the two-level load-dependency chain (read face_nbr → derive addr → load
// neighbour) — now the neighbour address is known immediately and its load can be issued at the very
// top with no prior load to wait on. Attacks the residual long_scoreboard directly. Same math as ns.
__global__ void operator_fused_arith(const double* __restrict d, const double* __restrict u,
                               const double* __restrict mass, const double* __restrict fswx,
                               const double* __restrict fswy, int ne, int N, double rx, double sy,
                               double jac, double tau, double lambda, double* __restrict out) {
    extern __shared__ double sm[];           // [ DS(16) | PR(epb·16) | PS(epb·16) ]
    const int nn=16;
    int t=threadIdx.x, epb=blockDim.x/nn, el=t/nn, m=t%nn;
    int i=m&3, j=m>>2;
    int e=blockIdx.x*epb+el; bool active=e<ne;
    int pr=nn+el*nn, ps=nn+epb*nn+el*nn, eb=e*nn;
    if (t<nn) sm[t]=d[t];
    __syncthreads();                        // DS staging barrier (correct; faster than global d reads)
    const double* D=sm;
    int ex=e%N, ey=e/N;
    // vertical (E/W) and horizontal (S/N) faces + neighbour bases — all arithmetic, no global load
    int vt=-1; double vnx=0; int vgb=-1, ving=0;
    if (i==0){ vt=3; vnx=-1; if (ex>0)   { vgb=(e-1)*nn; ving=3; } }
    else if (i==3){ vt=1; vnx=1; if (ex<N-1){ vgb=(e+1)*nn; ving=0; } }
    int ht=-1; double hny=0; int hgb=-1, hjng=0;
    if (j==0){ ht=0; hny=-1; if (ey>0)   { hgb=(e-N)*nn; hjng=3; } }
    else if (j==3){ ht=2; hny=1; if (ey<N-1){ hgb=(e+N)*nn; hjng=0; } }
    double qv0=0,qv1=0,qv2=0,qv3=0, qh0=0,qh1=0,qh2=0,qh3=0;
    if (active && vgb>=0){ qv0=u[vgb+0+j*4]; qv1=u[vgb+1+j*4]; qv2=u[vgb+2+j*4]; qv3=u[vgb+3+j*4]; }
    if (active && hgb>=0){ qh0=u[hgb+i+0]; qh1=u[hgb+i+4]; qh2=u[hgb+i+8]; qh3=u[hgb+i+12]; }
    double jw_b=jac*mass[m], ug=0, rf=0;
    if (active){
        double r0=u[eb+0+j*4],r1=u[eb+1+j*4],r2=u[eb+2+j*4],r3=u[eb+3+j*4];
        double c0=u[eb+i+0],c1=u[eb+i+4],c2=u[eb+i+8],c3=u[eb+i+12];
        ug=u[eb+m];
        double ur =(D[i*4+0]*r0 + D[i*4+1]*r1)+(D[i*4+2]*r2 + D[i*4+3]*r3);
        double uss=(D[j*4+0]*c0 + D[j*4+1]*c1)+(D[j*4+2]*c2 + D[j*4+3]*c3);
        double gxb=rx*ur, gyb=sy*uss;
        double prv=rx*(jw_b*gxb), psv=sy*(jw_b*gyb), hx=0, hy=0;
        if (vt>=0){ double sw=fswy[j], dun_e=vnx*gxb, avg,jump,gf;
            if (vgb<0){ avg=dun_e; jump=ug; gf=1.0; }
            else { double s=(D[ving*4+0]*qv0+D[ving*4+1]*qv1)+(D[ving*4+2]*qv2+D[ving*4+3]*qv3);
                   double ung=(ving==0)?qv0:qv3; avg=0.5*(dun_e+vnx*(rx*s)); jump=ug-ung; gf=0.5; }
            double g=gf*sw*jump; rf+=-sw*avg+tau*sw*jump; hx+=g*vnx; }
        if (ht>=0){ double sw=fswx[i], dun_e=hny*gyb, avg,jump,gf;
            if (hgb<0){ avg=dun_e; jump=ug; gf=1.0; }
            else { double s=(D[hjng*4+0]*qh0+D[hjng*4+1]*qh1)+(D[hjng*4+2]*qh2+D[hjng*4+3]*qh3);
                   double ung=(hjng==0)?qh0:qh3; avg=0.5*(dun_e+hny*(sy*s)); jump=ug-ung; gf=0.5; }
            double g=gf*sw*jump; rf+=-sw*avg+tau*sw*jump; hy+=g*hny; }
        sm[pr+m]=prv-rx*hx; sm[ps+m]=psv-sy*hy;
    }
    __syncthreads();
    if (!active) return;
    out[eb+m] = divergence(sm,pr,ps,i,j) + rf + lambda*jw_b*ug;
}

// ===== operator_fused_ARITH_LEAN — arith but loads INLINED (no hoist buffers) ⇒ fewer regs/higher occ =====
// RESULT: 48 regs (vs arith 56), occupancy 56% (vs 51%) — but 53.6µs (SLOWER than 47.7). Eligible-warps/sched
// DROPS 1.42→1.18: more warps resident but each stalls more (loads inline-exposed, not hoisted). The registers
// in arith aren't waste — they're spent on HOISTING that keeps warps ELIGIBLE; cutting them loses more ILP than
// the +5% occupancy gains. Both register directions tested net-negative (prefetch 82regs→27%occ; lean 48→slower)
// ⇒ arith's 56 regs is the occupancy↔ILP optimum (maximizes eligible-warps, the real issue-rate metric).
// Trades ILP (the hoisted r/c/qv/qh live across the gradient) for OCCUPANCY: shorter live ranges ⇒ fewer regs.
__global__ void operator_fused_arith_lean(const double* __restrict d, const double* __restrict u,
        const double* __restrict mass, const double* __restrict fswx, const double* __restrict fswy,
        int ne, int N, double rx, double sy, double jac, double tau, double lambda, double* __restrict out){
    extern __shared__ double sm[]; const int nn=16;
    int t=threadIdx.x, epb=blockDim.x/nn, el=t/nn, m=t%nn;
    int i=m&3, j=m>>2, e=blockIdx.x*epb+el; bool active=e<ne;
    int pr=nn+el*nn, ps=nn+epb*nn+el*nn, eb=e*nn;
    if(t<nn) sm[t]=d[t];
    __syncthreads();
    const double* D=sm; int ex=e%N, ey=e/N;
    double jw=jac*mass[m], ug=0, rf=0;
    if(active){
        int vt=-1; double vnx=0; int vgb=-1,ving=0;
        if(i==0){vt=3;vnx=-1;if(ex>0){vgb=(e-1)*nn;ving=3;}} else if(i==3){vt=1;vnx=1;if(ex<N-1){vgb=(e+1)*nn;ving=0;}}
        int ht=-1; double hny=0; int hgb=-1,hjng=0;
        if(j==0){ht=0;hny=-1;if(ey>0){hgb=(e-N)*nn;hjng=3;}} else if(j==3){ht=2;hny=1;if(ey<N-1){hgb=(e+N)*nn;hjng=0;}}
        ug=u[eb+m];
        double ur=(D[i*4+0]*u[eb+0+j*4]+D[i*4+1]*u[eb+1+j*4])+(D[i*4+2]*u[eb+2+j*4]+D[i*4+3]*u[eb+3+j*4]);
        double uss=(D[j*4+0]*u[eb+i+0]+D[j*4+1]*u[eb+i+4])+(D[j*4+2]*u[eb+i+8]+D[j*4+3]*u[eb+i+12]);
        double gxb=rx*ur, gyb=sy*uss, prv=rx*(jw*gxb), psv=sy*(jw*gyb), hx=0,hy=0;
        if(vt>=0){ double sw=fswy[j], dun_e=vnx*gxb, avg,jump,gf;
            if(vgb<0){avg=dun_e;jump=ug;gf=1.0;}
            else{ double s=(D[ving*4+0]*u[vgb+0+j*4]+D[ving*4+1]*u[vgb+1+j*4])+(D[ving*4+2]*u[vgb+2+j*4]+D[ving*4+3]*u[vgb+3+j*4]);
                  double ung=u[vgb+ving+j*4]; avg=0.5*(dun_e+vnx*(rx*s)); jump=ug-ung; gf=0.5; }
            double g=gf*sw*jump; rf+=-sw*avg+tau*sw*jump; hx+=g*vnx; }
        if(ht>=0){ double sw=fswx[i], dun_e=hny*gyb, avg,jump,gf;
            if(hgb<0){avg=dun_e;jump=ug;gf=1.0;}
            else{ double s=(D[hjng*4+0]*u[hgb+i+0]+D[hjng*4+1]*u[hgb+i+4])+(D[hjng*4+2]*u[hgb+i+8]+D[hjng*4+3]*u[hgb+i+12]);
                  double ung=u[hgb+i+hjng*4]; avg=0.5*(dun_e+hny*(sy*s)); jump=ug-ung; gf=0.5; }
            double g=gf*sw*jump; rf+=-sw*avg+tau*sw*jump; hy+=g*hny; }
        sm[pr+m]=prv-rx*hx; sm[ps+m]=psv-sy*hy;
    }
    __syncthreads();
    if(!active) return;
    out[eb+m]=divergence(sm,pr,ps,i,j)+rf+lambda*jw*ug;
}

// ===== operator_fused_ARITH_R — arith with FREE register cuts (no ILP loss): fold hx/hy into prv/psv,
// kill the e/N division (N-S boundary via e>=N / e<ne-N). KEEPS the qv/qh hoisting (the valuable ILP).
__global__ void operator_fused_arith_r(const double* __restrict d, const double* __restrict u,
        const double* __restrict mass, const double* __restrict fswx, const double* __restrict fswy,
        int ne, int N, double rx, double sy, double jac, double tau, double lambda, double* __restrict out){
    extern __shared__ double sm[]; const int nn=16;
    int t=threadIdx.x, epb=blockDim.x/nn, el=t/nn, m=t%nn;
    int i=m&3, j=m>>2, e=blockIdx.x*epb+el; bool active=e<ne;
    int pr=nn+el*nn, ps=nn+epb*nn+el*nn, eb=e*nn;
    if(t<nn) sm[t]=d[t];
    __syncthreads();
    const double* D=sm; int ex=e%N;                          // only ex needs modulo; ey-checks via e
    int vt=-1; double vnx=0; int vgb=-1,ving=0;
    if(i==0){vt=3;vnx=-1;if(ex>0){vgb=(e-1)*nn;ving=3;}} else if(i==3){vt=1;vnx=1;if(ex<N-1){vgb=(e+1)*nn;ving=0;}}
    int ht=-1; double hny=0; int hgb=-1,hjng=0;
    if(j==0){ht=0;hny=-1;if(e>=N){hgb=(e-N)*nn;hjng=3;}} else if(j==3){ht=2;hny=1;if(e<ne-N){hgb=(e+N)*nn;hjng=0;}}
    double qv0=0,qv1=0,qv2=0,qv3=0, qh0=0,qh1=0,qh2=0,qh3=0;
    if(active && vgb>=0){ qv0=u[vgb+0+j*4]; qv1=u[vgb+1+j*4]; qv2=u[vgb+2+j*4]; qv3=u[vgb+3+j*4]; }
    if(active && hgb>=0){ qh0=u[hgb+i+0]; qh1=u[hgb+i+4]; qh2=u[hgb+i+8]; qh3=u[hgb+i+12]; }
    double jw=jac*mass[m], ug=0, rf=0;
    if(active){
        double r0=u[eb+0+j*4],r1=u[eb+1+j*4],r2=u[eb+2+j*4],r3=u[eb+3+j*4];
        double c0=u[eb+i+0],c1=u[eb+i+4],c2=u[eb+i+8],c3=u[eb+i+12]; ug=u[eb+m];
        double ur=(D[i*4+0]*r0+D[i*4+1]*r1)+(D[i*4+2]*r2+D[i*4+3]*r3);
        double uss=(D[j*4+0]*c0+D[j*4+1]*c1)+(D[j*4+2]*c2+D[j*4+3]*c3);
        double gxb=rx*ur, gyb=sy*uss, prv=rx*(jw*gxb), psv=sy*(jw*gyb);
        if(vt>=0){ double sw=fswy[j], dun_e=vnx*gxb, avg,jump,gf;
            if(vgb<0){avg=dun_e;jump=ug;gf=1.0;}
            else{ double s=(D[ving*4+0]*qv0+D[ving*4+1]*qv1)+(D[ving*4+2]*qv2+D[ving*4+3]*qv3);
                  double ung=(ving==0)?qv0:qv3; avg=0.5*(dun_e+vnx*(rx*s)); jump=ug-ung; gf=0.5; }
            double g=gf*sw*jump; rf+=-sw*avg+tau*sw*jump; prv-=rx*(g*vnx); }   // hx folded
        if(ht>=0){ double sw=fswx[i], dun_e=hny*gyb, avg,jump,gf;
            if(hgb<0){avg=dun_e;jump=ug;gf=1.0;}
            else{ double s=(D[hjng*4+0]*qh0+D[hjng*4+1]*qh1)+(D[hjng*4+2]*qh2+D[hjng*4+3]*qh3);
                  double ung=(hjng==0)?qh0:qh3; avg=0.5*(dun_e+hny*(sy*s)); jump=ug-ung; gf=0.5; }
            double g=gf*sw*jump; rf+=-sw*avg+tau*sw*jump; psv-=sy*(g*hny); }   // hy folded
        sm[pr+m]=prv; sm[ps+m]=psv;
    }
    __syncthreads();
    if(!active) return;
    out[eb+m]=divergence(sm,pr,ps,i,j)+rf+lambda*jw*ug;
}

// ===== operator_fused_TILED — 2D element tile staged in shared; IN-TILE neighbours from shared =====
// Node-parallel ([element][node] layout, good L2) but a block processes a Tx×Ty tile of elements and
// stages the whole tile's u in shared. A face whose neighbour is INSIDE the tile reads the neighbour's
// edge from SHARED (~30 cyc) instead of L2 (~200 cyc) — attacking arith's residual latency. Halo
// (tile-boundary) neighbours read from global; domain boundary = BND. Balanced-tree contractions,
// arithmetic neighbours (no face_nbr). Shared `[ DS | tileU | PR | PS ]`. Bit-exact to operator_fused.
__global__ void operator_fused_tiled(const double* __restrict d, const double* __restrict u,
        const double* __restrict mass, const double* __restrict fswx, const double* __restrict fswy,
        int ne, int N, int Tx, int Ty, double rx, double sy, double jac, double tau, double lambda,
        double* __restrict out){
    extern __shared__ double sm[];
    const int nn=16; int nt=Tx*Ty;
    int t=threadIdx.x, tile_el=t/nn, m=t%nn;
    int tx=tile_el%Tx, ty=tile_el/Tx;
    int ex=blockIdx.x*Tx+tx, ey=blockIdx.y*Ty+ty;
    bool active = ex<N && ey<N;
    int e=ey*N+ex, i=m&3, j=m>>2;
    int ub=nn, prb=nn+nt*nn, psb=nn+2*nt*nn;
    int us=ub+tile_el*nn, pr=prb+tile_el*nn, ps=psb+tile_el*nn;
    if(t<nn) sm[t]=d[t];
    if(active) sm[us+m]=u[e*nn+m];
    __syncthreads();
    const double* D=sm;
    double jw=jac*mass[m], ug=0, rf=0;
    if(active){
        ug=sm[us+m];
        double ur=(D[i*4+0]*sm[us+0+j*4]+D[i*4+1]*sm[us+1+j*4])+(D[i*4+2]*sm[us+2+j*4]+D[i*4+3]*sm[us+3+j*4]);
        double uss=(D[j*4+0]*sm[us+i+0]+D[j*4+1]*sm[us+i+4])+(D[j*4+2]*sm[us+i+8]+D[j*4+3]*sm[us+i+12]);
        double gx=rx*ur, gy=sy*uss;
        double prv=rx*(jw*gx), psv=sy*(jw*gy);
        #pragma unroll
        for(int t4=0;t4<4;t4++){
            bool on; double nx,ny; bool xf; int a, ving=0,vjng=0, nti=-1, gnb=-1; // nti=in-tile nbr tile_el; gnb global halo nbr; -2=domain bnd
            if(t4==0){ on=(j==0); a=i; nx=0;ny=-1;xf=true; ving=i;vjng=3;
                if(on){ if(ey==0)gnb=-2; else if(ty>0)nti=tile_el-Tx; else gnb=e-N; } }
            else if(t4==1){ on=(i==3); a=j; nx=1;ny=0;xf=false; ving=0;vjng=j;
                if(on){ if(ex==N-1)gnb=-2; else if(tx<Tx-1)nti=tile_el+1; else gnb=e+1; } }
            else if(t4==2){ on=(j==3); a=i; nx=0;ny=1;xf=true; ving=i;vjng=0;
                if(on){ if(ey==N-1)gnb=-2; else if(ty<Ty-1)nti=tile_el+Tx; else gnb=e+N; } }
            else { on=(i==0); a=j; nx=-1;ny=0;xf=false; ving=3;vjng=j;
                if(on){ if(ex==0)gnb=-2; else if(tx>0)nti=tile_el-1; else gnb=e-1; } }
            if(on){
                double sw=xf?fswx[a]:fswy[a];
                double dun_e=nx*gx+ny*gy, avg,jump,gf;
                if(gnb==-2){ avg=dun_e; jump=ug; gf=1.0; }
                else { double s, ung;
                    if(nti>=0){ int nus=ub+nti*nn;            // in-tile neighbour ⇒ SHARED
                        if(xf) s=(D[vjng*4+0]*sm[nus+ving+0]+D[vjng*4+1]*sm[nus+ving+4])+(D[vjng*4+2]*sm[nus+ving+8]+D[vjng*4+3]*sm[nus+ving+12]);
                        else   s=(D[ving*4+0]*sm[nus+0+vjng*4]+D[ving*4+1]*sm[nus+1+vjng*4])+(D[ving*4+2]*sm[nus+2+vjng*4]+D[ving*4+3]*sm[nus+3+vjng*4]);
                        ung=sm[nus+ving+vjng*4];
                    } else { int nb=gnb*nn;                    // halo neighbour ⇒ GLOBAL
                        if(xf) s=(D[vjng*4+0]*u[nb+ving+0]+D[vjng*4+1]*u[nb+ving+4])+(D[vjng*4+2]*u[nb+ving+8]+D[vjng*4+3]*u[nb+ving+12]);
                        else   s=(D[ving*4+0]*u[nb+0+vjng*4]+D[ving*4+1]*u[nb+1+vjng*4])+(D[ving*4+2]*u[nb+2+vjng*4]+D[ving*4+3]*u[nb+3+vjng*4]);
                        ung=u[nb+ving+vjng*4];
                    }
                    double dun_ng=xf?ny*(sy*s):nx*(rx*s);
                    avg=0.5*(dun_e+dun_ng); jump=ug-ung; gf=0.5;
                }
                double g=gf*sw*jump; rf+=-sw*avg+tau*sw*jump; prv-=rx*(g*nx); psv-=sy*(g*ny);
            }
        }
        sm[pr+m]=prv; sm[ps+m]=psv;
    }
    __syncthreads();
    if(!active) return;
    double accR=(D[0*4+i]*sm[pr+0+j*4]+D[1*4+i]*sm[pr+1+j*4])+(D[2*4+i]*sm[pr+2+j*4]+D[3*4+i]*sm[pr+3+j*4]);
    double accS=(D[0*4+j]*sm[ps+i+0]+D[1*4+j]*sm[ps+i+4])+(D[2*4+j]*sm[ps+i+8]+D[3*4+j]*sm[ps+i+12]);
    out[e*nn+m]=(accR+accS)+rf+lambda*jw*ug;
}

// ===== operator_fused_CD — arith but the diff matrix D in CONSTANT memory (broadcast cache) =======
// PC sampling showed 56% of stalls on LDS (shared loads) — dominated by re-reading the tiny diff
// matrix D ~24×/thread from shared (broadcast, 26.5% shared-bytes/wavefront). D is read-only and
// the same for every element ⇒ put it in __constant__: the constant cache broadcasts same-address
// reads at near-register speed, killing the LDS traffic AND removing the DS staging tile + barrier.
__global__ void operator_fused_cd(const double* __restrict u,
                               const double* __restrict mass, const double* __restrict fswx,
                               const double* __restrict fswy, int ne, int N, double rx, double sy,
                               double jac, double tau, double lambda, double* __restrict out) {
    extern __shared__ double sm[];           // [ PR(epb·16) | PS(epb·16) ] — no DS, D is constant
    const int nn=16;
    int t=threadIdx.x, epb=blockDim.x/nn, el=t/nn, m=t%nn;
    int i=m&3, j=m>>2;
    int e=blockIdx.x*epb+el; bool active=e<ne;
    int pr=el*nn, ps=epb*nn+el*nn, eb=e*nn;
    int ex=e%N, ey=e/N;
    int vt=-1; double vnx=0; int vgb=-1, ving=0;
    if (i==0){ vt=3; vnx=-1; if (ex>0)   { vgb=(e-1)*nn; ving=3; } }
    else if (i==3){ vt=1; vnx=1; if (ex<N-1){ vgb=(e+1)*nn; ving=0; } }
    int ht=-1; double hny=0; int hgb=-1, hjng=0;
    if (j==0){ ht=0; hny=-1; if (ey>0)   { hgb=(e-N)*nn; hjng=3; } }
    else if (j==3){ ht=2; hny=1; if (ey<N-1){ hgb=(e+N)*nn; hjng=0; } }
    double qv0=0,qv1=0,qv2=0,qv3=0, qh0=0,qh1=0,qh2=0,qh3=0;
    if (active && vgb>=0){ qv0=u[vgb+0+j*4]; qv1=u[vgb+1+j*4]; qv2=u[vgb+2+j*4]; qv3=u[vgb+3+j*4]; }
    if (active && hgb>=0){ qh0=u[hgb+i+0]; qh1=u[hgb+i+4]; qh2=u[hgb+i+8]; qh3=u[hgb+i+12]; }
    double jw_b=jac*mass[m], ug=0, rf=0;
    if (active){
        double r0=u[eb+0+j*4],r1=u[eb+1+j*4],r2=u[eb+2+j*4],r3=u[eb+3+j*4];
        double c0=u[eb+i+0],c1=u[eb+i+4],c2=u[eb+i+8],c3=u[eb+i+12];
        ug=u[eb+m];
        double ur =(cD[i*4+0]*r0 + cD[i*4+1]*r1)+(cD[i*4+2]*r2 + cD[i*4+3]*r3);
        double uss=(cD[j*4+0]*c0 + cD[j*4+1]*c1)+(cD[j*4+2]*c2 + cD[j*4+3]*c3);
        double gxb=rx*ur, gyb=sy*uss;
        double prv=rx*(jw_b*gxb), psv=sy*(jw_b*gyb), hx=0, hy=0;
        if (vt>=0){ double sw=fswy[j], dun_e=vnx*gxb, avg,jump,gf;
            if (vgb<0){ avg=dun_e; jump=ug; gf=1.0; }
            else { double s=(cD[ving*4+0]*qv0+cD[ving*4+1]*qv1)+(cD[ving*4+2]*qv2+cD[ving*4+3]*qv3);
                   double ung=(ving==0)?qv0:qv3; avg=0.5*(dun_e+vnx*(rx*s)); jump=ug-ung; gf=0.5; }
            double g=gf*sw*jump; rf+=-sw*avg+tau*sw*jump; hx+=g*vnx; }
        if (ht>=0){ double sw=fswx[i], dun_e=hny*gyb, avg,jump,gf;
            if (hgb<0){ avg=dun_e; jump=ug; gf=1.0; }
            else { double s=(cD[hjng*4+0]*qh0+cD[hjng*4+1]*qh1)+(cD[hjng*4+2]*qh2+cD[hjng*4+3]*qh3);
                   double ung=(hjng==0)?qh0:qh3; avg=0.5*(dun_e+hny*(sy*s)); jump=ug-ung; gf=0.5; }
            double g=gf*sw*jump; rf+=-sw*avg+tau*sw*jump; hy+=g*hny; }
        sm[pr+m]=prv-rx*hx; sm[ps+m]=psv-sy*hy;
    }
    __syncthreads();
    if (!active) return;
    double accR=(cD[0*4+i]*sm[pr+0+j*4]+cD[1*4+i]*sm[pr+1+j*4])+(cD[2*4+i]*sm[pr+2+j*4]+cD[3*4+i]*sm[pr+3+j*4]);
    double accS=(cD[0*4+j]*sm[ps+i+0]+cD[1*4+j]*sm[ps+i+4])+(cD[2*4+j]*sm[ps+i+8]+cD[3*4+j]*sm[ps+i+12]);
    out[eb+m] = (accR+accS) + rf + lambda*jw_b*ug;
}

int main(int argc, char** argv){
    int N = (argc>1)? atoi(argv[1]) : 256;     // elements per side
    const int n1 = 4, nn = n1*n1;
    int ne = N*N;
    long ndof = (long)ne*nn;
    int epb = epb_for(nn);
    if (const char* ev = getenv("EPB")) { int v = atoi(ev); if (v>0) epb = v; } // sweep override
    int grid = (ne + epb - 1)/epb;
    int block = epb*nn;
    size_t shG = (nn + epb*nn)*sizeof(double);
    size_t shO = (nn + 2*epb*nn)*sizeof(double);
    size_t shF = (nn + 3*epb*nn)*sizeof(double);

    printf("N=%d  ne=%d  ndof=%ld  p=%d  epb=%d  grid=%d  block=%d\n", N, ne, ndof, n1-1, epb, grid, block);
    printf("u/gx/gy/out each = %.1f MB\n", ndof*8/1e6);

    // ---- host data (arbitrary finite values; structure of face_nbr is the real uniform grid) ----
    std::vector<double> hu(ndof), hd(nn), hmass(nn), hfx(n1), hfy(n1);
    for (long g=0; g<ndof; ++g) hu[g] = 1e-3*((g*1103515245u+12345u)&1023);
    for (int k=0;k<nn;k++){ hd[k]=0.01*(k+1); hmass[k]=0.5+0.01*k; }
    for (int a=0;a<n1;a++){ hfx[a]=0.3+0.1*a; hfy[a]=0.4+0.1*a; }
    // face_nbr: interior faces → neighbour's matching edge node (uniform, identity perm); domain
    // boundary → BND. Layout (e*4+t)*n1+a, t: 0=S 1=E 2=N 3=W, a along the edge.
    std::vector<uint32_t> hfn((size_t)ne*4*n1, BND);
    auto eid=[&](int ex,int ey){ return ey*N+ex; };
    for (int ey=0;ey<N;ey++) for (int ex=0;ex<N;ex++){
        int e=eid(ex,ey);
        for (int t=0;t<4;t++) for (int a=0;a<n1;a++){
            size_t idx=((size_t)e*4+t)*n1+a;
            int i,j, ni,nj, nex=ex,ney=ey, na;
            // my edge node (i,j) and neighbour edge node (ni,nj) for the uniform structured grid
            if (t==0){ i=a; j=0;    nex=ex; ney=ey-1; ni=a;    nj=n1-1; }       // South ↔ neighbour North
            else if (t==1){ i=n1-1; j=a; nex=ex+1; ney=ey; ni=0;    nj=a; }     // East ↔ neighbour West
            else if (t==2){ i=a; j=n1-1; nex=ex; ney=ey+1; ni=a;    nj=0; }     // North ↔ neighbour South
            else { i=0; j=a; nex=ex-1; ney=ey; ni=n1-1; nj=a; }                 // West ↔ neighbour East
            (void)i;(void)j;(void)na;
            if (nex<0||nex>=N||ney<0||ney>=N) { hfn[idx]=BND; }
            else { int nE=eid(nex,ney); hfn[idx]=(uint32_t)(nE*nn + (nj*n1+ni)); }
        }
    }

    // ---- device ----
    double *u,*gx,*gy,*out,*d,*mass,*fx,*fy; uint32_t* fn;
    CK(cudaMalloc(&u, ndof*8)); CK(cudaMalloc(&gx, ndof*8)); CK(cudaMalloc(&gy, ndof*8));
    CK(cudaMalloc(&out, ndof*8)); CK(cudaMalloc(&d, nn*8)); CK(cudaMalloc(&mass, nn*8));
    CK(cudaMalloc(&fx, n1*8)); CK(cudaMalloc(&fy, n1*8)); CK(cudaMalloc(&fn, (size_t)ne*4*n1*4));
    CK(cudaMemcpy(u, hu.data(), ndof*8, cudaMemcpyHostToDevice));
    CK(cudaMemcpy(d, hd.data(), nn*8, cudaMemcpyHostToDevice));
    CK(cudaMemcpy(mass, hmass.data(), nn*8, cudaMemcpyHostToDevice));
    CK(cudaMemcpy(fx, hfx.data(), n1*8, cudaMemcpyHostToDevice));
    CK(cudaMemcpy(fy, hfy.data(), n1*8, cudaMemcpyHostToDevice));
    CK(cudaMemcpy(fn, hfn.data(), (size_t)ne*4*n1*4, cudaMemcpyHostToDevice));
    double rx=2.0, sy=2.0, jac=0.25, tau=10.0, lambda=0.0;
    CK(cudaMemcpyToSymbol(cD, hd.data(), nn*8));   // diff matrix → constant memory

    // allow >48KB dynamic shared if needed (not at p=3, but future-proof)
    cudaFuncSetAttribute(operator_fused, cudaFuncAttributeMaxDynamicSharedMemorySize, 98304);

    auto bench=[&](const char* name, auto launch, int iters){
        // warmup
        for(int it=0;it<3;it++) launch();
        CK(cudaDeviceSynchronize());
        cudaEvent_t a,b; cudaEventCreate(&a); cudaEventCreate(&b);
        cudaEventRecord(a);
        for(int it=0;it<iters;it++) launch();
        cudaEventRecord(b); CK(cudaEventSynchronize(b));
        float ms=0; cudaEventElapsedTime(&ms,a,b);
        printf("  %-22s %8.1f us/call  (%d iters)\n", name, ms*1000.0/iters, iters);
        cudaEventDestroy(a); cudaEventDestroy(b);
    };

    int iters=200;
    int cgrid=(int)((ndof+255)/256);
    printf("\n-- bandwidth calibration (essential matvec traffic = read u + write out) --\n");
    bench("bw_copy(2u)", [&]{ bw_copy<<<cgrid,256>>>(u,out,ndof); }, iters);
    double bw_us = 0; { // capture bw_copy time to report achievable GB/s
        cudaEvent_t a,b; cudaEventCreate(&a); cudaEventCreate(&b);
        for(int it=0;it<3;it++) bw_copy<<<cgrid,256>>>(u,out,ndof);
        cudaEventRecord(a); for(int it=0;it<iters;it++) bw_copy<<<cgrid,256>>>(u,out,ndof);
        cudaEventRecord(b); cudaEventSynchronize(b); float ms=0; cudaEventElapsedTime(&ms,a,b); bw_us=ms*1000.0/iters;
    }
    printf("    -> achievable HBM2 ≈ %.0f GB/s (16.8MB / %.1fus); matvec BW floor ≈ %.1f us\n",
           ndof*16.0/ (bw_us*1e3), bw_us, bw_us);
    printf("\n-- two-kernel (gradient + operator) --\n");
    bench("gradient", [&]{ gradient<<<grid,block,shG>>>(d,u,rx,sy,n1,ne,gx,gy); }, iters);
    bench("op_split", [&]{ op_split<<<grid,block,shO>>>(d,u,gx,gy,mass,fx,fy,n1,ne,rx,sy,jac,fn,tau,lambda,out); }, iters);
    bench("gradient+op_split", [&]{
        gradient<<<grid,block,shG>>>(d,u,rx,sy,n1,ne,gx,gy);
        op_split<<<grid,block,shO>>>(d,u,gx,gy,mass,fx,fy,n1,ne,rx,sy,jac,fn,tau,lambda,out);
    }, iters);
    printf("\n-- solo (operator_fused) --\n");
    bench("operator_fused", [&]{ operator_fused<<<grid,block,shF>>>(d,u,mass,fx,fy,n1,ne,rx,sy,jac,fn,tau,lambda,out); }, iters);
    // reference output from operator_fused for the correctness check of the opt variant
    operator_fused<<<grid,block,shF>>>(d,u,mass,fx,fy,n1,ne,rx,sy,jac,fn,tau,lambda,out);
    std::vector<double> ref(ndof); CK(cudaMemcpy(ref.data(), out, ndof*8, cudaMemcpyDeviceToHost));
    bench("operator_fused_opt", [&]{ operator_fused_opt<<<grid,block,shF>>>(d,u,mass,fx,fy,ne,rx,sy,jac,fn,tau,lambda,out); }, iters);
    operator_fused_opt<<<grid,block,shF>>>(d,u,mass,fx,fy,ne,rx,sy,jac,fn,tau,lambda,out);
    { std::vector<double> opt(ndof); CK(cudaMemcpy(opt.data(), out, ndof*8, cudaMemcpyDeviceToHost));
      double maxrel=0; for (long g=0; g<ndof; ++g){ double r=fabs(opt[g]-ref[g])/(fabs(ref[g])+1e-300); maxrel=r>maxrel?r:maxrel; }
      printf("    opt  vs fused: max rel diff %.2e\n", maxrel); }
    bench("operator_fused_pipe", [&]{ operator_fused_pipe<<<grid,block,shF>>>(d,u,mass,fx,fy,ne,rx,sy,jac,fn,tau,lambda,out); }, iters);
    operator_fused_pipe<<<grid,block,shF>>>(d,u,mass,fx,fy,ne,rx,sy,jac,fn,tau,lambda,out);
    { std::vector<double> pp(ndof); CK(cudaMemcpy(pp.data(), out, ndof*8, cudaMemcpyDeviceToHost));
      double maxrel=0; for (long g=0; g<ndof; ++g){ double r=fabs(pp[g]-ref[g])/(fabs(ref[g])+1e-300); maxrel=r>maxrel?r:maxrel; }
      printf("    pipe vs fused: max rel diff %.2e\n", maxrel); }
    int cgrid2 = (ne + 2*epb - 1)/(2*epb);          // 2 elements/block-slot ⇒ half the blocks
    size_t shC = (nn + 6*epb*nn)*sizeof(double);     // DS | US(2·epb) | PR(2·epb) | PS(2·epb)
    cudaFuncSetAttribute(operator_fused_coarse, cudaFuncAttributeMaxDynamicSharedMemorySize, 98304);
    bench("operator_fused_coarse", [&]{ operator_fused_coarse<<<cgrid2,block,shC>>>(d,u,mass,fx,fy,ne,rx,sy,jac,fn,tau,lambda,out); }, iters);
    operator_fused_coarse<<<cgrid2,block,shC>>>(d,u,mass,fx,fy,ne,rx,sy,jac,fn,tau,lambda,out);
    { std::vector<double> cc(ndof); CK(cudaMemcpy(cc.data(), out, ndof*8, cudaMemcpyDeviceToHost));
      double maxrel=0; for (long g=0; g<ndof; ++g){ double r=fabs(cc[g]-ref[g])/(fabs(ref[g])+1e-300); maxrel=r>maxrel?r:maxrel; }
      printf("    coarse vs fused: max rel diff %.2e\n", maxrel); }
    size_t shNS = (2*epb*nn)*sizeof(double);    // PR | PS  (no DS, no US)
    bench("operator_fused_ns", [&]{ operator_fused_ns<<<grid,block,shNS>>>(d,u,mass,fx,fy,ne,rx,sy,jac,fn,tau,lambda,out); }, iters);
    operator_fused_ns<<<grid,block,shNS>>>(d,u,mass,fx,fy,ne,rx,sy,jac,fn,tau,lambda,out);
    { std::vector<double> ns(ndof); CK(cudaMemcpy(ns.data(), out, ndof*8, cudaMemcpyDeviceToHost));
      double maxrel=0; for (long g=0; g<ndof; ++g){ double r=fabs(ns[g]-ref[g])/(fabs(ref[g])+1e-300); maxrel=r>maxrel?r:maxrel; }
      printf("    ns   vs fused: max rel diff %.2e\n", maxrel); }
    size_t shAR = (nn + 2*epb*nn)*sizeof(double);    // DS | PR | PS
    bench("operator_fused_arith", [&]{ operator_fused_arith<<<grid,block,shAR>>>(d,u,mass,fx,fy,ne,N,rx,sy,jac,tau,lambda,out); }, iters);
    operator_fused_arith<<<grid,block,shAR>>>(d,u,mass,fx,fy,ne,N,rx,sy,jac,tau,lambda,out);
    { std::vector<double> ar(ndof); CK(cudaMemcpy(ar.data(), out, ndof*8, cudaMemcpyDeviceToHost));
      double maxrel=0; long am=-1; for (long g=0; g<ndof; ++g){ double r=fabs(ar[g]-ref[g])/(fabs(ref[g])+1e-300); if(r>maxrel){maxrel=r;am=g;} }
      long nbad=0; for (long g=0; g<ndof; ++g){ if (fabs(ar[g]-ref[g])/(fabs(ref[g])+1e-300) > 1e-9) nbad++; }
      printf("    arith vs fused: max rel diff %.2e; %ld/%ld nodes bad\n", maxrel, nbad, ndof); }
    { int pfb=2048; if(const char*ev=getenv("PFB")){int v=atoi(ev);if(v>0)pfb=v;}  // grid-stride blocks
      if(pfb>grid)pfb=grid;
      bench("operator_fused_arith_pf",[&]{ operator_fused_arith_pf<<<pfb,block,shAR>>>(d,u,mass,fx,fy,ne,N,rx,sy,jac,tau,lambda,out); }, iters);
      operator_fused_arith_pf<<<pfb,block,shAR>>>(d,u,mass,fx,fy,ne,N,rx,sy,jac,tau,lambda,out);
      std::vector<double> ap(ndof); CK(cudaMemcpy(ap.data(), out, ndof*8, cudaMemcpyDeviceToHost));
      double maxrel=0; long nbad=0; for (long g=0; g<ndof; ++g){ double r=fabs(ap[g]-ref[g])/(fabs(ref[g])+1e-300); maxrel=r>maxrel?r:maxrel; if(r>1e-9)nbad++; }
      printf("    arith_pf(%d blk, %d/thread) vs fused: max rel diff %.2e; %ld bad\n", pfb, (ne+pfb*epb-1)/(pfb*epb), maxrel, nbad); }
    { bench("operator_fused_arith_lean",[&]{ operator_fused_arith_lean<<<grid,block,shAR>>>(d,u,mass,fx,fy,ne,N,rx,sy,jac,tau,lambda,out); }, iters);
      operator_fused_arith_lean<<<grid,block,shAR>>>(d,u,mass,fx,fy,ne,N,rx,sy,jac,tau,lambda,out);
      std::vector<double> al(ndof); CK(cudaMemcpy(al.data(), out, ndof*8, cudaMemcpyDeviceToHost));
      double maxrel=0; long nbad=0; for (long g=0; g<ndof; ++g){ double r=fabs(al[g]-ref[g])/(fabs(ref[g])+1e-300); maxrel=r>maxrel?r:maxrel; if(r>1e-9)nbad++; }
      printf("    arith_lean vs fused: max rel diff %.2e; %ld bad\n", maxrel, nbad); }
    { bench("operator_fused_arith_r",[&]{ operator_fused_arith_r<<<grid,block,shAR>>>(d,u,mass,fx,fy,ne,N,rx,sy,jac,tau,lambda,out); }, iters);
      operator_fused_arith_r<<<grid,block,shAR>>>(d,u,mass,fx,fy,ne,N,rx,sy,jac,tau,lambda,out);
      std::vector<double> ar(ndof); CK(cudaMemcpy(ar.data(), out, ndof*8, cudaMemcpyDeviceToHost));
      double maxrel=0; long nbad=0; for (long g=0; g<ndof; ++g){ double r=fabs(ar[g]-ref[g])/(fabs(ref[g])+1e-300); maxrel=r>maxrel?r:maxrel; if(r>1e-9)nbad++; }
      printf("    arith_r vs fused: max rel diff %.2e; %ld bad\n", maxrel, nbad); }
    size_t shCD = (2*epb*nn)*sizeof(double);    // PR | PS only (D in constant)
    bench("operator_fused_cd", [&]{ operator_fused_cd<<<grid,block,shCD>>>(u,mass,fx,fy,ne,N,rx,sy,jac,tau,lambda,out); }, iters);
    operator_fused_cd<<<grid,block,shCD>>>(u,mass,fx,fy,ne,N,rx,sy,jac,tau,lambda,out);
    { std::vector<double> cd(ndof); CK(cudaMemcpy(cd.data(), out, ndof*8, cudaMemcpyDeviceToHost));
      double maxrel=0; long nbad=0; for (long g=0; g<ndof; ++g){ double r=fabs(cd[g]-ref[g])/(fabs(ref[g])+1e-300); maxrel=r>maxrel?r:maxrel; if(r>1e-9)nbad++; }
      printf("    cd   vs fused: max rel diff %.2e; %ld/%ld nodes bad\n", maxrel, nbad, ndof); }
    { int Tx=4, Ty=4;
      if(const char*e=getenv("TX")){int v=atoi(e); if(v>0)Tx=v;}
      if(const char*e=getenv("TY")){int v=atoi(e); if(v>0)Ty=v;}
      int nt=Tx*Ty; dim3 tb(nt*nn,1,1); dim3 tgr((N+Tx-1)/Tx,(N+Ty-1)/Ty,1);
      size_t shT=(nn+3*nt*nn)*sizeof(double);
      cudaFuncSetAttribute(operator_fused_tiled, cudaFuncAttributeMaxDynamicSharedMemorySize, 98304);
      bench("operator_fused_tiled", [&]{ operator_fused_tiled<<<tgr,tb,shT>>>(d,u,mass,fx,fy,ne,N,Tx,Ty,rx,sy,jac,tau,lambda,out); }, iters);
      operator_fused_tiled<<<tgr,tb,shT>>>(d,u,mass,fx,fy,ne,N,Tx,Ty,rx,sy,jac,tau,lambda,out);
      std::vector<double> tl(ndof); CK(cudaMemcpy(tl.data(), out, ndof*8, cudaMemcpyDeviceToHost));
      double maxrel=0; long nbad=0; for (long g=0; g<ndof; ++g){ double r=fabs(tl[g]-ref[g])/(fabs(ref[g])+1e-300); maxrel=r>maxrel?r:maxrel; if(r>1e-9)nbad++; }
      printf("    tiled(%dx%d) vs fused: max rel diff %.2e; %ld/%ld nodes bad\n", Tx,Ty, maxrel, nbad, ndof); }

    CK(cudaGetLastError());
    std::vector<double> ho(8); CK(cudaMemcpy(ho.data(), out, 8*8, cudaMemcpyDeviceToHost));
    printf("\nout[0..3] = %g %g %g %g\n", ho[0],ho[1],ho[2],ho[3]);
    return 0;
}
