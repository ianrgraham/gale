// LEAN matvec experiment: minimize live registers per element so a 2-element coarse kernel fits in a
// good occupancy budget. The heavy ns/pipe kernels (56 regs) get MLP by HOISTING ~16 doubles into
// registers; that's why coarse-of-heavy hit 94 regs / 27% occ. Here each element is processed LEAN
// (single-accumulator loops, inline neighbour reads, no hoisting → few live regs but latency-bound
// ALONE), and the coarse version gets its MLP from interleaving TWO independent lean elements.
// Same math as the straightforward DG-SIPG matvec (simple-loop association). p=3 (n1=4) specialized.
//
// Build: nvcc -O3 -arch=sm_70 --extended-lambda -Wno-deprecated-gpu-targets lean-matvec.cu -o lean
// Run:   EPB=6 ./lean 256

#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <vector>
#include <cmath>
#include <cuda_runtime.h>
#define CK(x) do{ cudaError_t e=(x); if(e!=cudaSuccess){ printf("CUDA %s @%d\n",cudaGetErrorString(e),__LINE__); exit(1);} }while(0)
static const uint32_t BND = 0xFFFFFFFFu;
static const uint32_t NEU = 0xFFFFFFFFu - 1;

__global__ void bw_copy(const double* __restrict u, double* __restrict out, long n){
    long i=(long)blockIdx.x*blockDim.x+threadIdx.x; if(i<n) out[i]=2.0*u[i];
}

// ---- lean per-element grad+face: simple loops, inline neighbour reads, fold lift into pr/ps.
// Writes pr[m],ps[m] to shared; returns rf; outputs ug. Few live doubles ⇒ low register pressure.
__device__ __forceinline__ double lean_gf(double* sm, int us, int pr, int ps, int m, int e,
        const double* __restrict u, const uint32_t* __restrict face_nbr,
        const double* __restrict fswx, const double* __restrict fswy,
        double rx, double sy, double jw, double tau, double& ug_out) {
    const double* D = sm; const double* US = sm + us;
    int i=m&3, j=m>>2;
    double ur=0, uss=0;
    #pragma unroll
    for (int k=0;k<4;k++){ ur += D[i*4+k]*US[k+j*4]; uss += D[j*4+k]*US[i+k*4]; }
    double gx=rx*ur, gy=sy*uss, ug=US[m];
    double prv=rx*(jw*gx), psv=sy*(jw*gy), rf=0;
    #pragma unroll
    for (int t4=0;t4<4;t4++){
        bool on; int a; double nx,ny; bool xf;
        if      (t4==0){on=(j==0);a=i;nx=0;ny=-1;xf=true;}
        else if (t4==1){on=(i==3);a=j;nx=1;ny=0;xf=false;}
        else if (t4==2){on=(j==3);a=i;nx=0;ny=1;xf=true;}
        else           {on=(i==0);a=j;nx=-1;ny=0;xf=false;}
        if (on){
            uint32_t nbr=face_nbr[(e*4+t4)*4+a];
            if (nbr!=NEU){
                double sw = xf?fswx[a]:fswy[a];
                double dun_e = nx*gx+ny*gy, avg, jump, gf;
                if (nbr==BND){ avg=dun_e; jump=ug; gf=1.0; }
                else { int ng=nbr,gb=(ng>>4)<<4,nl=ng&15,ig=nl&3,jg=nl>>2;
                       double s=0;
                       if (xf) {
                           #pragma unroll
                           for(int k=0;k<4;k++) s += D[jg*4+k]*u[gb+ig+k*4];
                       } else {
                           #pragma unroll
                           for(int k=0;k<4;k++) s += D[ig*4+k]*u[gb+k+jg*4];
                       }
                       double dun_ng = xf ? ny*(sy*s) : nx*(rx*s);
                       avg=0.5*(dun_e+dun_ng); jump=ug-u[ng]; gf=0.5; }
                double g=gf*sw*jump;
                rf += -sw*avg + tau*sw*jump;
                prv -= rx*(g*nx);          // fold the symmetry-lift straight into pr/ps (no hx/hy regs)
                psv -= sy*(g*ny);
            }
        }
    }
    sm[pr+m]=prv; sm[ps+m]=psv;
    ug_out=ug;
    return rf;
}
__device__ __forceinline__ double lean_div(const double* sm, int pr, int ps, int i, int j) {
    const double* D=sm;
    double acc=0;
    #pragma unroll
    for (int k=0;k<4;k++) acc += D[k*4+i]*sm[pr+k+j*4] + D[k*4+j]*sm[ps+i+k*4];
    return acc;
}

// ---- LEAN single element (1 thread = 1 node) ----
__global__ void op_lean(const double* __restrict d, const double* __restrict u,
        const double* __restrict mass, const double* __restrict fswx, const double* __restrict fswy,
        int ne, double rx, double sy, double jac, const uint32_t* __restrict face_nbr, double tau,
        double lambda, double* __restrict out) {
    extern __shared__ double sm[];
    const int nn=16;
    int t=threadIdx.x, epb=blockDim.x/nn, el=t/nn, m=t%nn;
    int e=blockIdx.x*epb+el; bool active=e<ne;
    int us=nn+el*nn, pr=nn+epb*nn+el*nn, ps=nn+2*epb*nn+el*nn, b=e*nn+m;
    int i=m&3, j=m>>2;
    if (t<nn) sm[t]=d[t];
    if (active) sm[us+m]=u[b];
    __syncthreads();
    double rf=0, ug=0, jw=jac*mass[m];
    if (active) rf = lean_gf(sm,us,pr,ps,m,e,u,face_nbr,fswx,fswy,rx,sy,jw,tau,ug);
    __syncthreads();
    if (!active) return;
    out[b] = lean_div(sm,pr,ps,i,j) + rf + lambda*jw*ug;
}

// ---- COARSE of the lean kernel: 2 elements/thread, interleaved (MLP from 2 elements, not hoisting) ----
__global__ void op_coarse_lean(const double* __restrict d, const double* __restrict u,
        const double* __restrict mass, const double* __restrict fswx, const double* __restrict fswy,
        int ne, double rx, double sy, double jac, const uint32_t* __restrict face_nbr, double tau,
        double lambda, double* __restrict out) {
    extern __shared__ double sm[];               // DS | US(2·epb) | PR(2·epb) | PS(2·epb)
    const int nn=16;
    int t=threadIdx.x, epb=blockDim.x/nn, el=t/nn, m=t%nn;
    int i=m&3, j=m>>2;
    int eA=blockIdx.x*(2*epb)+el, eB=eA+epb;
    bool aA=eA<ne, aB=eB<ne;
    int usA=nn+el*nn, usB=nn+(epb+el)*nn;
    int prb=nn+2*epb*nn, psb=nn+4*epb*nn;
    int prA=prb+el*nn, prB=prb+(epb+el)*nn, psA=psb+el*nn, psB=psb+(epb+el)*nn;
    if (t<nn) sm[t]=d[t];
    if (aA) sm[usA+m]=u[eA*nn+m];
    if (aB) sm[usB+m]=u[eB*nn+m];
    __syncthreads();
    double jw=jac*mass[m], rfA=0, rfB=0, ugA=0, ugB=0;
    if (aA) rfA = lean_gf(sm,usA,prA,psA,m,eA,u,face_nbr,fswx,fswy,rx,sy,jw,tau,ugA);
    if (aB) rfB = lean_gf(sm,usB,prB,psB,m,eB,u,face_nbr,fswx,fswy,rx,sy,jw,tau,ugB);
    __syncthreads();
    if (aA) out[eA*nn+m] = lean_div(sm,prA,psA,i,j) + rfA + lambda*jw*ugA;
    if (aB) out[eB*nn+m] = lean_div(sm,prB,psB,i,j) + rfB + lambda*jw*ugB;
}

int main(int argc, char** argv){
    int N=(argc>1)?atoi(argv[1]):256;
    const int n1=4, nn=16; int ne=N*N; long ndof=(long)ne*nn;
    int epb=6; if (const char* ev=getenv("EPB")){ int v=atoi(ev); if(v>0) epb=v; }
    int grid=(ne+epb-1)/epb, block=epb*nn;
    size_t shL=(nn+3*epb*nn)*8;                 // DS|US|PR|PS for lean single
    size_t shC=(nn+6*epb*nn)*8;                 // DS|US(2)|PR(2)|PS(2) for coarse
    int cgrid=(ne+2*epb-1)/(2*epb);
    printf("N=%d ne=%d ndof=%ld epb=%d block=%d\n",N,ne,ndof,epb,block);

    std::vector<double> hu(ndof),hd(nn),hmass(nn),hfx(n1),hfy(n1);
    for (long g=0;g<ndof;++g) hu[g]=1e-3*((g*1103515245u+12345u)&1023);
    for (int k=0;k<nn;k++){hd[k]=0.01*(k+1);hmass[k]=0.5+0.01*k;}
    for (int a=0;a<n1;a++){hfx[a]=0.3+0.1*a;hfy[a]=0.4+0.1*a;}
    std::vector<uint32_t> hfn((size_t)ne*4*n1,BND);
    auto eid=[&](int ex,int ey){return ey*N+ex;};
    for (int ey=0;ey<N;ey++) for (int ex=0;ex<N;ex++){ int e=eid(ex,ey);
        for (int t=0;t<4;t++) for (int a=0;a<n1;a++){ size_t idx=((size_t)e*4+t)*n1+a;
            int nex=ex,ney=ey,ni,nj;
            if (t==0){nex=ex;ney=ey-1;ni=a;nj=n1-1;} else if (t==1){nex=ex+1;ney=ey;ni=0;nj=a;}
            else if (t==2){nex=ex;ney=ey+1;ni=a;nj=0;} else {nex=ex-1;ney=ey;ni=n1-1;nj=a;}
            if (nex<0||nex>=N||ney<0||ney>=N) hfn[idx]=BND;
            else hfn[idx]=(uint32_t)(eid(nex,ney)*nn+(nj*n1+ni)); } }

    double *u,*out,*d,*mass,*fx,*fy; uint32_t* fn;
    CK(cudaMalloc(&u,ndof*8));CK(cudaMalloc(&out,ndof*8));CK(cudaMalloc(&d,nn*8));
    CK(cudaMalloc(&mass,nn*8));CK(cudaMalloc(&fx,n1*8));CK(cudaMalloc(&fy,n1*8));CK(cudaMalloc(&fn,(size_t)ne*4*n1*4));
    CK(cudaMemcpy(u,hu.data(),ndof*8,cudaMemcpyHostToDevice));CK(cudaMemcpy(d,hd.data(),nn*8,cudaMemcpyHostToDevice));
    CK(cudaMemcpy(mass,hmass.data(),nn*8,cudaMemcpyHostToDevice));CK(cudaMemcpy(fx,hfx.data(),n1*8,cudaMemcpyHostToDevice));
    CK(cudaMemcpy(fy,hfy.data(),n1*8,cudaMemcpyHostToDevice));CK(cudaMemcpy(fn,hfn.data(),(size_t)ne*4*n1*4,cudaMemcpyHostToDevice));
    double rx=2.0,sy=2.0,jac=0.25,tau=10.0,lambda=0.0;
    cudaFuncSetAttribute(op_coarse_lean,cudaFuncAttributeMaxDynamicSharedMemorySize,98304);

    auto bench=[&](const char* nm, auto L, int it){ for(int w=0;w<3;w++) L(); CK(cudaDeviceSynchronize());
        cudaEvent_t a,b;cudaEventCreate(&a);cudaEventCreate(&b);cudaEventRecord(a);
        for(int k=0;k<it;k++) L(); cudaEventRecord(b);CK(cudaEventSynchronize(b));
        float ms=0;cudaEventElapsedTime(&ms,a,b);printf("  %-18s %7.1f us/call\n",nm,ms*1000.0/it); };
    int iters=200, cg=(ndof+255)/256;
    bench("bw_copy",[&]{bw_copy<<<cg,256>>>(u,out,ndof);},iters);
    double bw; { cudaEvent_t a,b;cudaEventCreate(&a);cudaEventCreate(&b);for(int w=0;w<3;w++)bw_copy<<<cg,256>>>(u,out,ndof);
        cudaEventRecord(a);for(int k=0;k<iters;k++)bw_copy<<<cg,256>>>(u,out,ndof);cudaEventRecord(b);cudaEventSynchronize(b);
        float ms=0;cudaEventElapsedTime(&ms,a,b);bw=ms*1000.0/iters; }
    printf("    BW floor ~%.1f us (%.0f GB/s)\n",bw,ndof*16.0/(bw*1e3));

    bench("op_lean",[&]{op_lean<<<grid,block,shL>>>(d,u,mass,fx,fy,ne,rx,sy,jac,fn,tau,lambda,out);},iters);
    op_lean<<<grid,block,shL>>>(d,u,mass,fx,fy,ne,rx,sy,jac,fn,tau,lambda,out);
    std::vector<double> ref(ndof); CK(cudaMemcpy(ref.data(),out,ndof*8,cudaMemcpyDeviceToHost));
    bench("op_coarse_lean",[&]{op_coarse_lean<<<cgrid,block,shC>>>(d,u,mass,fx,fy,ne,rx,sy,jac,fn,tau,lambda,out);},iters);
    op_coarse_lean<<<cgrid,block,shC>>>(d,u,mass,fx,fy,ne,rx,sy,jac,fn,tau,lambda,out);
    { std::vector<double> c(ndof); CK(cudaMemcpy(c.data(),out,ndof*8,cudaMemcpyDeviceToHost));
      double mr=0; for(long g=0;g<ndof;++g){double r=fabs(c[g]-ref[g])/(fabs(ref[g])+1e-300); mr=r>mr?r:mr;}
      printf("    coarse_lean vs lean: max rel diff %.2e\n",mr); }
    CK(cudaGetLastError());
    return 0;
}
