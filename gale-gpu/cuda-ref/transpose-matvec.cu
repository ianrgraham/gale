// NEW STRATEGY: one-thread-per-element + TRANSPOSED [node][element] layout.
// Each thread does a whole p=3 element (16 nodes) entirely in REGISTERS — no shared, no barrier, no
// cross-thread LDS. Transposed layout uT[node*ne + e] ⇒ consecutive threads read consecutive elements
// ⇒ all global loads/stores COALESCED (own + neighbours). D read uniformly by every thread ⇒ CONSTANT
// broadcast cache works (the thing that was divergent/slow in the node-parallel kernel). Trades high
// register pressure / low occupancy for 16× per-thread ILP+MLP. Same math (arithmetic neighbours).
//
// Build: nvcc -O3 -arch=sm_70 --extended-lambda -Wno-deprecated-gpu-targets transpose-matvec.cu -o tr
// Run:   ./tr 256

#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <vector>
#include <cmath>
#include <cuda_runtime.h>
#define CK(x) do{ cudaError_t e=(x); if(e!=cudaSuccess){ printf("CUDA %s @%d\n",cudaGetErrorString(e),__LINE__); exit(1);} }while(0)
static const uint32_t BND=0xFFFFFFFFu, NEU=0xFFFFFFFFu-1;
__constant__ double cD[16];
__constant__ double cM[16];
__constant__ double cA[256];      // A_self: element-local operator (volume + self-face), all-interior
__constant__ double cB[4*256];    // B_dir[4]: neighbour coupling per direction (S,E,N,W)

__global__ void bw_copy(const double* __restrict u, double* __restrict out, long n){
    long i=(long)blockIdx.x*blockDim.x+threadIdx.x; if(i<n) out[i]=2.0*u[i];
}

// ---- reference: node-parallel [element][node], arithmetic neighbours, simple loops (correct) ----
__global__ void op_ref(const double* __restrict d, const double* __restrict u, const double* __restrict mass,
        const double* __restrict fswx, const double* __restrict fswy, int ne, int N, double rx, double sy,
        double jac, double tau, double lambda, double* __restrict out){
    extern __shared__ double sm[];      // DS|PR|PS
    const int nn=16; int t=threadIdx.x, epb=blockDim.x/nn, el=t/nn, m=t%nn;
    int i=m&3,j=m>>2, e=blockIdx.x*epb+el; bool active=e<ne;
    int pr=nn+el*nn, ps=nn+epb*nn+el*nn, eb=e*nn;
    if(t<nn) sm[t]=d[t]; __syncthreads();
    const double* D=sm; int ex=e%N, ey=e/N; double rf=0, ug=0, jw=jac*mass[m];
    if(active){
        double ur=0,uss=0;
        for(int k=0;k<4;k++){ ur+=D[i*4+k]*u[eb+k+j*4]; uss+=D[j*4+k]*u[eb+i+k*4]; }
        double gx=rx*ur, gy=sy*uss; ug=u[eb+m];
        double prv=rx*(jw*gx), psv=sy*(jw*gy);
        for(int t4=0;t4<4;t4++){
            bool on; int a,vng=-1,ving=0,vjng=0; double nx,ny; bool xf;
            if(t4==0){on=(j==0);a=i;nx=0;ny=-1;xf=true;  if(ey>0){vng=(e-N)*nn;ving=i;vjng=3;}}
            else if(t4==1){on=(i==3);a=j;nx=1;ny=0;xf=false; if(ex<N-1){vng=(e+1)*nn;ving=0;vjng=j;}}
            else if(t4==2){on=(j==3);a=i;nx=0;ny=1;xf=true;  if(ey<N-1){vng=(e+N)*nn;ving=i;vjng=0;}}
            else {on=(i==0);a=j;nx=-1;ny=0;xf=false; if(ex>0){vng=(e-1)*nn;ving=3;vjng=j;}}
            if(on){ double sw=xf?fswx[a]:fswy[a]; double dun_e=nx*gx+ny*gy, avg,jump,gf;
                if(vng<0){ avg=dun_e; jump=ug; gf=1.0; }
                else{ double s=0;
                    if(xf) for(int k=0;k<4;k++) s+=D[vjng*4+k]*u[vng+ving+k*4];
                    else   for(int k=0;k<4;k++) s+=D[ving*4+k]*u[vng+k+vjng*4];
                    double dun_ng=xf?ny*(sy*s):nx*(rx*s);
                    double ung=u[vng+ving+vjng*4];
                    avg=0.5*(dun_e+dun_ng); jump=ug-ung; gf=0.5; }
                double g=gf*sw*jump; rf+=-sw*avg+tau*sw*jump; prv-=rx*(g*nx); psv-=sy*(g*ny);
            }
        }
        sm[pr+m]=prv; sm[ps+m]=psv;
    }
    __syncthreads(); if(!active) return;
    double acc=0; for(int k=0;k<4;k++) acc+=D[k*4+i]*sm[pr+k+j*4]+D[k*4+j]*sm[ps+i+k*4];
    out[eb+m]=acc+rf+lambda*jw*ug;
}

// ---- one-thread-per-element, transposed layout, constant D ----
__global__ void op_transpose(const double* __restrict uT, int ne, int N, double rx, double sy,
        double jac, double tau, double lambda, double* __restrict outT){
    int e=blockIdx.x*blockDim.x+threadIdx.x; if(e>=ne) return;
    int ex=e%N, ey=e/N;
    double ue[16];
    #pragma unroll
    for(int k=0;k<16;k++) ue[k]=uT[k*ne+e];   // coalesced (consecutive e)
    double PR[16], PS[16], rf[16];
    #pragma unroll
    for(int m=0;m<16;m++){
        int i=m&3,j=m>>2;
        double ur=(cD[i*4+0]*ue[0+j*4]+cD[i*4+1]*ue[1+j*4])+(cD[i*4+2]*ue[2+j*4]+cD[i*4+3]*ue[3+j*4]);
        double uss=(cD[j*4+0]*ue[i+0]+cD[j*4+1]*ue[i+4])+(cD[j*4+2]*ue[i+8]+cD[j*4+3]*ue[i+12]);
        double gx=rx*ur, gy=sy*uss, jw=jac*cM[m];
        double prv=rx*(jw*gx), psv=sy*(jw*gy), r=0;
        // 4 faces (arithmetic neighbour; load neighbour edge row from transposed layout, coalesced)
        #pragma unroll
        for(int t4=0;t4<4;t4++){
            bool on; double nx,ny; bool xf; int ng=-1; int ving=0,vjng=0;
            if(t4==0){on=(j==0);nx=0;ny=-1;xf=true;  if(ey>0){ng=e-N;ving=i;vjng=3;}}
            else if(t4==1){on=(i==3);nx=1;ny=0;xf=false; if(ex<N-1){ng=e+1;ving=0;vjng=j;}}
            else if(t4==2){on=(j==3);nx=0;ny=1;xf=true;  if(ey<N-1){ng=e+N;ving=i;vjng=0;}}
            else {on=(i==0);nx=-1;ny=0;xf=false; if(ex>0){ng=e-1;ving=3;vjng=j;}}
            if(on){
                double fsw = xf ? (0.3+0.1*i) : (0.4+0.1*j); // fswx[i] / fswy[j] (uniform-mesh constants)
                double dun_e=nx*gx+ny*gy, avg,jump,gf;
                if(ng<0){ avg=dun_e; jump=ue[m]; gf=1.0; }
                else{ double s=0;
                    if(xf){
                        #pragma unroll
                        for(int k=0;k<4;k++) s+=cD[vjng*4+k]*uT[(ving+k*4)*ne+ng];
                    } else {
                        #pragma unroll
                        for(int k=0;k<4;k++) s+=cD[ving*4+k]*uT[(k+vjng*4)*ne+ng];
                    }
                    double dun_ng=xf?ny*(sy*s):nx*(rx*s);
                    double ung=uT[(ving+vjng*4)*ne+ng];
                    avg=0.5*(dun_e+dun_ng); jump=ue[m]-ung; gf=0.5; }
                double g=gf*fsw*jump; r+=-fsw*avg+tau*fsw*jump; prv-=rx*(g*nx); psv-=sy*(g*ny);
            }
        }
        PR[m]=prv; PS[m]=psv; rf[m]=r;
    }
    #pragma unroll
    for(int m=0;m<16;m++){
        int i=m&3,j=m>>2;
        double accR=(cD[0*4+i]*PR[0+j*4]+cD[1*4+i]*PR[1+j*4])+(cD[2*4+i]*PR[2+j*4]+cD[3*4+i]*PR[3+j*4]);
        double accS=(cD[0*4+j]*PS[i+0]+cD[1*4+j]*PS[i+4])+(cD[2*4+j]*PS[i+8]+cD[3*4+j]*PS[i+12]);
        outT[m*ne+e]=(accR+accS)+rf[m]+lambda*(jac*cM[m])*ue[m];
    }
}

// ---- op_transpose + GRID-STRIDE REGISTER PREFETCH (NVIDIA memory-prefetch blog pattern) ----
// Each thread loops over several elements; it prefetches element (e+stride)'s ue into registers while
// computing element e, so the load latency overlaps the (substantial) per-element compute — hides the
// LDG-latency that starved op_transpose at 17% occupancy, WITHOUT needing more occupancy (Volkov/ILP).
__global__ void op_transpose_pf(const double* __restrict uT, int ne, int N, double rx, double sy,
        double jac, double tau, double lambda, double* __restrict outT){
    int stride=blockDim.x*gridDim.x;
    int e0=blockIdx.x*blockDim.x+threadIdx.x; if(e0>=ne) return;
    double pue[16];
    #pragma unroll
    for(int k=0;k<16;k++) pue[k]=uT[k*ne+e0];          // prefetch the first element
    for(int e=e0;e<ne;e+=stride){
        double ue[16];
        #pragma unroll
        for(int k=0;k<16;k++) ue[k]=pue[k];            // consume prefetched
        int en=e+stride;
        if(en<ne){
            #pragma unroll
            for(int k=0;k<16;k++) pue[k]=uT[k*ne+en];  // ISSUE next element's loads (don't wait) → MLP
        }
        int ex=e%N, ey=e/N;
        double PR[16], PS[16], rf[16];
        #pragma unroll
        for(int m=0;m<16;m++){
            int i=m&3,j=m>>2;
            double ur=(cD[i*4+0]*ue[0+j*4]+cD[i*4+1]*ue[1+j*4])+(cD[i*4+2]*ue[2+j*4]+cD[i*4+3]*ue[3+j*4]);
            double uss=(cD[j*4+0]*ue[i+0]+cD[j*4+1]*ue[i+4])+(cD[j*4+2]*ue[i+8]+cD[j*4+3]*ue[i+12]);
            double gx=rx*ur, gy=sy*uss, jw=jac*cM[m];
            double prv=rx*(jw*gx), psv=sy*(jw*gy), r=0;
            #pragma unroll
            for(int t4=0;t4<4;t4++){
                bool on; double nx,ny; bool xf; int ng=-1; int ving=0,vjng=0;
                if(t4==0){on=(j==0);nx=0;ny=-1;xf=true;  if(ey>0){ng=e-N;ving=i;vjng=3;}}
                else if(t4==1){on=(i==3);nx=1;ny=0;xf=false; if(ex<N-1){ng=e+1;ving=0;vjng=j;}}
                else if(t4==2){on=(j==3);nx=0;ny=1;xf=true;  if(ey<N-1){ng=e+N;ving=i;vjng=0;}}
                else {on=(i==0);nx=-1;ny=0;xf=false; if(ex>0){ng=e-1;ving=3;vjng=j;}}
                if(on){ double fsw=xf?(0.3+0.1*i):(0.4+0.1*j); double dun_e=nx*gx+ny*gy, avg,jump,gf;
                    if(ng<0){avg=dun_e;jump=ue[m];gf=1.0;}
                    else{ double s=0;
                        if(xf){ for(int k=0;k<4;k++) s+=cD[vjng*4+k]*uT[(ving+k*4)*ne+ng]; }
                        else  { for(int k=0;k<4;k++) s+=cD[ving*4+k]*uT[(k+vjng*4)*ne+ng]; }
                        double dun_ng=xf?ny*(sy*s):nx*(rx*s); double ung=uT[(ving+vjng*4)*ne+ng];
                        avg=0.5*(dun_e+dun_ng); jump=ue[m]-ung; gf=0.5; }
                    double g=gf*fsw*jump; r+=-fsw*avg+tau*fsw*jump; prv-=rx*(g*nx); psv-=sy*(g*ny);
                }
            }
            PR[m]=prv; PS[m]=psv; rf[m]=r;
        }
        #pragma unroll
        for(int m=0;m<16;m++){
            int i=m&3,j=m>>2;
            double accR=(cD[0*4+i]*PR[0+j*4]+cD[1*4+i]*PR[1+j*4])+(cD[2*4+i]*PR[2+j*4]+cD[3*4+i]*PR[3+j*4]);
            double accS=(cD[0*4+j]*PS[i+0]+cD[1*4+j]*PS[i+4])+(cD[2*4+j]*PS[i+8]+cD[3*4+j]*PS[i+12]);
            outT[m*ne+e]=(accR+accS)+rf[m]+lambda*(jac*cM[m])*ue[m];
        }
    }
}

// ---- ASSEMBLED one-thread-per-element: out = cA·ue + Σ_dir cB[dir]·nbr_dir (matrices in constant) ----
// No PR/PS/rf intermediates ⇒ far fewer registers than op_transpose. All-interior assembly (the test
// measures perf + interior correctness; boundary elements would need a correction matrix — see notes).
__global__ void op_assembled(const double* __restrict uT, int ne, int N, double* __restrict outT){
    int e=blockIdx.x*blockDim.x+threadIdx.x; if(e>=ne) return;
    int ex=e%N, ey=e/N;
    double ue[16], out[16];
    #pragma unroll
    for(int k=0;k<16;k++) ue[k]=uT[(long)k*ne+e];
    // out = cA · ue  (dense 16×16 from constant broadcast)
    #pragma unroll
    for(int m=0;m<16;m++){ double a=0;
        #pragma unroll
        for(int n=0;n<16;n++) a+=cA[m*16+n]*ue[n];
        out[m]=a; }
    // + Σ_dir cB[dir] · nbr_dir   (dir 0=S 1=E 2=N 3=W); skip missing (boundary) neighbours
    long ngd[4]={ ey>0?(long)e-N:-1, ex<N-1?(long)e+1:-1, ey<N-1?(long)e+N:-1, ex>0?(long)e-1:-1 };
    #pragma unroll
    for(int dir=0;dir<4;dir++){ long ng=ngd[dir]; if(ng<0) continue;
        double nb[16];
        #pragma unroll
        for(int k=0;k<16;k++) nb[k]=uT[(long)k*ne+ng];
        const double* B=cB+dir*256;
        #pragma unroll
        for(int m=0;m<16;m++){ double a=0;
            #pragma unroll
            for(int n=0;n<16;n++) a+=B[m*16+n]*nb[n];
            out[m]+=a; }
    }
    #pragma unroll
    for(int m=0;m<16;m++) outT[(long)m*ne+e]=out[m];
}

// host replica of the element-local operator (all faces interior) — used to ASSEMBLE cA, cB by
// applying it to unit vectors (the operator is linear ⇒ superposition is exact).
static double hD[16],hMv[16],hFx[4],hFy[4],g_rx,g_sy,g_jac,g_tau,g_lam;
static void apply_local(const double ue[16], const double nbr[4][16], double out[16]){
    double PR[16],PS[16],rf[16];
    for(int m=0;m<16;m++){ int i=m&3,j=m>>2;
        double ur=0,uss=0; for(int k=0;k<4;k++){ ur+=hD[i*4+k]*ue[k+j*4]; uss+=hD[j*4+k]*ue[i+k*4]; }
        double gx=g_rx*ur, gy=g_sy*uss, jw=g_jac*hMv[m], ug=ue[m];
        double prv=g_rx*(jw*gx), psv=g_sy*(jw*gy), r=0;
        for(int t4=0;t4<4;t4++){
            bool on; double nx,ny; bool xf; int ving=0,vjng=0; int a;
            if(t4==0){on=(j==0);a=i;nx=0;ny=-1;xf=true; ving=i;vjng=3;}
            else if(t4==1){on=(i==3);a=j;nx=1;ny=0;xf=false; ving=0;vjng=j;}
            else if(t4==2){on=(j==3);a=i;nx=0;ny=1;xf=true; ving=i;vjng=0;}
            else {on=(i==0);a=j;nx=-1;ny=0;xf=false; ving=3;vjng=j;}
            if(on){ double sw=xf?hFx[a]:hFy[a]; double dun_e=nx*gx+ny*gy;
                double s=0; const double* nv=nbr[t4];
                if(xf) for(int k=0;k<4;k++) s+=hD[vjng*4+k]*nv[ving+k*4];
                else   for(int k=0;k<4;k++) s+=hD[ving*4+k]*nv[k+vjng*4];
                double dun_ng=xf?ny*(g_sy*s):nx*(g_rx*s);
                double ung=nv[ving+vjng*4];
                double avg=0.5*(dun_e+dun_ng), jump=ug-ung, gf=0.5;
                double g=gf*sw*jump; r+=-sw*avg+g_tau*sw*jump; prv-=g_rx*(g*nx); psv-=g_sy*(g*ny);
            }
        }
        PR[m]=prv; PS[m]=psv; rf[m]=r;
    }
    for(int m=0;m<16;m++){ int i=m&3,j=m>>2; double acc=0;
        for(int k=0;k<4;k++) acc+=hD[k*4+i]*PR[k+j*4]+hD[k*4+j]*PS[i+k*4];
        out[m]=acc+rf[m]+g_lam*(g_jac*hMv[m])*ue[m]; }
}

int main(int argc,char**argv){
    int N=(argc>1)?atoi(argv[1]):256; const int nn=16,n1=4; int ne=N*N; long ndof=(long)ne*nn;
    int epb=6; if(const char*ev=getenv("EPB")){int v=atoi(ev);if(v>0)epb=v;}
    int grid=(ne+epb-1)/epb, block=epb*nn;
    std::vector<double> hu(ndof),hd(nn),hm(nn),hfx(n1),hfy(n1);
    for(long g=0;g<ndof;++g) hu[g]=1e-3*((g*1103515245u+12345u)&1023);
    for(int k=0;k<nn;k++){hd[k]=0.01*(k+1);hm[k]=0.5+0.01*k;}
    for(int a=0;a<n1;a++){hfx[a]=0.3+0.1*a;hfy[a]=0.4+0.1*a;}
    // transposed copy
    std::vector<double> huT(ndof);
    for(int e=0;e<ne;e++) for(int m=0;m<nn;m++) huT[(long)m*ne+e]=hu[(long)e*nn+m];
    double *u,*uT,*out,*outT,*d,*mass,*fx,*fy;
    CK(cudaMalloc(&u,ndof*8));CK(cudaMalloc(&uT,ndof*8));CK(cudaMalloc(&out,ndof*8));CK(cudaMalloc(&outT,ndof*8));
    CK(cudaMalloc(&d,nn*8));CK(cudaMalloc(&mass,nn*8));CK(cudaMalloc(&fx,n1*8));CK(cudaMalloc(&fy,n1*8));
    CK(cudaMemcpy(u,hu.data(),ndof*8,cudaMemcpyHostToDevice));CK(cudaMemcpy(uT,huT.data(),ndof*8,cudaMemcpyHostToDevice));
    CK(cudaMemcpy(d,hd.data(),nn*8,cudaMemcpyHostToDevice));CK(cudaMemcpy(mass,hm.data(),nn*8,cudaMemcpyHostToDevice));
    CK(cudaMemcpy(fx,hfx.data(),n1*8,cudaMemcpyHostToDevice));CK(cudaMemcpy(fy,hfy.data(),n1*8,cudaMemcpyHostToDevice));
    CK(cudaMemcpyToSymbol(cD,hd.data(),nn*8));CK(cudaMemcpyToSymbol(cM,hm.data(),nn*8));
    double rx=2.0,sy=2.0,jac=0.25,tau=10.0,lambda=0.0;
    // ---- ASSEMBLE cA (A_self) and cB (B_dir) by applying the local operator to unit vectors ----
    for(int k=0;k<16;k++){hD[k]=hd[k];hMv[k]=hm[k];} for(int a=0;a<4;a++){hFx[a]=hfx[a];hFy[a]=hfy[a];}
    g_rx=rx;g_sy=sy;g_jac=jac;g_tau=tau;g_lam=lambda;
    std::vector<double> hA(256), hB(4*256);
    { double zero[4][16]={{0}}, o[16], en[16];
      for(int n=0;n<16;n++){ for(int k=0;k<16;k++) en[k]=(k==n);
        apply_local(en, zero, o); for(int m=0;m<16;m++) hA[m*16+n]=o[m]; }       // A_self[:,n]
      for(int dir=0;dir<4;dir++) for(int n=0;n<16;n++){ double nb[4][16]={{0}};
        for(int k=0;k<16;k++) nb[dir][k]=(k==n); double z16[16]={0};
        apply_local(z16, nb, o); for(int m=0;m<16;m++) hB[dir*256+m*16+n]=o[m]; } // B_dir[:,n]
    }
    CK(cudaMemcpyToSymbol(cA,hA.data(),256*8)); CK(cudaMemcpyToSymbol(cB,hB.data(),4*256*8));
    size_t shR=(nn+2*epb*nn)*8;
    auto bench=[&](const char*nm,auto L,int it){for(int w=0;w<3;w++)L();CK(cudaDeviceSynchronize());
        cudaEvent_t a,b;cudaEventCreate(&a);cudaEventCreate(&b);cudaEventRecord(a);for(int k=0;k<it;k++)L();
        cudaEventRecord(b);CK(cudaEventSynchronize(b));float ms=0;cudaEventElapsedTime(&ms,a,b);
        printf("  %-14s %7.1f us\n",nm,ms*1000.0/it);};
    int iters=200,cg=(ndof+255)/256;
    bench("bw_copy",[&]{bw_copy<<<cg,256>>>(u,out,ndof);},iters);
    double bw;{cudaEvent_t a,b;cudaEventCreate(&a);cudaEventCreate(&b);for(int w=0;w<3;w++)bw_copy<<<cg,256>>>(u,out,ndof);
        cudaEventRecord(a);for(int k=0;k<iters;k++)bw_copy<<<cg,256>>>(u,out,ndof);cudaEventRecord(b);cudaEventSynchronize(b);
        float ms=0;cudaEventElapsedTime(&ms,a,b);bw=ms*1000.0/iters;}
    printf("    BW floor ~%.1f us\n",bw);
    bench("op_ref",[&]{op_ref<<<grid,block,shR>>>(d,u,mass,fx,fy,ne,N,rx,sy,jac,tau,lambda,out);},iters);
    op_ref<<<grid,block,shR>>>(d,u,mass,fx,fy,ne,N,rx,sy,jac,tau,lambda,out);
    std::vector<double> ref(ndof);CK(cudaMemcpy(ref.data(),out,ndof*8,cudaMemcpyDeviceToHost));
    int tg=(ne+127)/128;
    bench("op_transpose",[&]{op_transpose<<<tg,128>>>(uT,ne,N,rx,sy,jac,tau,lambda,outT);},iters);
    op_transpose<<<tg,128>>>(uT,ne,N,rx,sy,jac,tau,lambda,outT);
    std::vector<double> oT(ndof);CK(cudaMemcpy(oT.data(),outT,ndof*8,cudaMemcpyDeviceToHost));
    { double mr=0; long nbad=0;
      for(int e=0;e<ne;e++) for(int m=0;m<nn;m++){ double v=oT[(long)m*ne+e], r0=ref[(long)e*nn+m];
        double r=fabs(v-r0)/(fabs(r0)+1e-300); if(r>mr)mr=r; if(r>1e-9)nbad++; }
      printf("    transpose vs ref: max rel diff %.2e; %ld/%ld bad\n",mr,nbad,ndof); }
    { int pfe=4; if(const char*ev=getenv("PFE")){int v=atoi(ev);if(v>0)pfe=v;}   // elements/thread
      int pg=(ne+(128*pfe)-1)/(128*pfe);                                          // grid-stride grid
      bench("op_transpose_pf",[&]{op_transpose_pf<<<pg,128>>>(uT,ne,N,rx,sy,jac,tau,lambda,outT);},iters);
      op_transpose_pf<<<pg,128>>>(uT,ne,N,rx,sy,jac,tau,lambda,outT);
      std::vector<double> oP(ndof);CK(cudaMemcpy(oP.data(),outT,ndof*8,cudaMemcpyDeviceToHost));
      double mr=0; long nbad=0;
      for(int e=0;e<ne;e++) for(int m=0;m<nn;m++){ double v=oP[(long)m*ne+e], r0=ref[(long)e*nn+m];
        double r=fabs(v-r0)/(fabs(r0)+1e-300); if(r>mr)mr=r; if(r>1e-9)nbad++; }
      printf("    transpose_pf(pfe=%d) vs ref: max rel diff %.2e; %ld bad\n",pfe,mr,nbad); }
    int ag=(ne+63)/64;
    bench("op_assembled",[&]{op_assembled<<<ag,64>>>(uT,ne,N,outT);},iters);
    op_assembled<<<ag,64>>>(uT,ne,N,outT);
    std::vector<double> oA(ndof);CK(cudaMemcpy(oA.data(),outT,ndof*8,cudaMemcpyDeviceToHost));
    { double mr=0; long nbad=0, nint=0;   // INTERIOR elements only (all-interior assembly)
      for(int e=0;e<ne;e++){ int ex=e%N,ey=e/N; if(ex==0||ex==N-1||ey==0||ey==N-1) continue; nint++;
        for(int m=0;m<nn;m++){ double v=oA[(long)m*ne+e], r0=ref[(long)e*nn+m];
          double r=fabs(v-r0)/(fabs(r0)+1e-300); if(r>mr)mr=r; if(r>1e-9)nbad++; } }
      printf("    assembled vs ref (interior): max rel diff %.2e; %ld bad / %ld interior elems\n",mr,nbad,nint); }
    CK(cudaGetLastError());
    return 0;
}
