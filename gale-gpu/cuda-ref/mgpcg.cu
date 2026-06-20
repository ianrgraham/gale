// Phase-E GPU prototype: DG-SIPG Poisson MG-PCG in CUDA, measuring REAL ms/solve with damped-Jacobi vs
// Chebyshev smoothers + a profiling/optimization pass. Reuses the validated 44µs operator_fused_arith and
// the algorithm validated by the CPU prototype (mgpcg.cpp). Real DG-SIPG coefficients (dumped from Rust):
// d (diff matrix), w=[1/6,5/6,5/6,1/6], rx=sy=2N, jac=1/(4N^2), tau=16*alpha*N (alpha=5), fsw=w/(2N).
//
// build: nvcc -O3 -arch=sm_70 -Wno-deprecated-gpu-targets mgpcg.cu -o mgpcg_gpu && ./mgpcg_gpu 64
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <vector>
#define CK(x) do{ cudaError_t e=(x); if(e){ printf("CUDA %s:%d %s\n",__FILE__,__LINE__,cudaGetErrorString(e)); exit(1);} }while(0)
using namespace std;

static const int nn=16;
static const double HW[4]={1.0/6,5.0/6,5.0/6,1.0/6};
static const double HXI[4]={-1.0,-0.44721359549995793,0.44721359549995793,1.0};
static const double HD[16]={ -3.0,4.0450849718747363,-1.5450849718747370,0.5,
  -0.80901699437494734,0.0,1.1180339887498949,-0.30901699437494745,
   0.30901699437494745,-1.1180339887498949,0.0,0.80901699437494734,
  -0.5,1.5450849718747370,-4.0450849718747363,3.0 };
static const double ALPHA=5.0;

__device__ __forceinline__ double divergence(const double* sm,int pr,int ps,int i,int j){
    const double* D=sm;
    double accR=(D[0*4+i]*sm[pr+0+j*4]+D[1*4+i]*sm[pr+1+j*4])+(D[2*4+i]*sm[pr+2+j*4]+D[3*4+i]*sm[pr+3+j*4]);
    double accS=(D[0*4+j]*sm[ps+i+0]+D[1*4+j]*sm[ps+i+4])+(D[2*4+j]*sm[ps+i+8]+D[3*4+j]*sm[ps+i+12]);
    return accR+accS;
}
// validated 44us arith DG-SIPG matvec core (n1=4, Dirichlet). Returns (A u)_m; the three smoother kernels
// below share this body and differ only in the final write. g_out=global index, ug_out=u_m, act_out=active.
template<typename T>
__device__ __forceinline__ double matvec_m(const double* d,const T* u,const double* mass,
        const double* fswx,const double* fswy,int ne,int N,double rx,double sy,double jac,double tau,
        double lambda,double* sm,long& g_out,double& ug_out,bool& act_out){
    int t=threadIdx.x, epb=blockDim.x/nn, el=t/nn, m=t%nn, i=m&3, j=m>>2;
    int e=blockIdx.x*epb+el; bool active=e<ne; int pr=nn+el*nn, ps=nn+epb*nn+el*nn, eb=e*nn;
    if(t<nn) sm[t]=d[t];
    __syncthreads();
    const double* D=sm; int ex=e%N, ey=e/N;
    int vt=-1; double vnx=0; int vgb=-1,ving=0;
    if(i==0){vt=3;vnx=-1;if(ex>0){vgb=(e-1)*nn;ving=3;}} else if(i==3){vt=1;vnx=1;if(ex<N-1){vgb=(e+1)*nn;ving=0;}}
    int ht=-1; double hny=0; int hgb=-1,hjng=0;
    if(j==0){ht=0;hny=-1;if(ey>0){hgb=(e-N)*nn;hjng=3;}} else if(j==3){ht=2;hny=1;if(ey<N-1){hgb=(e+N)*nn;hjng=0;}}
    double qv0=0,qv1=0,qv2=0,qv3=0,qh0=0,qh1=0,qh2=0,qh3=0;
    if(active&&vgb>=0){qv0=u[vgb+0+j*4];qv1=u[vgb+1+j*4];qv2=u[vgb+2+j*4];qv3=u[vgb+3+j*4];}
    if(active&&hgb>=0){qh0=u[hgb+i+0];qh1=u[hgb+i+4];qh2=u[hgb+i+8];qh3=u[hgb+i+12];}
    double jw=jac*mass[m], ug=0, rf=0;
    if(active){
        double r0=u[eb+0+j*4],r1=u[eb+1+j*4],r2=u[eb+2+j*4],r3=u[eb+3+j*4];
        double c0=u[eb+i+0],c1=u[eb+i+4],c2=u[eb+i+8],c3=u[eb+i+12]; ug=u[eb+m];
        double ur=(D[i*4+0]*r0+D[i*4+1]*r1)+(D[i*4+2]*r2+D[i*4+3]*r3);
        double uss=(D[j*4+0]*c0+D[j*4+1]*c1)+(D[j*4+2]*c2+D[j*4+3]*c3);
        double gx=rx*ur, gy=sy*uss, prv=rx*(jw*gx), psv=sy*(jw*gy), hx=0, hy=0;
        if(vt>=0){ double sw=fswy[j], dun_e=vnx*gx, avg,jump,gf;
            if(vgb<0){avg=dun_e;jump=ug;gf=1.0;}
            else{ double s=(D[ving*4+0]*qv0+D[ving*4+1]*qv1)+(D[ving*4+2]*qv2+D[ving*4+3]*qv3);
                  double ung=(ving==0)?qv0:qv3; avg=0.5*(dun_e+vnx*(rx*s)); jump=ug-ung; gf=0.5; }
            double g=gf*sw*jump; rf+=-sw*avg+tau*sw*jump; hx+=g*vnx; }
        if(ht>=0){ double sw=fswx[i], dun_e=hny*gy, avg,jump,gf;
            if(hgb<0){avg=dun_e;jump=ug;gf=1.0;}
            else{ double s=(D[hjng*4+0]*qh0+D[hjng*4+1]*qh1)+(D[hjng*4+2]*qh2+D[hjng*4+3]*qh3);
                  double ung=(hjng==0)?qh0:qh3; avg=0.5*(dun_e+hny*(sy*s)); jump=ug-ung; gf=0.5; }
            double g=gf*sw*jump; rf+=-sw*avg+tau*sw*jump; hy+=g*hny; }
        sm[pr+m]=prv-rx*hx; sm[ps+m]=psv-sy*hy;
    }
    __syncthreads();
    g_out=eb+m; ug_out=ug; act_out=active;
    if(!active) return 0;
    return divergence(sm,pr,ps,i,j)+rf+lambda*jw*ug;
}
#define MV_ARGS const double*__restrict d,const double*__restrict u,const double*__restrict mass,const double*__restrict fswx,const double*__restrict fswy,int ne,int N,double rx,double sy,double jac,double tau,double lambda
#define MV_CALL d,u,mass,fswx,fswy,ne,N,rx,sy,jac,tau,lambda,sm,g,ug,act
__global__ void op_arith(MV_ARGS,double* __restrict out){
    extern __shared__ double sm[]; long g; double ug; bool act; double ap=matvec_m(MV_CALL); if(act) out[g]=ap; }
// fused damped-Jacobi step: out = u + omega*Dinv*(b - A u)   (out != u)
__global__ void op_jacobi(MV_ARGS,const double* b,const double* dinv,double omega,double* __restrict out){
    extern __shared__ double sm[]; long g; double ug; bool act; double ap=matvec_m(MV_CALL); if(act) out[g]=ug+omega*dinv[g]*(b[g]-ap); }
// fused Chebyshev step: dvec = c1*dvec + c2*Dinv*(b - A u); out = u + dvec   (out != u; dvec persists across the K steps)
__global__ void op_cheby(MV_ARGS,const double* b,const double* dinv,double* dvec,double c1,double c2,double* __restrict out){
    extern __shared__ double sm[]; long g; double ug; bool act; double ap=matvec_m(MV_CALL);
    if(act){ double dn=c1*dvec[g]+c2*dinv[g]*(b[g]-ap); dvec[g]=dn; out[g]=ug+dn; } }
// fused residual: out = b - A u  (replaces apply + k_sub; the matvec output never round-trips through DRAM)
__global__ void op_resid(MV_ARGS,const double* b,double* __restrict out){
    extern __shared__ double sm[]; long g; double ug; bool act; double ap=matvec_m(MV_CALL); if(act) out[g]=b[g]-ap; }
// fused CG iterate+residual update: x += alpha*p; r += nalpha*Ap  (one functional step, was 2 axpys)
__global__ void cg_xr(double* x,double* r,const double* p,const double* Ap,const double* al,const double* nal,long n){
    long i=(long)blockIdx.x*blockDim.x+threadIdx.x; if(i<n){ x[i]+=al[0]*p[i]; r[i]+=nal[0]*Ap[i]; } }
// templated Chebyshev step (fields type T) — for the FP32-vs-FP64 microbenchmark of op_cheby's bandwidth
template<typename T>
__global__ void op_cheby_t(const double* d,const T* u,const double* mass,const double* fswx,const double* fswy,int ne,int N,double rx,double sy,double jac,double tau,double lambda,const T* b,const T* dinv,T* dvec,double c1,double c2,T* out){
    extern __shared__ double sm[]; long g; double ug; bool act; double ap=matvec_m<T>(d,u,mass,fswx,fswy,ne,N,rx,sy,jac,tau,lambda,sm,g,ug,act);
    if(act){ double dn=c1*(double)dvec[g]+c2*(double)dinv[g]*((double)b[g]-ap); dvec[g]=(T)dn; out[g]=(T)(ug+dn); } }

// vector kernels
__global__ void k_axpy(double* y,double a,const double* x,long n){ long i=blockIdx.x*blockDim.x+threadIdx.x; if(i<n)y[i]+=a*x[i]; }       // y+=a x
__global__ void k_xpay(double* y,const double* x,double a,long n){ long i=blockIdx.x*blockDim.x+threadIdx.x; if(i<n)y[i]=x[i]+a*y[i]; }     // y=x+a y
__global__ void k_copy(double* y,const double* x,long n){ long i=blockIdx.x*blockDim.x+threadIdx.x; if(i<n)y[i]=x[i]; }
__global__ void k_setz(double* y,long n){ long i=blockIdx.x*blockDim.x+threadIdx.x; if(i<n)y[i]=0; }
__global__ void k_dot(const double* a,const double* b,long n,double* part){
    __shared__ double sh[256]; double s=0;
    for(long i=blockIdx.x*blockDim.x+threadIdx.x;i<n;i+=(long)gridDim.x*blockDim.x) s+=a[i]*b[i];
    sh[threadIdx.x]=s; __syncthreads();
    for(int k=blockDim.x/2;k>0;k>>=1){ if(threadIdx.x<k) sh[threadIdx.x]+=sh[threadIdx.x+k]; __syncthreads(); }
    if(threadIdx.x==0) part[blockIdx.x]=sh[0];
}

// Four 2:1 prolongation matrices [sub][fineNode*16 + coarseNode], in GLOBAL (not __constant__): each fine
// node reads a DIFFERENT cPQ row, so in constant memory the warp's reads diverge and the constant cache
// serializes them (one addr/cycle). Staged into SHARED per block instead → 32 parallel banks, no serialization.
static double* g_pq=0;
// f += P c — one BLOCK per coarse element (64 threads). cPQ (1024) + the 16 coarse values staged to shared.
__global__ void k_prolong(const double* c,const double* pq,double* f,int Nc){
    __shared__ double spq[1024]; __shared__ double cs[16];
    int t=threadIdx.x;
    for(int k=t;k<1024;k+=64) spq[k]=pq[k];
    int C=blockIdx.x, Nf=2*Nc, Cx=C%Nc, Cy=C/Nc;
    if(t<16) cs[t]=c[C*16+t];
    __syncthreads();
    int sub=t>>4, fnode=t&15, sx=sub&1, sy=sub>>1; long fe=(long)(2*Cy+sy)*Nf+(2*Cx+sx);
    double v=0; for(int cn=0;cn<16;cn++) v+=spq[sub*256+fnode*16+cn]*cs[cn]; f[fe*16+fnode]+=v;
}
// c = P^T f — one BLOCK per coarse element; cPQ (1024) + the 64 fine values staged to shared.
__global__ void k_restrict(const double* f,const double* pq,double* c,int Nc){
    __shared__ double spq[1024]; __shared__ double fs[64];
    int t=threadIdx.x;
    for(int k=t;k<1024;k+=64) spq[k]=pq[k];
    int C=blockIdx.x, Nf=2*Nc, Cx=C%Nc, Cy=C/Nc;
    int sub=t>>4, fnode=t&15, sx=sub&1, sy=sub>>1; long fe=(long)(2*Cy+sy)*Nf+(2*Cx+sx);
    fs[t]=f[fe*16+fnode];
    __syncthreads();
    if(t<16){ int cn=t; double s=0; for(int s2=0;s2<4;s2++) for(int fn=0;fn<16;fn++) s+=spq[s2*256+fn*16+cn]*fs[s2*16+fn]; c[C*16+cn]=s; }
}
__global__ void k_setcol(double* u,int p,double val,long ne){ long e=(long)blockIdx.x*blockDim.x+threadIdx.x; if(e<ne) u[e*16+p]=val; }
__global__ void k_getcol(const double* au,int p,double* dg,long ne){ long e=(long)blockIdx.x*blockDim.x+threadIdx.x; if(e<ne){ dg[e*16+p]=au[e*16+p]; } }
__global__ void k_sub(double* r,const double* a,const double* b,long n){ long i=(long)blockIdx.x*blockDim.x+threadIdx.x; if(i<n) r[i]=a[i]-b[i]; } // r=a-b

struct Lev { int N,ne; long ndof; double rx,sy,jac,tau; double *fswx,*fswy,*d,*mass; };
static const int EPB=6;
static int og(int ne){ return (ne+EPB-1)/EPB; }
static size_t osh(){ return (nn+2*EPB*nn)*sizeof(double); }

static double *g_d=0,*g_mass=0; static int DOTB=256;
static double host_dot(const double* a,const double* b,long n,double* part){
    int nb=min((long)1024,(n+255)/256); k_dot<<<nb,256>>>(a,b,n,part);
    static vector<double> hp; hp.resize(nb); CK(cudaMemcpy(hp.data(),part,nb*8,cudaMemcpyDeviceToHost));
    double s=0; for(int i=0;i<nb;i++) s+=hp[i]; return s;
}
static void vaxpy(double*y,double a,const double*x,long n){ k_axpy<<<(n+255)/256,256>>>(y,a,x,n); }
static void vxpay(double*y,const double*x,double a,long n){ k_xpay<<<(n+255)/256,256>>>(y,x,a,n); }
static void apply(const Lev& L,const double* u,double* out){ op_arith<<<og(L.ne),EPB*nn,osh()>>>(L.d,u,L.mass,L.fswx,L.fswy,L.ne,L.N,L.rx,L.sy,L.jac,L.tau,0.0,out); }

static Lev make_lev(int N){
    Lev L; L.N=N; L.ne=N*N; L.ndof=(long)L.ne*nn; L.rx=L.sy=2.0*N; L.jac=1.0/(4.0*N*N); L.tau=16.0*ALPHA*N;
    double hfx[4]; for(int a=0;a<4;a++) hfx[a]=HW[a]/(2.0*N);
    CK(cudaMalloc(&L.fswx,4*8)); CK(cudaMalloc(&L.fswy,4*8));
    CK(cudaMemcpy(L.fswx,hfx,4*8,cudaMemcpyHostToDevice)); CK(cudaMemcpy(L.fswy,hfx,4*8,cudaMemcpyHostToDevice));
    L.d=g_d; L.mass=g_mass; return L;
}

// plain CG (GPU), preallocated scratch r,p,Ap — validate operator, baseline, and coarsest-level solve
static int cg(const Lev& L,const double* b,double* x,double* r,double* p,double* Ap,double* part,double tol,int maxit){
    long n=L.ndof;
    k_setz<<<(n+255)/256,256>>>(x,n); CK(cudaMemcpy(r,b,n*8,cudaMemcpyDeviceToDevice)); CK(cudaMemcpy(p,b,n*8,cudaMemcpyDeviceToDevice));
    double rr=host_dot(r,r,n,part), bn=sqrt(host_dot(b,b,n,part))+1e-300; int it=0;
    for(;it<maxit;it++){ apply(L,p,Ap); double a=rr/host_dot(p,Ap,n,part);
        vaxpy(x,a,p,n); vaxpy(r,-a,Ap,n); double rr2=host_dot(r,r,n,part);
        if(sqrt(rr2)/bn<tol){ it++; break; } double beta=rr2/rr; vxpay(p,r,beta,n); rr=rr2; }
    return it;
}

static void vsetz(double*y,long n){ k_setz<<<(n+255)/256,256>>>(y,n); }
static void vcopy(double*y,const double*x,long n){ k_copy<<<(n+255)/256,256>>>(y,x,n); }

struct MG { vector<Lev> lev; vector<double*> xb,bb,rb,tb,db,dg; vector<double> lmax; int nu,deg; double eta; bool cheb; double* part; double* csc; int coarse_fixed; };

// diagonal via position-coloring (16 probe matvecs); u,au scratch (level-sized)
static void diagonal(const Lev& L,double* dg,double* u,double* au){
    long ne=L.ne; vsetz(u,L.ndof);
    for(int p=0;p<16;p++){ k_setcol<<<(ne+255)/256,256>>>(u,p,1.0,ne); apply(L,u,au);
        k_getcol<<<(ne+255)/256,256>>>(au,p,dg,ne); k_setcol<<<(ne+255)/256,256>>>(u,p,0.0,ne); }
}
// largest eigenvalue of D^{-1}A by power iteration (scratch v,Av; part for dots)
__global__ void k_divv(double* y,const double* d,long n){ long i=(long)blockIdx.x*blockDim.x+threadIdx.x; if(i<n) y[i]/=d[i]; }
__global__ void k_inv(double* d,long n){ long i=(long)blockIdx.x*blockDim.x+threadIdx.x; if(i<n) d[i]=1.0/d[i]; }
static double lam_max(const Lev& L,const double* dg,double* v,double* Av,double* part){
    vector<double> h(L.ndof); for(long i=0;i<L.ndof;i++) h[i]=1.0+1e-3*((i*1103515245u+12345u)&1023)/1024.0;
    CK(cudaMemcpy(v,h.data(),L.ndof*8,cudaMemcpyHostToDevice)); double lam=1;
    for(int it=0;it<15;it++){ apply(L,v,Av); k_divv<<<(L.ndof+255)/256,256>>>(Av,dg,L.ndof);
        double nv=sqrt(host_dot(Av,Av,L.ndof,part)); lam=nv/sqrt(host_dot(v,v,L.ndof,part));
        k_setz<<<(L.ndof+255)/256,256>>>(v,L.ndof); vaxpy(v,1.0/nv,Av,L.ndof); }
    return lam;
}
// one smoother application (nu Jacobi sweeps OR deg Chebyshev steps); ping-pong x<->tb, dvec persists.
static void smooth(MG& mg,int l,const double* b,double* x){
    const Lev& L=mg.lev[l]; long n=L.ndof; double* tb=mg.tb[l]; double *a=x,*c=tb;
    if(!mg.cheb){ double om=4.0/(3.0*mg.lmax[l]);
        for(int s=0;s<mg.nu;s++){ op_jacobi<<<og(L.ne),EPB*nn,osh()>>>(L.d,a,L.mass,L.fswx,L.fswy,L.ne,L.N,L.rx,L.sy,L.jac,L.tau,0.0,b,mg.dg[l],om,c); double*t=a;a=c;c=t; }
    } else { double aa=mg.eta*mg.lmax[l],bb=1.05*mg.lmax[l],th=(bb+aa)/2,de=(bb-aa)/2,sg=th/de,rho=1.0/sg;
        double* dv=mg.db[l];
        for(int k=0;k<mg.deg;k++){ double c1,c2; if(k==0){c1=0;c2=1.0/th;} else{ double rn=1.0/(2*sg-rho); c1=rho*rn; c2=2*rn/de; rho=rn; }
            op_cheby<<<og(L.ne),EPB*nn,osh()>>>(L.d,a,L.mass,L.fswx,L.fswy,L.ne,L.N,L.rx,L.sy,L.jac,L.tau,0.0,b,mg.dg[l],dv,c1,c2,c); double*t=a;a=c;c=t; }
    }
    if(a!=x) vcopy(x,a,n);   // ensure result lands in x
}
static void vcycle(MG& mg,int l,const double* b,double* x){
    const Lev& L=mg.lev[l]; long n=L.ndof;
    if(l==(int)mg.lev.size()-1){ vsetz(x,n); cg(L,b,x,mg.rb[l],mg.tb[l],mg.db[l],mg.part,1e-10,300); return; }
    vsetz(x,n);
    smooth(mg,l,b,x);
    double* r=mg.rb[l]; op_resid<<<og(L.ne),EPB*nn,osh()>>>(L.d,x,L.mass,L.fswx,L.fswy,L.ne,L.N,L.rx,L.sy,L.jac,L.tau,0.0,b,r);   // r = b - A x
    int Nc=mg.lev[l+1].N; int cne=mg.lev[l+1].ne;
    k_restrict<<<cne,64>>>(r,g_pq,mg.bb[l+1],Nc);
    vcycle(mg,l+1,mg.bb[l+1],mg.xb[l+1]);
    k_prolong<<<cne,64>>>(mg.xb[l+1],g_pq,x,Nc);                                  // x += P e_c
    smooth(mg,l,b,x);
}
static int mgpcg(MG& mg,const double* b,double* x,double* r,double* z,double* p,double* Ap,double* part,double tol,int maxit){
    const Lev& L=mg.lev[0]; long n=L.ndof;
    vsetz(x,n); vcopy(r,b,n);
    vcycle(mg,0,r,z); vcopy(p,z,n); double rz=host_dot(r,z,n,part), bn=sqrt(host_dot(b,b,n,part))+1e-300; int it=0;
    for(;it<maxit;it++){ apply(L,p,Ap); double a=rz/host_dot(p,Ap,n,part);
        vaxpy(x,a,p,n); vaxpy(r,-a,Ap,n); if(sqrt(host_dot(r,r,n,part))/bn<tol){ it++; break; }
        vcycle(mg,0,r,z); double rz2=host_dot(r,z,n,part),beta=rz2/rz; vxpay(p,z,beta,n); rz=rz2; }
    return it;
}

// ===== device-resident pieces (no host sync ⇒ capturable into a CUDA graph) =====
__global__ void k_reduce(const double* part,int nb,double* out){     // sum nb partials → out[0]
    __shared__ double sh[256]; int t=threadIdx.x; double s=0; for(int i=t;i<nb;i+=blockDim.x) s+=part[i];
    sh[t]=s; __syncthreads(); for(int k=128;k>0;k>>=1){ if(t<k) sh[t]+=sh[t+k]; __syncthreads(); } if(t==0) out[0]=sh[0]; }
__global__ void k_alpha(double* a,double* na,const double* num,const double* den){ if(!threadIdx.x){ double v=num[0]/den[0]; a[0]=v; na[0]=-v; } }
__global__ void k_beta(double* b,const double* num,const double* den){ if(!threadIdx.x) b[0]=num[0]/den[0]; }
__global__ void k_cps(double* d,const double* s){ if(!threadIdx.x) d[0]=s[0]; }
__global__ void k_axpyd(double* y,const double* s,double sc,const double* x,long n){ long i=(long)blockIdx.x*blockDim.x+threadIdx.x; if(i<n) y[i]+=sc*s[0]*x[i]; }
__global__ void k_xpayd(double* y,const double* x,const double* s,long n){ long i=(long)blockIdx.x*blockDim.x+threadIdx.x; if(i<n) y[i]=x[i]+s[0]*y[i]; }
static void ddot(const double* a,const double* b,long n,double* part,double* out){   // device-resident dot
    int nb=min((long)1024,(n+255)/256); k_dot<<<nb,256>>>(a,b,n,part); k_reduce<<<1,256>>>(part,nb,out);
}
// device-resident fixed-iter plain CG (coarsest solve), scratch r,p,Ap + 6 device scalars sc[rr,pAp,al,nal,be,rrn]
static void cg_dev(const Lev& L,const double* b,double* x,double* r,double* p,double* Ap,double* sc,double* part,int fixed){
    long n=L.ndof; vsetz(x,n); vcopy(r,b,n); vcopy(p,b,n); ddot(r,r,n,part,sc+0);
    for(int it=0;it<fixed;it++){ apply(L,p,Ap); ddot(p,Ap,n,part,sc+1); k_alpha<<<1,1>>>(sc+2,sc+3,sc+0,sc+1);
        cg_xr<<<(n+255)/256,256>>>(x,r,p,Ap,sc+2,sc+3,n);
        ddot(r,r,n,part,sc+5); k_beta<<<1,1>>>(sc+4,sc+5,sc+0); k_xpayd<<<(n+255)/256,256>>>(p,r,sc+4,n); k_cps<<<1,1>>>(sc+0,sc+5); }
}
static void vcycle_dev(MG& mg,int l,const double* b,double* x){     // like vcycle but coarse = cg_dev (device-resident)
    const Lev& L=mg.lev[l]; long n=L.ndof;
    if(l==(int)mg.lev.size()-1){ cg_dev(L,b,x,mg.rb[l],mg.tb[l],mg.db[l],mg.csc,mg.part,mg.coarse_fixed); return; }
    vsetz(x,n); smooth(mg,l,b,x);
    double* r=mg.rb[l]; op_resid<<<og(L.ne),EPB*nn,osh()>>>(L.d,x,L.mass,L.fswx,L.fswy,L.ne,L.N,L.rx,L.sy,L.jac,L.tau,0.0,b,r);
    int Nc=mg.lev[l+1].N, cne=mg.lev[l+1].ne;
    k_restrict<<<cne,64>>>(r,g_pq,mg.bb[l+1],Nc); vcycle_dev(mg,l+1,mg.bb[l+1],mg.xb[l+1]); k_prolong<<<cne,64>>>(mg.xb[l+1],g_pq,x,Nc);
    smooth(mg,l,b,x);
}
// device-resident fixed-iter MG-PCG (capturable). r,z,p,Ap scratch; osc[6] outer scalars.
static void mgpcg_dev(MG& mg,const double* b,double* x,double* r,double* z,double* p,double* Ap,double* osc,double* part,int fixed){
    const Lev& L=mg.lev[0]; long n=L.ndof;
    vsetz(x,n); vcopy(r,b,n); vcycle_dev(mg,0,r,z); vcopy(p,z,n); ddot(r,z,n,part,osc+0);
    for(int it=0;it<fixed;it++){ apply(L,p,Ap); ddot(p,Ap,n,part,osc+1); k_alpha<<<1,1>>>(osc+2,osc+3,osc+0,osc+1);
        cg_xr<<<(n+255)/256,256>>>(x,r,p,Ap,osc+2,osc+3,n);
        vcycle_dev(mg,0,r,z); ddot(r,z,n,part,osc+5); k_beta<<<1,1>>>(osc+4,osc+5,osc+0); k_xpayd<<<(n+255)/256,256>>>(p,z,osc+4,n); k_cps<<<1,1>>>(osc+0,osc+5); }
}

static double lagh(int k,double z){ double p=1; for(int m=0;m<4;m++) if(m!=k) p*=(z-HXI[m])/(HXI[k]-HXI[m]); return p; }
static void build_pq(){
    vector<double> h(4*256,0);
    for(int sx=0;sx<2;sx++)for(int sy=0;sy<2;sy++){ int sub=sy*2+sx;
        for(int b=0;b<4;b++)for(int a=0;a<4;a++){ int fn=b*4+a; double zx=-1+sx+(HXI[a]+1)/2, zy=-1+sy+(HXI[b]+1)/2;
            for(int l=0;l<4;l++)for(int k=0;k<4;k++) h[sub*256+fn*16+l*4+k]=lagh(k,zx)*lagh(l,zy); } }
    CK(cudaMalloc(&g_pq,4*256*8)); CK(cudaMemcpy(g_pq,h.data(),4*256*8,cudaMemcpyHostToDevice));
}

// time op_cheby_t<T> (fields type T) — measures the BW effect of FP32 vs FP64 on the smoother kernel
template<typename T>
static float micro_cheby(const Lev& L,long n){
    T *u,*b,*dv,*di,*out; CK(cudaMalloc(&u,n*sizeof(T)));CK(cudaMalloc(&b,n*sizeof(T)));CK(cudaMalloc(&dv,n*sizeof(T)));CK(cudaMalloc(&di,n*sizeof(T)));CK(cudaMalloc(&out,n*sizeof(T)));
    vector<T> h(n); for(long i=0;i<n;i++) h[i]=(T)(1.0+1e-3*((i*1103515245u+12345u)&1023)/1024.0);
    CK(cudaMemcpy(u,h.data(),n*sizeof(T),cudaMemcpyHostToDevice)); CK(cudaMemcpy(b,h.data(),n*sizeof(T),cudaMemcpyHostToDevice));
    for(long i=0;i<n;i++) h[i]=(T)0.5; CK(cudaMemcpy(di,h.data(),n*sizeof(T),cudaMemcpyHostToDevice)); CK(cudaMemset(dv,0,n*sizeof(T)));
    auto run=[&]{ op_cheby_t<T><<<og(L.ne),EPB*nn,osh()>>>(L.d,u,L.mass,L.fswx,L.fswy,L.ne,L.N,L.rx,L.sy,L.jac,L.tau,0.0,b,di,dv,0.3,0.5,out); };
    for(int w=0;w<5;w++) run(); cudaDeviceSynchronize();
    cudaEvent_t s,e; cudaEventCreate(&s); cudaEventCreate(&e); cudaEventRecord(s); for(int r=0;r<300;r++) run(); cudaEventRecord(e); cudaEventSynchronize(e);
    float ms; cudaEventElapsedTime(&ms,s,e); cudaFree(u);cudaFree(b);cudaFree(dv);cudaFree(di);cudaFree(out); return ms/300*1e3;
}

int main(int argc,char**argv){
    int N=(argc>1)?atoi(argv[1]):64;
    double hmass[16]; for(int a=0;a<4;a++)for(int b=0;b<4;b++) hmass[a+b*4]=HW[a]*HW[b];
    CK(cudaMalloc(&g_d,16*8)); CK(cudaMalloc(&g_mass,16*8));
    CK(cudaMemcpy(g_d,HD,16*8,cudaMemcpyHostToDevice)); CK(cudaMemcpy(g_mass,hmass,16*8,cudaMemcpyHostToDevice));
    build_pq();
    Lev L=make_lev(N); long n=L.ndof;
    if(getenv("MICRO")){ printf("op_cheby @N=%d: FP64 %.2f us | FP32 %.2f us\n", N, micro_cheby<double>(L,n), micro_cheby<float>(L,n)); return 0; }
    printf("=== GPU MG-PCG @ N=%d ndof=%ld ===\n",N,n);

    // manufactured Dirichlet Poisson: u=sin(pi x)sin(pi y), b_m=jac*mass*f, f=2pi^2 u
    vector<double> hb(n),huex(n); const double PI=M_PI;
    for(int e=0;e<L.ne;e++){ int ex=e%N,ey=e/N; for(int m=0;m<nn;m++){ int i=m&3,j=m>>2;
        double X=(ex+(HXI[i]+1)/2)/N, Y=(ey+(HXI[j]+1)/2)/N; huex[e*nn+m]=sin(PI*X)*sin(PI*Y);
        hb[e*nn+m]=L.jac*hmass[m]*(2*PI*PI*sin(PI*X)*sin(PI*Y)); } }
    double *b,*x,*part,*mr,*mz,*mp,*mAp;
    CK(cudaMalloc(&b,n*8)); CK(cudaMalloc(&x,n*8)); CK(cudaMalloc(&part,1024*8));
    CK(cudaMalloc(&mr,n*8)); CK(cudaMalloc(&mz,n*8)); CK(cudaMalloc(&mp,n*8)); CK(cudaMalloc(&mAp,n*8));
    CK(cudaMemcpy(b,hb.data(),n*8,cudaMemcpyHostToDevice));

    bool prof=getenv("PROF")!=0;
    if(!prof){ int it=cg(L,b,x,mr,mp,mAp,part,1e-10,5000); CK(cudaDeviceSynchronize());
        vector<double> hx(n); CK(cudaMemcpy(hx.data(),x,n*8,cudaMemcpyDeviceToHost));
        double err=0,nrm=0; for(int e=0;e<L.ne;e++)for(int m=0;m<nn;m++){ double w=L.jac*hmass[m],dd=hx[e*nn+m]-huex[e*nn+m]; err+=w*dd*dd; nrm+=w*huex[e*nn+m]*huex[e*nn+m]; }
        printf("plain CG (GPU): %d iters; L2 rel error = %.3e\n", it, sqrt(err/nrm)); }

    // ---- build MG hierarchy (2:1 down to N=2) ----
    MG mg; mg.nu=2; mg.deg=3; mg.eta=0.05; mg.part=part;
    int cfloor=getenv("CFLOOR")?atoi(getenv("CFLOOR")):2;
    for(int Nl=N;Nl>=cfloor;Nl/=2) mg.lev.push_back(make_lev(Nl));
    int nl=mg.lev.size();
    for(int l=0;l<nl;l++){ long m=mg.lev[l].ndof; double *xb,*bb,*rb,*tb,*db,*dg;
        CK(cudaMalloc(&xb,m*8));CK(cudaMalloc(&bb,m*8));CK(cudaMalloc(&rb,m*8));CK(cudaMalloc(&tb,m*8));CK(cudaMalloc(&db,m*8));CK(cudaMalloc(&dg,m*8));
        mg.xb.push_back(xb);mg.bb.push_back(bb);mg.rb.push_back(rb);mg.tb.push_back(tb);mg.db.push_back(db);mg.dg.push_back(dg); }
    for(int l=0;l<nl;l++){ diagonal(mg.lev[l],mg.dg[l],mg.xb[l],mg.tb[l]); mg.lmax.push_back(lam_max(mg.lev[l],mg.dg[l],mg.xb[l],mg.tb[l],part));
        k_inv<<<(mg.lev[l].ndof+255)/256,256>>>(mg.dg[l],mg.lev[l].ndof); }   // dg: D → 1/D for the smoother (multiply)
    double *osc,*csc; CK(cudaMalloc(&osc,6*8)); CK(cudaMalloc(&csc,6*8)); mg.csc=csc; mg.coarse_fixed=getenv("CFIX")?atoi(getenv("CFIX")):24;
    CK(cudaDeviceSynchronize());
    printf("MG: %d levels, lmax(fine)=%.3f\n\n", nl, mg.lmax[0]);
    printf("  smoother                host-iters   ms/solve(host)   ms/solve(graph)   resid\n");

    cudaEvent_t s,e2; cudaEventCreate(&s); cudaEventCreate(&e2);
    auto bench=[&](const char* name,bool cheb){
        mg.cheb=cheb;
        int iters; float msh=0;
        if(prof){ iters=cheb?16:29; }                                                            // PROF: skip host path → clean graph-only profile
        else {
            iters=mgpcg(mg,b,x,mr,mz,mp,mAp,part,1e-8,500); CK(cudaDeviceSynchronize());          // host path → iter count
            for(int w=0;w<2;w++) mgpcg(mg,b,x,mr,mz,mp,mAp,part,1e-8,500); CK(cudaDeviceSynchronize());
            int R=20; cudaEventRecord(s); for(int r=0;r<R;r++) mgpcg(mg,b,x,mr,mz,mp,mAp,part,1e-8,500); cudaEventRecord(e2); cudaEventSynchronize(e2);
            cudaEventElapsedTime(&msh,s,e2); msh/=R;
        }
        // device-resident fixed-iter solve, captured into a CUDA graph (1 launch vs ~4000)
        mgpcg_dev(mg,b,x,mr,mz,mp,mAp,osc,part,iters); CK(cudaDeviceSynchronize());
        cudaGraph_t graph; cudaGraphExec_t exec;
        CK(cudaStreamBeginCapture(cudaStreamPerThread,cudaStreamCaptureModeThreadLocal));
        mgpcg_dev(mg,b,x,mr,mz,mp,mAp,osc,part,iters);
        CK(cudaStreamEndCapture(cudaStreamPerThread,&graph)); CK(cudaGraphInstantiate(&exec,graph,0));
        cudaGraphLaunch(exec,cudaStreamPerThread); CK(cudaStreamSynchronize(cudaStreamPerThread));
        int RG=50; cudaEventRecord(s); for(int r=0;r<RG;r++) cudaGraphLaunch(exec,cudaStreamPerThread); cudaEventRecord(e2); cudaEventSynchronize(e2);
        float msg; cudaEventElapsedTime(&msg,s,e2);
        apply(mg.lev[0],x,mAp); k_sub<<<(n+255)/256,256>>>(mAp,b,mAp,n);
        double resid=sqrt(host_dot(mAp,mAp,n,part)/host_dot(b,b,n,part));
        printf("  %-22s  %3d         %7.3f          %7.3f          %.1e\n", name, iters, msh, msg/RG, resid);
        cudaGraphExecDestroy(exec); cudaGraphDestroy(graph);
    };
    if(!prof) bench("Jacobi(nu=2)", false);
    char nm[48]; snprintf(nm,48,"Chebyshev(deg=%d,eta=%.2f)",mg.deg,mg.eta);
    bench(nm, true);
    return 0;
}
