// Phase-E prototype: DG-SIPG Poisson MG-PCG, measuring iteration count with damped-Jacobi vs
// Chebyshev smoothers. CPU C++ first (iteration count is a GPU-independent numerical property) — the
// goal is to VALIDATE the algorithm + the Chebyshev win before porting the smoother to cuda-oxide.
//
// The operator is the REAL DG-SIPG Poisson (p=3/n1=4) with coefficients dumped from the validated Rust
// flatten_mesh (see DUMP_COEFFS): GLL weights w=[1/6,5/6,5/6,1/6], d (diff matrix, dumped), and per-level
// scaling rx=sy=2N, jac=1/(4N^2), tau=16*alpha*N (alpha=5), fsw[a]=w[a]/(2N). All-Dirichlet boundary.
//
// build:  g++ -O2 -std=c++17 mgpcg.cpp -o mgpcg && ./mgpcg 64
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <vector>
#include <functional>
using namespace std;

static const int n1 = 4, nn = 16;
static const double W[4]   = {1.0/6, 5.0/6, 5.0/6, 1.0/6};                 // GLL weights (p=3)
static const double XI[4]  = {-1.0, -0.44721359549995793, 0.44721359549995793, 1.0}; // GLL nodes
// diff matrix d[i*4+k] = L_k'(xi_i), dumped from Rust (row-major 4x4):
static const double D[16] = {
  -3.0, 4.0450849718747363, -1.5450849718747370, 0.5,
  -0.80901699437494734, 0.0, 1.1180339887498949, -0.30901699437494745,
   0.30901699437494745, -1.1180339887498949, 0.0, 0.80901699437494734,
  -0.5, 1.5450849718747370, -4.0450849718747363, 3.0 };
static double MASS[16];                                                    // mass[m=(a,b)] = w[a]*w[b]
static const double ALPHA = 5.0;                                           // SIPG penalty scale
static bool g_fp32 = false;                                                // simulate FP32 smoother storage
static inline double f32(double v){ return g_fp32 ? (double)(float)v : v; } // round to float precision

struct Level { int N, ne; long ndof; double rx, sy, jac, tau, fsw[4]; };
static long g_mv = 0;   // global operator-apply counter (the dominant per-step cost)

static Level make_level(int N){
    Level L; L.N=N; L.ne=N*N; L.ndof=(long)L.ne*nn;
    L.rx=L.sy=2.0*N; L.jac=1.0/(4.0*N*N); L.tau=16.0*ALPHA*N;
    for(int a=0;a<4;a++) L.fsw[a]=W[a]/(2.0*N);
    return L;
}

// y = A x  (DG-SIPG operator, lambda=0 pure Poisson, all-Dirichlet). Two passes: gradient-lift then div.
static void apply(const Level& L, const vector<double>& x, vector<double>& y){
    g_mv++;
    int N=L.N; double rx=L.rx, sy=L.sy, jac=L.jac, tau=L.tau;
    static vector<double> PR, PS, RF; PR.assign(L.ndof,0); PS.assign(L.ndof,0); RF.assign(L.ndof,0);
    for(int e=0;e<L.ne;e++){
        int ex=e%N, ey=e/N, eb=e*nn;
        for(int m=0;m<nn;m++){
            int i=m&3, j=m>>2; double jw=jac*MASS[m];
            double ur=0,uss=0;
            for(int k=0;k<4;k++){ ur+=D[i*4+k]*x[eb+k+j*4]; uss+=D[j*4+k]*x[eb+i+k*4]; }
            double gx=rx*ur, gy=sy*uss, ug=x[eb+m];
            double prv=rx*(jw*gx), psv=sy*(jw*gy), rf=0;
            // four faces (only those the node is on contribute)
            for(int f=0;f<4;f++){
                int on=0; double nx=0,ny=0,sw=0; int nbr=-1, ving=0,vjng=0; int xface=0;
                if(f==0){ on=(j==0); nx=0;ny=-1; sw=L.fsw[i]; xface=0; ving=i;vjng=3; if(on)nbr=(ey>0)?e-N:-1; }      // S
                else if(f==1){ on=(i==3); nx=1;ny=0; sw=L.fsw[j]; xface=1; ving=0;vjng=j; if(on)nbr=(ex<N-1)?e+1:-1; }// E
                else if(f==2){ on=(j==3); nx=0;ny=1; sw=L.fsw[i]; xface=0; ving=i;vjng=0; if(on)nbr=(ey<N-1)?e+N:-1; }// N
                else { on=(i==0); nx=-1;ny=0; sw=L.fsw[j]; xface=1; ving=3;vjng=j; if(on)nbr=(ex>0)?e-1:-1; }         // W
                if(!on) continue;
                double dun_e=nx*gx+ny*gy, avg,jump,gf;
                if(nbr<0){ avg=dun_e; jump=ug; gf=1.0; }                  // Dirichlet boundary
                else { int nb=nbr*nn; double s=0,ung;
                    if(xface){ for(int k=0;k<4;k++) s+=D[ving*4+k]*x[nb+k+vjng*4]; ung=x[nb+ving+vjng*4]; }
                    else     { for(int k=0;k<4;k++) s+=D[vjng*4+k]*x[nb+ving+k*4]; ung=x[nb+ving+vjng*4]; }
                    double dun_ng=xface? nx*(rx*s) : ny*(sy*s);
                    avg=0.5*(dun_e+dun_ng); jump=ug-ung; gf=0.5;
                }
                double g=gf*sw*jump; rf+=-sw*avg+tau*sw*jump; prv-=rx*(g*nx); psv-=sy*(g*ny);
            }
            PR[eb+m]=prv; PS[eb+m]=psv; RF[eb+m]=rf;
        }
    }
    for(int e=0;e<L.ne;e++){ int eb=e*nn;
        for(int m=0;m<nn;m++){ int i=m&3,j=m>>2; double accR=0,accS=0;
            for(int k=0;k<4;k++){ accR+=D[k*4+i]*PR[eb+k+j*4]; accS+=D[k*4+j]*PS[eb+i+k*4]; }
            y[eb+m]=accR+accS+RF[eb+m];
        }
    }
}

// diagonal via position-coloring: matching face nodes are at DIFFERENT positions (i=3<->i=0), so a unit
// vector at position p in every element has zero off-diagonal coupling at p ⇒ (A e_p)_p = D_pp exactly.
static void diagonal(const Level& L, vector<double>& dg){
    dg.assign(L.ndof,0); vector<double> u(L.ndof,0), au(L.ndof,0);
    for(int p=0;p<nn;p++){
        for(long e=0;e<L.ne;e++) u[e*nn+p]=1.0;
        apply(L,u,au);
        for(long e=0;e<L.ne;e++){ dg[e*nn+p]=au[e*nn+p]; u[e*nn+p]=0.0; }
    }
}

static double dot(const vector<double>& a,const vector<double>& b){ double s=0; for(size_t i=0;i<a.size();i++) s+=a[i]*b[i]; return s; }
static void axpy(vector<double>&y,double a,const vector<double>&x){ for(size_t i=0;i<y.size();i++) y[i]+=a*x[i]; }
static int cg(const Level& L, const vector<double>& b, vector<double>& x, double tol, int maxit); // fwd

// degree-3 Lagrange basis at GLL nodes, evaluated at zeta
static double lag(int k,double z){ double p=1; for(int m=0;m<4;m++) if(m!=k) p*=(z-XI[m])/(XI[k]-XI[m]); return p; }
// the four 2:1 prolongation matrices P[sub][(a,b)*16 + (k,l)] (fine node (a,b) ← coarse node (k,l))
static double PQ[4][256];
static void build_transfers(){
    for(int sx=0;sx<2;sx++)for(int sy=0;sy<2;sy++){ int sub=sy*2+sx;
        for(int b=0;b<4;b++)for(int a=0;a<4;a++){ int fn_=b*4+a;
            double zx=-1+sx+(XI[a]+1)/2, zy=-1+sy+(XI[b]+1)/2;
            for(int l=0;l<4;l++)for(int k=0;k<4;k++) PQ[sub][fn_*16 + l*4+k]=lag(k,zx)*lag(l,zy);
        }
    }
}
// prolong: fine += P * coarse  (Nf = 2*Nc)
static void prolong(int Nc,const vector<double>& c,vector<double>& f){
    int Nf=2*Nc;
    for(int Cy=0;Cy<Nc;Cy++)for(int Cx=0;Cx<Nc;Cx++){ int C=Cy*Nc+Cx;
        for(int sy=0;sy<2;sy++)for(int sx=0;sx<2;sx++){ int sub=sy*2+sx; int fe=(2*Cy+sy)*Nf+(2*Cx+sx);
            for(int fn_=0;fn_<16;fn_++){ double v=0; for(int cn=0;cn<16;cn++) v+=PQ[sub][fn_*16+cn]*c[C*16+cn]; f[fe*16+fn_]+=v; }
        }
    }
}
// restrict: coarse = P^T * fine  (variational)
static void restrict_(int Nc,const vector<double>& f,vector<double>& c){
    int Nf=2*Nc; c.assign((long)Nc*Nc*16,0);
    for(int Cy=0;Cy<Nc;Cy++)for(int Cx=0;Cx<Nc;Cx++){ int C=Cy*Nc+Cx;
        for(int sy=0;sy<2;sy++)for(int sx=0;sx<2;sx++){ int sub=sy*2+sx; int fe=(2*Cy+sy)*Nf+(2*Cx+sx);
            for(int fn_=0;fn_<16;fn_++){ double fv=f[fe*16+fn_]; for(int cn=0;cn<16;cn++) c[C*16+cn]+=PQ[sub][fn_*16+cn]*fv; }
        }
    }
}

// power iteration: largest eigenvalue of D^{-1} A
static double lam_max(const Level& L,const vector<double>& dg){
    vector<double> v(L.ndof), Av(L.ndof); for(long i=0;i<L.ndof;i++) v[i]=1.0+1e-3*((i*1103515245u+12345u)&1023)/1024.0;
    double lam=1;
    for(int it=0;it<50;it++){ apply(L,v,Av); for(long i=0;i<L.ndof;i++) Av[i]/=dg[i];
        double nv=sqrt(dot(Av,Av)); lam=nv/sqrt(dot(v,v)); for(long i=0;i<L.ndof;i++) v[i]=Av[i]/nv; }
    return lam;
}

struct MG { vector<Level> lev; vector<vector<double>> dg; vector<double> lmax; double omega; int nu, cheb_deg; double cheb_eta; bool use_cheb; };

static void jacobi(const Level& L,const vector<double>& dg,double omega,const vector<double>& b,vector<double>& x,int nu){
    vector<double> Ax(L.ndof);
    for(int s=0;s<nu;s++){ apply(L,x,Ax); for(long i=0;i<L.ndof;i++) x[i]=f32(x[i]+omega*(b[i]-Ax[i])/dg[i]); }
}
// Chebyshev smoother over high-freq range [a,b] = [lmax*eta, lmax*1.05], degree K
static void cheby(const Level& L,const vector<double>& dg,double lmax,double eta,const vector<double>& b,vector<double>& x,int K){
    double aa=eta*lmax, bb=1.05*lmax, theta=(bb+aa)/2, delta=(bb-aa)/2, sigma=theta/delta, rho=1.0/sigma;
    vector<double> r(L.ndof), Ax(L.ndof), d(L.ndof);
    apply(L,x,Ax); for(long i=0;i<L.ndof;i++){ r[i]=b[i]-Ax[i]; d[i]=f32(r[i]/dg[i]/theta); x[i]=f32(x[i]+d[i]); }
    for(int k=1;k<K;k++){ apply(L,x,Ax); double rho_n=1.0/(2*sigma-rho);
        for(long i=0;i<L.ndof;i++){ double z=(b[i]-Ax[i])/dg[i]; d[i]=f32(rho*rho_n*d[i]+(2*rho_n/delta)*z); x[i]=f32(x[i]+d[i]); }
        rho=rho_n; }
}
static void smooth(MG& mg,int l,const vector<double>& b,vector<double>& x){
    if(mg.use_cheb) cheby(mg.lev[l],mg.dg[l],mg.lmax[l],mg.cheb_eta,b,x,mg.cheb_deg);
    else jacobi(mg.lev[l],mg.dg[l],mg.omega/mg.lmax[l]*4.0/3.0,b,x,mg.nu);
}
static void vcycle(MG& mg,int l,const vector<double>& b,vector<double>& x){
    const Level& L=mg.lev[l];
    if(l==(int)mg.lev.size()-1){ cg(L,b,x,1e-12,500); return; }   // coarsest: direct-ish
    x.assign(L.ndof,0);
    smooth(mg,l,b,x);
    vector<double> Ax(L.ndof), r(L.ndof); apply(L,x,Ax); for(long i=0;i<L.ndof;i++) r[i]=b[i]-Ax[i];
    vector<double> rc; restrict_(mg.lev[l+1].N, r, rc);
    vector<double> ec(mg.lev[l+1].ndof,0); vcycle(mg,l+1,rc,ec);
    prolong(mg.lev[l+1].N, ec, x);                                // x += P ec
    smooth(mg,l,b,x);
}
// MG-preconditioned CG
static int mgpcg(MG& mg,const vector<double>& b,vector<double>& x,double tol,int maxit){
    const Level& L=mg.lev[0]; x.assign(L.ndof,0);
    vector<double> r=b, z(L.ndof), p, Ap(L.ndof);
    vcycle(mg,0,r,z); p=z; double rz=dot(r,z), bn=sqrt(dot(b,b))+1e-300;
    int it=0; for(; it<maxit; it++){ apply(L,p,Ap); double a=rz/dot(p,Ap);
        axpy(x,a,p); axpy(r,-a,Ap);
        if(sqrt(dot(r,r))/bn<tol){ it++; break; }
        vcycle(mg,0,r,z); double rz2=dot(r,z), beta=rz2/rz;
        for(long i=0;i<L.ndof;i++) p[i]=z[i]+beta*p[i]; rz=rz2;
    }
    return it;
}

// plain CG (no preconditioner) — validates the operator is SPD + gives a baseline iteration count.
static int cg(const Level& L, const vector<double>& b, vector<double>& x, double tol, int maxit){
    vector<double> r=b, p, Ap(L.ndof); x.assign(L.ndof,0);
    p=r; double rr=dot(r,r), bn=sqrt(dot(b,b))+1e-300;
    int it=0; for(; it<maxit; it++){
        apply(L,p,Ap); double a=rr/dot(p,Ap);
        for(long i=0;i<L.ndof;i++){ x[i]+=a*p[i]; r[i]-=a*Ap[i]; }
        double rr2=dot(r,r); if(sqrt(rr2)/bn<tol){ it++; break; }
        double beta=rr2/rr; for(long i=0;i<L.ndof;i++) p[i]=r[i]+beta*p[i]; rr=rr2;
    }
    return it;
}

int main(int argc,char**argv){
    int N=(argc>1)?atoi(argv[1]):64;
    for(int a=0;a<4;a++)for(int b=0;b<4;b++) MASS[a+b*4]=W[a]*W[b];
    Level L=make_level(N);
    printf("=== Stage 1: operator validation @ N=%d (ndof=%ld) ===\n", N, L.ndof);

    // Manufactured Dirichlet Poisson: u=sin(pi x)sin(pi y) (=0 on boundary ⇒ no lift), f=2 pi^2 u.
    // GLL-lumped RHS: b_m = jac*mass[m]*f(x_m).
    auto nodex=[&](int e,int m){ int ex=e%N,i=m&3; return (ex+(XI[i]+1)/2)/N; };
    auto nodey=[&](int e,int m){ int ey=e/N,j=m>>2; return (ey+(XI[j]+1)/2)/N; };
    const double PI=M_PI;
    vector<double> b(L.ndof), uex(L.ndof);
    for(int e=0;e<L.ne;e++)for(int m=0;m<nn;m++){
        double X=nodex(e,m),Y=nodey(e,m); uex[e*nn+m]=sin(PI*X)*sin(PI*Y);
        b[e*nn+m]=L.jac*MASS[m]*(2*PI*PI*sin(PI*X)*sin(PI*Y));
    }
    vector<double> x; int it=cg(L,b,x,1e-10,5000);
    double err=0,nrm=0; for(int e=0;e<L.ne;e++)for(int m=0;m<nn;m++){ double w=L.jac*MASS[m]; double d=x[e*nn+m]-uex[e*nn+m]; err+=w*d*d; nrm+=w*uex[e*nn+m]*uex[e*nn+m]; }
    printf("plain CG: %d iters; L2 rel error vs sin*sin = %.3e\n", it, sqrt(err/nrm));

    // ---- Stage 2: build MG hierarchy (2:1 h-coarsening down to N=2) ----
    build_transfers();
    MG mg; mg.omega=1.0; mg.nu=2; mg.cheb_deg=2;
    for(int Nl=N; Nl>=2; Nl/=2){ mg.lev.push_back(make_level(Nl)); }
    int nl=mg.lev.size();
    mg.dg.resize(nl); mg.lmax.resize(nl);
    for(int l=0;l<nl;l++){ diagonal(mg.lev[l],mg.dg[l]); mg.lmax[l]=lam_max(mg.lev[l],mg.dg[l]); }
    printf("MG: %d levels (N=%d..2), lmax(D^-1 A) fine=%.3f\n", nl, N, mg.lmax[0]);

    // transfer sanity: prolong a constant coarse field → fine should be ~constant (interpolation exact)
    { vector<double> c(mg.lev[1].ndof,1.0), f(mg.lev[0].ndof,0.0); prolong(mg.lev[1].N,c,f);
      double fmin=1e300,fmax=-1e300; for(double v:f){fmin=min(fmin,v);fmax=max(fmax,v);}
      printf("transfer check: prolong(const 1) → fine in [%.4f, %.4f] (want ~[1,1])\n", fmin, fmax); }

    vector<double> xs; double tol=1e-8; int mx=500;
    printf("\n  smoother              outer-iters   total-matvecs\n");
    auto run=[&](const char* name){ g_mv=0; int it=mgpcg(mg,b,xs,tol,mx); printf("  %-20s  %4d         %6ld\n", name, it, g_mv); return g_mv; };
    char buf[64];
    mg.use_cheb=false;
    for(int nu : {1,2,3}){ mg.nu=nu; snprintf(buf,64,"Jacobi(nu=%d)",nu); run(buf); }
    mg.use_cheb=true;
    for(double eta : {0.05,0.08,0.12})
      for(int K : {2,3,4}){ mg.cheb_deg=K; mg.cheb_eta=eta; snprintf(buf,64,"Cheby(deg=%d,eta=%.2f)",K,eta); run(buf); }
    printf("\n  FP32-smoother iter-hold check (Cheby deg=3 eta=0.05):\n");
    mg.cheb_deg=3; mg.cheb_eta=0.05;
    g_fp32=false; run("  FP64 smoother");
    g_fp32=true;  run("  FP32 smoother");
    return 0;
}
