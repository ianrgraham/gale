// Driver-API launcher for the Enzyme-differentiated gale `implicit_relax` kernel.
//
// Loads build/gale_kernel.cubin, runs:
//   primal_relax        @ (1/lambda)        -> Psi_out                (baseline)
//   primal_relax        @ (1/lambda + h)    -> Psi_out_h              (finite-difference)
//   d_implicit_dinvlam  @ (1/lambda)        -> Psi_out + d(Psi_out)/d(1/lambda)
// and checks that the Enzyme tangent matches the central/forward FD of the real kernel,
// and that the Enzyme primal equals the unperturbed primal exactly.
//
// Build/run via ./build.sh (which compiles and executes this).
#include <cuda.h>
#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#define CK(x) do{CUresult r=(x);if(r!=CUDA_SUCCESS){const char*s;cuGetErrorString(r,&s);printf("ERR %s @%d: %s\n",#x,__LINE__,s);exit(1);}}while(0)
int main(){
  int ne=4, nn=25, ndof=ne*nn; long n=ndof;
  double gamma=0.1, invlam=2.0, alpha=0.0, ext=1e308; int n1=5;     // Oldroyd-B branch (alpha=0, ext=inf)
  double *bxx=malloc(n*8),*bxy=malloc(n*8),*byy=malloc(n*8);
  for(int i=0;i<n;i++){ bxx[i]=0.5+0.3*sin(0.3*i); bxy[i]=0.1*cos(0.2*i); byy[i]=0.3+0.2*sin(0.1*i); }
  CK(cuInit(0)); CUdevice d; CK(cuDeviceGet(&d,0)); char nm[64]; cuDeviceGetName(nm,64,d); printf("device: %s\n",nm);
  CUcontext c; CK(cuCtxCreate(&c,0,d));
  CUmodule m; CK(cuModuleLoad(&m,"build/gale_kernel.cubin"));
  CUfunction fdiff,fprim; CK(cuModuleGetFunction(&fdiff,m,"d_implicit_dinvlam")); CK(cuModuleGetFunction(&fprim,m,"primal_relax"));
  CUdeviceptr dbx,dby,dbz,oxx,oxy,oyy,dxx,dxy,dyy,ob,oh,sxy,syy;
  CUdeviceptr *al[]={&dbx,&dby,&dbz,&oxx,&oxy,&oyy,&dxx,&dxy,&dyy,&ob,&oh,&sxy,&syy};
  for(int i=0;i<13;i++) CK(cuMemAlloc(al[i],n*8));
  CK(cuMemcpyHtoD(dbx,bxx,n*8));CK(cuMemcpyHtoD(dby,bxy,n*8));CK(cuMemcpyHtoD(dbz,byy,n*8));
  CK(cuMemsetD8(dxx,0,n*8));CK(cuMemsetD8(dxy,0,n*8));CK(cuMemsetD8(dyy,0,n*8));
  // FD base & +h via primal_relax (17 args)
  double h=1e-6, ilh=invlam+h;
  void* p0[]={&dbx,&n,&dby,&n,&dbz,&n,&gamma,&invlam,&alpha,&ext,&n1,&ob,&n,&sxy,&n,&syy,&n};
  CK(cuLaunchKernel(fprim,ne,1,1,nn,1,1,0,0,p0,0));
  void* ph[]={&dbx,&n,&dby,&n,&dbz,&n,&gamma,&ilh,&alpha,&ext,&n1,&oh,&n,&sxy,&n,&syy,&n};
  CK(cuLaunchKernel(fprim,ne,1,1,nn,1,1,0,0,ph,0));
  // forward-diff (15 args): primal oxx + tangent dxx = d(oxx)/d(invlam)
  void* ad[]={&dbx,&dby,&dbz,&n,&gamma,&invlam,&alpha,&ext,&n1,&oxx,&dxx,&oxy,&dxy,&oyy,&dyy};
  CK(cuLaunchKernel(fdiff,ne,1,1,nn,1,1,0,0,ad,0));
  CK(cuCtxSynchronize());
  double *vob=malloc(n*8),*voh=malloc(n*8),*vp=malloc(n*8),*vt=malloc(n*8);
  CK(cuMemcpyDtoH(vob,ob,n*8));CK(cuMemcpyDtoH(voh,oh,n*8));CK(cuMemcpyDtoH(vp,oxx,n*8));CK(cuMemcpyDtoH(vt,dxx,n*8));
  double maxprim=0,maxrel=0;
  for(int i=0;i<n;i++){ double fd=(voh[i]-vob[i])/h;
    maxprim=fmax(maxprim, fabs(vp[i]-vob[i]));
    double e=fabs(vt[i]-fd), r=e/(fabs(fd)+1e-9); if(r>maxrel)maxrel=r;
    if(i<4) printf("node %d: Psi_xx=%.6f  d(Psi_xx)/d(1/lambda): enzyme=%.6f  fd=%.6f\n", i, vob[i], vt[i], fd); }
  printf("max|diff_primal - primal| = %.2e ; max rel |enzyme - FD| = %.3e\n", maxprim, maxrel);
  int ok = maxprim<1e-9 && maxrel<1e-4;
  printf(ok?"\nPASS: gale's REAL implicit_relax kernel, Enzyme-differentiated, parameter gradient correct on the Titan V.\n":"\nFAIL\n");
  return ok?0:1;
}
