/* Does optimizer traffic hide under compute? The claim everything rests on.
 *
 *   nvcc -O3 -arch=sm_120 -lcublas -o overlap vram/probe/overlap.cu && ./overlap
 *   ./overlap [gemms] [params]        defaults: 24 GEMMs of 4096^3, 50M params
 *
 * `train.py` prices a step as max(compute, traffic). If it is really
 * compute + traffic then every token floor in this repository is out by up to
 * 2x and the design needs rethinking. So: one layer's worth of 8-bit Adam
 * state streaming host <-> device, against a GEMM chain standing in for the
 * backward pass it is supposed to hide under.
 *
 * Configurations, all interleaved within each round:
 *   GEMM alone / optimizer alone / both / GEMM+DMA only / GEMM+kernel only
 *   plus a CONTROL: DMA against a kernel that touches no memory at all.
 *
 * The control is the important one. It separates "the copy engine cannot
 * overlap" from "the copy engine overlaps fine but wrecks the GEMM's cache",
 * and on this machine the answer was emphatically the second: DMA against a
 * pure-ALU kernel overlaps 102%, and the same DMA against cuBLAS costs very
 * nearly its full standalone time. Streaming traffic evicts the tiles the
 * GEMM reuses out of L2.
 *
 * Two traps this file was built around, both of which caught the first
 * version. Measure each configuration once per round, not one configuration
 * at a time -- clocks drift enough on this card that whichever ran first
 * looked fastest. And warm to steady state before timing anything: burst
 * GEMM on this GPU is ~119 TFLOP/s and sustained is 35-60, so a cold harness
 * measures a machine that does not exist during training.
 *
 * The Adam arithmetic is not numerically faithful -- the moments are treated
 * as raw bytes. What is faithful is the traffic (8 bytes read and 8 written
 * per parameter) and the occupancy, which is what the answer depends on.
 */
#include <cstdio>
#include <algorithm>
#include <vector>
#include <chrono>
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cublas_v2.h>
static double now(){using namespace std::chrono;return duration<double>(steady_clock::now().time_since_epoch()).count();}
__global__ void adam8(const __nv_bfloat16*__restrict__ g,float*__restrict__ ma,unsigned char*__restrict__ m,
                      unsigned char*__restrict__ v,__nv_bfloat16*__restrict__ w,size_t n,float lr){
  size_t i=blockIdx.x*(size_t)blockDim.x+threadIdx.x, st=(size_t)gridDim.x*blockDim.x;
  for(;i<n;i+=st){ float gi=__bfloat162float(g[i]); float mi=m[i]*(1.f/255.f), vi=v[i]*(1.f/255.f);
    mi=0.9f*mi+0.1f*gi; vi=0.999f*vi+0.001f*gi*gi; float pp=ma[i]-lr*mi/(sqrtf(fabsf(vi))+1e-8f);
    ma[i]=pp; m[i]=(unsigned char)fminf(255.f,fabsf(mi)*255.f); v[i]=(unsigned char)fminf(255.f,fabsf(vi)*255.f);
    w[i]=__float2bfloat16(pp);} }
/* Control: no memory traffic at all, so cache contention cannot be blamed. */
__global__ void spin(long iters,float*sink){
  float a=threadIdx.x*1e-6f,b=1.0000001f;
  for(long i=0;i<iters;i++) a=fmaf(a,b,1e-7f);
  if(a==12345.f) sink[0]=a;
}

int main(int argc,char**argv){
  const int G=(argc>1)?atoi(argv[1]):24; const size_t N=(argc>2)?(size_t)atol(argv[2]):50000000;
  const int n=4096, ROUNDS=15;
  cudaDeviceProp p; cudaGetDeviceProperties(&p,0);
  cublasHandle_t cb; cublasCreate(&cb);
  __nv_bfloat16 *A,*B; float*C; cudaMalloc(&A,(size_t)n*n*2);cudaMalloc(&B,(size_t)n*n*2);cudaMalloc(&C,(size_t)n*n*4);
  __nv_bfloat16 *dg,*dw,*hg,*hw; float*dma,*hma; unsigned char*dm,*dv,*hm,*hv;
  cudaMalloc(&dg,N*2);cudaMallocHost(&hg,N*2);cudaMalloc(&dw,N*2);cudaMallocHost(&hw,N*2);
  cudaMalloc(&dma,N*4);cudaMallocHost(&hma,N*4);cudaMalloc(&dm,N);cudaMallocHost(&hm,N);
  cudaMalloc(&dv,N);cudaMallocHost(&hv,N);
  cudaStream_t sc,sk; cudaStreamCreate(&sc);cudaStreamCreate(&sk); cublasSetStream(cb,sc);
  float al=1.f,be=0.f;
  auto gemm=[&]{for(int k=0;k<G;k++) cublasGemmEx(cb,CUBLAS_OP_N,CUBLAS_OP_N,n,n,n,&al,A,CUDA_R_16BF,n,B,CUDA_R_16BF,n,&be,C,CUDA_R_32F,n,CUBLAS_COMPUTE_32F,CUBLAS_GEMM_DEFAULT_TENSOR_OP);};
  auto din=[&]{cudaMemcpyAsync(dg,hg,N*2,cudaMemcpyHostToDevice,sk);cudaMemcpyAsync(dma,hma,N*4,cudaMemcpyHostToDevice,sk);
               cudaMemcpyAsync(dm,hm,N,cudaMemcpyHostToDevice,sk);cudaMemcpyAsync(dv,hv,N,cudaMemcpyHostToDevice,sk);};
  auto dout=[&]{cudaMemcpyAsync(hma,dma,N*4,cudaMemcpyDeviceToHost,sk);cudaMemcpyAsync(hm,dm,N,cudaMemcpyDeviceToHost,sk);
                cudaMemcpyAsync(hv,dv,N,cudaMemcpyDeviceToHost,sk);cudaMemcpyAsync(hw,dw,N*2,cudaMemcpyDeviceToHost,sk);};
  /* Drive the card to steady-state clocks before any timing. Cold-clock
   * boost was worth 2.7x on this GPU and made the first config measured
   * always look best, whichever one it was. */
  printf("warming to steady clocks ...\n"); fflush(stdout);
  double t0=now(); while(now()-t0<6.0){ gemm(); cudaDeviceSynchronize(); }
  struct Cfg{const char*name;int gm,dm_,kn;std::vector<double> t;};
  Cfg cfg[]={{"GEMM alone",1,0,0,{}},{"optimizer alone",0,1,1,{}},{"GEMM + optimizer",1,1,1,{}},
             {"GEMM + DMA only",1,1,0,{}},{"GEMM + kernel only",1,0,1,{}},
             {"CONTROL spin alone",2,0,0,{}},{"CONTROL spin + DMA",2,1,0,{}}};
  const int NC=7;
  for(int r=0;r<ROUNDS;r++) for(int i=0;i<NC;i++){
    Cfg&c=cfg[i]; cudaDeviceSynchronize(); double t=now();
    if(c.gm==1) gemm();
    if(c.gm==2) spin<<<p.multiProcessorCount,256,0,sc>>>(2200000,(float*)dma);
    if(c.dm_) din();
    if(c.kn) adam8<<<p.multiProcessorCount*8,256,0,sk>>>(dg,dma,dm,dv,dw,N,1e-4f);
    if(c.dm_) dout();
    cudaDeviceSynchronize(); c.t.push_back(now()-t); }
  int clk=0; cudaDeviceGetAttribute(&clk,cudaDevAttrClockRate,0);
  printf("\n%s | %d x %d^3 GEMM, %.0f MB moved | %d rounds, median (p25-p75)\n\n",
         p.name,G,n,N*16.0/1e6,ROUNDS);
  double med[NC];
  for(int i=0;i<NC;i++){ auto&v=cfg[i].t; std::sort(v.begin(),v.end());
    med[i]=v[v.size()/2];
    printf("%-24s %7.1f ms  (%.1f - %.1f)  spread %.0f%%\n",cfg[i].name,med[i]*1e3,
           v[v.size()/4]*1e3,v[3*v.size()/4]*1e3, 100.0*(v[3*v.size()/4]-v[v.size()/4])/med[i]); }
  printf("\n  GEMM alone                 %7.1f ms  (%.0f TFLOP/s)\n",med[0]*1e3,2.0*G*(double)n*n*n/med[0]/1e12);
  printf("  optimizer alone            %7.1f ms  (%.1f GB/s)\n",med[1]*1e3,N*16.0/med[1]/1e9);
  printf("  max(the two)               %7.1f ms   <- traffic hides\n",std::max(med[0],med[1])*1e3);
  printf("  sum(the two)               %7.1f ms   <- traffic does not hide\n",(med[0]+med[1])*1e3);
  printf("  MEASURED together          %7.1f ms\n",med[2]*1e3);
  printf("\n  hidden fraction            %6.0f%%   (1.0 = free, 0.0 = fully serial)\n",
         100.0*((med[0]+med[1])-med[2])/med[1]);
  printf("  of which DMA costs         %+6.1f ms\n",(med[3]-med[0])*1e3);
  printf("  of which kernel costs      %+6.1f ms\n",(med[4]-med[0])*1e3);
  printf("\n  CONTROL, no cache pressure:\n");
  printf("    spin alone               %7.1f ms\n",med[5]*1e3);
  printf("    spin + DMA               %7.1f ms   vs sum %.1f\n",med[6]*1e3,(med[5]+med[1])*1e3);
  printf("    overlap achieved         %6.0f%%   <- if this is ~100 and the\n",
         100.0*((med[5]+med[1])-med[6])/med[1]);
  printf("                                       figure above is not, the copy\n");
  printf("                                       engine is fine and the GEMM's\n");
  printf("                                       cache is what is being lost.\n");
  return 0;}
