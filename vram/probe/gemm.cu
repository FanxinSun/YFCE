/* Achievable BF16 GEMM throughput -- the compute side of the training roofline.
 *
 *   nvcc -O3 -arch=sm_120 -lcublas -o gemm vram/probe/gemm.cu && ./gemm
 *
 * This is the number that decides everything in `train.py`: the token count at
 * which streamed weights hide under compute is FLOPS/bandwidth, so a wrong
 * FLOPS is a wrong plan.
 *
 * Accumulate is FP32 (CUBLAS_COMPUTE_32F) on purpose. Mixed-precision training
 * accumulates in FP32, and on consumer Blackwell the FP16-accumulate path is
 * about twice as fast -- quoting that figure here would halve every token floor
 * in the model and none of those plans would hold.
 */
#include <cstdio>
#include <chrono>
#include <cuda_runtime.h>
#include <cublas_v2.h>

static double now() {
    using namespace std::chrono;
    return duration<double>(steady_clock::now().time_since_epoch()).count();
}

int main() {
    cublasHandle_t h;
    cublasCreate(&h);
    cublasSetMathMode(h, CUBLAS_TENSOR_OP_MATH);

    for (int n : {4096, 8192, 16384}) {
        size_t N = n;
        __nv_bfloat16 *A, *B;
        float *C;
        if (cudaMalloc(&A, N * N * 2) || cudaMalloc(&B, N * N * 2) ||
            cudaMalloc(&C, N * N * 4)) {
            printf("%5d^3 : out of memory, skipped\n", n);
            continue;
        }
        cudaMemset(A, 1, N * N * 2);
        cudaMemset(B, 1, N * N * 2);
        float al = 1.f, be = 0.f;

        for (int w = 0; w < 3; w++)
            cublasGemmEx(h, CUBLAS_OP_N, CUBLAS_OP_N, n, n, n, &al,
                         A, CUDA_R_16BF, n, B, CUDA_R_16BF, n, &be,
                         C, CUDA_R_32F, n, CUBLAS_COMPUTE_32F,
                         CUBLAS_GEMM_DEFAULT_TENSOR_OP);
        cudaDeviceSynchronize();

        double best = 1e9;
        for (int r = 0; r < 5; r++) {
            double t = now();
            for (int i = 0; i < 10; i++)
                cublasGemmEx(h, CUBLAS_OP_N, CUBLAS_OP_N, n, n, n, &al,
                             A, CUDA_R_16BF, n, B, CUDA_R_16BF, n, &be,
                             C, CUDA_R_32F, n, CUBLAS_COMPUTE_32F,
                             CUBLAS_GEMM_DEFAULT_TENSOR_OP);
            cudaDeviceSynchronize();
            double dt = (now() - t) / 10;
            if (dt < best) best = dt;
        }
        printf("BF16 GEMM %5d^3, FP32 accumulate : %7.1f TFLOP/s\n",
               n, 2.0 * N * N * N / best / 1e12);
        cudaFree(A); cudaFree(B); cudaFree(C);
    }
    cublasDestroy(h);
    return 0;
}
