/* Link and VRAM bandwidth probe: the four numbers the tier model rests on.
 *
 *   /usr/local/cuda-13.2/bin/nvcc -O3 -arch=sm_120 -o bw vram/probe/bw.cu && ./bw
 *
 * Nothing is installed; nvcc and a GPU are all it needs. Pick the -arch your
 * card wants. Every figure is the best of five 1 GiB transfers, because the
 * worst case here is scheduler noise and the best case is the hardware.
 *
 * The pageable line exists because it is the one most offload paths actually
 * get: a cudaMemcpy from ordinary malloc'd memory has to bounce through a
 * staging buffer, and on this box that costs a factor of two for nothing.
 *
 * The 4-stream line exists to show what streams are *not* for: one large
 * copy already saturates the link, so streams buy overlap with compute, not
 * bandwidth. A residency engine needs them for the former.
 */
#include <cstdio>
#include <cstring>
#include <chrono>
#include <cuda_runtime.h>

#define CK(x) do { cudaError_t e_ = (x); if (e_) { \
    printf("CUDA error %s at line %d\n", cudaGetErrorString(e_), __LINE__); \
    return 1; } } while (0)

static double now() {
    using namespace std::chrono;
    return duration<double>(steady_clock::now().time_since_epoch()).count();
}

/* Read+write stream copy: the honest measure of achievable DRAM bandwidth,
 * which is what a weight read out of VRAM competes with. */
__global__ void vcopy(const float4 *__restrict__ src, float4 *__restrict__ dst,
                      size_t n) {
    size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x;
    size_t stride = (size_t)gridDim.x * blockDim.x;
    for (; i < n; i += stride) dst[i] = src[i];
}

int main() {
    const int dev = 0, R = 5;
    const size_t N = 1024ull * 1024 * 1024;

    cudaDeviceProp p;
    CK(cudaGetDeviceProperties(&p, dev));
    int mclk = 0, mbus = 0;
    /* cudaDeviceProp lost memoryClockRate in CUDA 13; the attributes stayed. */
    cudaDeviceGetAttribute(&mclk, cudaDevAttrMemoryClockRate, dev);
    cudaDeviceGetAttribute(&mbus, cudaDevAttrGlobalMemoryBusWidth, dev);
    printf("device: %s  sm_%d%d  %d SMs  %d-bit  peak %.1f GB/s\n",
           p.name, p.major, p.minor, p.multiProcessorCount, mbus,
           2.0 * (double)mclk * (mbus / 8) / 1.0e6);

    void *pinned = nullptr, *dev_a = nullptr, *dev_b = nullptr;
    CK(cudaMallocHost(&pinned, N));
    void *pageable = malloc(N);
    CK(cudaMalloc(&dev_a, N));
    CK(cudaMalloc(&dev_b, N));
    memset(pinned, 1, N);
    memset(pageable, 1, N);

    double best;
#define TIME(...) do { best = 1e9; for (int i = 0; i < R; i++) { \
        CK(cudaDeviceSynchronize()); double t0 = now(); __VA_ARGS__; \
        CK(cudaDeviceSynchronize()); double dt = now() - t0; \
        if (dt < best) best = dt; } } while (0)

    TIME(CK(cudaMemcpy(dev_a, pinned, N, cudaMemcpyHostToDevice)));
    printf("H2D pinned         : %6.2f GB/s\n", N / best / 1e9);

    TIME(CK(cudaMemcpy(dev_a, pageable, N, cudaMemcpyHostToDevice)));
    printf("H2D pageable       : %6.2f GB/s\n", N / best / 1e9);

    TIME(CK(cudaMemcpy(pinned, dev_a, N, cudaMemcpyDeviceToHost)));
    printf("D2H pinned         : %6.2f GB/s\n", N / best / 1e9);

    {
        const int S = 4;
        const size_t chunk = 64ull << 20, nch = N / chunk;
        cudaStream_t st[S];
        for (int i = 0; i < S; i++) CK(cudaStreamCreate(&st[i]));
        TIME(for (size_t c = 0; c < nch; c++)
                 CK(cudaMemcpyAsync((char *)dev_a + c * chunk,
                                    (char *)pinned + c * chunk, chunk,
                                    cudaMemcpyHostToDevice, st[c % S])));
        printf("H2D pinned, 4 str  : %6.2f GB/s  (64 MiB chunks)\n", N / best / 1e9);
        for (int i = 0; i < S; i++) cudaStreamDestroy(st[i]);
    }

    TIME(vcopy<<<p.multiProcessorCount * 16, 256>>>(
             (const float4 *)dev_a, (float4 *)dev_b, N / sizeof(float4)));
    printf("VRAM copy          : %6.2f GB/s  (read+write counted)\n",
           2.0 * N / best / 1e9);

    cudaFreeHost(pinned);
    free(pageable);
    cudaFree(dev_a);
    cudaFree(dev_b);
    return 0;
}
