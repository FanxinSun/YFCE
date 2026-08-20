// Fused decode-into-matrix-unit: y = W x with W never materialised as BF16.
//
// lmz's GPU kernel (lmz/scratchpad/gpu/cuda/gpu_fused.cu) decodes a coded
// BF16 plane into VRAM at 399 GB/s and stops there. vram/tiers.py's V-coded
// tier assumes the next step -- that a tile can be decoded straight into the
// matrix unit, so the only DRAM traffic is the coded read -- and labels it
// NOT MEASURED. This is that kernel, and the measurement.
//
// Three paths over the same weights, same tiling, same accumulation order:
//
//   k_raw     raw BF16 in VRAM -> shared -> ldmatrix -> mma          (control)
//   k_decode  coded in VRAM -> decode -> raw BF16 in VRAM            (lmz today)
//   k_fused   coded in VRAM -> decode -> shared -> ldmatrix -> mma   (this)
//
// k_fused and k_raw differ in exactly one thing: where the A-tile comes from.
// Everything downstream is identical instruction for instruction, so the gap
// between them IS the cost of decoding inside the matmul, and their outputs
// must be bit-identical.
//
// The coder is lmz's own, included from the submodule rather than restated:
// the same rANS, the same 8 interleaved states, the same shared table that
// lmz/scratchpad/gpu proved round-trips through lmz_rans_decode unmodified.
//
// Build:
//   /usr/local/cuda-13.2/bin/nvcc -O3 -arch=sm_120 -lcublas \
//       -Xcompiler -fopenmp -o fused_gemm fused_gemm.cu
#define _GNU_SOURCE
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cstdint>
#include <cmath>
#include <string>
#include <vector>
#include <algorithm>
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cuda_pipeline.h>
#include <cublas_v2.h>

// lmz's shared-table rANS encoder, verbatim from the submodule.
#include "../../lmz/scratchpad/gpu/shared_enc.c"

#define TILE 16            // mma.m16n8k16 -- the A tile is 16x16
#define TILE_ELEMS 256
#define LUT_BYTES (PROB_SCALE * 4)

// States per lane. lmz's format is 8 rANS states per stream, one per lane, and
// each state is a strictly serial dependency chain: state -> table lookup ->
// renormalise -> state. One chain per lane leaves the SM waiting on memory
// latency with nothing else to issue. SPL states per lane is SPL independent
// chains interleaved in one instruction stream -- and because they are states
// of the SAME stream, they share the tile, the ring buffer and the cursor, so
// the interleaving costs no shared memory at all. NST = 8 * SPL.
#define MAX_SPL 4

#define CK(x) do { cudaError_t e_ = (x); if (e_) { \
    fprintf(stderr, "%s:%d %s -> %s\n", __FILE__, __LINE__, #x, \
            cudaGetErrorString(e_)); exit(1); } } while (0)
#define CB(x) do { cublasStatus_t s_ = (x); if (s_ != CUBLAS_STATUS_SUCCESS) { \
    fprintf(stderr, "%s:%d %s -> cublas %d\n", __FILE__, __LINE__, #x, (int)s_); \
    exit(1); } } while (0)

// ===========================================================================
// The layout
// ===========================================================================
//
// W is [N, K] row-major. Cut it into row-blocks of 16 (the mma's M), and cut
// each row-block's K into streams of KS columns. One stream therefore holds
// 16 x KS elements, ordered as (KS/16) consecutive 16x16 A-tiles, each tile
// row-major:
//
//   stream(rb, j) element (t*256 + r*16 + c) = W[rb*16 + r][j*KS + t*16 + c]
//
// That ordering is the whole trick. A group of 8 lanes decoding STAGE_ITERS
// steps produces 256 symbols in linear order, which is one A-tile already in
// the layout ldmatrix wants -- so the decoder's output register file feeds
// the matrix unit with no transpose, no gather, and no round trip to DRAM.
//
// A warp owns one row-block. Its four 8-lane groups decode four streams that
// cover four different k-slices of that row-block, in lockstep, so the four
// tiles the warp holds at any moment accumulate into the same D tile.
//
// stream id = rb * NS + j,  NS = K / KS,  and NS must be a multiple of 4.

struct Layout {
    int N, K, KS, NS, NSG, TPS, PLANE, NRB, SPL, LANES;
    size_t nstreams, nelem;
};

static Layout make_layout(int N, int K, int KS, int SPL = 1, int LANES = 8)
{
    Layout L{};
    L.N = N; L.K = K; L.KS = KS; L.SPL = SPL; L.LANES = LANES;
    L.NS = K / KS;
    L.NSG = L.NS / (32 / LANES);
    L.TPS = KS / TILE;
    L.PLANE = TILE * KS;
    L.NRB = N / TILE;
    L.nstreams = (size_t)L.NRB * L.NS;
    L.nelem = (size_t)N * K;
    return L;
}

// ===========================================================================
// Device: the pieces shared by all three kernels
// ===========================================================================

__device__ __forceinline__ void build_lut(const uint8_t *src, uint32_t *lut,
                                          int nth, int me)
{
    __shared__ uint32_t starts[256];
    if (me == 0) {
        uint32_t s = 0;
        for (int i = 0; i < 256; i++) {
            starts[i] = s;
            s += (uint32_t)src[4 + 2 * i] | ((uint32_t)src[5 + 2 * i] << 8);
        }
    }
    __syncthreads();
    for (int i = me; i < 256; i += nth) {
        uint32_t f = (uint32_t)src[4 + 2 * i] | ((uint32_t)src[5 + 2 * i] << 8);
        uint32_t st = starts[i];
        uint32_t packed = ((f ? f - 1 : 0) << 20) | (st << 8) | (uint32_t)i;
        for (uint32_t j = 0; j < f; j++) lut[st + j] = packed;
    }
    __syncthreads();
}

// Two BF16 out of two exponent bytes and two sign+mantissa bytes, packed as
// one 32-bit word. lmz's merge_bf16_scalar, twice, with no memory in between.
template <int I>
__device__ __forceinline__ uint32_t mrg2(uint32_t e, uint32_t s)
{
    uint32_t ea = (e >> (I * 8)) & 0xffu, eb = (e >> ((I + 1) * 8)) & 0xffu;
    uint32_t sa = (s >> (I * 8)) & 0xffu, sb = (s >> ((I + 1) * 8)) & 0xffu;
    return (((sa & 0x80u) << 8) | (ea << 7) | (sa & 0x7Fu)) |
           ((((sb & 0x80u) << 8) | (eb << 7) | (sb & 0x7Fu)) << 16);
}

__device__ __forceinline__ void ldmatrix_x4(uint32_t (&r)[4], const void *sp)
{
    uint32_t a = (uint32_t)__cvta_generic_to_shared(sp);
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];"
                 : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3]) : "r"(a));
}

__device__ __forceinline__ void mma16816(float (&d)[4], const uint32_t (&a)[4],
                                         const uint32_t (&b)[2])
{
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
                 "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
                 : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
                 : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]),
                   "r"(b[0]), "r"(b[1]));
}

// SPL rANS states per lane, read from the stream header. State (j*LANES + k)
// is lane k's j-th, which is the order the encoder renormalises in.
#define INIT_STATES(SPL, LANES) do { \
        _Pragma("unroll") \
        for (int j_ = 0; j_ < (SPL); j_++) { \
            const uint8_t *q_ = ptr + 4 * (j_ * (LANES) + k); \
            st[j_] = (uint32_t)q_[0] | ((uint32_t)q_[1] << 8) | \
                     ((uint32_t)q_[2] << 16) | ((uint32_t)q_[3] << 24); \
        } ptr += 4 * (LANES) * (SPL); } while (0)

// Which lanes share a stream, and therefore share its input cursor.
#define GROUP_MASKS(LANES) \
    const uint32_t gbase = (uint32_t)(lane & ~((LANES) - 1)); \
    const uint32_t lmask = ((LANES) == 32) ? 0xffffffffu \
                                           : ((1u << ((LANES) & 31)) - 1u); \
    const uint32_t wmask = ((LANES) == 32) ? 0xffffffffu : (lmask << gbase)

// One round of the cp.async ring: Q bytes, CPB per lane.
#define ISSUE(OFFS, CPB, BUFB) do { \
        uint32_t o2_ = (OFFS) + k * (CPB); \
        __pipeline_memcpy_async(mybuf + (o2_ & ((BUFB) - 1)), ptr + o2_, (CPB)); \
        __pipeline_commit(); } while (0)

// One A-tile: 256 symbols, 256/LANES per lane, SPL at a time. The SPL lookups
// issue back to back and the SPL renormalisations resolve against one shared
// cursor, so SPL symbols cost roughly one symbol's latency. STORE runs every
// eighth symbol with IDX naming the tile offset and p0/p1 holding eight
// exponents.
//
// The cursor is lmz's: states renormalise in strict index order and each takes
// two bytes, so a lane's byte offset is the prefix popcount of the ballot --
// here SPL ballots, summed in order. `consumed` is even by construction, which
// is why the two bytes come back as one 16-bit load.
#define DECODE_TILE(SPL, LANES, CPB, BUFB, STORE) do { \
        uint32_t p0 = 0, p1 = 0; \
        _Pragma("unroll") \
        for (int s = 0; s < TILE_ELEMS / ((LANES) * (SPL)); s++) { \
            if (filled - consumed <= 2 * Q) { \
                __pipeline_wait_prior(0); \
                __syncwarp(wmask); \
                ISSUE(filled, CPB, BUFB); \
                filled += Q; \
            } \
            uint32_t ee[SPL], xx[SPL], nd[SPL], gm[SPL]; \
            _Pragma("unroll") \
            for (int j = 0; j < (SPL); j++) ee[j] = lut[st[j] & (PROB_SCALE - 1)]; \
            _Pragma("unroll") \
            for (int j = 0; j < (SPL); j++) { \
                uint32_t sv_ = (ee[j] >> 8) & 0xfff, fq_ = (ee[j] >> 20) + 1; \
                xx[j] = fq_ * (st[j] >> PROB_BITS) + \
                        (st[j] & (PROB_SCALE - 1)) - sv_; \
                nd[j] = (xx[j] < RANS_L); \
            } \
            _Pragma("unroll") \
            for (int j = 0; j < (SPL); j++) \
                gm[j] = (__ballot_sync(0xffffffffu, nd[j]) >> gbase) & lmask; \
            uint32_t bef_ = 0; \
            _Pragma("unroll") \
            for (int j = 0; j < (SPL); j++) { \
                uint32_t o_ = consumed + 2 * (bef_ + __popc(gm[j] & ((1u << k) - 1))); \
                uint32_t w_ = *(const uint16_t *)(mybuf + (o_ & ((BUFB) - 1))); \
                st[j] = nd[j] ? ((xx[j] << 16) | w_) : xx[j]; \
                bef_ += __popc(gm[j]); \
            } \
            consumed += 2 * bef_; \
            _Pragma("unroll") \
            for (int j = 0; j < (SPL); j++) { \
                const int N_ = s * (SPL) + j, U = N_ & 7; \
                uint32_t sym = ee[j] & 0xff; \
                if (U < 4) p0 = U ? (p0 | (sym << (U * 8))) : sym; \
                else       p1 = (U - 4) ? (p1 | (sym << ((U - 4) * 8))) : sym; \
                if (U == 7) { const int IDX = ((N_ >> 3) * (LANES) + k) * 8; STORE } \
            } \
        } } while (0)

// One A-tile in shared -> the m16n8k16 A fragment. Thread t reads row t%16 at
// column (t/16)*8, which is exactly ldmatrix's four 8x8 quadrants.
#define A_FRAG(regs, tilebase) \
    ldmatrix_x4((regs), (tilebase) + (lane & 15) * TILE + (lane >> 4) * 8)

// All MT mma's for one A fragment. B comes from X[m][k], which is already the
// m16n8k16 B layout: for column m the 16 k values are contiguous.
#define MMA_ALL(regs, kpos) \
    _Pragma("unroll") \
    for (int mt = 0; mt < MT; mt++) { \
        const __nv_bfloat16 *xr = X + (size_t)(mt * 8 + (lane >> 2)) * K \
                                    + (kpos) + (lane & 3) * 2; \
        uint32_t b[2]; \
        b[0] = *(const uint32_t *)(xr); \
        b[1] = *(const uint32_t *)(xr + 8); \
        mma16816(acc[mt], (regs), b); \
    }

// The same, split so the activation fragment is fetched before the tile is
// decoded instead of after it. Its address depends only on k, never on the
// decoder, so the load has a whole tile's worth of decode to complete in --
// which matters now that the decoder is fast enough for a global load on the
// critical path to cost 13%. Only worth it when a warp holds one stream, since
// otherwise it is GRPS activation fragments live across the decode.
#define LOAD_B(kpos) \
    uint32_t bfr[MT][2]; \
    _Pragma("unroll") \
    for (int mt = 0; mt < MT; mt++) { \
        const __nv_bfloat16 *xr = X + (size_t)(mt * 8 + (lane >> 2)) * K \
                                    + (kpos) + (lane & 3) * 2; \
        bfr[mt][0] = *(const uint32_t *)(xr); \
        bfr[mt][1] = *(const uint32_t *)(xr + 8); \
    }
#define MMA_B(regs) \
    _Pragma("unroll") \
    for (int mt = 0; mt < MT; mt++) mma16816(acc[mt], (regs), bfr[mt]);

#define STORE_Y \
    do { \
        const int n0 = rb * TILE, r = lane >> 2, c = (lane & 3) * 2; \
        _Pragma("unroll") \
        for (int mt = 0; mt < MT; mt++) { \
            size_t o = (size_t)(n0 + r) * (MT * 8) + mt * 8 + c; \
            Y[o] = acc[mt][0]; Y[o + 1] = acc[mt][1]; \
            size_t o8 = (size_t)(n0 + r + 8) * (MT * 8) + mt * 8 + c; \
            Y[o8] = acc[mt][2]; Y[o8 + 1] = acc[mt][3]; \
        } \
    } while (0)
// ===========================================================================
// k_fused -- the point of the file
// ===========================================================================
//
// LANES is how many lanes share one stream, and it is the occupancy knob. At
// LANES=8 (lmz's shape) a warp holds four streams, so four A-tiles and four
// ring buffers: 4 KB of shared per warp, which caps the SM at 16 warps. At
// LANES=32 one warp is one stream: one tile, one ring, 1 KB per warp, and the
// SM runs 32. The decode work per lane is identical either way.

template <int MT, int WPB, int BUFB, int SPL, int LANES, bool MMA = true>
__launch_bounds__(WPB * 32)
__global__ void k_fused(const uint8_t *__restrict__ coded,
                        const uint64_t *__restrict__ soff,
                        const uint8_t *__restrict__ smpl,
                        const uint8_t *__restrict__ shdr,
                        const __nv_bfloat16 *__restrict__ X,
                        float *__restrict__ Y,
                        int K, int KS, int NS)
{
    constexpr int GRPS = 32 / LANES;          // streams in flight per warp
    extern __shared__ __align__(16) uint8_t smem[];
    uint32_t *lut = (uint32_t *)smem;
    __nv_bfloat16 *atile_all = (__nv_bfloat16 *)(smem + LUT_BYTES);
    uint8_t *inbuf_all = (uint8_t *)(atile_all + WPB * GRPS * TILE_ELEMS);

    build_lut(shdr, lut, blockDim.x, threadIdx.x);

    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    const int grp = lane / LANES, k = lane & (LANES - 1);
    GROUP_MASKS(LANES);
    const int rb = blockIdx.x * WPB + warp;
    const int NSG = NS / GRPS, TPS = KS >> 4;
    const size_t PLANE = (size_t)KS * TILE;

    __nv_bfloat16 *atile = atile_all + (warp * GRPS + grp) * TILE_ELEMS;
    uint8_t *mybuf = inbuf_all + (warp * GRPS + grp) * BUFB;

    float acc[MT][4];
#pragma unroll
    for (int i = 0; i < MT; i++)
#pragma unroll
        for (int q = 0; q < 4; q++) acc[i][q] = 0.f;

    constexpr uint32_t Q = BUFB / 4;
    constexpr uint32_t CPB = Q / LANES;       // cp.async bytes per lane: 16 or 4
    const uint8_t *ptr = nullptr;
    uint32_t filled = 0, consumed = 0;

    for (int jj = 0; jj < NSG; jj++) {
        const int j = grp * NSG + jj;
        const size_t sid = (size_t)rb * NS + j;
        ptr = coded + soff[sid];
        uint32_t st[SPL];
        INIT_STATES(SPL, LANES);
        const size_t ebase = sid * PLANE;
        filled = 0; consumed = 0;
        ISSUE(0, CPB, BUFB); ISSUE(Q, CPB, BUFB); ISSUE(2 * Q, CPB, BUFB);
        filled = 3 * Q;
        __pipeline_wait_prior(0);
        __syncwarp(wmask);

        for (int t = 0; t < TPS; t++) {
            // ---- 256 symbols: one A-tile, decoded in registers and merged
            // into shared the moment eight of them are ready. The exponent
            // plane never lands in memory at all -- the stream order is
            // permuted so that a lane's eight consecutive symbols ARE eight
            // consecutive weights, which is what removes the staging buffer.
            const size_t tbase = ebase + (size_t)t * TILE_ELEMS;
            if (GRPS == 1 && MMA) {
                LOAD_B(jj * KS + t * TILE);
                DECODE_TILE(SPL, LANES, CPB, BUFB, {
                    uint2 S = *(const uint2 *)(smpl + tbase + IDX);
                    uint4 ot;
                    ot.x = mrg2<0>(p0, S.x); ot.y = mrg2<2>(p0, S.x);
                    ot.z = mrg2<0>(p1, S.y); ot.w = mrg2<2>(p1, S.y);
                    *(uint4 *)(atile + IDX) = ot;
                });
                __syncwarp();
                uint32_t a[4];
                A_FRAG(a, atile);
                MMA_B(a);
                __syncwarp();
                continue;
            }
            DECODE_TILE(SPL, LANES, CPB, BUFB, {
                uint2 S = *(const uint2 *)(smpl + tbase + IDX);
                uint4 ot;
                ot.x = mrg2<0>(p0, S.x); ot.y = mrg2<2>(p0, S.x);
                ot.z = mrg2<0>(p1, S.y); ot.w = mrg2<2>(p1, S.y);
                *(uint4 *)(atile + IDX) = ot;
            });
            __syncwarp();
            // ---- and into the matrix unit, without ever touching DRAM ----
#pragma unroll
            for (int g = 0; g < GRPS; g++) {
                uint32_t a[4];
                A_FRAG(a, atile_all + (warp * GRPS + g) * TILE_ELEMS);
                const int kpos = (g * NSG + jj) * KS + t * TILE;
                if (MMA) { MMA_ALL(a, kpos); }
                else {
                    acc[0][0] += (float)(a[0] & 0xff); acc[0][1] += (float)(a[1] & 0xff);
                    acc[0][2] += (float)(a[2] & 0xff); acc[0][3] += (float)(a[3] & 0xff);
                }
            }
            __syncwarp();
        }
    }
    STORE_Y;
}

// ===========================================================================
// k_raw -- the control. Identical from the A-tile onward.
// ===========================================================================

template <int MT, int WPB, int LANES>
__launch_bounds__(WPB * 32)
__global__ void k_raw(const __nv_bfloat16 *__restrict__ Wt,
                      const __nv_bfloat16 *__restrict__ X,
                      float *__restrict__ Y,
                      int K, int KS, int NS)
{
    constexpr int GRPS = 32 / LANES;
    extern __shared__ __align__(16) uint8_t smem[];
    __nv_bfloat16 *atile_all = (__nv_bfloat16 *)smem;

    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    const int grp = lane / LANES, k = lane & (LANES - 1);
    const int rb = blockIdx.x * WPB + warp;
    const int NSG = NS / GRPS, TPS = KS >> 4;
    const size_t PLANE = (size_t)KS * TILE;
    __nv_bfloat16 *atile = atile_all + (warp * GRPS + grp) * TILE_ELEMS;

    float acc[MT][4];
#pragma unroll
    for (int i = 0; i < MT; i++)
#pragma unroll
        for (int q = 0; q < 4; q++) acc[i][q] = 0.f;

    for (int jj = 0; jj < NSG; jj++) {
        const int j = grp * NSG + jj;
        const size_t sid = (size_t)rb * NS + j;
        const size_t ebase = sid * PLANE;
        for (int t = 0; t < TPS; t++) {
#pragma unroll
            for (int u = 0; u < TILE_ELEMS / LANES / 8; u++) {
                const int idx = (u * LANES + k) * 8;
                *(uint4 *)(atile + idx) =
                    *(const uint4 *)(Wt + ebase + (size_t)t * TILE_ELEMS + idx);
            }
            __syncwarp();
#pragma unroll
            for (int g = 0; g < GRPS; g++) {
                uint32_t a[4];
                A_FRAG(a, atile_all + (warp * GRPS + g) * TILE_ELEMS);
                const int kpos = (g * NSG + jj) * KS + t * TILE;
                MMA_ALL(a, kpos);
            }
            __syncwarp();
        }
    }
    STORE_Y;
}

// ===========================================================================
// k_decode -- lmz today: coded VRAM in, raw BF16 VRAM out, no matmul.
// ===========================================================================

template <int WPB, int BUFB, int SPL, int LANES>
__launch_bounds__(WPB * 32)
__global__ void k_decode(const uint8_t *__restrict__ coded,
                         const uint64_t *__restrict__ soff,
                         const uint8_t *__restrict__ smpl,
                         const uint8_t *__restrict__ shdr,
                         __nv_bfloat16 *__restrict__ out,
                         int KS, int NS)
{
    constexpr int GRPS = 32 / LANES;
    extern __shared__ __align__(16) uint8_t smem[];
    uint32_t *lut = (uint32_t *)smem;
    uint8_t *inbuf_all = smem + LUT_BYTES;

    build_lut(shdr, lut, blockDim.x, threadIdx.x);

    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    const int grp = lane / LANES, k = lane & (LANES - 1);
    GROUP_MASKS(LANES);
    const int rb = blockIdx.x * WPB + warp;
    const int NSG = NS / GRPS, TPS = KS >> 4;
    const size_t PLANE = (size_t)KS * TILE;
    uint8_t *mybuf = inbuf_all + (warp * GRPS + grp) * BUFB;

    constexpr uint32_t Q = BUFB / 4;
    constexpr uint32_t CPB = Q / LANES;
    const uint8_t *ptr = nullptr;
    uint32_t filled = 0, consumed = 0;

    for (int jj = 0; jj < NSG; jj++) {
        const int j = grp * NSG + jj;
        const size_t sid = (size_t)rb * NS + j;
        ptr = coded + soff[sid];
        uint32_t st[SPL];
        INIT_STATES(SPL, LANES);
        const size_t ebase = sid * PLANE;
        filled = 0; consumed = 0;
        ISSUE(0, CPB, BUFB); ISSUE(Q, CPB, BUFB); ISSUE(2 * Q, CPB, BUFB);
        filled = 3 * Q;
        __pipeline_wait_prior(0);
        __syncwarp(wmask);
        for (int t = 0; t < TPS; t++) {
            const size_t tbase = ebase + (size_t)t * TILE_ELEMS;
            DECODE_TILE(SPL, LANES, CPB, BUFB, {
                uint2 S = *(const uint2 *)(smpl + tbase + IDX);
                uint4 ot;
                ot.x = mrg2<0>(p0, S.x); ot.y = mrg2<2>(p0, S.x);
                ot.z = mrg2<0>(p1, S.y); ot.w = mrg2<2>(p1, S.y);
                *(uint4 *)(out + tbase + IDX) = ot;
            });
            __syncwarp(wmask);
        }
    }
}
// ===========================================================================
// Host: safetensors, the tile-order relayout, and lmz's coder over it
// ===========================================================================

static double now_s(void)
{
    struct timespec t; clock_gettime(CLOCK_MONOTONIC, &t);
    return t.tv_sec + t.tv_nsec * 1e-9;
}

struct TensorRef { size_t off, nbytes; long rows, cols; };

static bool st_lookup(const std::string &h, const std::string &name, TensorRef &tr)
{
    size_t p = h.find("\"" + name + "\":{");
    if (p == std::string::npos) return false;
    size_t e = h.find('}', p);
    std::string rec = h.substr(p, e - p);
    if (rec.find("\"BF16\"") == std::string::npos) return false;
    auto two = [&](const char *key, long &a, long &b) {
        size_t q = rec.find(key);
        if (q == std::string::npos) return false;
        q = rec.find('[', q) + 1;
        a = strtol(rec.c_str() + q, nullptr, 10);
        q = rec.find(',', q) + 1;
        b = strtol(rec.c_str() + q, nullptr, 10);
        return true;
    };
    long s, en;
    if (!two("\"shape\"", tr.rows, tr.cols)) return false;
    if (!two("\"data_offsets\"", s, en)) return false;
    tr.off = (size_t)s; tr.nbytes = (size_t)(en - s);
    return true;
}

// lmz's coder with the state count lifted out of the #define. At nst == 8
// this must emit byte-identical output to lmzx_encode_shared above, which
// main() checks on real data before trusting any other value of nst.
static long enc_shared_n(const uint8_t *src, size_t n, const uint16_t *freqs,
                         uint8_t *dst, size_t cap, int nst)
{
    EncSym syms[256];
    uint32_t start = 0;
    for (int s = 0; s < 256; s++) {
        if (freqs[s]) {
            syms[s].freq = freqs[s];
            syms[s].cmpl_freq = PROB_SCALE - freqs[s];
            syms[s].bias = start;
            start += freqs[s];
        } else {
            syms[s].freq = 0;
        }
    }
    if (start != PROB_SCALE) return -2;
    for (size_t i = 0; i < n; i++) if (!syms[src[i]].freq) return -3;

    uint8_t *end = dst + cap, *ptr = end;
    uint32_t state[32 * MAX_SPL];
    for (int j = 0; j < nst; j++) state[j] = RANS_L;
    size_t i = n;
    while (i > 0 && (i & (size_t)(nst - 1))) {
        i--; put(&state[i & (nst - 1)], &ptr, &syms[src[i]]);
    }
    while (i >= (size_t)nst) {
        i -= nst;
        for (int j = nst - 1; j >= 0; j--) put(&state[j], &ptr, &syms[src[i + j]]);
    }
    for (int j = nst - 1; j >= 0; j--) flush(&state[j], &ptr);
    size_t coded = (size_t)(end - ptr);
    memmove(dst, ptr, coded);
    return (long)coded;
}

// W[N][K] row-major -> the stream/tile order the kernels walk.
static void relayout(const uint16_t *W, const Layout &L,
                     std::vector<uint16_t> &Wt, std::vector<uint8_t> &expp,
                     std::vector<uint8_t> &smp)
{
    Wt.resize(L.nelem); expp.resize(L.nelem); smp.resize(L.nelem);
#pragma omp parallel for schedule(static)
    for (int rb = 0; rb < L.NRB; rb++) {
        for (int j = 0; j < L.NS; j++) {
            size_t base = ((size_t)rb * L.NS + j) * (size_t)L.PLANE;
            for (int t = 0; t < L.TPS; t++)
                for (int r = 0; r < TILE; r++)
                    for (int c = 0; c < TILE; c++) {
                        const int e = r * TILE + c;      // position in the tile
                        // Decode-order position of this weight. Lane
                        // k = (e/8) % LANES emits it as its n-th symbol in this
                        // tile; with SPL states per lane that is step n/SPL,
                        // state (n%SPL)*LANES + k.
                        const int kk = (e >> 3) % L.LANES;
                        const int nn = ((e >> 3) / L.LANES) * 8 + (e & 7);
                        const int p = (nn / L.SPL) * (L.LANES * L.SPL)
                                    + (nn % L.SPL) * L.LANES + kk;
                        size_t d = base + (size_t)t * TILE_ELEMS;
                        uint16_t w = W[(size_t)(rb * TILE + r) * L.K
                                       + j * L.KS + t * TILE + c];
                        Wt[d + e] = w;
                        expp[d + p] = (uint8_t)((w >> 7) & 0xFF);
                        smp[d + e] = (uint8_t)(((w >> 8) & 0x80) | (w & 0x7F));
                    }
        }
    }
}

struct Coded {
    std::vector<uint8_t> blob, hdr;
    std::vector<uint64_t> off;
    size_t payload = 0;
};

static void encode_streams(const std::vector<uint8_t> &expp, const Layout &L,
                           Coded &C)
{
    uint64_t counts[256] = {0};
    lmzx_hist(expp.data(), expp.size(), counts);
    uint16_t freqs[256];
    if (lmzx_normalize(counts, freqs) != 0) { fprintf(stderr, "normalise failed\n"); exit(1); }

    C.hdr.assign(516, 0);
    C.hdr[0] = 'R'; C.hdr[1] = '1';
    for (int s = 0; s < 256; s++) {
        C.hdr[4 + 2 * s] = (uint8_t)(freqs[s] & 0xFF);
        C.hdr[5 + 2 * s] = (uint8_t)(freqs[s] >> 8);
    }

    const size_t cap = (size_t)L.PLANE * 2 + 256;
    std::vector<uint8_t> tmp(L.nstreams * cap);
    std::vector<long> len(L.nstreams);
    int bad = 0;
#pragma omp parallel for schedule(static) reduction(+ : bad)
    for (long i = 0; i < (long)L.nstreams; i++) {
        long n = enc_shared_n(expp.data() + (size_t)i * L.PLANE, L.PLANE,
                              freqs, tmp.data() + (size_t)i * cap, cap,
                              L.LANES * L.SPL);
        if (n <= 0) bad++;
        len[i] = n;
    }
    if (bad) { fprintf(stderr, "%d streams failed to encode\n", bad); exit(1); }

    C.off.resize(L.nstreams);
    size_t total = 0;
    for (size_t i = 0; i < L.nstreams; i++) {
        C.off[i] = total;
        total += ((size_t)len[i] + 15) & ~(size_t)15;   // 16 B aligned starts
    }
    C.payload = total;
    C.blob.assign(total + 8192, 0);                     // cp.async runs ahead
    for (size_t i = 0; i < L.nstreams; i++)
        memcpy(C.blob.data() + C.off[i], tmp.data() + i * cap, (size_t)len[i]);
}

// ===========================================================================
// Host: the measurement
// ===========================================================================

struct Dev {
    uint8_t *coded = nullptr, *smpl = nullptr, *hdr = nullptr;
    uint64_t *off = nullptr;
    __nv_bfloat16 *Wt = nullptr, *Wnat = nullptr, *Wdec = nullptr, *X = nullptr;
    float *Y = nullptr, *Ycb = nullptr;
};

template <typename F> static float time_ms(F f, int reps = 7)
{
    cudaEvent_t a, b; CK(cudaEventCreate(&a)); CK(cudaEventCreate(&b));
    f(); CK(cudaDeviceSynchronize());
    float best = 1e30f;
    for (int r = 0; r < reps; r++) {
        CK(cudaEventRecord(a)); f(); CK(cudaEventRecord(b));
        CK(cudaEventSynchronize(b));
        CK(cudaGetLastError());
        float ms; CK(cudaEventElapsedTime(&ms, a, b));
        if (ms < best) best = ms;
    }
    CK(cudaEventDestroy(a)); CK(cudaEventDestroy(b));
    return best;
}

static size_t g_shm_max = 0;

// Ring size. Q = BUFB/4 bytes arrive per refill and one step can consume
// 2 * LANES * SPL, so Q must cover a whole step or the cursor runs past what
// cp.async has delivered. At most 3Q is ever in flight, which is why BUFB is
// four times Q and not three.
#define BUFB_OF(LANES, SPL) \
    ((8 * (LANES) * (SPL)) > 512 ? (8 * (LANES) * (SPL)) : 512)
#define SHM_FUSED(WPB, LANES, SPL) (LUT_BYTES + (size_t)(WPB) * (32 / (LANES)) \
                                    * (TILE_ELEMS * 2 + BUFB_OF(LANES, SPL)))
#define SHM_RAW(WPB, LANES)   ((size_t)(WPB) * (32 / (LANES)) * TILE_ELEMS * 2)
#define SHM_DEC(WPB, LANES, SPL) (LUT_BYTES + (size_t)(WPB) * (32 / (LANES)) \
                                  * BUFB_OF(LANES, SPL))

template <int MT, bool MMA, int SPL, int LANES, int WPB>
static float run_fused(const Layout &L, const Dev &D)
{
    constexpr int BUFB = BUFB_OF(LANES, SPL);
    size_t shm = SHM_FUSED(WPB, LANES, SPL);
    if (shm > g_shm_max || L.NRB % WPB || L.SPL != SPL || L.LANES != LANES)
        return 1e30f;
    CK(cudaFuncSetAttribute(k_fused<MT, WPB, BUFB, SPL, LANES, MMA>,
                            cudaFuncAttributeMaxDynamicSharedMemorySize, (int)shm));
    int blocks = L.NRB / WPB;
    return time_ms([&] {
        k_fused<MT, WPB, BUFB, SPL, LANES, MMA><<<blocks, WPB * 32, shm>>>(
            D.coded, D.off, D.smpl, D.hdr, D.X, D.Y, L.K, L.KS, L.NS);
    });
}

template <int MT, int LANES, int WPB> static float run_raw(const Layout &L, const Dev &D)
{
    size_t shm = SHM_RAW(WPB, LANES);
    if (shm > g_shm_max || L.NRB % WPB || L.LANES != LANES) return 1e30f;
    int blocks = L.NRB / WPB;
    return time_ms([&] {
        k_raw<MT, WPB, LANES><<<blocks, WPB * 32, shm>>>(D.Wt, D.X, D.Y,
                                                         L.K, L.KS, L.NS);
    });
}

template <int SPL, int LANES, int WPB>
static void launch_decode(const Layout &L, const Dev &D)
{
    constexpr int BUFB = BUFB_OF(LANES, SPL);
    size_t shm = SHM_DEC(WPB, LANES, SPL);
    CK(cudaFuncSetAttribute(k_decode<WPB, BUFB, SPL, LANES>,
                            cudaFuncAttributeMaxDynamicSharedMemorySize, (int)shm));
    k_decode<WPB, BUFB, SPL, LANES><<<L.NRB / WPB, WPB * 32, shm>>>(
        D.coded, D.off, D.smpl, D.hdr, D.Wdec, L.KS, L.NS);
}

template <int SPL, int LANES, int WPB>
static float run_decode(const Layout &L, const Dev &D)
{
    if (L.NRB % WPB || L.SPL != SPL || L.LANES != LANES) return 1e30f;
    return time_ms([&] { launch_decode<SPL, LANES, WPB>(L, D); });
}

// SPL and LANES are template parameters; this picks the instantiation the
// coded blob was actually built for.
#define DEC_ONE(SPL, LANES) do { \
        if (wpb == 8) launch_decode<SPL, LANES, 8>(L, D); \
        else          launch_decode<SPL, LANES, 4>(L, D); } while (0)
static void decode_now(const Layout &L, const Dev &D, int wpb = 4)
{
    if (L.LANES == 8) {
        if (L.SPL == 1) DEC_ONE(1, 8); else if (L.SPL == 2) DEC_ONE(2, 8);
        else DEC_ONE(4, 8);
    } else {
        if (L.SPL == 1) DEC_ONE(1, 32); else if (L.SPL == 2) DEC_ONE(2, 32);
        else DEC_ONE(4, 32);
    }
}

static const int WPBS[5] = {1, 2, 4, 8, 16};

// Smallest legal stream length for a shape, used only by the coder self-check.
static int KSC0(int K)
{
    for (int ks : {128, 256, 512, 1024, 2048})
        if (K % ks == 0 && (K / ks) % 4 == 0) return ks;
    return K / 4;
}

// Runtime warps-per-block over a compile-time templated kernel.
#define WPB4(fn, ...) \
    ((w) == 1 ? fn<__VA_ARGS__, 1>(L, D) : (w) == 2 ? fn<__VA_ARGS__, 2>(L, D) \
     : (w) == 4 ? fn<__VA_ARGS__, 4>(L, D) : (w) == 8 ? fn<__VA_ARGS__, 8>(L, D) \
     : fn<__VA_ARGS__, 16>(L, D))

template <int MT> static float best_raw(const Layout &L, const Dev &D, int *bw)
{
    float b = 1e30f;
    for (int w : WPBS) {
        float t = L.LANES == 8 ? WPB4(run_raw, MT, 8) : WPB4(run_raw, MT, 32);
        if (t < b) { b = t; *bw = w; }
    }
    return b;
}
// SPL is compile-time in the kernel and runtime in the Layout; these three
// pick the instantiation the coded blob was actually built for.
#define BY_SPL1(fn, LN, ...) (L.SPL == 1 ? WPB4(fn, __VA_ARGS__, 1, LN) \
                            : L.SPL == 2 ? WPB4(fn, __VA_ARGS__, 2, LN) \
                                         : WPB4(fn, __VA_ARGS__, 4, LN))
#define BY_SPL(fn, ...) (L.LANES == 8 ? BY_SPL1(fn, 8, __VA_ARGS__) \
                                      : BY_SPL1(fn, 32, __VA_ARGS__))

template <int MT, bool MMA>
static float best_fused(const Layout &L, const Dev &D, int *bw)
{
    float b = 1e30f;
    for (int w : WPBS) {
        float t = BY_SPL(run_fused, MT, MMA);
        if (t < b) { b = t; *bw = w; }
    }
    return b;
}

static float fused_now(const Layout &L, const Dev &D, int mt, int w)
{
    switch (mt) {
    case 1:  return BY_SPL(run_fused, 1, true);
    case 2:  return BY_SPL(run_fused, 2, true);
    case 4:  return BY_SPL(run_fused, 4, true);
    case 8:  return BY_SPL(run_fused, 8, true);
    default: return BY_SPL(run_fused, 16, true);
    }
}

// Alternate two kernels inside one loop. Clock drift on this box is worth more
// than most of what is being measured here, so the only safe comparison is one
// where both sides see the same clocks.
template <typename FA, typename FB>
static void duel(FA fa, FB fb, float *a, float *b, int rounds = 8)
{
    *a = 1e30f; *b = 1e30f;
    for (int r = 0; r < rounds; r++) {
        float x = fa(); if (x < *a) *a = x;
        float y = fb(); if (y < *b) *b = y;
    }
}

static float raw_now(const Layout &L, const Dev &D, int mt, int w)
{
#define RAW_LN(MT) (L.LANES == 8 ? WPB4(run_raw, MT, 8) : WPB4(run_raw, MT, 32))
    switch (mt) {
    case 1:  return RAW_LN(1);
    case 2:  return RAW_LN(2);
    case 4:  return RAW_LN(4);
    case 8:  return RAW_LN(8);
    default: return RAW_LN(16);
    }
#undef RAW_LN
}

static float best_decode(const Layout &L, const Dev &D, int *bw)
{
    float b = 1e30f;
    for (int w : WPBS) {
        float t = L.LANES == 8
                ? (L.SPL == 1 ? WPB4(run_decode, 1, 8)
                   : L.SPL == 2 ? WPB4(run_decode, 2, 8) : WPB4(run_decode, 4, 8))
                : (L.SPL == 1 ? WPB4(run_decode, 1, 32)
                   : L.SPL == 2 ? WPB4(run_decode, 2, 32) : WPB4(run_decode, 4, 32));
        if (t < b) { b = t; *bw = w; }
    }
    return b;
}

static void upload(const Layout &L, Dev &D, const Coded &C,
                   const std::vector<uint8_t> &smp,
                   const std::vector<uint16_t> &Wt, bool with_raw)
{
    CK(cudaFree(D.coded)); CK(cudaFree(D.off));
    CK(cudaMalloc(&D.coded, C.blob.size()));
    CK(cudaMemcpy(D.coded, C.blob.data(), C.blob.size(), cudaMemcpyHostToDevice));
    CK(cudaMalloc(&D.off, L.nstreams * 8));
    CK(cudaMemcpy(D.off, C.off.data(), L.nstreams * 8, cudaMemcpyHostToDevice));
    CK(cudaMemcpy(D.hdr, C.hdr.data(), 516, cudaMemcpyHostToDevice));
    CK(cudaMemcpy(D.smpl, smp.data(), L.nelem, cudaMemcpyHostToDevice));
    if (with_raw)
        CK(cudaMemcpy(D.Wt, Wt.data(), L.nelem * 2, cudaMemcpyHostToDevice));
}

int main(int argc, char **argv)
{
    const char *tfmt = argc > 1 ? argv[1] : "model.layers.%d.mlp.gate_proj.weight";
    int NLAY = argc > 2 ? atoi(argv[2]) : 8;
    const char *path = argc > 3 ? argv[3]
                                : "/home/rog/.cache/lmz-bench/model.safetensors";
    // "lanes,spl,ks" pins the format and skips the sweep. The sweep is 25
    // configurations over 268 MB each; running it first leaves the card warm
    // and drags every table after it, so the headline numbers are taken with
    // the format pinned and the GPU settled.
    int pinLN = 0, pinSPL = 0, pinKS = 0;
    if (argc > 4) sscanf(argv[4], "%d,%d,%d", &pinLN, &pinSPL, &pinKS);

    cudaDeviceProp prop; CK(cudaGetDeviceProperties(&prop, 0));
    int shm_opt = 0, l2 = 0;
    CK(cudaDeviceGetAttribute(&shm_opt, cudaDevAttrMaxSharedMemoryPerBlockOptin, 0));
    CK(cudaDeviceGetAttribute(&l2, cudaDevAttrL2CacheSize, 0));
    g_shm_max = (size_t)shm_opt;
    printf("GPU %s  sm_%d%d  %d SMs  %.0f KB shared/block  %d warps/SM max  "
           "%.0f MB L2\n", prop.name, prop.major, prop.minor,
           prop.multiProcessorCount, g_shm_max / 1024.0,
           prop.maxThreadsPerMultiProcessor / 32, l2 / 1048576.0);

    // ---- a stack of real weight matrices, big enough to miss L2 ----------
    FILE *f = fopen(path, "rb");
    if (!f) { fprintf(stderr, "cannot open %s\n", path); return 1; }
    uint64_t hlen; if (fread(&hlen, 8, 1, f) != 1) return 1;
    std::string hdr(hlen, '\0');
    if (fread(&hdr[0], 1, hlen, f) != hlen) return 1;

    std::vector<uint16_t> W;
    int N = 0, K = 0;
    for (int i = 0; i < NLAY; i++) {
        char nm[256]; snprintf(nm, sizeof nm, tfmt, i);
        TensorRef tr;
        if (!st_lookup(hdr, nm, tr)) { NLAY = i; break; }
        if (!K) { K = (int)tr.cols; }
        else if ((int)tr.cols != K) { NLAY = i; break; }
        size_t at = W.size();
        W.resize(at + tr.nbytes / 2);
        fseek(f, (long)(8 + hlen + tr.off), SEEK_SET);
        if (fread(W.data() + at, 1, tr.nbytes, f) != tr.nbytes) return 1;
        N += (int)tr.rows;
    }
    fclose(f);
    if (!NLAY) { fprintf(stderr, "no matching BF16 tensors\n"); return 1; }
    char nm0[256]; snprintf(nm0, sizeof nm0, tfmt, 0);
    printf("weights %d x %s  ->  [%d x %d] BF16  %.1f MB  (%.1fx L2)\n\n",
           NLAY, nm0, N, K, (double)N * K * 2 / 1e6, (double)N * K * 2 / l2);
    if (N % (TILE * 8) || K % 512) {
        fprintf(stderr, "shape not supported by this tiling\n"); return 1;
    }

    // ---- activations ------------------------------------------------------
    const int MTMAX = 16, MPAD = MTMAX * 8;
    std::vector<uint16_t> Xh((size_t)MPAD * K);
    uint32_t rs = 0x12345678u;
    for (size_t i = 0; i < Xh.size(); i++) {
        rs = rs * 1664525u + 1013904223u;
        float v = ((int)(rs >> 8) / (float)(1 << 23)) - 1.0f;
        __nv_bfloat16 b = __float2bfloat16(v * 0.5f);
        memcpy(&Xh[i], &b, 2);
    }

    Dev D;
    const size_t NE = (size_t)N * K;
    CK(cudaMalloc(&D.Wnat, NE * 2));
    CK(cudaMemcpy(D.Wnat, W.data(), NE * 2, cudaMemcpyHostToDevice));
    CK(cudaMalloc(&D.Wt, NE * 2));
    CK(cudaMalloc(&D.Wdec, NE * 2));
    CK(cudaMalloc(&D.smpl, NE));
    CK(cudaMalloc(&D.X, (size_t)MPAD * K * 2));
    CK(cudaMemcpy(D.X, Xh.data(), (size_t)MPAD * K * 2, cudaMemcpyHostToDevice));
    CK(cudaMalloc(&D.Y, (size_t)N * MPAD * 4));
    CK(cudaMalloc(&D.Ycb, (size_t)N * MPAD * 4));
    CK(cudaMalloc(&D.hdr, 516));

    // ---- the generic coder must be lmz's coder at 8 states ----------------
    {
        Layout Lc = make_layout(N, K, KSC0(K), 1, 8);
        std::vector<uint16_t> Wc; std::vector<uint8_t> ec, sc;
        relayout(W.data(), Lc, Wc, ec, sc);
        uint64_t cnt[256] = {0};
        lmzx_hist(ec.data(), ec.size(), cnt);
        uint16_t fq[256];
        if (lmzx_normalize(cnt, fq) != 0) { fprintf(stderr, "normalise\n"); return 1; }
        size_t cap = (size_t)Lc.PLANE * 2 + 256;
        std::vector<uint8_t> a(cap), b(cap);
        int bad = 0;
        for (size_t i = 0; i < Lc.nstreams; i += Lc.nstreams / 64 + 1) {
            long na = lmzx_encode_shared(ec.data() + i * Lc.PLANE, Lc.PLANE, fq,
                                         a.data(), cap);
            long nb = enc_shared_n(ec.data() + i * Lc.PLANE, Lc.PLANE, fq,
                                   b.data(), cap, 8);
            if (na <= 0 || na != nb || memcmp(a.data(), b.data(), (size_t)na)) bad++;
        }
        printf("generic coder vs lmz's at 8 states: %s\n",
               bad ? "*** DIFFERS ***" : "byte-identical");
        if (bad) return 1;
    }

    // ---- the three format knobs: stream length, lanes/stream, states/lane -
    printf("\n-- lanes per stream x states per lane, fused decode+mma at M=8 --\n");
    printf("%5s %5s %6s %10s %8s %8s %9s %9s %7s\n", "lanes", "SPL", "states",
           "elem/strm", "coded MB", "ratio", "fused GB/s", "raw GB/s", "rel");
    printf("      (raw is measured next to each fused figure, because this "
           "card's clocks\n       drift over a sweep this long; the ratio is "
           "what survives that)\n");
    int bestKS = 0, bestSPL = 1, bestLN = 8; float bestT = 1e30f;   // fused/raw
    std::vector<uint16_t> Wt, chk((size_t)N * K); std::vector<uint8_t> expp, smp;
    Coded C;
    for (int ln : {8, 32}) {
        if (pinLN && ln != pinLN) continue;
        std::vector<int> KSC;
        for (int ks : {128, 256, 512, 1024, 2048, 4096})
            if (K % ks == 0 && (K / ks) % (32 / ln) == 0 && ks <= K) KSC.push_back(ks);
        for (int spl : {1, 2, 4}) {
            if (pinSPL && spl != pinSPL) continue;
            for (int ks : KSC) {
                if (256 % (ln * spl)) continue;
                if (pinKS && ks != pinKS) continue;
                Layout L = make_layout(N, K, ks, spl, ln);
                relayout(W.data(), L, Wt, expp, smp);
                encode_streams(expp, L, C);
                upload(L, D, C, smp, Wt, false);
                double codedMB = (C.payload + (double)L.nelem) / 1e6;
                int bw = 4, bwr = 4;
                float b = 1e30f, r = 1e30f;
                for (int rep = 0; rep < 3; rep++) {      // interleaved
                    b = std::min(b, best_fused<1, true>(L, D, &bw));
                    r = std::min(r, best_raw<1>(L, D, &bwr));
                }
                // every point in this table is checked, not just the winner:
                // a format that decodes wrongly can easily decode fast
                CK(cudaMemset(D.Wdec, 0, L.nelem * 2));
                decode_now(L, D);
                CK(cudaDeviceSynchronize()); CK(cudaGetLastError());
                CK(cudaMemcpy(chk.data(), D.Wdec, L.nelem * 2, cudaMemcpyDeviceToHost));
                bool ok = memcmp(chk.data(), Wt.data(), L.nelem * 2) == 0;
                double gf = (double)L.nelem * 2 / (b * 1e-3) / 1e9;
                double gr = (double)L.nelem * 2 / (r * 1e-3) / 1e9;
                printf("%5d %5d %6d %10d %8.1f %7.3fx %9.1f %9.1f %6.2fx%s\n",
                       ln, spl, ln * spl, L.PLANE, codedMB,
                       (double)L.nelem * 2 / (C.payload + (double)L.nelem),
                       gf, gr, gf / gr,
                       ok ? "" : "  *** DECODE MISMATCH ***");
                if (ok && b / r < bestT) { bestT = b / r; bestKS = ks; bestSPL = spl; bestLN = ln; }
            }
        }
    }

    Layout L = make_layout(N, K, bestKS, bestSPL, bestLN);
    relayout(W.data(), L, Wt, expp, smp);
    encode_streams(expp, L, C);
    upload(L, D, C, smp, Wt, true);
    const double codedB = C.payload + (double)L.nelem;
    const double ratio = (double)L.nelem * 2 / codedB;
    printf("\nchosen: %d elements per stream, %d lanes x %d states = %d states "
           "per stream\n        %.1f MB coded vs %.1f MB raw, %.3fx lossless\n",
           L.PLANE, bestLN, bestSPL, bestLN * bestSPL, codedB / 1e6,
           L.nelem * 2 / 1e6, ratio);

    // ---- correctness ------------------------------------------------------
    CK(cudaMemset(D.Wdec, 0, L.nelem * 2));
    decode_now(L, D);
    CK(cudaDeviceSynchronize()); CK(cudaGetLastError());
    CK(cudaMemcpy(chk.data(), D.Wdec, L.nelem * 2, cudaMemcpyDeviceToHost));
    bool exact = memcmp(chk.data(), Wt.data(), L.nelem * 2) == 0;
    printf("decode vs the checkpoint's own bytes: %s\n",
           exact ? "byte-identical over the whole stack" : "*** MISMATCH ***");
    if (!exact) return 1;

    const int RREF = 64;
    std::vector<double> ref((size_t)RREF * MPAD);
    for (int n = 0; n < RREF; n++)
        for (int m = 0; m < MPAD; m++) {
            double s = 0;
            for (int k = 0; k < K; k++) {
                uint32_t a = (uint32_t)W[(size_t)n * K + k] << 16;
                uint32_t b = (uint32_t)Xh[(size_t)m * K + k] << 16;
                float fa, fb; memcpy(&fa, &a, 4); memcpy(&fb, &b, 4);
                s += (double)fa * fb;
            }
            ref[(size_t)n * MPAD + m] = s;
        }

    // ---- occupancy: the LUT is paid per block, so warps per block is it ---
    printf("\n-- warps per block, fused decode+mma at M=8 --\n");
    printf("%10s %9s %11s %10s\n", "warps/blk", "shared KB", "ms", "GB/s BF16");
    for (int w : WPBS) {
        float t = BY_SPL(run_fused, 1, true);
        if (t > 1e29f) continue;
        double shm = (LUT_BYTES + (size_t)w * (32 / L.LANES)
                      * (TILE_ELEMS * 2 + BUFB_OF(L.LANES, L.SPL))) / 1024.0;
        printf("%10d %9.1f %11.3f %10.1f\n", w, shm, t,
               (double)L.nelem * 2 / (t * 1e-3) / 1e9);
    }

    // ---- where the decoder's bandwidth goes -------------------------------
    printf("\n-- the decoder, by what it is asked to feed (M=8) --\n");
    printf("%-38s %8s %10s %10s\n", "", "ms", "GB/s BF16", "of standalone");
    int bw = 4, bwm = 4, bws = 4;
    best_decode(L, D, &bw);
    best_fused<1, false>(L, D, &bws);
    best_fused<1, true>(L, D, &bwm);
    float td, tm, ts, tm2;
    { int w = bwm; duel([&] { int w2 = bw; (void)w2;
                              return L.LANES == 8
                                ? (L.SPL == 1 ? WPB4(run_decode, 1, 8)
                                   : L.SPL == 2 ? WPB4(run_decode, 2, 8)
                                   : WPB4(run_decode, 4, 8))
                                : (L.SPL == 1 ? WPB4(run_decode, 1, 32)
                                   : L.SPL == 2 ? WPB4(run_decode, 2, 32)
                                   : WPB4(run_decode, 4, 32)); },
                     [&] { return fused_now(L, D, 1, bwm); }, &td, &tm); }
    { int w = bws; duel([&] { return BY_SPL(run_fused, 1, false); },
                        [&] { return fused_now(L, D, 1, bwm); }, &ts, &tm2); }
    if (tm2 < tm) tm = tm2;
    double rawB = (double)L.nelem * 2;
    printf("%-38s %8.3f %10.1f %10s\n", "decode -> VRAM (lmz today)", td,
           rawB / (td * 1e-3) / 1e9, "1.00x");
    printf("%-38s %8.3f %10.1f %9.2fx\n", "decode -> shared, no matmul", ts,
           rawB / (ts * 1e-3) / 1e9, td / ts);
    printf("%-38s %8.3f %10.1f %9.2fx\n", "decode -> shared -> mma (fused)", tm,
           rawB / (tm * 1e-3) / 1e9, td / tm);

    // ---- the layer, three ways -------------------------------------------
    cublasHandle_t cbh; CB(cublasCreate(&cbh));
    printf("\n-- one linear layer over %.0f MB of weights, three ways --\n",
           rawB / 1e6);
    printf("%5s | %8s %8s %8s %8s | %8s %8s | %s\n", "M", "raw ms", "fused",
           "2-pass", "cuBLAS", "raw TF/s", "fusd TF/s",
           "WPB r/f  fused/raw   check");
    std::vector<float> yf((size_t)N * MPAD), yr((size_t)N * MPAD);

    for (int mt : {1, 2, 4, 8, 16}) {
        const int M = mt * 8;
        int bwr = 4, bwf = 4;
        float tr_, tf;
        switch (mt) {
        case 1:  tr_ = best_raw<1>(L, D, &bwr);  tf = best_fused<1, true>(L, D, &bwf);  break;
        case 2:  tr_ = best_raw<2>(L, D, &bwr);  tf = best_fused<2, true>(L, D, &bwf);  break;
        case 4:  tr_ = best_raw<4>(L, D, &bwr);  tf = best_fused<4, true>(L, D, &bwf);  break;
        case 8:  tr_ = best_raw<8>(L, D, &bwr);  tf = best_fused<8, true>(L, D, &bwf);  break;
        default: tr_ = best_raw<16>(L, D, &bwr); tf = best_fused<16, true>(L, D, &bwf); break;
        }
        fused_now(L, D, mt, bwf);
        CK(cudaMemcpy(yf.data(), D.Y, (size_t)N * M * 4, cudaMemcpyDeviceToHost));
        raw_now(L, D, mt, bwr);
        CK(cudaMemcpy(yr.data(), D.Y, (size_t)N * M * 4, cudaMemcpyDeviceToHost));

        float t2 = time_ms([&] {
            decode_now(L, D);
            int blocks = L.NRB / 4;
#define RAW2(LN) do { \
            size_t shm = SHM_RAW(4, LN); \
            switch (mt) { \
            case 1:  k_raw<1, 4, LN><<<blocks, 128, shm>>>(D.Wdec, D.X, D.Y, L.K, L.KS, L.NS); break; \
            case 2:  k_raw<2, 4, LN><<<blocks, 128, shm>>>(D.Wdec, D.X, D.Y, L.K, L.KS, L.NS); break; \
            case 4:  k_raw<4, 4, LN><<<blocks, 128, shm>>>(D.Wdec, D.X, D.Y, L.K, L.KS, L.NS); break; \
            case 8:  k_raw<8, 4, LN><<<blocks, 128, shm>>>(D.Wdec, D.X, D.Y, L.K, L.KS, L.NS); break; \
            default: k_raw<16, 4, LN><<<blocks, 128, shm>>>(D.Wdec, D.X, D.Y, L.K, L.KS, L.NS); break; \
            } } while (0)
            if (L.LANES == 8) RAW2(8); else RAW2(32);
#undef RAW2
        });
        const float alpha = 1.f, beta = 0.f;
        float tc = time_ms([&] {
            cublasGemmEx(cbh, CUBLAS_OP_T, CUBLAS_OP_N, N, M, K, &alpha,
                         D.Wnat, CUDA_R_16BF, K, D.X, CUDA_R_16BF, K, &beta,
                         D.Ycb, CUDA_R_32F, N, CUBLAS_COMPUTE_32F,
                         CUBLAS_GEMM_DEFAULT);
        });

        bool bit = memcmp(yf.data(), yr.data(), (size_t)N * M * 4) == 0;
        double emax = 0, rmax = 0;
        for (int n = 0; n < RREF; n++)
            for (int m = 0; m < M; m++) {
                double r = ref[(size_t)n * MPAD + m];
                emax = std::max(emax, fabs(yf[(size_t)n * M + m] - r));
                rmax = std::max(rmax, fabs(r));
            }
        double flop = 2.0 * N * K * M;
        printf("%5d | %8.3f %8.3f %8.3f %8.3f | %8.1f %8.1f | %2d/%-2d %6.2fx  %s %.0e\n",
               M, tr_, tf, t2, tc, flop / (tr_ * 1e-3) / 1e12,
               flop / (tf * 1e-3) / 1e12, bwr, bwf, tf / tr_,
               bit ? "exact," : "*DIFFERS*,", emax / rmax);
    }

    printf("\n-- what a byte of VRAM buys --\n");
    printf("%-32s %9s %9s %11s\n", "", "VRAM MB", "GB/s BF16", "rel. speed");
    {
        int wr = 4, wf = 4;
        best_raw<1>(L, D, &wr);
        best_fused<1, true>(L, D, &wf);
        float tr1, tf1;
        duel([&] { return raw_now(L, D, 1, wr); },
             [&] { return fused_now(L, D, 1, wf); }, &tr1, &tf1, 12);
        printf("%-32s %9.1f %9.1f %10.2fx\n", "raw BF16 in VRAM", rawB / 1e6,
               rawB / (tr1 * 1e-3) / 1e9, 1.0);
        printf("%-32s %9.1f %9.1f %10.2fx\n", "coded in VRAM, fused decode",
               codedB / 1e6, rawB / (tf1 * 1e-3) / 1e9, tr1 / tf1);
        printf("\n%.3fx the model per byte of VRAM, at %.2fx the speed.\n",
               ratio, tr1 / tf1);
        double dram_f = codedB / (tf1 * 1e-3) / 1e9;
        double dram_r = rawB / (tr1 * 1e-3) / 1e9;
        printf("DRAM actually moved: %.0f GB/s coded vs %.0f GB/s raw -- the "
               "coded tier is decode-bound\nwith %.1fx of this card's read "
               "bandwidth still unused.\n", dram_f, dram_r, dram_r / dram_f);
    }
    CB(cublasDestroy(cbh));
    return 0;
}
