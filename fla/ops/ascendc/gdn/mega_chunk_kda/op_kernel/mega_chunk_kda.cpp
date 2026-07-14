// mega_chunk_kda.cpp — KDA (Kimi Delta Attention) Mega-Kernel: all PTO stages in one launch
//
// Fuses the six KDA stages into a single NPU launch, modelled on the GDN mega_kernel.cpp.
// Sub-kernels are reused verbatim via namespaced #include (the `call_kernel` macro trick),
// and their templated device functions are invoked in sequence with SyncAllImpl<false>()
// barriers between stages.
//
// Stages:
//   1. gate_cumsum (Vec)             — within-chunk prefix sum of g  [B,T,HV,K]
//   2. transpose   (Vec)             — g_sum BSND [T,HV,K] -> head-major [HV,T,K]
//   3. kkt         (Cube+Vec)        — gated K·K^T lower-tri matrix L
//   4. solve_tril  (Cube)            — (I + L)^{-1}  (shared tri_inverse kernel)
//   5. wy          (Vec+Cube)        — WY auxiliaries u, w
//   6. chunk_h     (Cube+Vec)        — recurrent state snapshots + v_corr
//   7. chunk_o     (Cube+Vec)        — output
//
// KDA difference from GDN: gates are per-dimension [B,T,HV,K] (not scalar [B,T,H]), and
// the sub-kernels read k/q/g_cs in head-major [HV,T,K] layout (beta in [HV,T]).  Static
// inputs (q, k, beta) are permuted to head-major in Python; only g_sum — produced inside
// the kernel by gate_cumsum — is transposed on-device here.
//
// Framework entry adapted from xllm_ops MegaChunkGdn: the original `launch_mega_kernel_kda`
// __global__ is renamed to `mega_kernel_kda_impl` (AICORE void), and a new
// `extern "C" __global__ __aicore__` entry unpacks tiling data + workspace offsets and
// delegates to it.

#ifndef GDN_D
#define GDN_D 128
#endif
#ifndef GDN_C
#define GDN_C 128
#endif
#ifndef MEMORY_BASE
#define MEMORY_BASE
#endif
#ifndef KDA_KERNEL_NAME
#define KDA_KERNEL_NAME mega_chunk_kda
#endif
// Note the codegen parser does not support arguments of form "type *name", only "type* name"
// clang-format off
#ifndef GM_ADDR
#define GM_ADDR __gm__ uint8_t*
#endif
// clang-format off

#include "acl/acl.h"
#include "kernel_operator.h"
#include <pto/pto-inst.hpp>
#include <runtime/rt_ffts.h>
#include <type_traits>
using namespace pto;

struct MegaChunkKdaKernelTilingData {
    uint32_t block_dim;
    uint32_t num_matrices;
    uint32_t num_heads;
    int64_t batch_size;
    int64_t seq_len;
    int64_t total_tokens;
    uint64_t ffts_addr;
};

// ===================================================================
// Device-only helpers (shared with standard mega-kernel)
// ===================================================================
#ifdef __CCE_AICORE__

// pto::SYNC_AIC_FLAG / SYNC_AIV_FLAG / SYNC_AIC_AIV_FLAG / SYNC_AIV_ONLY_ALL
// are provided by pto/common/type.hpp; only these shift constants are local.
constexpr uint16_t SYNC_MODE_SHIFT_VALUE = 4;
constexpr uint16_t SYNC_FLAG_SHIFT_VALUE = 8;

AICORE inline uint16_t GetffstMsg(uint16_t mode, uint16_t flagId)
{
    return (0x1 + ((mode & 0x3) << SYNC_MODE_SHIFT_VALUE) +
            ((flagId & 0xf) << SYNC_FLAG_SHIFT_VALUE));
}

// Full Cube+Vec all-core barrier.  Uses FFTS flags 11-14, distinct from the sub-kernels'
// internal sync flags (6-9), so a sub-kernel's exit-sync immediately followed by this
// barrier never reuses the same flag back-to-back.
template <bool isAIVOnly = true>
AICORE inline void SyncAllImpl()
{
    pipe_barrier(PIPE_ALL);
    if constexpr (isAIVOnly)
    {
        ffts_cross_core_sync(PIPE_MTE3, GetffstMsg(0x0, SYNC_AIV_ONLY_ALL));
        wait_flag_dev(SYNC_AIV_ONLY_ALL);
        return;
    }
#if defined(__DAV_C220_CUBE__)
    wait_flag_dev(SYNC_AIV_FLAG);
    ffts_cross_core_sync(PIPE_FIX, GetffstMsg(0x0, SYNC_AIC_FLAG));
    wait_flag_dev(SYNC_AIC_FLAG);
    ffts_cross_core_sync(PIPE_MTE3, GetffstMsg(0x02, SYNC_AIC_AIV_FLAG));
#elif defined(__DAV_C220_VEC__)
    ffts_cross_core_sync(PIPE_MTE3, GetffstMsg(0x02, SYNC_AIV_FLAG));
    wait_flag_dev(SYNC_AIC_AIV_FLAG);
#endif
}

// Strided GM->GM copy reordering a per-dimension tensor from BSND [T,HV,K] to head-major
// [HV,T,K].  K stays innermost-contiguous; only the (T,HV) axes swap, so this is a pure
// gather/scatter of K-contiguous rows (no element transpose).  Layout-only — independent
// of cu_seqlens.
template <typename T, int32_t KD>
AICORE void mega_permute_THK_to_HTK(
    __gm__ T *src, __gm__ T *dst, int64_t T_len, int32_t HV)
{
#if defined(__DAV_C220_VEC__)
    if (get_subblockid() != 0)
        return;
    set_mask_norm();
    set_vector_mask(-1, -1);

    auto cid = get_block_idx();
    auto block_num = get_block_num();

    constexpr int32_t BLOCK = 128; // tokens per UB tile
    constexpr int32_t UB0 = 0;

    using UBTileDyn = Tile<TileType::Vec, T, BLOCK, KD, BLayout::RowMajor,
                           DYNAMIC, DYNAMIC, SLayout::NoneBox, 512, PadValue::Zero>;
    using Gm2D = Shape<1, 1, 1, DYNAMIC, DYNAMIC>;
    using GmSrcS = Stride<1, 1, 1, DYNAMIC, 1>; // row stride = HV*K (runtime; skip other heads)
    using GmDstS = Stride<1, 1, 1, KD, 1>;      // contiguous
    GmSrcS src_stride(HV * KD);

    int64_t num_tok_blocks = (T_len + BLOCK - 1) / BLOCK;
    int64_t total = static_cast<int64_t>(HV) * num_tok_blocks;

    for (int64_t wi = static_cast<int64_t>(cid); wi < total;
         wi += static_cast<int64_t>(block_num))
    {
        int64_t h = wi / num_tok_blocks;
        int64_t bi = wi % num_tok_blocks;
        int64_t t0 = bi * BLOCK;
        int32_t valid = (t0 + BLOCK <= T_len)
                            ? BLOCK
                            : static_cast<int32_t>(T_len - t0);

        {
            Gm2D gs;
            gs.shape[3] = valid;
            gs.shape[4] = KD;
            GlobalTensor<T, Gm2D, GmSrcS> gm(
                src + (t0 * static_cast<int64_t>(HV) + h) * KD, gs, src_stride);
            UBTileDyn ld(valid, KD);
            TASSIGN(ld, UB0);
            TLOAD(ld, gm);
        }
        set_flag(PIPE_MTE2, PIPE_MTE3, EVENT_ID0);
        wait_flag(PIPE_MTE2, PIPE_MTE3, EVENT_ID0);
        {
            Gm2D gs;
            gs.shape[3] = valid;
            gs.shape[4] = KD;
            GlobalTensor<T, Gm2D, GmDstS> gm(
                dst + (h * T_len + t0) * KD, gs);
            UBTileDyn st(valid, KD);
            TASSIGN(st, UB0);
            TSTORE(gm, st);
        }
        set_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID0);
        wait_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID0);
    }
#endif
}

#endif // __CCE_AICORE__

// ===================================================================
// Include original KDA sub-kernel implementations in separate namespaces.
// The `call_kernel` macro trick renames each file's host wrapper so they
// don't collide (matches mega_kernel.cpp).
// ===================================================================

#define call_kernel _mk_unused_kda_gc
namespace mk_gc
{
#include "gate_cumsum_kda.cpp"
}
#undef call_kernel

#define call_kernel _mk_unused_kda_kkt
namespace mk_kkt
{
#include "kkt_kda.cpp"
}
#undef call_kernel

namespace mk_solve
{
#include "tri_inverse_impl.cpp"
}

#define call_kernel _mk_unused_kda_wy
namespace mk_wy
{
#include "wy_kda.cpp"
}
#undef call_kernel

#define call_kernel _mk_unused_kda_h
namespace mk_h
{
#include "chunk_h_kda.cpp"
}
#undef call_kernel

#define call_kernel _mk_unused_kda_o
namespace mk_o
{
#include "chunk_o_kda.cpp"
}
#undef call_kernel

// Shared triangular-inverse dispatch (identical to GDN mega_kernel.cpp).
AICORE inline void mega_solve_tril(
    __gm__ half *out, __gm__ half *in, __gm__ half *minus_id,
    uint32_t matrix_size, uint32_t num_matrices,
    uint32_t num_bsnd_heads,
    __gm__ int32_t *cu_seqlens, uint32_t is_lower)
{
    if (num_matrices <= get_block_num())
        mk_solve::runKernelTriInvRecUnroll<half, float, GDN_C, 1, true, half>(
            out, in, minus_id, num_matrices,
            num_bsnd_heads, cu_seqlens, is_lower);
    else if (num_matrices <= 2u * get_block_num())
        mk_solve::runKernelTriInvRecUnroll<half, float, GDN_C, 2, true, half>(
            out, in, minus_id, num_matrices,
            num_bsnd_heads, cu_seqlens, is_lower);
    else
        mk_solve::runKernelTriInvRecUnroll<half, float, GDN_C, 4, true, half>(
            out, in, minus_id, num_matrices,
            num_bsnd_heads, cu_seqlens, is_lower);
}

// ===================================================================
// Fused pipeline (renamed from launch_mega_kernel_kda to avoid __global__
// collision with the framework entry below).
// ===================================================================
AICORE inline void mega_kernel_kda_impl(
    // ── inputs ──────────────────────────────────────────────────────
    __gm__ uint8_t *q_hm_ptr,    // [1, HV, T, K] fp16 (head-major, scaled)
    __gm__ uint8_t *k_hm_ptr,    // [1, HV, T, K] fp16 (head-major)
    __gm__ uint8_t *v_ptr,       // [1, T, HV, V] fp16 (BSND)
    __gm__ uint8_t *g_in_ptr,    // [1, T, HV, K] fp16 (BSND, raw gate)
    __gm__ uint8_t *beta_hm_ptr, // [1, HV, T]    fp16 (head-major)
    // ── masks / constants ───────────────────────────────────────────
    __gm__ uint8_t *mask_strict_ptr, // [C, C] fp32 (rows >  cols)
    __gm__ uint8_t *mask_incl_ptr,   // [C, C] fp32 (rows >= cols)
    __gm__ uint8_t *minus_id_ptr,    // [C, C] fp16 (-I)
    __gm__ uint8_t *cu_seqlens_ptr,  // int32
    // ── output ──────────────────────────────────────────────────────
    __gm__ uint8_t *o_ptr, // [1, T, HV, V] fp16 (BSND)
    // ── intermediate buffers ────────────────────────────────────────
    __gm__ uint8_t *g_sum_ptr,   // [1, T, HV, K] fp32 (BSND)
    __gm__ uint8_t *g_cs_hm_ptr, // [1, HV, T, K] fp32 (head-major)
    __gm__ uint8_t *L_ptr,       // [1, T, HV, C] fp16
    __gm__ uint8_t *A_inv_ptr,   // [1, T, HV, C] fp16
    __gm__ uint8_t *u_ptr,       // [1, T, HV, V] fp16
    __gm__ uint8_t *w_ptr,       // [1, T, HV, K] fp16
    __gm__ uint8_t *s_ptr,       // [tc, HV, K, V] fp16
    __gm__ uint8_t *v_corr_ptr,  // [1, T, HV, V] fp16
    // ── per-core workspaces ─────────────────────────────────────────
    __gm__ uint8_t *kkt_ws_in_ptr,  // [bd*2, 2C, K] fp32 (stages exp(±g_cs))
    __gm__ uint8_t *kkt_ws_out_ptr, // [bd*2, C, C]  fp32 (unmasked gated K·K^T)
    __gm__ uint8_t *wy_ws_a2_ptr,   // [bd, C, C]    fp16
    __gm__ uint8_t *wy_ws_keff_ptr, // [bd, C, K]    fp16
    __gm__ uint8_t *h_ws_ptr,       // [bd*5, K, K]  fp16
    __gm__ uint8_t *o_ws_ptr,       // [bd*7, K, K]  fp32 (gated q/k + GEMM I/O)
    // ── scalars ─────────────────────────────────────────────────────
    int64_t batch_size,
    int64_t seq_len,
    int64_t total_tokens,
    uint32_t num_matrices,
    int32_t num_heads,
    uint64_t ffts_addr)
{
    set_ffts_base_addr(ffts_addr);

    // Head count (HV) is a runtime argument; only GDN_D (K) and GDN_C (C) stay
    // compile-time, so the fused .so is head-count-agnostic like the staged ones.
    const int32_t HV = num_heads;
    constexpr int32_t KD = GDN_D;
    constexpr int32_t C = GDN_C;

    // ── Stage 1: gate_cumsum (BSND -> BSND) ──────────────────────────
    mk_gc::gate_cumsum_kda_kernel<KD, C>(
        reinterpret_cast<__gm__ half *>(g_in_ptr),
        reinterpret_cast<__gm__ float *>(g_sum_ptr),
        reinterpret_cast<__gm__ int32_t *>(cu_seqlens_ptr),
        batch_size, seq_len, HV, ffts_addr);

#ifdef MEGA_STOP_AFTER_CUMSUM
    pipe_barrier(PIPE_ALL);
    return;
#endif

    SyncAllImpl<false>();

    // ── Stage 2: transpose g_sum -> head-major g_cs ──────────────────
    mega_permute_THK_to_HTK<float, KD>(
        reinterpret_cast<__gm__ float *>(g_sum_ptr),
        reinterpret_cast<__gm__ float *>(g_cs_hm_ptr),
        total_tokens, HV);

#ifdef MEGA_STOP_AFTER_TRANSPOSE
    pipe_barrier(PIPE_ALL);
    return;
#endif

    SyncAllImpl<false>();

    // ── Stage 3: kkt (gated K·K^T lower-tri matrix) ──────────────────
    mk_kkt::kkt_kda_kernel<KD, C>(
        reinterpret_cast<__gm__ half *>(k_hm_ptr),
        reinterpret_cast<__gm__ float *>(g_cs_hm_ptr),
        reinterpret_cast<__gm__ half *>(beta_hm_ptr),
        reinterpret_cast<__gm__ float *>(mask_strict_ptr),
        reinterpret_cast<__gm__ float *>(kkt_ws_in_ptr),
        reinterpret_cast<__gm__ float *>(kkt_ws_out_ptr),
        reinterpret_cast<__gm__ half *>(L_ptr),
        reinterpret_cast<__gm__ int32_t *>(cu_seqlens_ptr),
        batch_size, seq_len, total_tokens, HV, ffts_addr);

#ifdef MEGA_STOP_AFTER_KKT
    pipe_barrier(PIPE_ALL);
    return;
#endif

    SyncAllImpl<false>();

    // ── Stage 4: solve_tril ((I + L)^{-1}) ───────────────────────────
    mega_solve_tril(
        reinterpret_cast<__gm__ half *>(A_inv_ptr),
        reinterpret_cast<__gm__ half *>(L_ptr),
        reinterpret_cast<__gm__ half *>(minus_id_ptr),
        C, num_matrices, HV,
        reinterpret_cast<__gm__ int32_t *>(cu_seqlens_ptr), 1);

#ifdef MEGA_STOP_AFTER_SOLVE
    pipe_barrier(PIPE_ALL);
    return;
#endif

    SyncAllImpl<false>();

    // ── Stage 5: wy (auxiliaries u, w) ───────────────────────────────
    mk_wy::wy_kda_kernel<KD, C>(
        reinterpret_cast<__gm__ half *>(k_hm_ptr),
        reinterpret_cast<__gm__ half *>(v_ptr),
        reinterpret_cast<__gm__ half *>(beta_hm_ptr),
        reinterpret_cast<__gm__ float *>(g_cs_hm_ptr),
        reinterpret_cast<__gm__ half *>(A_inv_ptr),
        reinterpret_cast<__gm__ half *>(wy_ws_a2_ptr),
        reinterpret_cast<__gm__ half *>(wy_ws_keff_ptr),
        reinterpret_cast<__gm__ half *>(u_ptr),
        reinterpret_cast<__gm__ half *>(w_ptr),
        reinterpret_cast<__gm__ int32_t *>(cu_seqlens_ptr),
        batch_size, seq_len, total_tokens, HV, ffts_addr);

#ifdef MEGA_STOP_AFTER_WY
    pipe_barrier(PIPE_ALL);
    return;
#endif

    SyncAllImpl<false>();

    // ── Stage 6: chunk_h (state snapshots + v_corr) ──────────────────
    mk_h::chunk_h_kda_kernel<KD, C>(
        reinterpret_cast<__gm__ half *>(k_hm_ptr),
        reinterpret_cast<__gm__ half *>(w_ptr),
        reinterpret_cast<__gm__ half *>(u_ptr),
        reinterpret_cast<__gm__ float *>(g_cs_hm_ptr),
        reinterpret_cast<__gm__ half *>(s_ptr),
        reinterpret_cast<__gm__ half *>(v_corr_ptr),
        reinterpret_cast<__gm__ half *>(h_ws_ptr),
        reinterpret_cast<__gm__ int32_t *>(cu_seqlens_ptr),
        batch_size, seq_len, total_tokens, HV, ffts_addr);

#ifdef MEGA_STOP_AFTER_H
    pipe_barrier(PIPE_ALL);
    return;
#endif

    SyncAllImpl<false>();

    // ── Stage 7: chunk_o (output) ────────────────────────────────────
    mk_o::chunk_o_kda_kernel<KD, C>(
        reinterpret_cast<__gm__ half *>(q_hm_ptr),
        reinterpret_cast<__gm__ half *>(k_hm_ptr),
        reinterpret_cast<__gm__ half *>(v_corr_ptr),
        reinterpret_cast<__gm__ half *>(s_ptr),
        reinterpret_cast<__gm__ float *>(g_cs_hm_ptr),
        reinterpret_cast<__gm__ float *>(mask_incl_ptr),
        reinterpret_cast<__gm__ float *>(o_ws_ptr),
        reinterpret_cast<__gm__ half *>(o_ptr),
        reinterpret_cast<__gm__ int32_t *>(cu_seqlens_ptr),
        batch_size, seq_len, total_tokens, HV, ffts_addr);
}

// ===================================================================
// Framework entry — unpacks tiling data + workspace, delegates to pipeline.
// Parameter order must match mega_chunk_kda_def.cpp (9 inputs, 9 outputs).
// ===================================================================
extern "C" __global__ __aicore__ void
KDA_KERNEL_NAME(GM_ADDR q_ptr, GM_ADDR k_ptr, GM_ADDR v_ptr, GM_ADDR g_in_ptr, GM_ADDR beta_ptr,
                GM_ADDR mask_strict_ptr, GM_ADDR mask_incl_ptr, GM_ADDR minus_id_ptr, GM_ADDR cu_seqlens_ptr,
                GM_ADDR o_ptr, GM_ADDR g_sum_ptr, GM_ADDR g_cs_ptr, GM_ADDR L_ptr, GM_ADDR A_inv_ptr,
                GM_ADDR u_ptr, GM_ADDR w_ptr, GM_ADDR s_ptr, GM_ADDR v_corr_ptr,
                GM_ADDR workspace, GM_ADDR tiling)
{
    REGISTER_TILING_DEFAULT(MegaChunkKdaKernelTilingData);
    GET_TILING_DATA_WITH_STRUCT(MegaChunkKdaKernelTilingData, tiling_data, tiling);
    GM_ADDR user_ws = AscendC::GetUserWorkspace(workspace);

    // KDA workspace layout (mixed precision — must match tiling.cpp CalcUserWorkspaceBytes).
    const uint64_t bd = static_cast<uint64_t>(tiling_data.block_dim);
    constexpr uint64_t C = static_cast<uint64_t>(GDN_C);
    constexpr uint64_t K = static_cast<uint64_t>(GDN_D);

    GM_ADDR kkt_ws_in_ptr   = user_ws;
    GM_ADDR kkt_ws_out_ptr  = kkt_ws_in_ptr  + bd * 2 * 2 * C * K * sizeof(float);
    GM_ADDR wy_ws_a2_ptr    = kkt_ws_out_ptr + bd * 2 * C * C * sizeof(float);
    GM_ADDR wy_ws_keff_ptr  = wy_ws_a2_ptr   + bd * C * C * sizeof(half);
    GM_ADDR h_ws_ptr        = wy_ws_keff_ptr + bd * C * K * sizeof(half);
    GM_ADDR o_ws_ptr        = h_ws_ptr       + bd * 5 * K * K * sizeof(half);

    mega_kernel_kda_impl(
        q_ptr, k_ptr, v_ptr, g_in_ptr, beta_ptr,
        mask_strict_ptr, mask_incl_ptr, minus_id_ptr, cu_seqlens_ptr,
        o_ptr,
        g_sum_ptr, g_cs_ptr, L_ptr, A_inv_ptr,
        u_ptr, w_ptr, s_ptr, v_corr_ptr,
        kkt_ws_in_ptr, kkt_ws_out_ptr, wy_ws_a2_ptr, wy_ws_keff_ptr,
        h_ws_ptr, o_ws_ptr,
        tiling_data.batch_size, tiling_data.seq_len, tiling_data.total_tokens,
        tiling_data.num_matrices, static_cast<int32_t>(tiling_data.num_heads),
        tiling_data.ffts_addr);
}

// The CANN wrapper generated for mixed AIC/AIV kernels calls matmul::clearWorkspace
// after including this source. Keep this include after PTO code so CANN's DYNAMIC
// enum does not collide with pto::DYNAMIC in the kernel templates above.
#include "lib/matmul_intf.h"
