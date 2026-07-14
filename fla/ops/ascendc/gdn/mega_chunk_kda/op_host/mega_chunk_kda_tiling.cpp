/**
 * Copyright (c) 2026 Tianjin University, Ltd.
 * Adapted from xllm_ops MegaChunkGdn (Apache-2.0) and megagdn-pto KDA kernels.
 * This program is free software under CANN Open Software License Agreement Version 2.0.
 */

#include "mega_chunk_kda_tiling.h"

#include <algorithm>
#include <cstdint>
#include <iostream>
#include <limits>
#include <sstream>

#include "register/op_def_registry.h"
#include "tiling/platform/platform_ascendc.h"

namespace {
constexpr uint32_t kHeadDim = 128;    // K (== V)
constexpr uint32_t kChunkSize = 128;  // C
constexpr uint32_t kFloatBytes = 4;
constexpr uint32_t kHalfBytes = 2;

// Input indices — must match mega_chunk_kda_def.cpp declaration order.
enum InputIndex {
    Q_INDEX = 0,
    K_INDEX,
    V_INDEX,
    G_INDEX,
    BETA_INDEX,
    MASK_STRICT_INDEX,
    MASK_INCL_INDEX,
    MINUS_IDENTITY_INDEX,
    CU_SEQLENS_INDEX,
};

// Attr indices — must match mega_chunk_kda_def.cpp declaration order.
enum AttrIndex {
    NUM_MATRICES_ATTR = 0,
    FFTS_ADDR_ATTR,
};

uint32_t CeilDiv(uint32_t value, uint32_t divisor)
{
    return divisor == 0 ? 0 : (value + divisor - 1) / divisor;
}

// KDA workspace layout (mixed precision — differs from GDN's uniform fp16 tiles).
// The kernel entry computes per-pointer offsets from this total; the layout must
// match op_kernel/mega_chunk_kda.cpp exactly.
//
//   kkt_ws_in   [bd*2, 2C, K]  fp32  — stages exp(±g_cs), overflows fp16
//   kkt_ws_out  [bd*2, C,  C]  fp32  — unmasked gated K·K^T
//   wy_ws_a2    [bd,    C,  C]  fp16  — WY auxiliary matrix
//   wy_ws_keff  [bd,    C,  K]  fp16  — WY k-effective
//   h_ws        [bd*5,  K,  K]  fp16  — recurrent state snapshots
//   o_ws        [bd*7,  K,  K]  fp32  — gated q/k GEMM I/O, overflows fp16
uint64_t CalcUserWorkspaceBytes(uint32_t blockDim)
{
    const uint64_t bd = static_cast<uint64_t>(blockDim);
    const uint64_t C = kChunkSize;
    const uint64_t K = kHeadDim;

    const uint64_t kkt_ws_in_bytes   = bd * 2 * 2 * C * K * kFloatBytes;
    const uint64_t kkt_ws_out_bytes  = bd * 2 * C * C * kFloatBytes;
    const uint64_t wy_ws_a2_bytes    = bd * C * C * kHalfBytes;
    const uint64_t wy_ws_keff_bytes  = bd * C * K * kHalfBytes;
    const uint64_t h_ws_bytes        = bd * 5 * K * K * kHalfBytes;
    const uint64_t o_ws_bytes        = bd * 7 * K * K * kFloatBytes;

    return kkt_ws_in_bytes + kkt_ws_out_bytes + wy_ws_a2_bytes +
           wy_ws_keff_bytes + h_ws_bytes + o_ws_bytes;
}

std::string ShapeToString(const gert::Shape &shape)
{
    std::ostringstream oss;
    oss << "[";
    for (size_t i = 0; i < shape.GetDimNum(); ++i) {
        if (i > 0) {
            oss << ",";
        }
        oss << shape.GetDim(i);
    }
    oss << "]";
    return oss.str();
}

ge::graphStatus FailTiling(const std::string &reason)
{
    std::cerr << "[MegaChunkKdaTiling] " << reason << std::endl;
    return ge::GRAPH_FAILED;
}
}  // namespace

namespace optiling {
static ge::graphStatus TilingFunc(gert::TilingContext *context)
{
    auto platformInfo = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
    uint32_t blockDim = platformInfo.GetCoreNumAic();
    if (blockDim == 0) {
        blockDim = platformInfo.GetCoreNumAiv();
    }
    blockDim = std::max<uint32_t>(blockDim, 1);

    const gert::StorageShape *qShape = context->GetInputShape(Q_INDEX);
    const gert::StorageShape *vShape = context->GetInputShape(V_INDEX);
    const gert::StorageShape *cuShape = context->GetInputShape(CU_SEQLENS_INDEX);
    if (qShape == nullptr || vShape == nullptr || cuShape == nullptr) {
        return FailTiling("missing q/v/cu_seqlens shape");
    }

    // q is head-major [1, HV, T, K]; v is BSND [1, T, HV, V].
    // Read T and HV from v (BSND) to avoid layout ambiguity, then cross-validate with q.
    const auto &qOriginShape = qShape->GetOriginShape();
    const auto &vOriginShape = vShape->GetOriginShape();
    const auto &cuOriginShape = cuShape->GetOriginShape();
    if (qOriginShape.GetDimNum() != 4 || vOriginShape.GetDimNum() != 4 ||
        qOriginShape.GetDim(3) != kHeadDim || vOriginShape.GetDim(3) != kHeadDim) {
        return FailTiling("invalid q/v shape q=" + ShapeToString(qOriginShape) + " v=" +
                          ShapeToString(vOriginShape) + " cu=" + ShapeToString(cuOriginShape));
    }
    // q: [1, HV, T, K]  v: [1, T, HV, V] — cross-check consistency
    const uint32_t numHeads = static_cast<uint32_t>(vOriginShape.GetDim(2));
    const uint32_t totalTokens = static_cast<uint32_t>(vOriginShape.GetDim(1));
    if (qOriginShape.GetDim(1) != numHeads || qOriginShape.GetDim(2) != totalTokens) {
        return FailTiling("q/v head/token mismatch: q=" + ShapeToString(qOriginShape) +
                          " v=" + ShapeToString(vOriginShape));
    }
    if (numHeads == 0) {
        return FailTiling("num_heads is zero");
    }
    const int64_t batchSize = cuOriginShape.GetDim(0) - 1;

    // Attributes
    int64_t numMatricesAttr = 0;
    auto attrs = context->GetAttrs();
    if (attrs != nullptr && attrs->GetAttrPointer<int64_t>(NUM_MATRICES_ATTR) != nullptr) {
        numMatricesAttr = *attrs->GetAttrPointer<int64_t>(NUM_MATRICES_ATTR);
    }
    uint64_t fftsAddr = 0;
    if (attrs != nullptr && attrs->GetAttrPointer<int64_t>(FFTS_ADDR_ATTR) != nullptr) {
        fftsAddr = static_cast<uint64_t>(*attrs->GetAttrPointer<int64_t>(FFTS_ADDR_ATTR));
    }

    uint32_t numMatrices = 0;
    if (numMatricesAttr > 0 && numMatricesAttr <= std::numeric_limits<uint32_t>::max()) {
        numMatrices = static_cast<uint32_t>(numMatricesAttr);
    }
    if (numMatrices == 0) {
        numMatrices = CeilDiv(totalTokens, kChunkSize) * numHeads;
    }

    MegaChunkKdaTilingData tiling;
    tiling.set_block_dim(blockDim);
    tiling.set_num_matrices(numMatrices);
    tiling.set_num_heads(numHeads);
    tiling.set_batch_size(batchSize);
    tiling.set_seq_len(totalTokens);
    tiling.set_total_tokens(totalTokens);
    tiling.set_ffts_addr(fftsAddr);

    context->SetBlockDim(blockDim);
    tiling.SaveToBuffer(context->GetRawTilingData()->GetData(),
                        context->GetRawTilingData()->GetCapacity());
    context->GetRawTilingData()->SetDataSize(tiling.GetDataSize());

    size_t *workspaceSizes = context->GetWorkspaceSizes(1);
    if (workspaceSizes == nullptr) {
        return FailTiling("workspace size pointer is null H=" + std::to_string(numHeads) +
                          " T=" + std::to_string(totalTokens) +
                          " num_matrices=" + std::to_string(numMatrices));
    }
    workspaceSizes[0] = CalcUserWorkspaceBytes(blockDim) + platformInfo.GetLibApiWorkSpaceSize();
    return ge::GRAPH_SUCCESS;
}

struct MegaChunkKdaCompileInfo {};

static ge::graphStatus TilingParseForMegaChunkKda(gert::TilingParseContext *context)
{
    (void)context;
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_OPTILING(MegaChunkKda)
    .Tiling(TilingFunc)
    .TilingParse<MegaChunkKdaCompileInfo>(TilingParseForMegaChunkKda);
}  // namespace optiling
