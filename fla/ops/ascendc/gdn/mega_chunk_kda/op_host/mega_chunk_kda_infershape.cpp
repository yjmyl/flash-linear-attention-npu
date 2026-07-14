/**
 * Copyright (c) 2026 Tianjin University, Ltd.
 * Adapted from xllm_ops MegaChunkGdn (Apache-2.0) and megagdn-pto KDA kernels.
 * This program is free software under CANN Open Software License Agreement Version 2.0.
 */

#include <initializer_list>

#include "register/op_def_registry.h"

namespace {
constexpr uint32_t kHeadDim = 128;
constexpr uint32_t kChunkSize = 128;

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

enum OutputIndex {
    OUT_INDEX = 0,
    G_SUM_INDEX,
    G_CS_INDEX,
    L_INDEX,
    A_INV_INDEX,
    U_INDEX,
    W_INDEX,
    S_INDEX,
    V_CORR_INDEX,
};

enum AttrIndex {
    NUM_MATRICES_ATTR = 0,
};

void SetShape(gert::Shape *shape, std::initializer_list<int64_t> dims)
{
    shape->SetDimNum(dims.size());
    size_t index = 0;
    for (auto dim : dims) {
        shape->SetDim(index++, dim);
    }
}
}  // namespace

namespace ge {
static ge::graphStatus InferShape(gert::InferShapeContext *context)
{
    const gert::Shape *qShape = context->GetInputShape(Q_INDEX);
    const gert::Shape *vShape = context->GetInputShape(V_INDEX);
    const gert::Shape *gShape = context->GetInputShape(G_INDEX);
    const gert::Shape *cuShape = context->GetInputShape(CU_SEQLENS_INDEX);
    if (qShape == nullptr || vShape == nullptr || gShape == nullptr || cuShape == nullptr) {
        return GRAPH_FAILED;
    }

    // q is head-major [1, HV, T, K]
    const int64_t totalTokens = qShape->GetDim(2);
    const int64_t numHeads = qShape->GetDim(1);

    int64_t numMatrices = 0;
    if (context->GetAttrs() != nullptr &&
        context->GetAttrs()->GetAttrPointer<int64_t>(NUM_MATRICES_ATTR) != nullptr) {
        numMatrices = *context->GetAttrs()->GetAttrPointer<int64_t>(NUM_MATRICES_ATTR);
    }
    const int64_t inferredMatrices =
        numMatrices > 0 ? numMatrices : ((totalTokens + kChunkSize - 1) / kChunkSize) * numHeads;
    const int64_t totalChunks =
        numMatrices > 0 ? (numMatrices / numHeads) : ((totalTokens + kChunkSize - 1) / kChunkSize);

    // out: same as v [1, T, HV, V]
    *context->GetOutputShape(OUT_INDEX) = *vShape;
    // g_sum: same as g [1, T, HV, K] (fp32)
    *context->GetOutputShape(G_SUM_INDEX) = *gShape;
    // g_cs: head-major [1, HV, T, K] (fp32)
    SetShape(context->GetOutputShape(G_CS_INDEX), {1, numHeads, totalTokens, kHeadDim});
    // L: [1, T, HV, C]
    SetShape(context->GetOutputShape(L_INDEX), {1, totalTokens, numHeads, kChunkSize});
    // A_inv: [1, T, HV, C]
    SetShape(context->GetOutputShape(A_INV_INDEX), {1, totalTokens, numHeads, kChunkSize});
    // u: same as v [1, T, HV, V]
    *context->GetOutputShape(U_INDEX) = *vShape;
    // w: same as g [1, T, HV, K] (fp16)
    *context->GetOutputShape(W_INDEX) = *gShape;
    // s: [tc, HV, K, V]
    SetShape(context->GetOutputShape(S_INDEX), {totalChunks, numHeads, kHeadDim, kHeadDim});
    // v_corr: same as v [1, T, HV, V]
    *context->GetOutputShape(V_CORR_INDEX) = *vShape;
    return GRAPH_SUCCESS;
}

static ge::graphStatus InferDataType(gert::InferDataTypeContext *context)
{
    const ge::DataType qDtype = context->GetInputDataType(Q_INDEX);

    context->SetOutputDataType(OUT_INDEX, qDtype);
    context->SetOutputDataType(G_SUM_INDEX, ge::DT_FLOAT);
    context->SetOutputDataType(G_CS_INDEX, ge::DT_FLOAT);
    context->SetOutputDataType(L_INDEX, qDtype);
    context->SetOutputDataType(A_INV_INDEX, qDtype);
    context->SetOutputDataType(U_INDEX, qDtype);
    context->SetOutputDataType(W_INDEX, qDtype);
    context->SetOutputDataType(S_INDEX, qDtype);
    context->SetOutputDataType(V_CORR_INDEX, qDtype);
    return GRAPH_SUCCESS;
}

IMPL_OP_INFERSHAPE(MegaChunkKda)
    .InferShape(InferShape)
    .InferDataType(InferDataType);
}  // namespace ge
