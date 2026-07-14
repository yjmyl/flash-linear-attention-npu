/**
 * Copyright (c) 2026 Tianjin University, Ltd.
 * Adapted from xllm_ops MegaChunkGdn (Apache-2.0) and megagdn-pto KDA kernels.
 * This program is free software under CANN Open Software License Agreement Version 2.0.
 */

#include "mega_chunk_kda_l0.h"

#include "aclnn_kernels/common/op_error_check.h"
#include "opdev/op_def.h"
#include "opdev/op_dfx.h"
#include "opdev/op_log.h"
#include "runtime/rt_ffts.h"

using namespace op;

namespace l0op {
OP_TYPE_REGISTER(MegaChunkKda);

using MegaChunkKdaOutputs =
    std::tuple<const aclTensor *, const aclTensor *, const aclTensor *, const aclTensor *, const aclTensor *,
               const aclTensor *, const aclTensor *, const aclTensor *, const aclTensor *>;

MegaChunkKdaOutputs MakeNullOutputs()
{
    return {nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr};
}

MegaChunkKdaOutputs
MegaChunkKda(const aclTensor *q, const aclTensor *k, const aclTensor *v, const aclTensor *g,
             const aclTensor *beta, const aclTensor *maskStrict, const aclTensor *maskIncl,
             const aclTensor *minusIdentity, const aclTensor *cuSeqlens, int64_t numMatrices,
             const aclTensor *out, const aclTensor *gSum, const aclTensor *gCs, const aclTensor *L,
             const aclTensor *aInv, const aclTensor *u, const aclTensor *w, const aclTensor *s,
             const aclTensor *vCorr, aclOpExecutor *executor)
{
    L0_DFX(MegaChunkKda, q, k, v, g, beta, maskStrict, maskIncl, minusIdentity, cuSeqlens, numMatrices);

    auto outRet = executor->AllocTensor(out->GetViewShape(), out->GetDataType(), Format::FORMAT_ND);
    auto gSumRet = executor->AllocTensor(gSum->GetViewShape(), gSum->GetDataType(), Format::FORMAT_ND);
    auto gCsRet = executor->AllocTensor(gCs->GetViewShape(), gCs->GetDataType(), Format::FORMAT_ND);
    auto LRet = executor->AllocTensor(L->GetViewShape(), L->GetDataType(), Format::FORMAT_ND);
    auto aInvRet = executor->AllocTensor(aInv->GetViewShape(), aInv->GetDataType(), Format::FORMAT_ND);
    auto uRet = executor->AllocTensor(u->GetViewShape(), u->GetDataType(), Format::FORMAT_ND);
    auto wRet = executor->AllocTensor(w->GetViewShape(), w->GetDataType(), Format::FORMAT_ND);
    auto sRet = executor->AllocTensor(s->GetViewShape(), s->GetDataType(), Format::FORMAT_ND);
    auto vCorrRet = executor->AllocTensor(vCorr->GetViewShape(), vCorr->GetDataType(), Format::FORMAT_ND);

    OP_CHECK(outRet != nullptr && gSumRet != nullptr && gCsRet != nullptr && LRet != nullptr &&
                 aInvRet != nullptr && uRet != nullptr && wRet != nullptr && sRet != nullptr &&
                 vCorrRet != nullptr,
             OP_LOGE(ACLNN_ERR_INNER_NULLPTR, "MegaChunkKda AllocTensor failed."),
             return MakeNullOutputs());

    uint32_t fftsLen = 0;
    uint64_t fftsAddr = 0;
    (void)rtGetC2cCtrlAddr(&fftsAddr, &fftsLen);

    auto ret = ADD_TO_LAUNCHER_LIST_AICORE(
        MegaChunkKda, OP_INPUT(q, k, v, g, beta, maskStrict, maskIncl, minusIdentity, cuSeqlens),
        OP_OUTPUT(outRet, gSumRet, gCsRet, LRet, aInvRet, uRet, wRet, sRet, vCorrRet),
        OP_ATTR(numMatrices, static_cast<int64_t>(fftsAddr)));
    OP_CHECK_ADD_TO_LAUNCHER_LIST_AICORE(ret != ACLNN_SUCCESS,
                                         return MakeNullOutputs(),
                                         "MegaChunkKda ADD_TO_LAUNCHER_LIST_AICORE failed.");

    return {outRet, gSumRet, gCsRet, LRet, aInvRet, uRet, wRet, sRet, vCorrRet};
}
}  // namespace l0op
