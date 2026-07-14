/**
 * Copyright (c) 2026 Tianjin University, Ltd.
 * Adapted from xllm_ops MegaChunkGdn (Apache-2.0) and megagdn-pto KDA kernels.
 * This program is free software under CANN Open Software License Agreement Version 2.0.
 */

#include "aclnn_mega_chunk_kda.h"

#include "aclnn_kernels/common/op_error_check.h"
#include "aclnn_kernels/contiguous.h"
#include "mega_chunk_kda_l0.h"
#include "opdev/make_op_executor.h"
#include "opdev/op_dfx.h"
#include "opdev/op_executor.h"
#include "opdev/op_log.h"

using namespace op;

#ifdef __cplusplus
extern "C" {
#endif

aclnnStatus aclnnMegaChunkKdaGetWorkspaceSize(
    const aclTensor *q, const aclTensor *k, const aclTensor *v, const aclTensor *g, const aclTensor *beta,
    const aclTensor *maskStrict, const aclTensor *maskIncl, const aclTensor *minusIdentity,
    const aclTensor *cuSeqlens, int64_t numMatrices, aclTensor *out, aclTensor *gSum, aclTensor *gCs,
    aclTensor *L, aclTensor *aInv, aclTensor *u, aclTensor *w, aclTensor *s, aclTensor *vCorr,
    uint64_t *workspaceSize, aclOpExecutor **executor)
{
    L2_DFX_PHASE_1(aclnnMegaChunkKda,
                   DFX_IN(q, k, v, g, beta, maskStrict, maskIncl, minusIdentity, cuSeqlens, numMatrices),
                   DFX_OUT(out, gSum, gCs, L, aInv, u, w, s, vCorr));

    auto uniqueExecutor = CREATE_EXECUTOR();
    CHECK_RET(uniqueExecutor.get() != nullptr, ACLNN_ERR_INNER_CREATE_EXECUTOR);

    auto qContiguous = l0op::Contiguous(q, uniqueExecutor.get());
    auto kContiguous = l0op::Contiguous(k, uniqueExecutor.get());
    auto vContiguous = l0op::Contiguous(v, uniqueExecutor.get());
    auto gContiguous = l0op::Contiguous(g, uniqueExecutor.get());
    auto betaContiguous = l0op::Contiguous(beta, uniqueExecutor.get());
    auto cuSeqlensContiguous = l0op::Contiguous(cuSeqlens, uniqueExecutor.get());

    auto outputs = l0op::MegaChunkKda(qContiguous, kContiguous, vContiguous, gContiguous, betaContiguous,
                                      maskStrict, maskIncl, minusIdentity, cuSeqlensContiguous, numMatrices,
                                      out, gSum, gCs, L, aInv, u, w, s, vCorr, uniqueExecutor.get());
    if (std::get<0>(outputs) == nullptr) {
        return ACLNN_ERR_INNER_NULLPTR;
    }

    const aclTensor *retTensors[] = {std::get<0>(outputs), std::get<1>(outputs), std::get<2>(outputs),
                                     std::get<3>(outputs), std::get<4>(outputs), std::get<5>(outputs),
                                     std::get<6>(outputs), std::get<7>(outputs), std::get<8>(outputs)};
    aclTensor *outTensors[] = {out, gSum, gCs, L, aInv, u, w, s, vCorr};
    for (size_t i = 0; i < 9; ++i) {
        auto viewCopy = l0op::ViewCopy(retTensors[i], outTensors[i], uniqueExecutor.get());
        if (viewCopy == nullptr) {
            return ACLNN_ERR_INNER_NULLPTR;
        }
    }

    *workspaceSize = uniqueExecutor->GetWorkspaceSize();
    uniqueExecutor.ReleaseTo(executor);
    return ACLNN_SUCCESS;
}

aclnnStatus aclnnMegaChunkKda(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, aclrtStream stream)
{
    L2_DFX_PHASE_2(aclnnMegaChunkKda);
    return CommonOpExecutorRun(workspace, workspaceSize, executor, stream);
}

#ifdef __cplusplus
}
#endif
