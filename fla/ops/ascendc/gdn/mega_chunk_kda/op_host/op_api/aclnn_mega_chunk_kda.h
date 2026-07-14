/**
 * Copyright (c) 2026 Tianjin University, Ltd.
 * Adapted from xllm_ops MegaChunkGdn (Apache-2.0) and megagdn-pto KDA kernels.
 * This program is free software under CANN Open Software License Agreement Version 2.0.
 */

#pragma once

#include "aclnn/aclnn_base.h"

#ifdef __cplusplus
extern "C" {
#endif

__attribute__((visibility("default"))) aclnnStatus aclnnMegaChunkKdaGetWorkspaceSize(
    const aclTensor *q, const aclTensor *k, const aclTensor *v, const aclTensor *g, const aclTensor *beta,
    const aclTensor *maskStrict, const aclTensor *maskIncl, const aclTensor *minusIdentity,
    const aclTensor *cuSeqlens, int64_t numMatrices, aclTensor *out, aclTensor *gSum, aclTensor *gCs,
    aclTensor *L, aclTensor *aInv, aclTensor *u, aclTensor *w, aclTensor *s, aclTensor *vCorr,
    uint64_t *workspaceSize, aclOpExecutor **executor);

__attribute__((visibility("default"))) aclnnStatus aclnnMegaChunkKda(void *workspace, uint64_t workspaceSize,
                                                                     aclOpExecutor *executor, aclrtStream stream);

#ifdef __cplusplus
}
#endif
