/**
 * Copyright (c) 2026 Tianjin University, Ltd.
 * Adapted from xllm_ops MegaChunkGdn (Apache-2.0) and megagdn-pto KDA kernels.
 * This program is free software under CANN Open Software License Agreement Version 2.0.
 */

#pragma once

#include <tuple>

#include "opdev/make_op_executor.h"
#include "opdev/op_executor.h"

namespace l0op {
std::tuple<const aclTensor *, const aclTensor *, const aclTensor *, const aclTensor *, const aclTensor *,
           const aclTensor *, const aclTensor *, const aclTensor *, const aclTensor *>
MegaChunkKda(const aclTensor *q, const aclTensor *k, const aclTensor *v, const aclTensor *g,
             const aclTensor *beta, const aclTensor *maskStrict, const aclTensor *maskIncl,
             const aclTensor *minusIdentity, const aclTensor *cuSeqlens, int64_t numMatrices,
             const aclTensor *out, const aclTensor *gSum, const aclTensor *gCs, const aclTensor *L,
             const aclTensor *aInv, const aclTensor *u, const aclTensor *w, const aclTensor *s,
             const aclTensor *vCorr, aclOpExecutor *executor);
}  // namespace l0op
