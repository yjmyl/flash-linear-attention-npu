/**
 * Copyright (c) 2026 Tianjin University, Ltd.
 * Adapted from xllm_ops MegaChunkGdn (Apache-2.0) and megagdn-pto KDA kernels.
 * This program is free software under CANN Open Software License Agreement Version 2.0.
 */

#pragma once

#include "register/tilingdata_base.h"

namespace optiling {
BEGIN_TILING_DATA_DEF(MegaChunkKdaTilingData)
TILING_DATA_FIELD_DEF(uint32_t, block_dim);
TILING_DATA_FIELD_DEF(uint32_t, num_matrices);
TILING_DATA_FIELD_DEF(uint32_t, num_heads);
TILING_DATA_FIELD_DEF(int64_t, batch_size);
TILING_DATA_FIELD_DEF(int64_t, seq_len);
TILING_DATA_FIELD_DEF(int64_t, total_tokens);
TILING_DATA_FIELD_DEF(uint64_t, ffts_addr);
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(MegaChunkKda, MegaChunkKdaTilingData)
}  // namespace optiling
