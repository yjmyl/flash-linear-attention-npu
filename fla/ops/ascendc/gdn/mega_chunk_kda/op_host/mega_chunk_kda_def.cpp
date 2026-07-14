/**
 * Copyright (c) 2026 Tianjin University, Ltd.
 * Adapted from xllm_ops MegaChunkGdn (Apache-2.0) and megagdn-pto KDA kernels.
 * This program is free software under CANN Open Software License Agreement Version 2.0.
 */

/*!
 *\file mega_chunk_kda_def.cpp
 *\brief OpDef for KDA (Kimi Delta Attention) fused mega-kernel.
 *
 * Inputs (9):
 *   q               [1, HV, T, K]  fp16  (head-major, pre-scaled)
 *   k               [1, HV, T, K]  fp16  (head-major)
 *   v               [1, T, HV, V]  fp16  (BSND)
 *   g               [1, T, HV, K]  fp16  (BSND, raw per-dimension gate logits)
 *   beta            [1, HV, T]     fp16  (head-major, post-sigmoid)
 *   mask_strict     [C, C]         fp32  (rows >  cols)
 *   mask_incl       [C, C]         fp32  (rows >= cols)
 *   minus_identity  [C, C]         fp16  (-I)
 *   cu_seqlens      [N_seq+1]      int32
 *
 * Outputs (9):
 *   out             [1, T, HV, V]  fp16
 *   g_sum           [1, T, HV, K]  fp32  (within-chunk prefix sum of g)
 *   g_cs            [1, HV, T, K]  fp32  (head-major, cumulative gate)
 *   L               [1, T, HV, C]  fp16  (gated K*K^T lower-tri)
 *   A_inv           [1, T, HV, C]  fp16  ((I+L)^{-1})
 *   u               [1, T, HV, V]  fp16  (WY auxiliary)
 *   w               [1, T, HV, K]  fp16  (WY auxiliary)
 *   s               [tc, HV, K, V] fp16  (recurrent state snapshots)
 *   v_corr          [1, T, HV, V]  fp16  (gated v)
 *
 * Attrs (2):
 *   num_matrices    Int(0)   optional — auto-computed as ceil(T/C)*HV if 0
 *   ffts_addr       Int(0)   optional — set by l0 op via rtGetC2cCtrlAddr
 */

#include "register/op_def_registry.h"

namespace ops {
class MegaChunkKda : public OpDef {
public:
    explicit MegaChunkKda(const char *name) : OpDef(name)
    {
        // ── Inputs (9) ──────────────────────────────────────────────────
        this->Input("q").ParamType(REQUIRED).DataType({ge::DT_FLOAT16}).Format({ge::FORMAT_ND}).AutoContiguous();
        this->Input("k").ParamType(REQUIRED).DataType({ge::DT_FLOAT16}).Format({ge::FORMAT_ND}).AutoContiguous();
        this->Input("v").ParamType(REQUIRED).DataType({ge::DT_FLOAT16}).Format({ge::FORMAT_ND}).AutoContiguous();
        this->Input("g").ParamType(REQUIRED).DataType({ge::DT_FLOAT16}).Format({ge::FORMAT_ND}).AutoContiguous();
        this->Input("beta").ParamType(REQUIRED).DataType({ge::DT_FLOAT16}).Format({ge::FORMAT_ND}).AutoContiguous();
        this->Input("mask_strict").ParamType(REQUIRED).DataType({ge::DT_FLOAT}).Format({ge::FORMAT_ND});
        this->Input("mask_incl").ParamType(REQUIRED).DataType({ge::DT_FLOAT}).Format({ge::FORMAT_ND});
        this->Input("minus_identity").ParamType(REQUIRED).DataType({ge::DT_FLOAT16}).Format({ge::FORMAT_ND});
        this->Input("cu_seqlens").ParamType(REQUIRED).DataType({ge::DT_INT32}).Format({ge::FORMAT_ND}).AutoContiguous();

        // ── Outputs (9) ─────────────────────────────────────────────────
        this->Output("out").ParamType(REQUIRED).DataType({ge::DT_FLOAT16}).Format({ge::FORMAT_ND});
        this->Output("g_sum").ParamType(REQUIRED).DataType({ge::DT_FLOAT}).Format({ge::FORMAT_ND});
        this->Output("g_cs").ParamType(REQUIRED).DataType({ge::DT_FLOAT}).Format({ge::FORMAT_ND});
        this->Output("L").ParamType(REQUIRED).DataType({ge::DT_FLOAT16}).Format({ge::FORMAT_ND});
        this->Output("A_inv").ParamType(REQUIRED).DataType({ge::DT_FLOAT16}).Format({ge::FORMAT_ND});
        this->Output("u").ParamType(REQUIRED).DataType({ge::DT_FLOAT16}).Format({ge::FORMAT_ND});
        this->Output("w").ParamType(REQUIRED).DataType({ge::DT_FLOAT16}).Format({ge::FORMAT_ND});
        this->Output("s").ParamType(REQUIRED).DataType({ge::DT_FLOAT16}).Format({ge::FORMAT_ND});
        this->Output("v_corr").ParamType(REQUIRED).DataType({ge::DT_FLOAT16}).Format({ge::FORMAT_ND});

        // ── Attrs (2) ───────────────────────────────────────────────────
        this->Attr("num_matrices").AttrType(OPTIONAL).Int(0);
        this->Attr("ffts_addr").AttrType(OPTIONAL).Int(0);

        this->AICore().AddConfig("ascend910b");
        this->AICore().AddConfig("ascend910_93");
    }
};

OP_ADD(MegaChunkKda);
}  // namespace ops
