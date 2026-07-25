# GDN Core Ablation

Every fusion step must compare against the same six-operator AscendC baseline:

`local_cumsum -> KKT -> solve_tri -> recompute_w_u -> fwd_h -> fwd_o`

The benchmark records bit-exact output checks, host-side ACLNN call count, NPU
event latency, peak allocated NPU memory, per-ACLNN workspace sizes, and optional
device kernel count from a profiler trace.

```bash
TEST_DEVICE_ID=2 python torch_custom/fla_npu/test/benchmark_gdn_core_ablation.py \
  --batch 1 --key-heads 4 --value-heads 8 --tokens 1024 \
  --chunk-size 64 --dtype bf16 --warmup 3 --iterations 10 --profile \
  --output test_output/gdn_ablation/dense_bf16_c64.json
```

For variable-length input, add canonical cumulative sequence lengths:

```bash
--cu-seqlens 0,384,1024
```

Do not compare invalid tail-padding columns in `A`; only the lower-triangular
region belonging to real tokens is part of the numerical contract. Keep all
other outputs bit exact. Add future fused implementations as named variants in
the same script rather than creating independent benchmarks.

Permanent one-ACLNN checkpoints are named
`phase1_one_aclnn_six_kernels` and
`phase2_one_aclnn_fused_kkt_solve`. The unversioned `composite_one_aclnn`
tracks the current default and must not be used as the only historical A/B.

Phase 2 also adds the five-call `fused_kkt_solve` variant. It replaces only
`chunk_scaled_dot_kkt -> solve_tri`; local cumsum, layout handling and the four
following GDN stages stay unchanged. The first fused kernel accepts the
Phase 1 physical contract where q/k have already been expanded to `Hv` heads.
