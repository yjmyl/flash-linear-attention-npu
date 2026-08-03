# Learnings

Corrections, insights, and knowledge gaps captured during development.

**Categories**: correction | insight | knowledge_gap | best_practice

---

## [LRN-20260731-001] correction

**Logged**: 2026-07-31T00:00:00+08:00
**Priority**: medium
**Status**: resolved
**Area**: docs

### Summary
GDN 阶段验收报告应与仓内既有进度和方案文档统一使用中文。

### Details
Phase 5 验收快照误用了英文模板。虽然数据、结论和证据路径正确，但语言不一致会增加汇报和交叉核对成本。

### Suggested Action
新建或刷新 GDN 阶段验收文档时，先检查同目录文档语言和术语风格；技术符号、算子名与公式保留英文，其余叙述和表头使用中文。

### Metadata
- Source: user_feedback
- Related Files: fla/ops/ascendc/gdn/docs/GDN_PHASE5_ACCEPTANCE_A2.md
- Tags: gdn, docs, language-consistency, acceptance

### Resolution
- **Resolved**: 2026-07-31T00:00:00+08:00
- **Notes**: 已将 Phase 5 验收报告完整转换为中文，并保持数据、结论、路径和哈希不变。

---
## [LRN-20260730-001] correction

**Logged**: 2026-07-30T00:00:00+08:00
**Priority**: medium
**Status**: pending
**Area**: docs

### Summary
Reserve a future shape capability separately from its formal acceptance gate.

### Details
The current Phase 4 acceptance scope is V=128. The implementation may retain tiling and scheduling hooks for V=256, but that does not mean V=256 performance and precision must be fully aligned before moving to the next fusion phase. A future full V=256 gate should not be treated as a current Phase 4/P1 blocker unless the release plan explicitly requires V=256 support now.

### Suggested Action
Keep the current P1 checkpoint frozen, record V=256 as reserved/deferred capability, and proceed with the next fusion work on the frozen V=128 contract. Before production release, run a dedicated V=256 compatibility/accuracy/performance gate; do not combine it with the Phase 5 fusion change.

### Metadata
- Source: user_feedback
- Related Files: fla/ops/ascendc/gdn/docs/GDN_CURRENT_STATUS_A2.md
- Tags: gdn, phase4, v256, scope, acceptance

---

## [LRN-20260728-001] correction

**Logged**: 2026-07-28T20:12:00+08:00
**Priority**: high
**Status**: resolved
**Area**: tests

### Summary
When a same-card control completes Phase 3 but Phase 4 hangs, treat the Phase 4-specific path as the primary defect suspect until a discriminating test disproves it.

### Details
The initial diagnosis over-weighted dirty-device and workload-contention evidence. Those factors remain confounders, but a successful Phase 3 control already rules out a generic Python, OPP, and card-startup failure. The Phase 4 kernel, tiling, synchronization, and artifact contract must therefore be investigated first rather than being provisionally cleared.

### Suggested Action
Use the same card, input, installed hashes, launcher, and timeout for adjacent phase controls; capture per-variant exit codes and classify the first divergent device path before attributing the result to shared-server state.

### Resolution
- **Resolved**: 2026-07-28T21:02:00+08:00
- **Notes**: On idle device 2, Phase 3 completed, archived Phase 4 timed out after 180 seconds, and the new pipeline Phase 4 completed after an artifact-only swap and archive restoration.

### Metadata
- Source: user_feedback
- Related Files: fla/ops/ascendc/gdn/docs/GDN_CURRENT_STATUS_A2.md
- Tags: gdn, phase4, deadlock, control-test, diagnosis

---

## [LRN-20260725-002] correction

**Logged**: 2026-07-25T12:00:00+08:00
**Priority**: high
**Status**: promoted
**Area**: backend

### Summary
Each accepted GDN fusion Phase must keep an independently callable ACLNN; later phases must not overwrite the previous composite entry.

### Details
Phase 2 changed the shared `aclnnGdnCoreFwd` implementation from six independent kernels to fused KKT + solve_tri. Although the ABI stayed stable, the Phase 1 one-ACLNN baseline disappeared, preventing a fair same-package Phase 1/2 comparison.

### Suggested Action
Export immutable `aclnnGdnCoreFwdPhaseN` checkpoints in one complete wheel, keep the unversioned symbol only as the current default, and benchmark versioned entries on identical inputs and environment.

### Metadata
- Source: user_feedback
- Related Files: fla/ops/ascendc/gdn/docs/GDN_PHASE_VERSION_ARCHIVE_A2.md
- Tags: gdn, phase-versioning, ablation, aclnn
- Promoted: fla/ops/ascendc/gdn/docs/GDN_FUSION_DEVELOPMENT_PLAYBOOK_A2.md

---

## [LRN-20260725-001] insight

**Logged**: 2026-07-25T09:30:00+08:00
**Priority**: high
**Status**: resolved
**Area**: backend

### Summary
The fused KKT + solve_tri MIX kernel cannot safely alias its typed KKT hand-off buffer with the final A output using only per-core-group ready flags.

### Details
Dense cases happened to pass, but varlen BF16/C64 produced NaNs. Each AIC waits only for the two paired AIV writers in its own core group. If KKT and solve share one GM tensor, one group can begin solve output writes while another group is still writing KKT rows. A separate hand-off buffer avoids this cross-group write/read race.

### Suggested Action
Keep the private `aWorkspace` until the kernel introduces a true global stage barrier or is redesigned as a tile-local producer/consumer pipeline that proves ownership of every aliased region.

### Resolution
- **Resolved**: 2026-07-25T11:00:00+08:00
- **Notes**: Phase 2 acceptance keeps a separate private hand-off workspace, and the rule is now mandatory in `GDN_FUSION_DEVELOPMENT_PLAYBOOK_A2.md`.

### Metadata
- Source: error
- Related Files: fla/ops/ascendc/gdn/chunk_gdn_fwd/chunk_kkt_solve_tri/op_kernel/chunk_kkt_solve_tri.cpp
- Tags: ascendc, mix-kernel, synchronization, varlen

---

## [LRN-20260725-001] best_practice

**Logged**: 2026-07-25T08:02:00+08:00
**Priority**: high
**Status**: resolved
**Area**: backend

### Summary
In an A2 MIX_AIC_1_2 kernel, two `CrossCoreSetFlag<0x2>` calls from paired AIV sub-blocks produce one merged event for the paired AIC.

### Details
The first formal `ChunkKktSolveTri` stage barrier waited twice on the AIC after both AIV sub-blocks set the same flag. Repository-proven MIX kernels wait once because the `0x2` mode performs the two-AIV rendezvous. The second wait deadlocked.

### Suggested Action
For a paired AIV-to-AIC stage barrier, let both AIV sub-blocks call `CrossCoreSetFlag<0x2, PIPE_MTE3>` and let the AIC call `CrossCoreWaitFlag` exactly once.

### Metadata
- Source: error
- Related Files: fla/ops/ascendc/gdn/chunk_gdn_fwd/chunk_kkt_solve_tri/op_kernel/chunk_kkt_solve_tri.cpp
- Tags: a2, mix-kernel, cross-core-sync

---
## [LRN-20260729-001] correction

**Logged**: 2026-07-29T10:15:00+08:00
**Priority**: high
**Status**: resolved
**Area**: docs

### Summary
A project progress report must show each completed phase and the cumulative outcome versus the original baseline, not mainly describe the current optimization phase.

### Details
The first standalone GDN progress report included a phase overview but devoted most detail to Phase 4/P1. It did not give Phase 2 and Phase 3 their own implementation, validation, and performance narratives, nor did it answer how current Phase 4 compares with the original unfused Phase 1 path.

### Suggested Action
For milestone reports, include a consistent per-phase template: change, validation, performance versus previous phase, resource impact, and conclusion. Add a cumulative baseline view, clearly distinguishing direct same-run A/B evidence from normalized estimates composed across separate phase acceptance runs.

### Metadata
- Source: user_feedback
- Related Files: D:/workspace/大融合算子测试/gdntest/GDN_当前进展汇报_20260729.md
- Tags: gdn, progress-report, phase-summary, cumulative-performance

### Resolution
- **Resolved**: 2026-07-29T10:15:00+08:00
- **Notes**: Expanded the standalone report with dedicated Phase 1/2/3/4 sections and a labeled Phase 1-to-Phase 4 cumulative performance view.

---
## [LRN-20260729-002] correction

**Logged**: 2026-07-29T10:30:00+08:00
**Priority**: high
**Status**: resolved
**Area**: docs

### Summary
GDN Phase 1 is the one-ACLNN composite executor, while the original unfused baseline is Phase 0 with six Python-side ACLNN calls.

### Details
The standalone progress report called Phase 1 the original unfused baseline. This erased the Phase 0-to-Phase 1 boundary and incorrectly described the normalized Phase 1-to-Phase 4 gain as the gain versus the earliest unfused implementation. Phase 1 still launches the six existing stage kernels, but consolidates host-side ACLNN construction, validation, workspace management, and execution into one executor.

### Suggested Action
Always show Phase 0 separately in GDN reports. Report Phase 0-to-Phase 1 ACLNN composition evidence independently from Phase 2+ kernel-fusion evidence, and never multiply pilot data into a formal end-to-end claim without labeling the result as an estimate.

### Metadata
- Source: user_feedback
- Related Files: D:/workspace/大融合算子测试/gdntest/GDN_当前进展汇报_20260729.md
- Tags: gdn, phase0, phase1, aclnn-composite, performance-baseline
- See Also: LRN-20260729-001

### Resolution
- **Resolved**: 2026-07-29T10:30:00+08:00
- **Notes**: Corrected the standalone report to separate Phase 0, Phase 1 pilot evidence, Phase 1-to-Phase 4 normalized gains, and the absence of a formal Phase 0-to-Phase 4 direct A/B.

---
## [LRN-20260729-003] correction

**Logged**: 2026-07-29T10:45:00+08:00
**Priority**: high
**Status**: resolved
**Area**: docs

### Summary
Untracked benchmark JSON without build identity or measurement metadata must not be promoted into a GDN progress report headline.

### Details
Two local Phase 1 JSON files contain numerically valid Phase 0 six-call versus unversioned composite measurements, but they are untracked, use `composite_one_aclnn` instead of the immutable Phase 1 symbol, omit package/library hashes and launch-blocking state, cover only two dense BF16 cases, use 20 fixed-order samples, and contain severe Phase 0 long tails. Matching workspace sizes strongly suggest the composite was the Phase 1 implementation, but do not provide immutable provenance.

### Suggested Action
Classify such files as unverified historical observations. Do not quote their speedups in the executive summary, do not use them to derive a Phase 0-to-current total, and state that formal Phase 0-to-Phase 1 performance is unknown until rerun with versioned symbols, hashes, paired AB/BA, and the frozen matrix.

### Metadata
- Source: user_feedback
- Related Files: .phase1_dense_bf16_c64.json, .phase1_dense_bf16_c128.json, D:/workspace/大融合算子测试/gdntest/GDN_当前进展汇报_20260729.md
- Tags: gdn, evidence-quality, benchmark-provenance, phase0, phase1
- See Also: LRN-20260729-001, LRN-20260729-002

### Resolution
- **Resolved**: 2026-07-29T10:45:00+08:00
- **Notes**: Removed the unverified pilot speedups and derived Phase 0-to-Phase 4 estimate from the report's conclusions; retained only a provenance audit note.

---
## [LRN-20260729-004] correction

**Logged**: 2026-07-29T11:05:00+08:00
**Priority**: high
**Status**: resolved
**Area**: testing

### Summary
Cross-phase performance runs must use the frozen identity contract, not a nearby representative optimization shape.

### Details
The first Phase 0-to-P1 direct pilot reused the pipeline representative shape `D_FP16_C64,T=1024`. The historical formal matrix defines FP16 dense cases as the tail-block shape `T=1025`; `T=1024` is the BF16 dense token count. The pilot proved the benchmark path but was not eligible for the formal matrix. A contract audit caught the mismatch before the result was promoted.

### Suggested Action
Keep one explicit eight-identity contract table and validate every matrix result against it before aggregation. For five-variant order balancing, require warmup and iteration counts to cover the complete forward and reverse rotation cycle (`2 * variant_count`), not only one positional rotation.

### Metadata
- Source: user_feedback
- Related Files: torch_custom/fla_npu/test/benchmark_gdn_core_ablation.py, fla/ops/ascendc/gdn/docs/GDN_CURRENT_STATUS_A2.md
- Tags: gdn, benchmark, shape-contract, ab-ba, phase-matrix
- See Also: LRN-20260729-003

### Resolution
- **Resolved**: 2026-07-29T11:05:00+08:00
- **Notes**: Audited all eight Phase 2/3/4 contracts, reran `D_FP16_C64` at `T=1025` for three 200-sample processes, and tightened the benchmark's order-cycle validation.

---
