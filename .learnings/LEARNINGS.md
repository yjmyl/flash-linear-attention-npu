# Learnings

Corrections, insights, and knowledge gaps captured during development.

**Categories**: correction | insight | knowledge_gap | best_practice

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
