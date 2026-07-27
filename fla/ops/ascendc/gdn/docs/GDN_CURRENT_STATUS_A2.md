# A2 GDN 当前状态

> 最后更新：2026-07-27
>
> 文档定位：**GDN A2 唯一可变进度源**。其他文档分别承载路线、开发规则、验收快照、版本归档或 API contract，不再重复播报当前进度。

## 1. 当前结论

- **Phase 1：已归档。** 保留六个 GDN core 逻辑阶段的版本化对照入口。旧路径已知的非有限值和长序列问题按历史边界记录，不扩成当前融合路径的修复任务。
- **Phase 2：已验收并归档。** 融合 `KKT + solve_tri`，保留独立 `local_cumsum`。
- **Phase 3：已验收并归档。** 在 Phase 2 基础上继续吸收 `local_cumsum`，最终为单个累积融合 `ChunkCumsumKktSolveTri`。
- **Phase 4：尚未启动。** 下一候选边界是吸收 `recompute_w_u`；必须先冻结启动卡，再按单点 smoke -> 精度 -> 性能逐层前进。

当前没有未收口的 Phase 2/3 实现或验收项。新工作不应继续改写 Phase 3，而应从不可变 `gdn-a2-phase3` 里程碑追加 Phase 4 的新 commit 和新版本化入口。

## 2. 融合边界和性能口径

令：

```text
A = ChunkLocalCumsum
B = ChunkScaledDotKkt
C = SolveTri
D = RecomputeW/U
E = FwdH
F = FwdO
```

三个已归档阶段的逻辑边界是：

```text
Phase 1: A + B + C + D + E + F
Phase 2: A + (B + C) + D + E + F
Phase 3: (A + B + C) + D + E + F
```

因此 Phase 3 不是用 `A+B` 替代 Phase 2 已完成的 `B+C`。早期拆分候选 `(A+B)+C` 会丢失 Phase 2 的累积融合收益，已因性能回退和 kernel 数增加被淘汰。

| 证据名称 | 比较对象 | 用途 | 是否是 Phase 3 最终性能结论 |
| --- | --- | --- | --- |
| 共享 helper 局部微基准 | `A+B` 两小算子 vs `ChunkCumsumKkt` | 验证 cumsum+KKT 共享实现的正确性和局部收益 | 否，是辅助证据 |
| 目标段 profiler | Phase 2 的 `A + (B+C)` vs Phase 3 的 `(A+B+C)` | 证明目标段由 2 个 kernel 降为 1 个 | 否，是机理证据 |
| **GDN core 生产性能** | `aclnnGdnCoreFwdPhase2` vs `aclnnGdnCoreFwdPhase3` | 比较由 A-F 六个逻辑阶段组成的完整 forward core 链路 | **是** |
| 完整 Demo/模型性能 | causal conv、RMSNorm/gate 等也纳入 | 更外层端到端吞吐 | 本 Phase 未验收 |

文档中的“GDN core 生产性能”可以理解为原六个 GDN core 小算子的完整 forward core 性能，但不等于包含 causal conv、RMSNorm/gate 或整个模型的“整网性能”。

## 3. 已冻结验收摘要

| Phase | 最终路径 | 功能/精度 | 最终生产性能 | 归档状态 |
| --- | --- | --- | --- | --- |
| Phase 1 | `A+B+C+D+E+F` | 历史冻结范围已关闭；旧路径已知问题单独记录 | 作为版本化历史对照 | 已归档 |
| Phase 2 | `A+(B+C)+D+E+F` | 独立融合、dense/varlen 和 core 验收完成 | 完整 core 相对 Phase 1 的可比较 case 主判据无可复现回退 | `gdn-a2-phase2` |
| Phase 3 | `(A+B+C)+D+E+F` | 局部 `80/80 exact`，core dense/varlen `8/8` + state `1/1` bit-exact/有限 | 完整 core 相对 Phase 2 主判据 `8/8` median 改善，NPU kernel 数 `9 -> 8` | `gdn-a2-phase3` |

Phase 3 的共享 helper 局部微基准也是 `8/8` median 改善，但只用于证明 `A+B` helper 有效，不代替上表中的完整 core 生产性能结论。

## 4. 当前范围边界

已验收的 Phase 3 仍限于：

- A2（Ascend 910B3）、`K==V==128`、head-first ND；
- dense/varlen，FP16/BF16，`chunk_size=64/128`；
- 现有 transpose/contiguous 和 canonical varlen metadata；
- GVA 依旧由外部扩头，不是 kernel 内原生 `Hk != Hv`。

未由 Phase 3 宣称完成：`V=256`、原生 GVA、causal conv、RMSNorm/gate、backward、完整 Demo/模型性能，以及单 ACLNN 绝对 workspace `<=50 MB`。

## 5. 下一小步

Phase 4 目前只是路线候选，尚未启动。启动时只做一张冻结卡：

1. 从 `gdn-a2-phase3` 锁定不可变 Phase 3 基线。
2. 确认 `ChunkCumsumKktSolveTri + recompute_w_u` 的精确输入、输出、dtype/layout 和 workspace 生命周期。
3. 定义一个最小 representative case 和停止条件；在启动卡确认前不改 kernel。

后续仍执行“一小步、一层证据、失败就停”，不直接铺开大矩阵。

## 6. 文档职责和更新规则

| 文档 | 职责 | 何时更新 |
| --- | --- | --- |
| `GDN_CURRENT_STATUS_A2.md` | **唯一当前进度源**：当前 Phase、本次结论、下一小步、当前 blocker | 每个有效小步后；Phase 关闭时压缩成快照，不累积实验日志 |
| `GDN_FUSION_PLAN_A2.md` | 稳定路线、Phase 边界、冻结启动/修正卡 | 只有路线或边界变更时 |
| `GDN_FUSION_DEVELOPMENT_PLAYBOOK_A2.md` | 开发和验收方法 | 只有流程规则变更时 |
| `GDN_PHASE2_ACCEPTANCE_A2.md` / `GDN_PHASE3_ACCEPTANCE_A2.md` | 已关闭 Phase 的冻结验收快照 | 原则上不随日常进度更新；只修正事实或表述错误 |
| `GDN_PHASE_VERSION_ARCHIVE_A2.md` | commit/tag/产物身份和归档规则 | 只在新里程碑或归档身份变化时 |
| `docs/aclnn*.md` / `gdn_core_ablation.md` | API contract 和 benchmark 使用方法 | 只在接口、路径或测量方法变更时 |

日常小步默认只更新本文档。Phase 关闭时才有一次有意识的多文档收口：写验收快照、登记归档身份、把本页压缩到新的当前状态。过程失败记录由 Git 历史和结构化证据保留，不再堆叠到“当前状态”中。

## 7. 正式证据索引

- Phase 2 验收：`GDN_PHASE2_ACCEPTANCE_A2.md`
- Phase 3 验收：`GDN_PHASE3_ACCEPTANCE_A2.md`
- 版本与 tag：`GDN_PHASE_VERSION_ARCHIVE_A2.md`
- 总体路线：`GDN_FUSION_PLAN_A2.md`
- 小步开发方法：`GDN_FUSION_DEVELOPMENT_PLAYBOOK_A2.md`
- 统一 core API：`../chunk_gdn_fwd/chunk_gated_delta_rule_fwd_h/docs/aclnnGdnCoreFwd.md`
- 性能脚本口径：`../chunk_gdn_fwd/chunk_gated_delta_rule_fwd_h/docs/gdn_core_ablation.md`
