# A2 GDN 当前状态

> 最后更新：2026-07-28
>
> 文档定位：**GDN A2 唯一可变进度源**。其他文档分别承载路线、开发规则、验收快照、版本归档或 API contract，不再重复播报当前进度。

## 1. 当前结论

- **Phase 1：已归档。** 保留六个 GDN core 逻辑阶段的版本化对照入口。旧路径已知的非有限值和长序列问题按历史边界记录，不扩成当前融合路径的修复任务。
- **Phase 2：已验收并归档。** 融合 `KKT + solve_tri`，保留独立 `local_cumsum`。
- **Phase 3：已验收并归档。** 在 Phase 2 基础上继续吸收 `local_cumsum`，最终为单个累积融合 `ChunkCumsumKktSolveTri`。
- **Phase 4：冻结矩阵已通过，待验收归档。** 边界为 `fwd_h + fwd_o`，目标 core 路径是
  `(A+B+C) + D + (E+F)`。单个 MIX kernel、版本化 Phase 4 core 入口、A2 整包构建、
  dense `4/4` + varlen `4/4` + state `1/1` 精度均已通过；dense/varlen 共 `8/8` 性能点
  当前均无可复现的实质回退，workspace 全部下降。下一步只做验收快照、版本身份和文档收口，
  不再扩测试变量或修改 kernel。

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

未由 Phase 3 宣称完成：`V=256`、原生 GVA、causal conv、RMSNorm/gate、backward、完整 Demo/模型性能，以及单 ACLNN 绝对 workspace `<=50 MB`。其中绝对 workspace 作为持续优化目标和报告项，不作为阻断融合路线的硬门槛；性能优先，若能在不伤害性能的前提下降低 workspace 则继续优化。

已达成但尚未实施的后续顺序为：

```text
Phase 4: (A+B+C) + D + (E+F)
    -> 规格闸门: V=256，再原生 GVA
    -> Phase 5: (A+B+C) + (D+E+F)
    -> Phase 6: 仅在 profiler 支持时尝试 (A+B+C+D+E+F)
    -> Phase 7: transpose/layout
    -> Phase 8: causal_conv1d、RMSNorm/gate
    -> Phase 9: 完整 Demo/模型收口
```

生产目标是最快且满足显存/精度的拓扑，不要求为了形式把 `ABC + DEF` 强制合成单 kernel。

后续默认只推进一条生产路线，不按 dtype、chunk、layout 或 shape 预设运行时分支。每个 Phase 的版本化 ACLNN 仅用于不可变 A/B 和归档，不代表生产入口需要同时维护多条规格路由。若冻结用例出现超出测量噪声的性能回退，或融合后没有获得预期收益，先在同一路线上最多做三轮有明确假设的单变量优化；三轮后仍无解，再携带基线差距、profiler 瓶颈、已尝试方案和预计工作量反馈决策，不自动新增分支。

## 5. Phase 4 验收收口与下一小步

Phase 4 启动卡已冻结：

1. 不可变 core 基线为 tag `gdn-a2-phase3` 对应的 `aclnnGdnCoreFwdPhase3`；局部逻辑基线为同包
   `ChunkGatedDeltaRuleFwdH -> ChunkFwdO`。
2. 首版 fused kernel 在一次 MIX launch 内顺序执行完整 `FwdH`、全核同步、完整 `FwdO`；`h/v_new`
   只存在于该算子的 user workspace，不再作为 Phase 4 core executor 中间 tensor。首版仍允许
   `FwdH` 写 GM、`FwdO` 从 GM 读取，不宣称片上直通。
3. 首个 smoke 固定为 dense FP16、`B=1,H=4,T=128,K=V=128,C=64`、无 initial/final state；
   首个生产性能 pilot 固定为 dense FP16、`B=1,H=8,T=1025,K=V=128,C=64`。首轮不同时修改
   transpose、原生 GVA、`V=256` 或 workspace 别名。
4. 实现保留 `V=256` 和 `Hv % Hk == 0` 的 tiling/调度表达能力，但 Phase 4 首轮只验收
   `V=128`、core 入口外部扩头后的 `Hk==Hv`。
5. 输出/state 必须 bit-exact 或达到已批准 L1 且全部有限，profiler 中目标段 kernel 数必须
   `2 -> 1`。功能、精度或有限性失败立即停；生产性能超出噪声回退或未取得预期收益时，最多做三轮
   有明确假设的单变量优化，仍无解再反馈决策，不直接扩成规格分支，也不进入完整矩阵。

首轮 pilot 结果：

- A2 干净整包构建成功；`aclnnGdnCoreFwdPhase1` 至 `Phase4` 共 8 个
  workspace/launch 符号齐全，`ChunkGatedDeltaRuleFwdHO` 的 8 组 kernel `.o/.json` 已进产物。
- dense FP16 `B=1,H=4,T=128,C=64` 无 state smoke 和打开 initial/final state 的单变量
  smoke 都与 Phase 3 的 `o/final_state/g_cumsum/A` bit-exact，`max_abs=0`，且全部有限。
- 冻结性能 shape 对应的 dense FP16 `B=1,physical H=8,T=1025,C=64` 也完成
  Phase 3/4 全输出 bit-exact，外部扩头后 core 内 `Hk==Hv==8`。
- 继续保持只测精度，dense 剩余 `BF16+C64`、`FP16+C128`、`BF16+C128` 三个组合
  也全部 bit-exact、`max_abs=0`、Phase 3/4 全有限。因此当前 dense 首轮为 `4/4`，另有 state `1/1`。
- varlen `FP16/BF16 x C64/C128` 四个组合也全部通过；`o`、valid-A、`g_cumsum`
  均与 Phase 3 bit-exact，`output_max_abs=0`，Phase 3/4 全有限。当前 varlen 精度为 `4/4`。
- 首个性能点采用 2 轮交替顺序、每轮每变体 50 个 NPU Event 样本。Phase 3/4 合并
  100 个样本后 median 为 `0.991740/0.983540 ms`（Phase 4 `-0.83%`），P90 为
  `1.025320/1.009740 ms`（`-1.52%`），mean 为 `0.993697/0.983854 ms`（`-0.99%`）。
- profiler 中 core 设备任务数 `8 -> 7`，`FwdH + FwdO` 由 2 个 kernel 变为 1 个
  `ChunkGatedDeltaRuleFwdHO`。单次 trace 中原两 kernel 耗时和约 `142.18 us`，融合 kernel 约
  `146.38 us`，所以当前收益主要来自减少 launch/中间张量管理，尚不宣称 kernel body 本身变快。
- 单 ACLNN workspace `137055744 -> 104017920 B`（`-24.11%`），peak allocated delta
  `140239360 -> 108040704 B`（`-22.96%`）。这是首版自然结果，不计为性能优化轮次。
- varlen FP16 `T=259,C=64,cu_seqlens=[0,1,66,259]` 代表点同样使用 2 轮交替顺序和
  每变体 100 个合并样本。Phase 3/4 median 为 `0.950280/0.905000 ms`（`-4.76%`），
  P90 为 `1.003120/0.931080 ms`（`-7.18%`），mean 为 `0.956605/0.905655 ms`（`-5.33%`）。
  设备任务数仍为 `8 -> 7`；workspace `130229760 -> 94570496 B`（`-27.38%`），
  peak allocated delta `131035648 -> 95376384 B`（`-27.21%`）。
- dense 剩余三个性能点随后改用正式的同进程 Phase 3/4 交替测量；每个 case 运行 3 个独立进程，
  每进程每变体 200 个样本。合并 600 个样本后，`BF16+C64` median 为
  `0.974270/0.967570 ms`（Phase 4 `-0.69%`）；`FP16+C128` 为
  `0.983820/0.988150 ms`（`+0.44%`）；`BF16+C128` 为
  `0.974930/0.981580 ms`（`+0.68%`）。两个 C128 点的逐轮配对 median 变化分别只有
  `+0.46%/+0.38%`，显著小于先前跨进程独立测量显示的 `2%~3%` 漂移。
- C128 profiler 中 Phase 3 的 `FwdH + FwdO` 为 `115.52 us`，Phase 4 的 fused
  `FwdHO` 为 `112.24 us`，目标 kernel 本体没有回退；core 设备任务仍为 `8 -> 7`。
  结合 AB/BA、配对结果和 profiler，当前将 C128 的小于 `0.7%` 差异判为测量噪声范围，
  不启动无 profiler 假设的 kernel 优化轮次。
- 上述三个点的 workspace 分别为 `136917504 -> 103617536 B`（`-24.32%`）、
  `152240640 -> 113175040 B`（`-25.66%`）、`151708160 -> 112380416 B`（`-25.92%`）。
- varlen 剩余三点同样采用同进程 Phase 3/4 交替测量，每变体 200 个样本：`BF16+C64`
  median 为 `1.042500/1.029260 ms`（Phase 4 `-1.27%`），`FP16+C128` 为
  `0.962640/0.946080 ms`（`-1.72%`），`BF16+C128` 为
  `0.912510/0.906950 ms`（`-0.61%`）。逐轮配对 median 变化分别为
  `-2.45%/-1.75%/-0.91%`，三点均改善；C128 P90 也分别改善，C64 P90 基本持平。
  workspace 为 `130229760 -> 94570496 B`（`-27.38%`）和
  `143843840 -> 103729664 B`（两个 C128 点均为 `-27.89%`）。

冻结功能/精度、dense/varlen `8/8` 性能、workspace/peak 和 profiler 证据现已齐全。
当前唯一下一小步：写 Phase 4 验收快照，登记最终 run 包/host 库与 Git 里程碑身份，刷新版本归档和
当前状态文档。该步只收口现有证据，不修改 kernel，也不提前进入 `V=256` 规格闸门。

后续仍执行“一小步、一层证据”，不直接铺开大矩阵。单纯复测、扩大迭代数或确认噪声不计入三轮优化。

## 6. 文档职责和更新规则

| 文档 | 职责 | 何时更新 |
| --- | --- | --- |
| `GDN_CURRENT_STATUS_A2.md` | **唯一当前进度源**：当前 Phase、本次结论、下一小步、当前 blocker | 每个有效小步后；Phase 关闭时压缩成快照，不累积实验日志 |
| `GDN_FUSION_PLAN_A2.md` | 稳定路线、Phase 边界、冻结启动/修正卡 | 只有路线或边界变更时 |
| `GDN_FUSION_DEVELOPMENT_PLAYBOOK_A2.md` | 开发和验收方法 | 只有流程规则变更时 |
| `GDN_PHASE2/3/4_ACCEPTANCE_A2.md` | 已关闭 Phase 的冻结验收快照 | 原则上不随日常进度更新；只修正事实或表述错误 |
| `GDN_PHASE_VERSION_ARCHIVE_A2.md` | commit/tag/产物身份和归档规则 | 只在新里程碑或归档身份变化时 |
| `docs/aclnn*.md` / `gdn_core_ablation.md` | API contract 和 benchmark 使用方法 | 只在接口、路径或测量方法变更时 |

日常小步默认只更新本文档。Phase 关闭时才有一次有意识的多文档收口：写验收快照、登记归档身份、把本页压缩到新的当前状态。过程失败记录由 Git 历史和结构化证据保留，不再堆叠到“当前状态”中。

## 7. 正式证据索引

- Phase 2 验收：`GDN_PHASE2_ACCEPTANCE_A2.md`
- Phase 3 验收：`GDN_PHASE3_ACCEPTANCE_A2.md`
- Phase 4 验收：`GDN_PHASE4_ACCEPTANCE_A2.md`
- 版本与 tag：`GDN_PHASE_VERSION_ARCHIVE_A2.md`
- 总体路线：`GDN_FUSION_PLAN_A2.md`
- 小步开发方法：`GDN_FUSION_DEVELOPMENT_PLAYBOOK_A2.md`
- 统一 core API：`../chunk_gdn_fwd/chunk_gated_delta_rule_fwd_h/docs/aclnnGdnCoreFwd.md`
- 性能脚本口径：`../chunk_gdn_fwd/chunk_gated_delta_rule_fwd_h/docs/gdn_core_ablation.md`
