# A2 GDN 全融合计划归档

本文档是 A2 平台 GDN 全融合工作的主计划。后续实现、消融和性能结论都以本文档为准；如果方案发生变化，先更新本文档，再修改代码。

## 1. 最终目标

在 A2 平台将 Demo 使用的完整 GDN 路径逐步融合为大融合算子，最终覆盖：

```text
输入投影/布局处理
    -> causal_conv1d
    -> local_cumsum
    -> KKT
    -> solve_tri
    -> recompute_w_u
    -> fwd_h
    -> fwd_o
    -> RMSNorm + gated SiLU
```

最终要求来自 `gdn.md`：

- 硬件：A2
- 数据格式：ND
- 数据类型：FP16/BF16
- `bs`、头数泛化
- GVA 支持，即 `Hk <= Hv`，通常要求 `Hv % Hk == 0`
- `k_dim = 128`
- `v_dim = 128/256`
- `chunk_size = 64/128`
- 额外显存不超过 50 MB
- 融合版本精度达到 L1，并与六个 AscendC 小算子拼接版本比较
- 全场景性能不劣于小算子版本

当前第一阶段先锁定 `K == V == 128`，但接口和内部数据结构不能阻碍后续支持 `K=128, V=256`。

## 2. 两个层次必须区分

### 2.1 统一 ACLNN 入口

`aclnnGdnCoreFwd` 的第一阶段定位是稳定的复合入口：

```text
一个 ACLNN executor
    -> local_cumsum
    -> KKT
    -> solve_tri
    -> recompute_w_u
    -> fwd_h
    -> fwd_o
```

它解决的是 Python 侧统一调用、输入输出契约、元数据校验、测试基线和后续替换边界。内部仍可以调用六个已有 AscendC kernel，因此它不等于最终的单 kernel 大融合。

### 2.2 真正的 kernel 融合

真正融合是减少阶段之间的 GM 中间结果、算子启动和同步边界。融合实现应当在不改变 `aclnnGdnCoreFwd` 高层 ABI 的前提下，逐阶段替换已有小算子。

**首个真正融合目标固定为：**

```text
chunk_scaled_dot_kkt + solve_tri
```

不能把“六算子串成一个 ACLNN”误称为“六算子已经融合”，也不能把当前探索性的 `local_cumsum + KKT` 实现改称为首个正式融合目标。

## 3. 正式实施阶段

### Phase 0：六算子基线

固定以下顺序和语义，作为所有后续方案的唯一基线：

```text
local_cumsum -> KKT -> solve_tri -> recompute_w_u -> fwd_h -> fwd_o
```

基线必须覆盖：

- dense 定长
- 等长和不等长 varlen
- FP16/BF16
- `chunk_size=64/128`
- 尾块
- 普通多头和 GVA
- 输出、必要的中间结果和完整 GDN 精度

每轮记录：

- 输出精度和中间结果精度
- ACLNN 调用次数
- NPU 端 kernel 数量
- 端到端 latency
- 每个阶段 workspace 和峰值显存
- 尾块、varlen、GVA 的通过情况

### Phase 1：统一复合 ACLNN

保留六个已有 kernel 的计算实现，在 `aclnnGdnCoreFwd` 内完成统一调度。此阶段的验收重点是：

- Python GDN 可以切换到统一入口
- 与六算子 Python 拼接路径结果一致
- dense/varlen、FP16/BF16、尾块均可运行
- transpose 和 layout 转换位置固定、可追踪
- 复合入口成为后续所有融合实现的共同调用入口

### Phase 2：首个真正融合，KKT + solve_tri

新增独立的 fused kernel 或等价内部实现，计算边界为：

```text
输入 k、g_cumsum、beta
    -> 生成 KKT/A
    -> 直接完成 solve_tri
    -> 输出求解后的 A
```

第一轮不同时改外部布局，不同时融合 transpose，不同时引入 `K != V`。目的是隔离变量，清楚回答：

1. KKT 和 solve_tri 的中间 GM 往返能否消除。
2. 融合后精度是否满足 L1。
3. A2 上性能是否优于两个独立 kernel。
4. workspace 和峰值显存是否下降或至少不恶化。

### Phase 3：吸收 local_cumsum

在 Phase 2 稳定后，评估将 `local_cumsum` 与 KKT 前处理放入同一 kernel。此阶段重点检查：

- gate 累加的 FP32 精度边界
- varlen 序列边界
- 尾块和无效 padding 区域
- KKT 输入的数值顺序是否改变

当前工作区中的 `ChunkCumsumKkt` 属于这个方向的探索性实现，不改变 Phase 2 的正式优先级。

### Phase 4：吸收 recompute_w_u

目标是让求解后的 A 尽量直接供 `recompute_w_u` 使用，减少：

- A 的 GM 写回/读取
- w/u 的中间落盘
- 阶段间同步

此阶段需要重新评估 A、w、u 的存储精度和 workspace 复用，不能只以算子数量减少作为成功标准。

### Phase 5：吸收 fwd_h

重点是状态矩阵 h、`v_new` 和前一阶段 w/u 的片上复用。需要特别关注：

- state 的 `[Hv, K, V]` 规模
- GVA 下 `Hk` 与 `Hv` 的映射
- `initial_state` 和 `final_state`
- dense 与 varlen 的 chunk 索引
- 50 MB 额外显存限制

### Phase 6：吸收 fwd_o

重点是 q/k/v_new/h 的读取和输出布局，评估是否可以避免中间 transpose 或重复 contiguous。此阶段结束后，核心 GDN 路径应接近：

```text
local_cumsum/KKT/solve_tri/recompute_w_u/fwd_h/fwd_o
```

的单次大融合执行。

### Phase 7：纳入 causal_conv1d、RMSNorm 和门控

完整 Demo 的顶层融合还包括：

```text
causal_conv1d -> GDN core -> RMSNorm -> gated SiLU
```

它们不是当前核心六算子首个融合点，但属于最终全 GDN 目标。纳入顺序应以数据依赖和显存收益为依据，不能为了“算子数量看起来更少”破坏已经稳定的 GDN core。

## 4. transpose 策略

第一轮真正融合不跳过 transpose，也不把 transpose 和 KKT/solve_tri 同时修改。

暂时采用：

```text
外部输入布局
    -> ACLNN wrapper 中完成必要 transpose/contiguous
    -> fused KKT + solve_tri 使用已验证的内部 head-first 布局
    -> 保持现有输出布局
```

这样可以隔离第一轮融合变量。只有当 `KKT + solve_tri` 本身通过精度和性能验收后，才单独评估：

- fused kernel 是否直接接受外部布局
- 是否可以消除 transpose
- 是否可以把 transpose 与前后阶段合并

## 5. 消融和验收规则

开始任何新 Phase 前，必须先按 `GDN_FUSION_DEVELOPMENT_PLAYBOOK_A2.md` 填写阶段启动卡，并执行其规定的分层构建、版本哈希、精度和验收流程。

### 5.1 Phase 版本固化规则

从 2026-07-25 起，每个进入验收的 Phase 必须保留独立、可构建、可调用的版本化 ACLNN，禁止在同一个 ACLNN 实现上原地覆盖上一 Phase：

| Phase | 固定 ACLNN | 固定计算边界 |
| --- | --- | --- |
| Phase 1 | `aclnnGdnCoreFwdPhase1` | 一个 executor 内调度六个独立 kernel |
| Phase 2 | `aclnnGdnCoreFwdPhase2` | 用 `ChunkKktSolveTri` 替换独立 KKT + solve_tri |
| 当前默认 | `aclnnGdnCoreFwd` | 兼容入口，当前指向 Phase 2，可随验收阶段前移 |

后续 Phase 3 必须新增 `aclnnGdnCoreFwdPhase3`，不能修改 Phase 1/2 的阶段调度来承载 Phase 3。版本化入口必须同时存在于同一个完整验收包中，使用同一输入、同一设备和同一 benchmark 直接 A/B；不得用不同安装包的先后测试冒充同环境对照。

版本化入口只允许修复已证明影响该版本正确性、构建性或 ABI 的缺陷。修复前后必须归档原因和回归结果；性能优化或新融合边界一律进入新 Phase。源代码提交或归档点、安装包 SHA256、验收报告共同构成 Phase 快照，只有函数改名而没有测试和版本证据不算归档完成。

每个融合阶段都必须在同一个 benchmark 中增加命名 variant，并与 Phase 0 六算子基线比较。不能为每个版本创建互不兼容的测试脚本。

除版本化 ACLNN 外，每个 Phase 还必须在 `chw/gdn-a2-phase-archive` 上形成一个只追加、不改写的里程碑 commit。下一 Phase 新增 commit，不 amend/rebase/force-push 已归档 Phase；commit SHA、验收 wheel/OPP SHA256 和报告一起登记到 `GDN_PHASE_VERSION_ARCHIVE_A2.md`。Phase 1/2 因规则建立前共处同一未提交工作区，由首个 Phase 2 归档 commit 共同固化；Phase 3 起严格一 Phase 一新增 commit。

最低验收维度：

| 维度 | 要求 |
| --- | --- |
| 数值 | 输出和关键中间结果达到 L1；有效三角区域比较，忽略尾块无效 padding |
| 功能 | dense、varlen、尾块、FP16、BF16、GVA |
| 规格 | `K=128`，第一阶段 `V=128`；后续 `V=256` |
| chunk | `64/128` |
| 性能 | 全场景不劣于六算子基线，重点看端到端 latency |
| 显存 | 额外显存不超过 50 MB |
| 稳定性 | 至少 warmup 后多次测量，记录均值、P90 和异常情况 |

每次融合只有在以下条件同时满足后才能进入下一阶段：

1. 精度通过。
2. 目标场景功能通过。
3. 性能没有全场景回退，或回退有明确且获批准的取舍。
4. workspace、同步和尾块行为已解释清楚。
5. benchmark 结果和变更说明已归档。

## 6. 当前边界和不混淆事项

- 平台是 A2，不是 A5。
- 第一阶段先做 `K == V == 128`，这是阶段性范围，不是 GDN 数学上的永久限制。
- `V=256` 需要后续扩展状态、w/u、h、final state 和 fwd_o 的 shape/tiling。
- 统一 ACLNN 入口不代表单 kernel 融合。
- `GVA` 当前复合入口可以依赖外部 q/k head 扩展，但最终融合应原生处理 `Hk != Hv`，避免复制带来的额外流量。
- TND 是变长逻辑布局；当前 wrapper 可能使用物理 `B=1` 的 head-first rank-4 张量并通过 `cu_seqlens` 表达逻辑 batch。
- RMSNorm 不在当前六个 GDN core 小算子列表中，但属于最终完整 Demo 融合目标。
- backward 暂不作为首轮融合目标；先稳定 forward，保持现有 backward 链路。

## 7. 当前归档状态

截至 2026-07-24，Phase 1 已通过 A2 验收：

- GDN core 复合入口 A/B：`6/6 PASS`，覆盖 dense/varlen、FP16/BF16、尾块、`chunk_size=64/128`
- 完整 Demo 复合入口 A/B：`2/2 PASS`，覆盖 dense 与等长 varlen 的输出和梯度
- Phase 1 因此关闭，后续默认从 Phase 2 开始
- Phase 1 的六独立 kernel 复合调度已恢复并固化为 `aclnnGdnCoreFwdPhase1`；不再由通用入口的当前实现隐式代表

截至 2026-07-25，Phase 2 已完成实现、功能/精度验收和代表性性能筛查，但完整性能门禁尚未关闭：

- 独立 `ChunkKktSolveTri` A/B：`6/6 PASS`，覆盖 dense/varlen、FP16/BF16、`chunk_size=64/128`、多 batch/头数和尾块
- 4 组端到端 GDN core 消融场景通过；输出、`g_cumsum` 和有效 `A` 区域与基线一致
- 4 个代表 case 中，Phase 2 局部阶段 median latency 下降 `45.2%–57.3%`，完整 core 五调用路径相对六调用 median 改善 `1.2%–27.2%`
- profiler 确认 KKT + solve_tri 的 NPU kernel 数由 3 降为 1，完整 core 路径由 12 降为 10
- 局部融合 ACLNN 最大 workspace 为 `24,252,928` bytes；完整 core 峰值分配较基线下降 `8.2%–9.0%`
- 当前只有 4 个对角性能组合，且使用了 `ASCEND_LAUNCH_BLOCKING=1`；完整规格矩阵与生产性能模式待补
- 完整结果见 `GDN_PHASE2_ACCEPTANCE_A2.md`

当前工作区已经有：

- 六算子路径和统一 `aclnnGdnCoreFwd` 的实现/测试改动
- `gdn_core_ablation.md` 消融 benchmark 说明
- `aclnnGdnCoreFwd.md` 复合 ACLNN 接口说明
- `ChunkCumsumKkt` 探索性实现
- 已验收的 Phase 2 `ChunkKktSolveTri` 实现和独立 A/B 测试
- 可并存构建和调用的 `aclnnGdnCoreFwdPhase1` / `aclnnGdnCoreFwdPhase2` 固定入口

这些改动仍可能处于未提交状态。它们是当前工作区状态，不代表本文档中的后续阶段都已经完成。下一步先补齐 **Phase 2 完整性能门禁**；可并行分析 Phase 3，但性能门禁收口前不替换已验证路径。

## 8. 相关文件

- 总体背景与规格：工作区外部任务说明 `gdn.md`（不纳入仓库）
- 统一 ACLNN 接口说明：`fla/ops/ascendc/gdn/chunk_gdn_fwd/chunk_gated_delta_rule_fwd_h/docs/aclnnGdnCoreFwd.md`
- 消融 benchmark 说明：`fla/ops/ascendc/gdn/chunk_gdn_fwd/chunk_gated_delta_rule_fwd_h/docs/gdn_core_ablation.md`
- 消融脚本：`torch_custom/fla_npu/test/benchmark_gdn_core_ablation.py`
- GDN Python 入口：`examples/flash_gated_delta_rule.py`
- Phase 2 A2 验收报告：`fla/ops/ascendc/gdn/docs/GDN_PHASE2_ACCEPTANCE_A2.md`
- A2 GDN 融合开发手册：`fla/ops/ascendc/gdn/docs/GDN_FUSION_DEVELOPMENT_PLAYBOOK_A2.md`
