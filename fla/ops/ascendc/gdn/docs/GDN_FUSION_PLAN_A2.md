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

#### Phase 3 启动卡（2026-07-25 冻结）

| 项目 | 冻结内容 |
| --- | --- |
| 基线 | 局部基线为同一完整包中的 `aclnnChunkLocalCumsum -> aclnnChunkScaledDotKkt`；端到端基线为不可变 `aclnnGdnCoreFwdPhase2` |
| 首个融合边界 | 仅 `raw g -> chunk-local FP32 forward cumsum` 与 `KKT`；首版独立输出 `g_cumsum` 和未求解 `A_raw` |
| 输入 | `k=[B,H,T,128]` FP16/BF16；`g,beta=[B,H,T]` FP32；dense 时无可选元数据，varlen 时同时提供 INT64 `cu_seqlens` 与 `[seq,local_chunk]` `chunk_indices` |
| 输出 | `g_cumsum=[B,H,T]` FP32；`A_raw=[B,H,T,C]` FP32，`C=64/128` |
| 物理布局/头契约 | head-first ND；探索实现的 tiling 明确要求物理 `k.H == g.H == beta.H`，首版不实现原生 `Hk != Hv` |
| 保留项 | 保留公开 `g_cumsum` GM 输出、KKT score workspace、现有 transpose/contiguous、外部 GVA 扩头和 Phase 2 `ChunkKktSolveTri` 路径 |
| 不做项 | 不接 `solve_tri`，不创建 Phase 3 core 调度，不改 Phase 1/2，不做 reverse/scale cumsum，不扩 `V=256`、原生 GVA、causal conv、RMSNorm/gate 或 workspace 别名 |
| 独立入口 | 新增 `aclnnChunkCumsumKkt`、torch/Python wrapper 和专项测试；现有探索实现只有 OpDef/tiling/kernel/L0，尚不能直接做独立 ACLNN A/B |

静态执行语义按源码冻结为：每个物理 head、每个序列、每个 chunk 内从 raw FP32 `g` 按 token
顺序做 `values[row] += values[row-1]`，同一 FP32 UB 结果既写出公开 `g_cumsum`，也直接进入 KKT
gate epilogue。验收时必须把公开输出和内部 compute view 分开记录；只有源码和逐位证据同时证明
二者一致后，才允许以公开 `g_cumsum` 代表内部视图。

路由身份不能只写算子名。A2/CANN `9.1.0.beta1` 下首轮冻结 8 个完整身份：

```text
(FP16/BF16 k) × (C64/C128) × (dense/varlen optional-input state)
```

源码 tiling key 为 FP16 C64=`10`、BF16 C64=`20`、FP16 C128=`778`、BF16 C128=`788`；
dense/varlen 共用 key 但走不同 `isVarlen` 子路径，因此仍分开验收。实际二进制必须用 profiler 再
确认 `ChunkCumsumKkt` kernel 名、tiling key 和 block dim，不能只由源码公式推定。

最低验收矩阵和成功标准：

1. 先做 1 个 dense FP16/C64 最小 smoke，只证明独立 ACLNN 可调用、两个输出 shape/dtype 正确且有限。
2. 再覆盖 dense/varlen、FP16/BF16、C64/C128、整 chunk/尾块、多 batch/多头；每个完整路由身份至少 `10` 个 exact case。
3. `g_cumsum` 全有效元素对两小算子 NPU 拼接基线逐位一致；`A_raw` 严格下三角有效值逐位一致，并单独断言对角、上三角和尾块 padding 为零。基线非有限时改用 CPU FP64 参考，不把 NaN/Inf 当 golden。
4. profiler 证明局部 ACLNN 数 `2 -> 1` 且实际 NPU kernel 数减少；生产计时关闭 launch blocking，固定 warmup/iteration，AB/BA 测 median、P90、min。
5. 局部融合 median 不劣于两小算子基线；随后新增不可变 `aclnnGdnCoreFwdPhase3`，端到端 median 不劣于 Phase 2。若局部无收益，停止接入 core并保留 Phase 2。
6. 记录 fused/baseline workspace max/sum、peak delta 和绝对值；相对基线额外显存不得超过 `50 MB`，绝对 `50 MB` 口径继续单列，不混淆。

Phase 3 的第一个实现小步只补 `ChunkCumsumKkt` 独立 ACLNN、torch/Python wrapper、ABI/静态测试
和最小 smoke 脚本，不修改现有 kernel、tiling、数学顺序或 Phase 2 调度。该入口通过最小 smoke
前，不开始完整 exact 矩阵，更不接入 `solve_tri` 或 `GdnCoreFwdPhase3`。

#### Phase 3 core 累积融合修正卡（2026-07-25 冻结）

局部 `ChunkCumsumKkt` 已完成 frozen 8 identities 共 `80/80` exact，随后探索性 Phase 3 core
使用 `ChunkCumsumKkt -> Cast -> SolveTri`。生产 pilot 和完整 core profiler 证明该路由相对
Phase 2 median 回退 `3.682%`，且将 core NPU kernel 数从 `9` 增至 `10`；根因是吸收 cumsum 时
丢失了 Phase 2 已完成的 KKT + solve_tri 融合。因此 Phase 3 最终验收边界修正为累积融合：

```text
raw FP32 g -> chunk-local FP32 cumsum -> KKT -> low-precision hand-off -> solve_tri
```

| 项目 | 冻结内容 |
| --- | --- |
| Phase 2 不可变基线 | `aclnnGdnCoreFwdPhase2` 保持 `ChunkLocalCumsum -> ChunkKktSolveTri`，不原地修改 |
| Phase 3 专用 kernel | 新增 `ChunkCumsumKktSolveTri`；不改写已完成独立精度证据的 `ChunkCumsumKkt` contract |
| 输入 | 与 Phase 3 局部算子一致：`k=[B,H,T,128]` FP16/BF16，raw `g/beta=[B,H,T]` FP32，dense/varlen canonical metadata |
| 输出 | 公开 `g_cumsum=[B,H,T]` FP32；已求解 `A=[B,H,T,C]` 与 `k` 同 dtype，`C=64/128` |
| workspace | 复用 Phase 2 已验证布局：FP32 score、低精度 A hand-off、每个 MIX core group 的 solve workspace；不做输出/workspace 别名 |
| 同步 | AIC 生成 score；配对 AIV 使用已 `80/80 exact` 的 cumsum/KKT epilogue并写低精度 A；AIV 以 `PIPE_MTE3` 发出 `KKT_READY`，配对 AIC 只等待一次后进入 solve |
| public/compute cumsum | 继续使用已验证的 `MTE2_V -> V_MTE3 -> MTE3_MTE2 -> MTE3_V` 顺序；公开 FP32 输出写回后再读作 KKT compute view |
| core 调度 | `aclnnGdnCoreFwdPhase3` 直接消费该 kernel 的两个输出，不再调度独立 Cast 或 SolveTri |
| 保留边界 | head-first ND、外部 GVA 扩头、`K==V==128`、现有 transpose、dense/varlen metadata 和 Phase 1/2 固定入口不变 |
| 性能门槛 | profiler 预期完整 core kernel 数 `9 -> 8`；生产 median 必须不劣于 Phase 2，否则 Phase 3 不收口 |

实现顺序固定为：新增 op/kernel 静态注册与 L0 -> Phase 3 core 改接 -> 完整包最小 smoke -> 独立
精度/主路由/state 回归 -> 单点生产性能门禁 -> 全矩阵 -> profiler/workspace -> 验收与归档。任何一层
失败先刷新 `GDN_CURRENT_STATUS_A2.md`，不跨层继续。

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

### 5.2 Phase 2 性能对照的进程隔离例外

`V_BF16_C64` 的因果 A/B 已证明：Phase 1/2 各自在干净进程中连续 `10/10 PASS`，但 Phase 2
释放 workspace 后，Phase 1 若在同一进程复用同址的非零 workspace 会触发 `507015`/AIV MTE
越界；迫使 Phase 1 换址或清零 Phase 1/2 workspace 重叠区后故障消失。这是 Phase 1 对 workspace
初值的未声明依赖，继续用同进程交替会把旧路径生命周期缺陷混入 Phase 2 性能门禁。

因此 Phase 2 剩余性能收口采用以下受控例外：Phase 1/2 仍使用同一个完整安装包、同一输入、
同一 device、同一环境和同一 benchmark 参数，但每个 variant 在干净进程中分别测量；case 级
执行顺序按 `Phase 1 -> Phase 2` 与 `Phase 2 -> Phase 1` 进程级换序，固定 warmup/iteration，
分别记录 median、P90 和 min，用重复批次检查设备漂移。该例外只改变进程生命周期，不允许
更换构建产物、输入 contract 或计时方法。旧基线非有限的用例只使用 Phase 1 latency，Phase 2
精度必须独立检查有限性并与 CPU FP64 或其他已验证高精度参考比较。

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

截至 2026-07-25，Phase 2 已按冻结范围完成实现、功能/精度和生产性能门禁：

- 独立 `ChunkKktSolveTri` A/B：`6/6 PASS`，覆盖 dense/varlen、FP16/BF16、`chunk_size=64/128`、多 batch/头数和尾块
- 4 组端到端 GDN core 消融场景通过；输出、`g_cumsum` 和有效 `A` 区域与基线一致
- 4 个代表 case 中，Phase 2 局部阶段 median latency 下降 `45.2%–57.3%`，完整 core 五调用路径相对六调用 median 改善 `1.2%–27.2%`
- profiler 确认 KKT + solve_tri 的 NPU kernel 数由 3 降为 1，完整 core 路径由 12 降为 10
- 局部融合 ACLNN 最大 workspace 为 `24,252,928` bytes；完整 core 峰值分配较基线下降 `8.2%–9.0%`
- 关闭 `ASCEND_LAUNCH_BLOCKING` 后，dense/varlen 的 FP16/BF16 × C64/C128 交叉点和 `B=4,H=4,T=4096` 扩展点的 Phase 2 median 均无可复现回退；varlen standalone 的 Phase 2 输出/state 均通过独立有限性检查
- `T=32768` varlen 上 Phase 2 单路径输出有限且 median 为 `6.911 ms`；Phase 1 首次同步 MTE 越界，故该点的相对性能/显存不可判定，不扩入 Phase 1 修复
- 生产可比较 case 的最大相对增量为 `8.86 MB` workspace / `8.39 MB` peak；单 ACLNN 绝对 workspace `<=50 MB` 口径未满足，需与相对增量口径区分
- 完整结果见 `GDN_PHASE2_ACCEPTANCE_A2.md`

截至 2026-07-26，Phase 3 已按累积融合修正卡完成实现和 A2 验收：

- 最终累积边界为单 `ChunkCumsumKktSolveTri`：raw FP32 cumsum + KKT + low-precision hand-off + solve_tri；
- 独立 `ChunkCumsumKkt` 的最终共享 helper 完成 `80/80 exact`，core dense/varlen `8/8` 加 state
  `1/1` 均对不可变 Phase 2 bit-exact/有限；
- 独立局部和完整 core 的生产性能矩阵均 `8/8` median 改善；
- profiler 证明完整 core NPU kernel 数 `9 -> 8`，目标段由两个 kernel 合并为一个；
- 完整 run 包 SHA256 为 `837742c6731143ec0ea55c517338e0daca3d3f295b9b7a71f079805b9b62bfdb`，
  安装 host 库 SHA256 为 `645336f9f65c522a06d74cf8b3df0ca6db2ae493fcd6fec980719eca7c8af07f`；
- `aclnnGdnCoreFwdPhase1/2/3` 与默认入口可在同一包内并存，默认入口仍保持 Phase 2 兼容行为；
- 完整结果见 `GDN_PHASE3_ACCEPTANCE_A2.md`。

Phase 3 当前仍处于未提交工作区，尚未创建 `gdn-a2-phase3` 不可变 tag。只有正式提交清单、归档前
门禁、里程碑 commit、tag 和远端只读回查全部完成后，才能将 Phase 3 标记为 Git 已归档；Phase 4
不得在此之前启动。

## 8. 相关文件

- 总体背景与规格：工作区外部任务说明 `gdn.md`（不纳入仓库）
- 统一 ACLNN 接口说明：`fla/ops/ascendc/gdn/chunk_gdn_fwd/chunk_gated_delta_rule_fwd_h/docs/aclnnGdnCoreFwd.md`
- 消融 benchmark 说明：`fla/ops/ascendc/gdn/chunk_gdn_fwd/chunk_gated_delta_rule_fwd_h/docs/gdn_core_ablation.md`
- 消融脚本：`torch_custom/fla_npu/test/benchmark_gdn_core_ablation.py`
- GDN Python 入口：`examples/flash_gated_delta_rule.py`
- Phase 2 A2 验收报告：`fla/ops/ascendc/gdn/docs/GDN_PHASE2_ACCEPTANCE_A2.md`
- Phase 3 A2 验收报告：`fla/ops/ascendc/gdn/docs/GDN_PHASE3_ACCEPTANCE_A2.md`
- A2 GDN 融合开发手册：`fla/ops/ascendc/gdn/docs/GDN_FUSION_DEVELOPMENT_PLAYBOOK_A2.md`
