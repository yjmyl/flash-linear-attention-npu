# A2 GDN 当前状态

> 最后更新：2026-07-29
>
> 文档定位：**GDN A2 唯一可变进度源**。其他文档分别承载路线、开发规则、验收快照、版本归档或 API contract，不再重复播报当前进度。

## 1. 当前结论

- **Phase 1：已归档。** 保留六个 GDN core 逻辑阶段的版本化对照入口。旧路径已知的非有限值和长序列问题按历史边界记录，不扩成当前融合路径的修复任务。
- **Phase 2：已验收并归档。** 融合 `KKT + solve_tri`，保留独立 `local_cumsum`。
- **Phase 3：已验收并归档。** 在 Phase 2 基础上继续吸收 `local_cumsum`，最终为单个累积融合 `ChunkCumsumKktSolveTri`。
- **Phase 4：已验收并归档。** 边界为 `fwd_h + fwd_o`，目标 core 路径是
  `(A+B+C) + D + (E+F)`。单个 MIX kernel、版本化 Phase 4 core 入口、A2 整包构建、
  dense `4/4` + varlen `4/4` + state `1/1` 精度均已通过；dense/varlen 共 `8/8` 性能点
  均无可复现的实质回退，workspace 全部下降。Git 里程碑为
  `9719f2701f62ec7ef3d67751af52d1a1ea3c9435`，tag 为 `gdn-a2-phase4`。
- **Phase 4 流水 P1：已验收并完成 Git 归档。**
  dense C64/C128 采用 `H AIV 生产者 -> IB -> O AIV 消费者 -> 本地 O AIC`
  的 chunk-ready 流水，其他 shape 采用 H/O 任务生产核亲和调度；外层 H/O
  `SyncAll<false>()` 已移除。FP16/BF16 产物都包含 tiling key 1/2，device 7
  dense/varlen `8/8` 加 state `1/1` 对 Phase 3 bit-exact/finite；正式性能 8 点
  相对 Phase 0 全部大幅改善，目标 fused kernel 在 profiler 中更快。干净 run 包、
  完整 wheel、安装文件/符号/import 和正式 Example/ST 回归已通过。里程碑 commit 为
  `57c3ba03a3c15a797cedb9a712f02d3957de94f2`，tag 为 `gdn-a2-phase4-pipeline-p1`。

当前没有未收口的 Phase 2/3/4/P1 验收项。`gdn-a2-phase4` 保持不可变；
P1 作为归档后的独立性能 checkpoint 使用新 tag，没有回写旧 tag，也没有修改
Phase 1/2/3 历史路径。下一生产小步进入冻结 `V=128` 验收口径下的 Phase 5；
`V=256` 目前只保留实现扩展口，不作为 Phase 5 的前置门槛。

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

Phase 1~3 的逻辑边界是：

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
| Phase 4 | `(A+B+C)+D+(E+F)` | core dense/varlen `8/8` + state `1/1` bit-exact/有限 | 完整 core `8/8` 无可复现实质回退，任务数 `8 -> 7`，workspace 全部下降 | `gdn-a2-phase4` |
| Phase 4 流水 P1 | 同 Phase 4 融合边界，换成活性安全流水/亲和调度 | clean 产物 dense/varlen `8/8` + state `1/1` bit-exact/有限 | 8 点相对 Phase 0 改善 `76.25%~80.74%`；目标 kernel 更快，workspace 8/8 下降 | `gdn-a2-phase4-pipeline-p1` |

Phase 3 的共享 helper 局部微基准也是 `8/8` median 改善，但只用于证明 `A+B` helper 有效，不代替上表中的完整 core 生产性能结论。

## 4. 当前范围边界

已验收的 Phase 3/4/P1 共享以下范围边界：

- A2（Ascend 910B3）、`K==V==128`、head-first ND；
- dense/varlen，FP16/BF16，`chunk_size=64/128`；
- 现有 transpose/contiguous 和 canonical varlen metadata；
- GVA 依旧由外部扩头，不是 kernel 内原生 `Hk != Hv`。

上述已验收范围未宣称完成：`V=256`、原生 GVA、causal conv、RMSNorm/gate、backward、完整 Demo/模型性能，以及单 ACLNN 绝对 workspace `<=50 MB`。其中绝对 workspace 作为持续优化目标和报告项，不作为阻断融合路线的硬门槛；性能优先，若能在不伤害性能的前提下降低 workspace 则继续优化。

已达成但尚未实施的融合主线为：

```text
Phase 4: (A+B+C) + D + (E+F)
    -> Phase 5: (A+B+C) + (D+E+F)
    -> Phase 6: 仅在 profiler 支持时尝试 (A+B+C+D+E+F)
    -> Phase 7: transpose/layout
    -> Phase 8: causal_conv1d、RMSNorm/gate
    -> Phase 9: 完整 Demo/模型收口
```

`V=256` 和原生 GVA 是独立的后续规格扩展：当前只保留 tiling/调度表达能力，
不要求在进入 Phase 5 前完成全量精度和性能对齐；在产品计划要求正式声明支持时，
再分别启动独立规格闸门，避免与融合改动同时进行。

生产目标是最快且满足显存/精度的拓扑，不要求为了形式把 `ABC + DEF` 强制合成单 kernel。

后续默认只推进一条生产路线，不按 dtype、chunk、layout 或 shape 预设运行时分支。每个 Phase 的版本化 ACLNN 仅用于不可变 A/B 和归档，不代表生产入口需要同时维护多条规格路由。若冻结用例出现超出测量噪声的性能回退，或融合后没有获得预期收益，先在同一路线上最多做三轮有明确假设的单变量优化；三轮后仍无解，再携带基线差距、profiler 瓶颈、已尝试方案和预计工作量反馈决策，不自动新增分支。

## 5. Phase 4 快照、流水 P0/P1 与下一小步

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

归档后流水 P0/P1 结果（不改变 Phase 4 验收快照）：

- CANN `9.1.0.beta1` 实际头文件
  `asc/impl/basic_api/kernel_operator_block_sync_intf_impl.h` 在 A2 (`__NPU_ARCH__==2201`)
  的 AIC 分支中对 `IBSet/IBWait` 直接 `return`。因此禁止设计“AIV 直接 IB 唤醒
  cube”；cube 不得作为 IB 的 set/wait 端。
- 隔离探针先只保留 MIX kernel 中 AIV0/1 生产者与另一 MIX block 的 AIV0/1
  消费者，连续 64 轮 `IBSet<false>/IBWait<false>` 双向握手通过；说明 A2 的
  AIV 侧官方 IB 可用。
- 首个 AIV->AIC 桥接探针使用 `PIPE_S` 发事件，120 s 超时；按仓内 H/O
  已验证的 MIX 方向修正为 AIV `PIPE_MTE3` 发 ready、AIC `PIPE_FIX` 回 ack 后，
  `AIV 生产者 -> IB -> AIV 消费者 -> 本地 AIC` 64 轮通过。
- 通过版探针的 dense FP16 smoke 中 `o/g_cumsum/A` 与 Phase 3 bit-exact，全部有限；
  实验后已恢复 SHA256
  `a297168f3b5d14a09afd23acd060d3ab546bab9cbd71e7c8d301ebe3ce9b9206` 的归档 Phase 4 包，
  恢复后 smoke 再次通过。探针不保留在正式 HO kernel 源码中。
- P1 只在 dense `B=1,Hv=8,V=128,C=64` 且 A2 为 24 cube core 时开启：8 个 H 生产 core
  每头一个，16 个 O 消费 core 每头两个，分别处理偶数/奇数 chunk。ready 使用
  `chunk_idx % 2` 的两个 IB event，同步区只在 `v_new` 后增加 3072 B；其他 shape
  保持归档 Phase 4 的全核同步路径。
- 代表 smoke 为 dense FP16 `B=1,Hk=4,Hv=8,T=128,K=V=128,C=64`；Phase 4 单 ACLNN
  执行成功、全部有限，与 Phase 3 的 `o/g_cumsum/A/final_state` 全部 bit-exact，
  `output_max_abs=0`。
- 代表性能点 dense FP16 `B=1,Hk=4,Hv=8,T=1024,C=64` 的 50 轮同进程配对结果为
  Phase 3/Phase 4 median `1.038500/1.008300 ms`（Phase 4 `-2.91%`），逐轮配对
  median 改善 `-2.55%`。归档旧 Phase 4 的两轮历史数据相对同期 Phase 3 归一化约为
  `-0.3%`，与本轮差值指向约 `2.6%` 的流水净收益；但由于共享 A2 服务器连续外部
  `SIGKILL` 并留下设备任务，旧/新 Phase 4 同卡直接 A/B 未形成有效证据，`2.6%`
  只作为当前估计，不作归档结论。
- 补测时先在空闲 device 4/2 分别运行归档旧 Phase 4 的 `T=128,C=64` 最小 smoke；两次均未
  生成结果 JSON，AIC/AIV 持续 100%，进程终止后各留下一个不可中断 `D` 状态线程。`dmesg`
  在 device 4 测试前已有 `stranded cqe`，device 2 的普通 PyTorch NPU 加法虽返回正确值，
  退出时也产生同类 `stranded cqe`。待 `D` 状态线程由驱动自行回收后，又在 device 4 完整
  等待一次 300 s `timeout`，结果仍为持续满载、无 JSON。
- 随后按“允许竞争、只判定能否完成”的口径补测 device 5/6：相同输入的 Phase 3 控制组均
  `status=0`、生成 JSON 且全部有限；归档旧 Phase 4 在 device 6 约 60 s 后被杀，
  `status=137`，明显早于脚本设置的 180 s timeout。再在只有 `xllm` 常驻且启动前 AICore=0
  的 device 1 跑单迭代旧 Phase 4，仍进入 AIC/AIV 100% 且无结果，诊断充分后主动终止为
  `status=143`。归档旧 Phase 4 固定 `blockDim=aicCoreNum=24`，且 H/O 中间包含全核
  `SyncAll<false>()`；因此“竞争只增加延迟”不成立，当前现象与竞争或脏设备使全核 barrier
  无法完成一致。它不是新流水代码卡死的证据：这些失败全部发生在归档旧 Phase 4 哈希下，
  而新流水代表路径此前已完成 T128 bit-exact smoke 和 T1024 的 50 轮配对测量。
- device 2 无运行进程且启动前 AICore=0 时，使用归档 Phase 4 原仓 benchmark 做同脚本判别：
  Phase 3 单迭代 `status=0`、生成 JSON 且全部有限；紧接着归档旧 Phase 4 持续 AIC/AIV
  100%，180 s 后 `status=124`，无结果 JSON。随后在同一 device 临时覆盖新流水代表产物，
  新 Phase 4 单迭代 `status=0`、全部有限，并生成结果 JSON。由此当前主诊断改为：归档旧
  Phase 4 的全核 `SyncAll<false>()` 路径存在活性问题；新流水代表路径没有复现该卡死。
- device 2 随后完成 3 个独立进程、每进程 50 轮的 Phase 3/新 Phase 4 同进程交替测量，三轮
  median 变化依次为 `+0.05%/-1.06%/-1.12%`；合并 150 个样本后 median 为
  `1.077810/1.069810 ms`（新 Phase 4 `-0.74%`），逐对百分比的聚合 median 为 `-1.05%`。
  三个进程均 `status=0`、全部有限。结合先前 device 1 的 `-2.91%`，性能方向为正但收益幅度
  应按约 `1%` 级而不是先前单轮估计的 `2.6%` 描述。
- 上述诊断只作用于本轮精确 PID；未暂停其他用户进程、未 reset 共享设备、未覆盖安装产物，
  也没有把竞争卡上的挂起或 `137/143` 当作性能样本。
- 早期真机只构建了代表 FP16/G=float/state=float 的 tiling key 1 内核；本地正式源码仍保留
  key 1/2 和非代表 shape fallback。实验后已恢复归档安装环境，opmaster/代表 HO
  `.o/.json` SHA256 分别回到 `97399baa...` / `80de8119...` / `4c8e58b2...`；补测后再次
  核对仍保持这三个归档哈希。
- 本轮对 BF16/G=float/state=float 签名
  `b06684e29552ec07bcf59d49c832c773` 执行单目标构建成功；生成的 `.json` 同时包含
  tiling key 1/2，`.o/.json` SHA256 为 `aa0e5854...` / `dea2dc43...`。dense BF16 C64
  `T=128` smoke 为单 ACLNN 且全部 finite；正式 `T=1024` 下 Phase 3/P1 的
  `o/g_cumsum/valid-A/final_state` 全部 bit-exact，`output_max_abs=0`。三进程各 200 样本
  池化后 Phase 3/P1 median 为 `0.989290/0.981570 ms`（P1 `-0.78%`），逐对百分比
  聚合 median 为 `-0.74%`。工作区仍为 `136917504 -> 103617536 B`（`-24.32%`）。
  第一轮 P90 受共享机干扰波动，因此不用单轮 `-4.17%` 作收益结论。覆盖测试后
  安装环境的 opmaster/BF16 HO `.o/.json` 已精确恢复为 `97399baa...` /
  `208eaae4...` / `948e6c8b...`。
- Phase 0/1/2/3/新 P1 的同设备、同输入、同进程直测链路已经打通。首个与历史正式矩阵
  对齐的 `D_FP16_C64` 尾块点固定为
  `B=1,Hk=4,Hv=8,T=1025,K=V=128,C=64`，在 device 2 关闭 launch blocking，采用
  5 变体正反向轮转顺序；3 个独立进程各 warmup 20、每变体 200 样本均正常结束、全部
  finite，Phase 1/2/3/P1 对 Phase 0 bit-exact。合并 600 样本的 Phase 0/1/2/3/P1 median
  为 `3.942610/1.044950/1.014590/0.990840/0.990720 ms`；P1 相对 Phase 0 改善
  `74.87%`，相对 Phase 3 为 `-0.01%`，即该尾块点基本持平。三轮 P1 相对 Phase 3
  分别为 `+0.61%/-0.60%/+0.35%`，不构成稳定回退或稳定收益。
- 第二个正式直测 identity `D_BF16_C64` 固定为
  `B=1,Hk=4,Hv=8,T=1024,K=V=128,C=64`；同样在 device 2 执行 3 个独立进程，
  每进程 warmup 20、每变体 200 样本的 5 变体正反向轮转。三轮均 `status=0`、
  全部 finite，Phase 1/2/3/P1 对 Phase 0 bit-exact。合并 600 样本的 Phase 0/1/2/3/P1
  median 为 `4.289460/1.089440/1.051320/1.031320/1.031750 ms`；P1 相对 Phase 0
  改善 `75.95%`，相对 Phase 3 为 `+0.04%`，也属持平。三轮 P1 相对 Phase 3
  分别为 `-0.54%/+0.70%/+0.20%`。每轮覆盖测试后的安装哈希均与测试前一致。
- 已逐字段审计 Phase 2/3/4 历史 8 个正式性能 identity：除验收批次使用的 device 不同外，
  同 identity 的 ACLNN 输入 contract 一致。dense 固定
  `B=1,Hk=4,Hv=8,K=V=128`，BF16 用 `T=1024`、FP16 用尾块 `T=1025`；varlen 固定
  `T=259`，C64 为 `cu_seqlens=[0,1,66,259]`，C128 为
  `[0,1,130,259]`。因此旧阶段比值的 shape 身份可直接对应，但它们不是一次运行内的
  Phase 0~4 绝对毫秒横比。

冻结 Phase 4 验收快照保持不变。P1 已补齐 key 1/2 完整产物、替换所有非流水
shape 的全核同步 fallback，并完成 C128/varlen 正式矩阵和交付门禁。本 checkpoint
没有同时修改 transpose、workspace 别名、`V=256`、原生 GVA 或数学路径。因此下一小步
不再是补测 P1，也不提前验收后续规格，而是在冻结 `V=128` 口径下启动 Phase 5。
`V=256` 仅保留扩展口；若 Phase 5 触及共享 tiling/调度代码，可做编译或单点 smoke 防止
扩展口被意外破坏，但不在本阶段执行完整精度/性能矩阵。

**P1 最终收口（2026-07-29）：** 工作树已实现活性安全替代路径：dense C64/C128
使用分离 H 生产核和 O 消费核的 chunk-ready 流水，其余 shape 按 H 任务生产核
做 O 任务亲和调度，已移除 H/O 之间的外层 `SyncAll<false>()`。FP16/BF16、
`G=float/state=float` 两个签名均已在 A2 构建成功，产物都同时包含 tiling key 1/2；
设备产物 SHA256 分别为 FP16 `5ba5d16a.../1a57587e...`、BF16
`aa0e5854.../dea2dc43...`（`.o/.json`）。

使用已通过 Phase 0~P1 同进程门禁的 `7bdd4323...` opmaster 覆盖上述新 kernel 后，
device 7 的正式功能精度矩阵 dense `4/4`、varlen `4/4`、state `1/1` 均对
Phase 3 bit-exact 且全部 finite；额外 `B=4` 和四段 varlen 压力点也通过。含空序列
的额外 varlen 点在进入 P1 前即被 Phase 3 workspace 查询以 `169109` 拒绝，因此不记作
新 fallback 失败。正式性能均在 device 7，使用 3 个独立进程、每阶段池化 600 样本：
dense 为完整 `Phase 0/1/2/3/P1` 平衡轮转；首个 varlen 长稳进程在 warmup 第 2 轮、
明确标识为历史 `phase1_one_aclnn_six_kernels` 时复现 AIV MTE 越界，换 device 6
同输入复现到相同变体且 P1 尚未执行，因此按冻结决策不修旧算子，varlen 改为
`Phase 0/2/3/P1` 平衡轮转。

8 点池化 P1 相对 Phase 0 全部改善 `76.25%~80.74%`。相对 Phase 3 的 median 变化按
`D_BF16_C128/D_BF16_C64/D_FP16_C128/D_FP16_C64/V_BF16_C128/V_BF16_C64/`
`V_FP16_C128/V_FP16_C64` 顺序为
`+0.94%/-0.88%/-1.63%/-2.14%/+0.45%/-0.34%/-1.81%/+0.20%`；
三个正差均小于 1%。P1 workspace 为 `94,570,496~113,175,040 B`，对应 Phase 3
`130,229,760~152,240,640 B`，8 点均明显下降。profiler 中设备任务数均
`8 -> 7`：C64 目标段 `137.96 -> 132.70 us`（约 `-3.81%`），最差池化点
BF16/C128 的目标段 `107.30 -> 105.76 us`（约 `-1.44%`）。因此小于 1% 的整段
正差判为外围测量噪声，不启动无瓶颈假设的优化轮次。

当前 build 目录单独增量生成的 `12e062fa...` opmaster 在 Phase 3 tiling 时报告
`ChunkCumsumKktSolveTri compile info not contain [_pattern]`，不可作为发布产物；
覆盖失败后安装环境已按哈希恢复，最终结论没有使用该增量产物。全新源码副本的 A2
`bash build.sh --pkg --soc=ascend910b --vendor_name=fla_npu` 已成功：run 包 SHA256 为
`5525c89fbf7563c34fa5375a666326448928b1b06dd8759de731291af9795c86`，`libcust_opapi.so`
为 `8904e82de928a6a0eb137ddfcd457430cff85ef0679e3d052d4abc18e77f8fda`，opmaster 为
`771393f67bf931c47b2e20a9627b84fd2d5eae11c1d62525b207ffb716bbd47b`，HO 产物为
8 个 `.o` + 8 个 `.json`。使用这些 clean 产物反向覆盖后，正式 dense `4/4`、
varlen `4/4`、state `1/1` 和两个额外压力点全部返回 0。

基于 tag commit 重打的最终完整 wheel SHA256 为
`f374213f88ae3fae73179543fc097cbb95c2ac1ed2a122146b473833681bc550`，内含上述 clean OPP
和 P1 Python wrapper。wheel + run 成对安装后，安装 Python 文件与 tag 源码一致，
host/opmaster/HO 16 个文件与 clean 包一致，Phase 1~4 的 8 个 ACLNN 符号齐全，
`site-packages` 可直接 import `gdn_core_fwd_phase4`。安装态 dense FP16/C64 对 Phase 3
bit-exact；正式 Example/ST `gdr_accuracy_varlen_64_64_h2_d128_fp16` 的
`o/dq/dk/dv/dbeta/dg` 均 finite/allclose/cosine 通过。启用 qk-l2norm 的另一个
example 会在进入 GDN 前因当前 `triton-ascend` 与 CANN 9.1 runtime 枚举名不兼容而失败；
它属于 Phase 4 已冻结范围外的 Triton/L2Norm 环境门，不作为 P1 kernel 阻断项。

## 6. 文档职责和更新规则

| 文档 | 职责 | 何时更新 |
| --- | --- | --- |
| `GDN_CURRENT_STATUS_A2.md` | **唯一当前进度源**：当前 Phase、本次结论、下一小步、当前 blocker | 每个有效小步后；Phase 关闭时压缩成快照，不累积实验日志 |
| `GDN_FUSION_PLAN_A2.md` | 稳定路线、Phase 边界、冻结启动/修正卡 | 只有路线或边界变更时 |
| `GDN_FUSION_DEVELOPMENT_PLAYBOOK_A2.md` | 开发和验收方法 | 只有流程规则变更时 |
| `GDN_PHASE2/3/4_ACCEPTANCE_A2.md` | 已关闭 Phase 的冻结验收快照 | 原则上不随日常进度更新；只修正事实或表述错误 |
| `GDN_PHASE4_PIPELINE_P1_ACCEPTANCE_A2.md` | Phase 4 流水 P1 的独立性能 checkpoint 验收快照 | P1 关闭时一次性生成；不覆盖 Phase 4 原始快照 |
| `GDN_PHASE_VERSION_ARCHIVE_A2.md` | commit/tag/产物身份和归档规则 | 只在新里程碑或归档身份变化时 |
| `docs/aclnn*.md` / `gdn_core_ablation.md` | API contract 和 benchmark 使用方法 | 只在接口、路径或测量方法变更时 |

日常小步默认只更新本文档。Phase 关闭时才有一次有意识的多文档收口：写验收快照、登记归档身份、把本页压缩到新的当前状态。过程失败记录由 Git 历史和结构化证据保留，不再堆叠到“当前状态”中。

## 7. 正式证据索引

- Phase 2 验收：`GDN_PHASE2_ACCEPTANCE_A2.md`
- Phase 3 验收：`GDN_PHASE3_ACCEPTANCE_A2.md`
- Phase 4 验收：`GDN_PHASE4_ACCEPTANCE_A2.md`
- Phase 4 流水 P1 验收：`GDN_PHASE4_PIPELINE_P1_ACCEPTANCE_A2.md`
- Phase 4 流水 P1 正式性能：`gdn-phase0-p1-safe-formal-summary-d7-r1/`
- Phase 4 流水 P1 profiler：`gdn-phase4-p1-profiler-d7-r1/`
- Phase 4 流水 P1 clean 构建/产物矩阵：`gdn-phase4-p1-clean-full-build-r2/`、
  `gdn-phase4-p1-clean-artifact-matrix-d7-r1/`
- Phase 4 流水 P1 Git wheel/安装回归：`gdn-phase4-p1-git-wheel-build-r5/`、
  `gdn-phase4-p1-git-install-regression-d7-r3/`
- 版本与 tag：`GDN_PHASE_VERSION_ARCHIVE_A2.md`
- 总体路线：`GDN_FUSION_PLAN_A2.md`
- 小步开发方法：`GDN_FUSION_DEVELOPMENT_PLAYBOOK_A2.md`
- 统一 core API：`../chunk_gdn_fwd/chunk_gated_delta_rule_fwd_h/docs/aclnnGdnCoreFwd.md`
- 性能脚本口径：`../chunk_gdn_fwd/chunk_gated_delta_rule_fwd_h/docs/gdn_core_ablation.md`
