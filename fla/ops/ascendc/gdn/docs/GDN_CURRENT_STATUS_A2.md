# A2 GDN 当前工作状态

> 本文件是 GDN A2 融合工作的唯一续跑入口。每个小步开始前必须先读取本文件；
> 每个小步完成、失败或被中断后，必须先刷新本文件，再开始下一步。

## 1. 工作协议

1. 一次只执行“下一小步”中的一项，不从聊天记忆推断其他工作。
2. 开始前核对当前分支、HEAD、工作树和本文件记录是否一致。
3. 完成后记录实际命令或产物、结论、失败点和工作树变化。
4. 无论成功还是失败，都将“下一小步”改成唯一的一项；不得同时展开后续 Phase。
5. 正式里程碑仍以 Git commit、不可变 tag、验收包哈希和验收报告为准；本文件只记录进行中的工作状态。

## 2. 当前目标

归档 Phase 2 的 A2 性能收口增量；归档完成后填写 Phase 3 启动卡，在启动卡冻结前不修改 kernel。

## 3. 已冻结基线

- 分支：`gdn-a2-phase-archive`
- Phase 1/2 归档 commit：`f2a4b46`
- Phase 2 tag：`gdn-a2-phase2`
- remote：`chw/gdn-a2-phase-archive`
- Phase 1 固定入口：`aclnnGdnCoreFwdPhase1` / `gdn_core_fwd_phase1`
- Phase 2 固定入口：`aclnnGdnCoreFwdPhase2` / `gdn_core_fwd_phase2`
- 已归档验收包 SHA256：`ebcc8ecf20b8030acf51f75e0f62fe155f28d33379686cce647b6ecdd6e7c93c`

## 4. 当前进展

Phase 2 的实现、功能/精度验收、版本化入口和首个 Git 归档点已经完成。
本轮正在补正式生产性能口径：关闭 `ASCEND_LAUNCH_BLOCKING`，Phase 1/2 使用相同输入、
相同安装包、相同设备和相同参数，在互相隔离的干净进程中分别测量；每个 variant
warmup 10 轮、计时 50 轮。Phase 1 只作为性能基线，不再作为非有限用例的精度 golden。

截至本轮证据审计，Phase 2 已按冻结范围完成生产性能收口。正式验收报告、主计划和版本归档
已同步记录 dense/varlen 交叉点、`B=4,H=4,T=4096` 扩展点、`T=32768` Phase 2 单路径、
workspace/peak、profiler 和未覆盖边界。Phase 1 的长序列失败、原生 GVA、`V=256`、完整 Demo
性能和绝对 workspace `<=50 MB` 未被误写为已完成，也不再扩入本 Phase。

已完成 4 个 dense 交叉 case：

| case | Phase 2 相对 Phase 1 core median | 融合阶段相对独立 KKT+solve median | peak allocated delta |
| --- | ---: | ---: | ---: |
| `D_BF16_C64` | `-2.582%` | `-41.020%` | `-14,076,928 B` |
| `D_FP16_C64` | `-3.644%` | `-42.278%` | `-13,938,688 B` |
| `D_BF16_C128` | `-1.194%` | `-42.730%` | `-10,452,992 B` |
| `D_FP16_C128` | `-2.992%` | `-42.872%` | `-10,485,760 B` |

四例均通过 bit-exact 检查，Phase 2 core 未回退，局部融合收益稳定。

`V_BF16_C64` 已按单 case、独立进程复现，证明全矩阵终止不是 runner 跳过或日志丢失。

## 5. 当前停点

`V_BF16_C64` 在 Phase 1/2 成对交替计时的 warmup 阶段失败。异常发生在
`measure_paired_latency -> run_synchronized -> torch.npu.synchronize`，返回码为 `507015`；
设备日志报告 AIV core 22 的 MTE DDR 地址越界（`error code = 0x800000`）。精度比较和单次
workspace 测量已经在该异常前完成，但 case 没有生成最终 JSON，因此没有可归档的性能结论。

本次隔离运行的 benchmark 退出码为 `1`。完整 stdout/stderr 保存在：

- `/opt/chw/gdn-phase2-close-v-bf16-c64-isolated.log`
- `/opt/chw/gdn-phase2-close-v-bf16-c64-isolated.rc`

首次启动 Phase 1 单路径诊断时，远端命令封装错误地使用 `tr -d "\\r"`，删除了脚本中的
所有字母 `r`，使路径、`source` 和 `printf` 损坏；该次命令未进入 Python/NPU，不产生算子
结论，也不计为 Phase 1 重复调用结果。

修正后的临时脚本 `.codex_phase2_phase1_repeat_remote.sh` 已上传到远端源码目录。本地 Windows
环境没有 `bash`，因此本地 `bash -n` 未执行；下一次启动必须先在 A2 远端执行 `bash -n`，
通过后才能运行脚本。

远端 `bash -n` 校验通过后，`V_BF16_C64` 的 `aclnnGdnCoreFwdPhase1` 单路径在新进程中
连续同步调用 `10/10 PASS`，进程退出码为 `0`，未复现 `507015` 或 MTE 地址越界。证据文件：

- `/opt/chw/gdn-phase2-v-bf16-c64-phase1-repeat.log`
- `/opt/chw/gdn-phase2-v-bf16-c64-phase1-repeat.rc`

因此当前可以排除“Phase 1 单路径在 10 次重复调用内必然失败”，但尚不能排除 Phase 2 单路径
或 Phase 1/2 交替生命周期触发问题。

`V_BF16_C64` 的 `aclnnGdnCoreFwdPhase2` 单路径也在新进程中连续同步调用
`10/10 PASS`，进程退出码为 `0`，未复现 `507015` 或 MTE 地址越界。证据文件：

- `/opt/chw/gdn-phase2-v-bf16-c64-phase2-repeat.log`
- `/opt/chw/gdn-phase2-v-bf16-c64-phase2-repeat.rc`

因此 Phase 1、Phase 2 各自单路径的 10 次重复调用均稳定；当前故障范围收窄为 Phase 1/2
交替调用，或正式 benchmark 在成对计时前执行的精度/workspace 调用造成的累积状态。

交替诊断脚本已通过本地 `python -m py_compile` 并成功上传。首次远端启动命令在 conda 环境
加载前执行了 `python -m py_compile`，登录环境没有默认 `python`，因此 `&&` 提前终止；
该次命令未运行诊断脚本、未进入 NPU，不产生交替调用结论。

随后在新进程执行逐轮换序交替诊断，结果为：

```text
round 1: Phase 1 PASS -> Phase 2 PASS
round 2: Phase 2 PASS -> Phase 1 FAIL
```

失败发生在第二轮第二次调用的 `torch.npu.synchronize()`，返回 `507015`；设备日志再次报告
AIV MTE DDR 地址越界（本次 core 37，`error code = 0x800000`）。进程退出码为 `1`。证据文件：

- `/opt/chw/gdn-phase2-v-bf16-c64-paired-alternating.log`
- `/opt/chw/gdn-phase2-v-bf16-c64-paired-alternating.rc`

结合 Phase 1/2 各自单路径 `10/10 PASS`，故障目前高度集中在 `Phase 2 -> Phase 1` 的切换边界；
仍需用固定方向的新进程复现，排除逐轮换序脚本或更长调用序列的影响。

固定方向 `Phase 2 -> Phase 1` 已在新进程第一轮复现：Phase 2 同步通过，紧接的 Phase 1
第一次同步即返回 `507015`，设备日志再次报告 AIV MTE DDR 地址越界（本次 core 7，
`error code = 0x800000`）。进程退出码为 `1`。证据文件：

- `/opt/chw/gdn-phase2-v-bf16-c64-phase2-then-phase1.log`
- `/opt/chw/gdn-phase2-v-bf16-c64-phase2-then-phase1.rc`

该结果证明 `Phase 2 -> Phase 1` 切换本身即可触发故障，不需要正式 benchmark 前置流程，也不需要
多轮累积。尚需固定反方向 A/B，确认问题是否具有方向性。

固定反方向 `Phase 1 -> Phase 2` 的结果为：第一轮 Phase 1、Phase 2 均通过；第二轮刚回到
Phase 1 时同步失败，返回同一 `507015`，设备日志再次报告 AIV MTE DDR 地址越界（本次 core 34，
`error code = 0x800000`）。进程退出码为 `1`。证据文件：

- `/opt/chw/gdn-phase2-v-bf16-c64-phase1-then-phase2.log`
- `/opt/chw/gdn-phase2-v-bf16-c64-phase1-then-phase2.rc`

两种方向合并后的更准确结论是：Phase 1/2 单独重复均稳定；Phase 2 一旦执行过，同一进程中
随后首次 Phase 1 就会失败。现阶段还不能区分 Phase 2 修改/破坏复用输入，还是污染了进程级
executor、workspace 或 allocator 状态。

`Phase 2 -> 全新输入 -> Phase 1` 仍然失败：Phase 2 同步通过；重新创建的同 contract、同 seed
输入与旧输入之间没有任何 NPU `data_ptr` 复用（诊断输出为 `{}`）；Phase 1 随后同步返回
`507015`，设备日志再次报告 AIV MTE DDR 地址越界（本次 core 34，`error code = 0x800000`）。
进程退出码为 `1`。证据文件：

- `/opt/chw/gdn-phase2-v-bf16-c64-phase2-then-phase1-fresh-inputs.log`
- `/opt/chw/gdn-phase2-v-bf16-c64-phase2-then-phase1-fresh-inputs.rc`

因此可以排除“仅由复用输入被 Phase 2 修改”这一解释；故障更符合进程级 executor、workspace、
allocator 或 kernel/runtime 状态污染。下一步需要判断 Phase 2 新增的融合 kernel 单独执行是否足以触发。

独立融合阶段 `ChunkKktSolveTri -> 全新输入 -> Phase 1` 通过：融合阶段同步通过；新旧输入和
融合阶段输入之间没有 NPU `data_ptr` 复用（诊断输出为 `{}`）；随后 Phase 1 同步通过，进程
退出码为 `0`。证据文件：

- `/opt/chw/gdn-phase2-v-bf16-c64-fused-stage-then-phase1-fresh-inputs.log`
- `/opt/chw/gdn-phase2-v-bf16-c64-fused-stage-then-phase1-fresh-inputs.rc`

因此 `ChunkKktSolveTri` 单独执行不足以污染后续 Phase 1。剩余差异是 Phase 2 的完整阶段组合，
以及这些阶段位于统一 `aclnnGdnCoreFwdPhase2` executor 内的调度和 workspace 生命周期。

Python 五 ACLNN Phase 2 等价链 `fused_pipeline -> 全新输入 -> Phase 1` 也通过：五调用链同步
通过；新旧输入之间没有 NPU `data_ptr` 复用（诊断输出为 `{}`）；随后统一 Phase 1 同步通过，
进程退出码为 `0`。证据文件：

- `/opt/chw/gdn-phase2-v-bf16-c64-fused-pipeline-then-phase1-fresh-inputs.log`
- `/opt/chw/gdn-phase2-v-bf16-c64-fused-pipeline-then-phase1-fresh-inputs.rc`

因此污染不来自 Phase 2 的数学阶段组合，而特指统一 `aclnnGdnCoreFwdPhase2` executor 或其共享
workspace 生命周期。下一步需要区分 Phase 2 workspace 释放/地址复用与执行期间越界写入。

保留统一 Phase 2 输出和 runtime 保活队列时，`Phase 2 retained -> 全新输入 -> Phase 1` 通过：
Phase 2 同步后 `_RECENT_LAUNCH_STORAGE` 长度为 `1`；新旧输入/输出之间没有 NPU `data_ptr`
复用（诊断输出为 `{}`）；Phase 1 随后同步通过，进程退出码为 `0`。证据文件：

- `/opt/chw/gdn-phase2-v-bf16-c64-phase2-retained-then-phase1-fresh-inputs.log`
- `/opt/chw/gdn-phase2-v-bf16-c64-phase2-retained-then-phase1-fresh-inputs.rc`

结合“清理保活队列后失败”的 A/B，故障优先指向 Phase 2 workspace 释放后的地址复用或生命周期
问题，而不是 Phase 2 执行本身必然污染进程。尚需直接比对前后 workspace 地址。

workspace 地址取证直接命中同址复用：

```text
Phase 2 workspace: address=20616941928448, bytes=130229760
Phase 1 workspace: address=20616941928448, bytes=130449920
workspace address reused: True
Phase 1 synchronize: 507015 / AIV MTE DDR address out of range
```

Phase 1 workspace 比 Phase 2 大 `220,160 B`，但 allocator 复用了完全相同的基地址，随后同步失败。
进程退出码为 `1`。证据文件：

- `/opt/chw/gdn-phase2-v-bf16-c64-phase2-phase1-workspace-addresses.log`
- `/opt/chw/gdn-phase2-v-bf16-c64-phase2-phase1-workspace-addresses.rc`

该证据把故障与 workspace 同址复用强关联，但仍需通过占址 guard 迫使 Phase 1 换地址，完成因果 A/B。

占址 guard 因果 A/B 通过：

```text
Phase 2 workspace: address=20616941928448, bytes=130229760
guard:             address=20616941928448, bytes=130229760, reused_phase2=True
Phase 1 workspace: address=20617076146176, bytes=130449920, reused_phase2=False
Phase 1 synchronize: PASS
```

guard 抢占 Phase 2 旧基址并迫使 Phase 1 workspace 换址后，故障消失，进程退出码为 `0`。证据文件：

- `/opt/chw/gdn-phase2-v-bf16-c64-phase2-guard-then-phase1.log`
- `/opt/chw/gdn-phase2-v-bf16-c64-phase2-guard-then-phase1.rc`

至此已证明故障依赖 Phase 1 对 Phase 2 workspace 基址的直接复用。尚需区分该地址上的脏内容，
与 allocator/runtime 对该地址保留的状态。

整块 workspace 清零 A/B 通过，且三次分配使用完全相同的基址：

```text
Phase 2 workspace: address=20616941928448, bytes=130229760
scrub tensor:      address=20616941928448, bytes=130449920
Phase 1 workspace: address=20616941928448, bytes=130449920
Phase 1 synchronize: PASS
```

进程退出码为 `0`。证据文件：

- `/opt/chw/gdn-phase2-v-bf16-c64-phase2-scrub-then-phase1.log`
- `/opt/chw/gdn-phase2-v-bf16-c64-phase2-scrub-then-phase1.rc`

因此可以排除“allocator/runtime 禁止复用该地址”；故障由 Phase 2 遗留的 workspace 内容触发，
而 Phase 1 对 workspace 初值存在未声明依赖。尚需判断脏数据位于两者重叠区还是 Phase 1 新增尾部。

只清零 Phase 1/2 workspace 重叠区的 A/B 也通过：

```text
Phase 2 workspace: address=20616941928448, bytes=130229760
overlap scrub:     address=20616941928448, bytes=130449920
zeroed=130229760, tail_uninitialized=220160
Phase 1 workspace: address=20616941928448, bytes=130449920
Phase 1 synchronize: PASS
```

进程退出码为 `0`。证据文件：

- `/opt/chw/gdn-phase2-v-bf16-c64-phase2-scrub-overlap-then-phase1.log`
- `/opt/chw/gdn-phase2-v-bf16-c64-phase2-scrub-overlap-then-phase1.rc`

因此触发故障的 Phase 2 遗留内容确定在前 `130,229,760 B` 重叠区，不在 Phase 1 新增的
`220,160 B` 尾部。

已完成 Phase 1/2 workspace 的静态语义映射。本 case 的统一 ACLNN 实际收到的是已展开到
`Hv=8` 的 q/k，因此 contract 为 `B=1,Hk=Hv=8,T=259,K=V=128,C=64,totalChunks=7`。
两条路径共享的显式中间张量包括：

```text
gCumsum=8,288 B; aSolved=265,216 B; w=530,432 B; u=530,432 B
h=1,835,008 B; vNew=530,432 B; finalStateDummy=4 B
g/beta transpose or cast temporaries=8,288 B each
```

Phase 1 额外包含 `aRaw(fp32)=530,432 B` 以及 varlen `cast/transpose/solve/transpose`
形成的多个 `265,216 B` 临时张量；Phase 2 用 `ChunkKktSolveTri` 取代这些对象。各 kernel
workspace 的静态公式也已核对：KKT 为 `lib + score`，solve 为 `lib + shared + per-core`，
融合阶段为 `lib + score + a hand-off + per-core solve`，后续三个 kernel 的 workspace 规划在
两 Phase 间相同。由于 `aclOpExecutor` 会按生命周期复用线性 workspace，源码中的申请顺序
不能直接推出这些对象在总 `130 MB` workspace 中的绝对 offset；需要读取 executor 的实际
`GetWorkspaceOffsets()` 才能完成绝对地址映射。

静态阅读同时发现更强的根因候选：`fwd_h` 的 varlen 调度器把 `numSeq` 和 `numChunks` 放在
两个约 `512 B` 的 workspace 区。A2/C64/V128 下，其 kernel 内语义 offset 约为
`38,799,360 B` 和 `38,799,872 B`（最终以运行时 tiling/executor offset 为准）。每个 AIC/AIV
都会在 `BlockSchedulerGdnFwdH::Init()` 中并发清零、递推读写这两个共享数组，而全核
`SyncAll` 位于 `Init()` 返回后的 `Process()`。因此调度元数据在初始化完成前就可能被其他核读取；
新分配/清零 workspace 会掩盖该初值依赖，Phase 2 脏内容同址复用后可能生成错误的 token/chunk
计数并最终导致 MTE 越界。该结论目前是高优先级假设，尚未经过精准子区清零 A/B 证明。

已在本地准备临时只读 shim、Python 诊断脚本和远端编译运行脚本，本地
`python -m py_compile`、远端 `bash -n` 和远端 `python -m py_compile` 均通过；三份上传文件的
SHA256 已由远端 `sha256sum` 记录。首次启动 `dump_phase1` 时，外层 PowerShell/SSH 日志重定向
命令引号未闭合，远端 shell 在进入诊断脚本前即报 `unexpected EOF`。该次未编译 shim、
未启动 Python、未触发 NPU，不产生 executor/workspace 结论。

日志和退出码捕获改入远端 wrapper 后，第二次 `dump_phase1` 启动在 shim 编译阶段停止：
`op_executor.h` 间接包含的 `opdev/op_log.h` 未被当前单一 `${cann}/include` 搜索路径找到。
同时 wrapper 在 `source set_env.sh` 前开启 `set -u`，对尚未定义的 `LD_LIBRARY_PATH/PYTHONPATH/
CMAKE_PREFIX_PATH` 产生告警。该次未生成 shim、未启动 Python、未触发 NPU，不产生
executor/workspace 结论。

修复环境加载和 include 搜索路径后，`dump_phase1` 在 A2 上执行成功，进程退出码为 `0`：

```text
aclnnGdnCoreFwdPhase1: workspace_size=130449920 offset_count=0 offsets=[]
aclnnGdnCoreFwdPhase1: workspace_address=20616941928448
dump_phase1: PASS
```

证据文件：

- `/opt/chw/gdn-phase2-v-bf16-c64-phase1-workspace-offsets.log`
- `/opt/chw/gdn-phase2-v-bf16-c64-phase1-workspace-offsets.rc`

该结果证明临时 shim 与当前 `aclOpExecutor` ABI 可用，但 `GetWorkspaceOffsets()` 对该复合
executor 返回空向量。它没有暴露 `GetWorkspaceSize()` 内部线性内存规划的逐 launcher offset，
因此不能用空向量猜测 `fwd_h` 的绝对位置，本小步未执行精准子区清零 A/B。

已完成 CANN/opbase 只读 ABI 检查，可行的替代取证路径如下：

- `aclOpExecutor` 公开头文件中保留了内部 `allocatedTensorList_`；
- `aclTensor` 公开 `GetViewShape()`、`GetDataType()`、`IsFromWorkspace()` 和 `GetWorkspaceOffset()`；
- A2 当前 `libnnopbase.so` 导出 `aclTensor::GetWorkspaceOffset()` 及上述 shape/dtype 查询符号；
- 因此可用临时 shim 只读枚举 executor 的 workspace tensor 形状、dtype、numel 和绝对 offset，
  再用 `fwd_h` 独有的 `h=[1,8,7,128,128]`、`vNew=[1,8,259,128]` 及其 kernel workspace
  大小识别相关区间。

该路径不需要修改 runtime/kernel，也不再依赖猜测 launcher index。

`dump_phase1_tensors` 已在 A2 上执行成功，进程退出码为 `0`。Phase 1 executor 共枚举出
`51` 个内部 tensor，其中关键 workspace 规划为：

```text
tensor[47]: from_workspace=1, offset=20,070,400, bytes=55,577,088, shape=[55,577,088]
tensor[50]: from_workspace=1, offset=76,178,944, bytes=54,270,464, shape=[54,270,464]
```

`tensor[47]` 的大小与 `fwd_h` tiling 公式完全一致，`tensor[50]` 是后续 `fwd_o` workspace。
因此 `fwd_h` 在统一 Phase 1 workspace 中的基址 offset 确定为 `20,070,400`。根据
`fwd_h` 实际总 workspace `55,577,088 B`、尾部保留区 `16,777,216 B` 和两个对齐后
`512 B` 元数据区反推，先前静态记录的局部 offset 各偏大 `512 B`，精确值为：

```text
numSeq local offset    = 38,798,848
numChunks local offset = 38,799,360
absolute scrub range   = [58,869,248, 58,870,272)
```

证据文件：

- `/opt/chw/gdn-phase2-v-bf16-c64-phase1-workspace-tensors.log`
- `/opt/chw/gdn-phase2-v-bf16-c64-phase1-workspace-tensors.rc`

本小步只执行了 Phase 1 张量枚举，未执行 Phase 2 或精准清零 A/B。

`Phase 2 -> 同址 Phase 1` 的 `fwd_h` 调度元数据精准清零 A/B 已执行。Phase 1 确实
复用 Phase 2 完全相同的 workspace 基址 `20,616,941,928,448`，但结果为：

```text
targeted_scrub range=[58,869,248, 58,870,272)
nonzero_bytes_before=0
phase1 reused_phase2=True
phase1 synchronize=507015 / AIV MTE DDR address out of range
```

进程退出码为 `1`，证据文件：

- `/opt/chw/gdn-phase2-v-bf16-c64-targeted-fwd-h-scrub.log`
- `/opt/chw/gdn-phase2-v-bf16-c64-targeted-fwd-h-scrub.rc`

这是一个明确的否定结论：`numSeq/numChunks` 两个元数据区在 Phase 1 launch 前本来就是
全零，精准清零不改变失败。因此“这 `1024 B` 的 Phase 2 脏初值导致 `fwd_h` 调度器越界”
已被排除。已知整个 `130,229,760 B` 重叠区清零可以消除故障，所以真正的触发内容
仍位于其他重叠子区；目前不能说根因已经找到。

日志同时出现 `CausalConv1dBwd` 元信息初始化告警，但当前调用链不包含 causal conv；在没有
进一步证据前，不将其认定为本次 GDN 地址越界的根因。

2026-07-25 已确认收口策略调整：停止继续定位或修复 Phase 1 的 workspace 初值依赖。
现有证据已足以将其归档为旧路径遗留问题：Phase 1/2 单路径各自连续 `10/10 PASS`；只有
Phase 2 释放 workspace 后，Phase 1 在同一进程复用同址脏 workspace 时失败；换址或清零
重叠区可消除故障。该行为不再阻塞 Phase 2，但禁止用同进程 Phase 1/2 交替执行作为性能门禁。

旧独立 `KKT + solve_tri` 基线在部分 varlen 输入上会产生 NaN/Inf，因此这类用例必须独立
检查 Phase 2 输出有限性，并使用 CPU FP64 或其他已验证高精度参考判断精度；不得把 Phase 1
非有限输出当作 golden。Phase 1 仍可在干净进程中作为同输入性能参照。

2026-07-25 在继续性能门禁前重新完整阅读了 `docs/` 下五份权威文档。文档审计发现：
`GDN_FUSION_DEVELOPMENT_PLAYBOOK_A2.md` 和 `GDN_FUSION_PLAN_A2.md` 仍要求同一进程内交替
Phase 1/2，但上述已证明的 Phase 1 workspace 初值依赖使该方法不能产生有效的 varlen 性能
对照。开发手册要求任何偏离先写入主计划，因此启动 NPU 性能测试前必须先在主计划中记录
“相同包/输入/device、干净进程分别测量、进程级换序控制漂移”的证据化例外；本条只记录
文档前置条件，尚未启动新的性能运行。

上述例外已写入 `GDN_FUSION_PLAN_A2.md` 的“5.2 Phase 2 性能对照的进程隔离例外”：保留
同包、同输入、同 device、同环境和同 benchmark 参数，Phase 1/2 在干净进程中分别测量，
并按 case 级进程顺序换序和重复批次控制设备漂移。本小步只修改主计划和本状态文件，未修改
kernel/runtime/测试脚本，未启动 NPU。

`V_BF16_C64` Phase 1 启动前预检发现，现有 `benchmark_gdn_core_ablation.py --paired-only`
并不是单 variant 模式：它仍会在精度准备、workspace 测量和 `paired_latency` 中调用 Phase 2，
随后才写 JSON。因此不能用现有参数直接生成 Phase 1 干净进程基线。本次预检只读取脚本并
确认远端没有本项目残留测试进程，未启动 Python/NPU，未产生性能结论。

正式 `benchmark_gdn_core_ablation.py` 已新增 `--standalone-variant`，可选
`phase1_one_aclnn_six_kernels` 或 `phase2_one_aclnn_fused_kkt_solve`。standalone 分支在
`make_kkt_solve_inputs` 和完整消融控制流之前提前返回，只运行指定版本化入口，记录 output、
`g_cumsum`、有效 `A`、final state 的有限性，以及单次 ACLNN/workspace/peak memory 和 NPU
Event latency。本地 `python -m py_compile`、AST 禁止调用集合/提前返回控制流断言和
`git diff --check` 均通过；本小步未启动 NPU，也未修改 kernel/runtime 或矩阵 runner。

`V_BF16_C64` Phase 1 standalone 正式性能基线已在 A2 device 2 完成，退出码为 `0`。
生产计时关闭 `ASCEND_LAUNCH_BLOCKING`，warmup `10`、计时 `50`，进程内只调用
`aclnnGdnCoreFwdPhase1` 一次/轮。结果为：mean `1.389969 ms`、median `1.339530 ms`、
P90 `1.574000 ms`、min `1.153120 ms`；workspace max/sum 均为 `130,449,920 B`，
peak allocated delta 为 `131,255,808 B`。output、`g_cumsum` 和有效 `A` 的非有限计数均为
`0`，本 case 未请求 final state。运行结束后无残留 benchmark 进程。

证据文件及 SHA256：

- `/opt/chw/gdn-phase2-close-20260725-production/standalone/V_BF16_C64_phase1.json`：
  `ea03e4313678bdef4e7f89311add630b1d3f9eb0a9eb70e293b4e87e4d2180f2`
- `/opt/chw/gdn-phase2-v-bf16-c64-phase1-standalone.log`：
  `2696462f7c9c2f7df63096241663dddad361189016b76286b1760f0dc5be18e4`
- `/opt/chw/gdn-phase2-v-bf16-c64-phase1-standalone.rc`：`0`

本次运行还确认：远端当前 `build_out/fla-npu-fla_npu_linux-aarch64.run` SHA256 为
`ad3d614ca03844246a97c234cd04fa804855ac637d5a385fa8471bddb70f0cd1`，不等于旧报告记录的
归档包 `ebcc8e...`，因此不得将当前 `build_out` 文件表述为旧归档包；实际加载的已安装
`libcust_opapi.so` SHA256 为 `87c387757fcc19227a166aa90be149b4ce02bc386f8982ac731fbb8741a3349e`，
与 Phase 1/2 恢复验证归档值一致，且同时导出 Phase 1/2 执行和 GetWorkspaceSize 符号。

匹配的 `V_BF16_C64` Phase 2 standalone 正式性能已完成，退出码为 `0`。输入、已安装库、
device、seed、环境、warmup/iteration 和 benchmark SHA256 与 Phase 1 相同，进程内只调用
`aclnnGdnCoreFwdPhase2`。结果为：mean `1.058995 ms`、median `1.051950 ms`、P90
`1.118360 ms`、min `0.957040 ms`；workspace max/sum 均为 `130,229,760 B`，peak allocated
delta 为 `131,035,648 B`。output、`g_cumsum` 和有效 `A` 的非有限计数均为 `0`。
相对首轮 Phase 1，Phase 2 median 改善 `21.469%`、P90 改善 `28.948%`、min 改善
`17.004%`，workspace 和 peak allocated delta 均减少 `220,160 B`。运行结束后无残留进程。

证据文件及 SHA256：

- `/opt/chw/gdn-phase2-close-20260725-production/standalone/V_BF16_C64_phase2.json`：
  `13a9a47a5b042959a32109fa7c0d06f7bd927325407d3b3610b8871ddc7d1c2a`
- `/opt/chw/gdn-phase2-v-bf16-c64-phase2-standalone.log`：
  `8a920518a32f4251b72cc02033b78126d95287a0395475cfedaaa47d43bf728a`
- `/opt/chw/gdn-phase2-v-bf16-c64-phase2-standalone.rc`：`0`

`V_BF16_C64` Phase 2 standalone 的反向批次首半 `r2` 已完成，退出码为 `0`，所有有限性、
ACLNN 数和 workspace 检查再次通过。结果为：mean `1.154882 ms`、median `1.145580 ms`、
P90 `1.185960 ms`、min `1.003240 ms`；workspace/peak 与首轮完全一致。该次 median 比首轮
Phase 2 高约 `8.9%`，说明跨进程/时间漂移不可忽略，必须完成紧邻的 Phase 1 `r2` 后再按
批次比较，不能只用首轮绝对值下结论。

证据文件及 SHA256：

- `/opt/chw/gdn-phase2-close-20260725-production/standalone/V_BF16_C64_phase2_r2.json`：
  `6c5405451632a7f8ac21742a39b255fe95bcc006f1e216c55a00ce4f4dc3404f`
- `/opt/chw/gdn-phase2-v-bf16-c64-phase2-standalone-r2.log`：
  `acc3c8ba75b65e0783125f92961035882df50a7714df501f29c776fe7a8c2dd6`
- `/opt/chw/gdn-phase2-v-bf16-c64-phase2-standalone-r2.rc`：`0`

匹配的 `V_BF16_C64` Phase 1 standalone `r2` 已完成，退出码为 `0`，所有有限性、ACLNN 数和
workspace 检查通过。结果为：mean `1.072514 ms`、median `1.067020 ms`、P90 `1.124240 ms`、
min `0.978280 ms`；workspace/peak 与首轮 Phase 1 完全一致。反向批次内 Phase 2 `r2`
相对 Phase 1 `r2` 的 median 为回退 `7.363%`，与正向批次的改善 `21.469%` 方向相反；同时
Phase 1 两轮 median 漂移 `-20.344%`，Phase 2 两轮漂移 `+8.901%`。因此两轮独立进程数据
只能证明测量时序漂移显著，不能证明 Phase 2 core 提升或回退，当前性能结论为不确定。

证据文件及 SHA256：

- `/opt/chw/gdn-phase2-close-20260725-production/standalone/V_BF16_C64_phase1_r2.json`：
  `6cfc113861f0c215766f458594cd1e67a6019e950a7e324c67d0ff08b4e17fc4`
- `/opt/chw/gdn-phase2-v-bf16-c64-phase1-standalone-r2.log`：
  `75e7d01ade989362c1ddf9fb9cf71e5b69be2154b3ab537a9a3af63c0a279daa`
- `/opt/chw/gdn-phase2-v-bf16-c64-phase1-standalone-r2.rc`：`0`

正式 `run_gdn_phase2_performance_matrix.py` 已新增可选
`--measurement-mode standalone --standalone-rounds N`。standalone 模式要求偶数轮且至少两轮，
每轮按 AB/BA 交替顺序启动全新的 benchmark 子进程；每个子进程只运行一个版本化 Phase，
逐运行 JSON 保留原始 samples。runner 汇总全部 samples 和 batch medians 的 mean/median/P90/min，
记录进程顺序、有限性和 workspace/peak 一致性，并直接计算 Phase 2 相对 Phase 1 的变化。
默认 `paired` 模式及原有输出结构未改变。本地 `py_compile`、纯函数 aggregate/command 检查和
`git diff --check` 均通过；本小步未启动 NPU，未修改 benchmark/kernel/runtime。

`V_BF16_C64` standalone balanced 正式性能已完成，退出码为 `0`。共 4 轮 AB/BA、8 个干净
子进程，每个 variant 汇总 `200` 个 NPU Event 样本。Phase 1 median/P90/min 为
`1.116150/1.308640/0.939600 ms`，4 个 batch median 为
`[0.983990, 1.256780, 1.149050, 1.039830] ms`；Phase 2 median/P90/min 为
`0.958540/1.108880/0.821220 ms`，batch median 为
`[0.954390, 0.959050, 0.918000, 1.078200] ms`。聚合后 Phase 2 median 改善 `14.121%`、
P90 改善 `15.265%`、min 改善 `12.599%`。两 Phase 的 4 次 workspace/peak 签名各自一致，
Phase 2 所有运行 output/state 均有限。该平衡结果替代前述两轮方向矛盾的试运行结论。

证据文件及 SHA256：

- `/opt/chw/gdn-phase2-close-20260725-production/standalone-balanced-v-bf16-c64/summary.json`：
  `cfe52c55347b153d3bf73d49cc0552b214967f277bbde1dbe2d0048da110a419`
- `/opt/chw/gdn-phase2-v-bf16-c64-standalone-balanced.log`：
  `254a39ef620c49866f98683b922b79471f13efe09a79a81f70c07dca0ec11054`
- `/opt/chw/gdn-phase2-v-bf16-c64-standalone-balanced.rc`：`0`
- 本地摘要镜像：`.phase2_v_bf16_c64_balanced_summary.json`，SHA256 与远端 summary 一致。

`V_FP16_C64` standalone balanced 正式性能已完成，退出码为 `0`。Phase 1 median/P90/min 为
`1.056100/1.143760/0.666140 ms`，batch median 为
`[1.043240, 1.106390, 1.077610, 1.002120] ms`；Phase 2 median/P90/min 为
`0.966450/1.040980/0.762060 ms`，batch median 为
`[0.963170, 0.926100, 0.991460, 0.985930] ms`。Phase 2 median 改善 `8.489%`、P90 改善
`8.986%`；min 回退 `14.399%` 是由 Phase 1 单个异常低样本主导，不作为主判据。两 Phase 的
4 次 workspace/peak 签名一致，Phase 2 所有运行均有限；workspace 和 peak delta 相对 Phase 1
各减少 `220,160 B`。

证据文件及 SHA256：

- `/opt/chw/gdn-phase2-close-20260725-production/standalone-balanced-v-fp16-c64/summary.json`：
  `c560e466d52a69542f9ab27bc6055ea438a70652d2dafec2234b49a2908b89b7`
- `/opt/chw/gdn-phase2-v-fp16-c64-standalone-balanced.log`：
  `45ae988f889f53717c9b9d4d51b72e576abcada28d5f35476a3f2a34cfe91b7a`
- `/opt/chw/gdn-phase2-v-fp16-c64-standalone-balanced.rc`：`0`
- 本地摘要镜像：`.phase2_v_fp16_c64_balanced_summary.json`，SHA256 与远端 summary 一致。

`V_BF16_C128` standalone balanced 正式性能已完成，退出码为 `0`。Phase 1 median/P90/min 为
`1.190180/1.378220/0.859740 ms`，batch median 为
`[1.220190, 1.242460, 1.213000, 0.974180] ms`；Phase 2 median/P90/min 为
`0.963580/1.062300/0.779200 ms`，batch median 为
`[1.052880, 0.979580, 0.949360, 0.881810] ms`。Phase 2 median/P90/min 分别改善
`19.039%/22.922%/9.368%`。两 Phase 的 4 次 workspace/peak 签名各自一致，Phase 2 所有运行
均有限。

证据文件及 SHA256：

- `/opt/chw/gdn-phase2-close-20260725-production/standalone-balanced-v-bf16-c128/summary.json`：
  `6f115b4e0c39ce43616bada2e585ddcb15a0f3874c0453422c6502992879de47`
- `/opt/chw/gdn-phase2-v-bf16-c128-standalone-balanced.log`：
  `9750d4b8a226947fdf035e4aecca4aefe7eabfad756396e72b925db238335e7c`
- `/opt/chw/gdn-phase2-v-bf16-c128-standalone-balanced.rc`：`0`
- 本地摘要镜像：`.phase2_v_bf16_c128_balanced_summary.json`，SHA256 与远端 summary 一致。

`V_FP16_C128` standalone balanced 首轮正式性能已完成，退出码为 `0`。Phase 1 median/P90/min
为 `1.143880/1.465060/0.916360 ms`，batch median 为
`[0.986960, 1.139430, 1.193430, 1.229930] ms`；Phase 2 median/P90/min 为
`0.987180/1.503960/0.812900 ms`，batch median 为
`[0.918470, 0.948590, 1.177360, 1.183170] ms`。四轮 batch median 均为 Phase 2 更快，聚合
median 改善 `13.699%`、min 改善 `11.290%`，但聚合 P90 回退 `2.655%`。逐轮样本显示后两轮
两 Phase 均出现 `2 ms+` 长尾，Phase 2 的长尾更集中；该 P90 回退需一组独立平衡重复确认，
当前不能宣称此 case 完整无回退。两 Phase workspace/peak 签名一致，Phase 2 全部运行有限。

证据文件及 SHA256：

- `/opt/chw/gdn-phase2-close-20260725-production/standalone-balanced-v-fp16-c128/summary.json`：
  `a7e7b0af9f9c4e99ce97cc6cd5bd72f978ccc85e9b2dd150098357db65766c66`
- `/opt/chw/gdn-phase2-v-fp16-c128-standalone-balanced.log`：
  `747789fc06a40a466506d36bc83c3e1c0c178c9891c863cced27e28941b624a6`
- `/opt/chw/gdn-phase2-v-fp16-c128-standalone-balanced.rc`：`0`
- 本地摘要镜像：`.phase2_v_fp16_c128_balanced_summary.json`，SHA256 与远端 summary 一致。

`V_FP16_C128 r2` 在启动前按协议停止，未进入 Python/NPU：wrapper 语法和本地/远端三文件
SHA256 均通过，但 `npu-smi info` 显示 A2 device 2 已被外部 `xllm` 进程占用约 `45,250 MB`，
AICore 利用率约 `65%`。未终止或干预外部进程，也未启动本项目 runner，因此没有新的 JSON、
退出码或性能结论。device 1 当时无运行进程且为相同 910B3 SoC。

`V_FP16_C128` 已在空闲 A2 device 1 完成独立复制，退出码为 `0`。Phase 1 median/P90/min 为
`1.003290/1.139840/0.844180 ms`，batch median 为
`[0.993960, 1.004990, 0.965350, 1.117730] ms`；Phase 2 median/P90/min 为
`0.944230/1.074140/0.380620 ms`，batch median 为
`[0.952640, 0.897890, 0.933450, 1.068150] ms`。Phase 2 median/P90 分别改善
`5.887%/5.764%`，device 2 首轮的 P90 `2.655%` 回退未跨设备复现；两次独立实验的主判据
median 均提升，因此该 case 按“median 无回退、P90 异常未复现”收口。两 Phase workspace/peak
签名一致，Phase 2 全部运行有限。device 1/2 原始样本未合并。

证据文件及 SHA256：

- `/opt/chw/gdn-phase2-close-20260725-production/standalone-balanced-v-fp16-c128-device1-r2/summary.json`：
  `8db4da5a2cbc7a4e71383c681543d0487e786d7a634dba655773622ba47b1722`
- `/opt/chw/gdn-phase2-v-fp16-c128-standalone-balanced-device1-r2.log`：
  `b6915b521edecbe5cc4417ce59a96691800778a265063eb2b1bc82cc5cb527d9`
- `/opt/chw/gdn-phase2-v-fp16-c128-standalone-balanced-device1-r2.rc`：`0`
- 本地摘要镜像：`.phase2_v_fp16_c128_device1_r2_summary.json`，SHA256 与远端 summary 一致。

至此 4 个 varlen dtype/chunk 交叉点均完成 standalone balanced 生产性能：Phase 2 的聚合
median 相对 Phase 1 分别改善 `14.121%`、`8.489%`、`19.039%`、以及
`13.699%`（device 2）/`5.887%`（device 1 复制）；所有 Phase 2 运行均有限且 workspace 一致。

`S_B4_H4` 已在 A2 device 1 完成首轮 standalone balanced，退出码为 `0`。Phase 1
median/P90/min 为 `2.119090/2.160520/1.982820 ms`，batch median 为
`[2.128010, 2.113380, 2.106840, 2.125330] ms`；Phase 2 median/P90/min 为
`2.142560/2.324960/1.375180 ms`，batch median 为
`[2.151210, 2.029580, 2.291530, 2.138510] ms`。Phase 2 median 回退 `1.108%`、P90 回退
`7.611%`，四轮 batch median 中三轮略慢、一轮更快；min 改善 `30.645%`。Phase 1/2
workspace 分别为 `194,351,616/203,207,168 B`，Phase 2 增加 `8,855,552 B`；peak delta
分别为 `220,464,640/228,853,248 B`，Phase 2 增加 `8,388,608 B`，仍低于额外 50 MB 门槛。
两 Phase 的 4 次签名各自一致且 Phase 2 全部运行有限。因 median/P90 均出现回退，必须独立
重复后才能决定是噪声还是可复现回退。

证据文件及 SHA256：

- `/opt/chw/gdn-phase2-close-20260725-production/standalone-balanced-s-b4-h4-device1/summary.json`：
  `bf1523ed952e40dbef01f605705c7f8cfe3d751ed3b98a73127037f560183efb`
- `/opt/chw/gdn-phase2-s-b4-h4-standalone-balanced-device1.log`：
  `d49630f1c65d174e18c2f0941c41417da59c1ce121cb0b0008920eb423985007`
- `/opt/chw/gdn-phase2-s-b4-h4-standalone-balanced-device1.rc`：`0`
- 本地摘要镜像：`.phase2_s_b4_h4_device1_summary.json`，SHA256 与远端 summary 一致。

`S_B4_H4` 独立重复 `r2` 已在同一 device 1 完成，退出码为 `0`。Phase 1 median/P90/min
为 `2.109400/2.322560/1.944160 ms`，Phase 2 为
`2.071290/2.136400/1.939540 ms`；Phase 2 median/P90/min 分别改善
`1.807%/8.015%/0.238%`，首轮 median/P90 回退未复现。合并同设备两组后每个 variant 共
`400` 样本，Phase 1/2 median 为 `2.115320/2.091060 ms`，Phase 2 改善 `1.147%`；P90
为 `2.194820/2.285540 ms`，Phase 2 回退 `4.133%`，该差异由首轮长尾主导且在 `r2` 方向
反转。因此本 case 的严格结论为“主判据 median 无回退；P90 长尾不稳定、首轮回退未复现”，
并保留 Phase 2 workspace 增加 `8.86 MB`、peak 增加 `8.39 MB` 的事实。

证据文件及 SHA256：

- `/opt/chw/gdn-phase2-close-20260725-production/standalone-balanced-s-b4-h4-device1-r2/summary.json`：
  `2fd1cd4dec61c5b01c6a81aac38fb94a602498d8bb7833f2cf9bfe0a2530b098`
- `/opt/chw/gdn-phase2-s-b4-h4-standalone-balanced-device1-r2.log`：
  `b3fb6209c95dbc872f1198bda4f23e0d378b505c2f9e793d64b102396b83cdb8`
- `/opt/chw/gdn-phase2-s-b4-h4-standalone-balanced-device1-r2.rc`：`0`
- 本地摘要镜像：`.phase2_s_b4_h4_device1_r2_summary.json`，用于同设备合并分析。

`L_VARLEN_T32768` standalone balanced 在 A2 device 1 按协议于首个干净子进程停止。预检时
本地/远端 benchmark、runner 和 wrapper SHA256 一致，远端 `bash -n`、两脚本 `py_compile`、
runner `--help` 均通过；device 1 无进程且 AICore/AIVector 利用率为 `0%`。首个子进程只调用
Phase 1，但第一次调用后的 `torch.npu.synchronize()` 即返回 `507015`，设备日志报告 AIV core 33
的 MTE DDR 地址越界（`error code = 0x800000`）。失败发生在有限性检查、workspace 测量和计时
之前，因此没有生成该轮 JSON；runner 未启动 Phase 2 或剩余 7 个子进程，退出码为 `1`，结束后
device 1 无本项目残留进程。

证据文件及 SHA256：

- `/opt/chw/gdn-phase2-l-varlen-t32768-standalone-balanced-device1.log`：
  `21ea02192faf359e106c50ec2f7ed442362de6a7275922b023cd051cd839be1f`
- `/opt/chw/gdn-phase2-l-varlen-t32768-standalone-balanced-device1.rc`：`1`，文件 SHA256：
  `4355a46b19d348dc2f57c046f8ef63d4538ebb936000f3c9ee954a27460dd865`

该结果只证明 Phase 1 旧路径在此长 varlen contract 上无法形成性能基线，尚不构成 Phase 2
功能、精度或性能结论；按已确认的收口策略不进入 Phase 1 kernel 调试。

同一 `L_VARLEN_T32768` contract 的 Phase 2 standalone 随后在 device 1 的全新进程中完成，
退出码为 `0`。进程内只调用 `aclnnGdnCoreFwdPhase2`，warmup `10`、计时 `50`，生产计时关闭
`ASCEND_LAUNCH_BLOCKING`。output `33,554,432` 个元素、`g_cumsum` `262,144` 个元素和有效
`A` `33,552,400` 个元素的非有限计数均为 `0`。latency mean/median/P90/min 为
`6.954240/6.910770/7.268100/6.681740 ms`；workspace max/sum 均为 `543,318,528 B`，
peak allocated delta 为 `679,634,432 B`。该绝对 workspace 明显超过 `50 MB`，但 Phase 1
同 contract 无法运行，不能计算相对基线的 workspace/peak 增量，也不能给出长序列相对性能。

证据文件及 SHA256：

- `/opt/chw/gdn-phase2-close-20260725-production/standalone-l-varlen-t32768-phase2-device1/L_VARLEN_T32768_phase2.json`：
  `12cca72ee4188bfe040c5e46b4879d447120c57610665d5977f1a68640ce0472`
- `/opt/chw/gdn-phase2-l-varlen-t32768-phase2-standalone-device1.log`：
  `49a66aa79ce57068d4b4b68b538239cd71fad3d97beef7e900c4db31d75aced3`
- `/opt/chw/gdn-phase2-l-varlen-t32768-phase2-standalone-device1.rc`：`0`，文件 SHA256：
  `9a271f2a916b0b6ee6cecb2426f0b3206ef074578be55d9bc94f6f3ab86aa`
- 本地摘要镜像：`.phase2_l_varlen_t32768_phase2_device1.json`，SHA256 与远端 JSON 一致。

因此长序列代表点目前可归档的严格结论是：Phase 2 功能、有限性和绝对性能可运行；由于
Phase 1 旧路径失败，长序列相对性能和相对显存门禁不可判定。按既定策略不把 Phase 1 修复
扩入 Phase 2 收口。

远端已有结果目录：`/opt/chw/gdn-phase2-close-20260725-production/`。

本地工作树包含尚未归档的性能门禁改动：

- `torch_custom/fla_npu/test/benchmark_gdn_core_ablation.py`：成对交替计时、state 校验和运行期保活清理。
- `torch_custom/fla_npu/test/run_gdn_phase2_performance_matrix.py`：冻结的性能矩阵 runner。
- `.codex_phase2_*`、`.phase*.json`：临时脚本和中间结果，不进入里程碑 commit。

### Phase 2 归档推送记录

只将 Phase 2 性能收口追加 commit `2b8161d` fast-forward 推送到
`chw/gdn-a2-phase-archive`。推送前确认本地 `HEAD=2b8161d`、远端分支仍为基线 `f2a4b46`、
`gdn-a2-phase2` tag 仍指向 `f2a4b46`；不得 force-push、不得推送或移动 tag。推送内容只包含
已核验的 7 个正式文件；本地 `.learnings/`、`.codex_phase2_*`、`.phase*.json` 保持未提交。
推送失败立即停止，不修改远端历史；成功后先刷新本文件，再将唯一下一小步改为 Phase 3 启动卡。

首次远端只读核对执行 `git ls-remote chw` 时，本机到 `github.com:443` 连接超时并返回
exit code `128`。该失败发生在 push 前，远端没有任何变更；本地 `HEAD=2b8161d`，且
`f2a4b46` 已确认是其祖先，本地 annotated tag `gdn-a2-phase2` 仍解析到 commit `f2a4b46`。
继续本小步前必须先恢复 GitHub 连通性并重新读取远端 branch/tag；不得跳过只读核对直接推送。

随后使用只在本机监听的临时 SOCKS 隧道经 A2 出网，本地 Git credential helper 仍直接完成
GitHub 认证。只读核对确认远端 branch 为 `f2a4b46`、annotated tag 对象为 `e13c6ed`；普通
fast-forward push 将 `chw/gdn-a2-phase-archive` 更新到 `2b8161d`，未使用 force、未推送或
移动 tag。推送后从 A2 再次只读确认 branch=`2b8161d`、tag object=`e13c6ed`；临时隧道已关闭。

Phase 2 性能收口至此完成并已远端归档。commit `2b8161d` 包含 7 个正式文件；本地
`.learnings/`、`.codex_phase2_*`、`.phase*.json` 和远端原始日志均未进入该 commit。

## 6. 下一小步

只填写并冻结 Phase 3 启动卡：阅读现有探索性 `ChunkCumsumKkt` 的 op_host/op_kernel/op_api、
Python wrapper 和测试，明确基线、融合边界、输入输出、首版保留项、不做项、最小验收矩阵与
成功标准，并将结论写入主计划和本状态文件。本小步只做静态证据分析和文档更新，不构建、
不运行 NPU、不修改 kernel/runtime/接口；发现探索实现与 Phase 3 边界不符时只记录差异，
不顺手修代码。完成后先刷新本文件，再决定首个独立 smoke 小步。

## 7. 收口判据

Phase 2 只有在以下证据均写入正式验收报告后才能关闭：

- dtype/chunk/layout 交叉组合通过；
- batch/head 和长序列代表点通过；
- launch blocking 关闭后，Phase 1/2 在匹配的干净进程中分别测量，Phase 2 相对基线无性能回退；
- Phase 2 单路径重复稳定，所有输出/state 有独立 NaN/Inf 检查，旧基线非有限用例改用 CPU FP64
  或其他已验证高精度参考；
- 精度、state、workspace/peak memory 和 profiler 结论归档；
- 正式测试脚本、报告、commit SHA、tag 和产物 SHA256 可追溯。
