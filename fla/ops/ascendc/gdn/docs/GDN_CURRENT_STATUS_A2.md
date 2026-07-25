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

按已冻结启动卡完成 Phase 3 累积融合的正式验收、不可变 Git 里程碑和远端归档，同时保留
Phase 1/2 固定入口可在同一包内复跑。

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

只提交本地 tag 创建记录：暂存本文件和 `GDN_PHASE_VERSION_ARCHIVE_A2.md`，创建普通 commit
`docs(gdn): record phase 3 tag`，登记 tag object `b2615917...`、peeled commit `7fb8f05...` 和固定
message。提交后核对两文件清单并刷新本文件；本小步不 push、不构建或运行 NPU。

### Phase 3 本地不可变 tag（已完成）

修正后的前置只读门禁确认：

- 本地、远端均不存在 `gdn-a2-phase3`；
- 本地与远端 `gdn-a2-phase2^{}` 均为
  `f2a4b467f37887824106633524bd0b1c45737e1c`，旧 tag 未移动；
- `TAG_PRECHECK=PASS`。

随后创建本地 annotated tag：

```text
name:    gdn-a2-phase3
object:  b26159171f8aa0b1340f3f927412795853dc72e9
type:    tag
peeled:  7fb8f05b59ab56a8392e0f6c9bef071714894826
message: A2 GDN Phase 3 cumulative fusion
```

tag 明确指向实现与验收里程碑，不指向后续元数据 commit。没有 push、构建或运行 NPU；下一层只
提交两份 tag 元数据文档。

### Phase 3 精确里程碑元数据 commit（已完成）

已创建普通元数据 commit：

```text
commit:  b6bf26dc5ecfa60290226becbe7ac8ecdbc3ffbe
parent:  7fb8f05b59ab56a8392e0f6c9bef071714894826
subject: docs(gdn): record phase 3 milestone
files:   GDN_CURRENT_STATUS_A2.md, GDN_PHASE_VERSION_ARCHIVE_A2.md
```

提交清单严格为两份文档，未 amend 实现里程碑，未创建 tag、push、构建或运行 NPU。下一层只读
核对 tag 命名空间后，在本地创建指向 `7fb8f05...` 的 annotated tag，不指向元数据 commit。

首次 tag 前置只读检查没有创建 tag。它确认本地/远端均不存在 `gdn-a2-phase3`，远端 Phase 2
annotated tag object 仍为 `e13c6ed...`；但检查器错误地把文档短 SHA `f2a4b46` 手工扩成了错误的
完整 SHA，从而把实际本地 peeled commit
`f2a4b467f37887824106633524bd0b1c45737e1c` 误报为 tag 移动。该次没有本地或远端变更。
重试必须直接比较本地与远端 `gdn-a2-phase2^{}` 的 Git 派生完整 SHA，不再硬编码猜测值。

### Phase 3 实现与验收里程碑 commit（已完成）

已创建普通、不可改写的实现里程碑：

```text
commit:  7fb8f05b59ab56a8392e0f6c9bef071714894826
parent:  c76ab5a15b28ea5d890cbf7939ad340ce6b875be
subject: feat(gdn): complete A2 phase 3 cumulative fusion
files:   31
```

提交后审计确认 `BAD_PATH_COUNT=0`，没有 `.learnings/`、`.codex_*`、`.phase*`、build、run 包、JSON、
日志或远端原始产物；`git diff-tree --check` 通过，parent、subject 和文件数均与冻结清单一致，
总门禁 `MILESTONE_COMMIT_AUDIT=PASS`。没有 amend、tag、push、构建或运行 NPU。

精确 SHA 通过后续普通元数据 commit 登记，避免里程碑自引用；实现里程碑本身保持不变。下一层只
提交本状态文件和版本归档两份元数据。

### Phase 3 staged pre-commit 门禁（已完成）

使用显式 31 文件清单执行 `git add -- <files>`，没有使用 `git add .`。暂存区门禁结果：

- staged file count：`31`；临时/原始/build 路径命中：`0`；
- `git diff --cached --check`：通过；
- 敏感模式（private key/password/token/secret/机器地址）命中：`0`；
- Phase 1/2/3 入口、Phase 2/default 路由、Phase 3 累积路由和验收报告必需项齐全；
- ctypes ABI：`11/11 PASS`；CPU contract：`2/2 PASS`；
- 五个正式 Python 测试/benchmark/runner `py_compile`：通过；
- 总门禁：`STAGED_PRECOMMIT=PASS`。

验收报告使用 `gdn-a2-phase3^{commit}` 作为不可变里程碑引用；commit 创建后再用普通元数据 commit
登记精确 SHA，禁止 amend 实现里程碑制造自引用，也禁止提前创建或移动 tag。本小步没有修改实现、
构建、运行 NPU、commit、tag 或 push；下一层只创建并审计里程碑 commit。

### Phase 3 里程碑提交清单与最终 C++ 审查（已完成）

提交候选精确收敛为 31 个正式文件：18 个 tracked 修改和 13 个正式新增，覆盖：

- Phase 3 累积 kernel、OpDef/InferShape/L0、共享 tiling/workspace 和 CMake 注册；
- 版本化 `aclnnGdnCoreFwdPhase3`、独立 `aclnnChunkCumsumKkt`、torch/ctypes/Python ABI；
- 独立与 core benchmark、正式 Phase 3 runner、ABI/CPU contract 测试和 `test.sh` 接入；
- `aclnnGdnCoreFwd`、独立 ACLNN、消融说明、主计划、版本归档、当前状态和正式验收报告。

明确排除 `.learnings/ERRORS.md`、全部 `.codex_*`/`.remote_*`、`.phase*`、远端日志、profiler 原始
目录和本地取回的 `.phase3_staging_local_perf_r2/`。候选中没有 build/CPack/安装产物或机器凭据。

按 `cpp-code-reviewer` 对最终 C++ 差异复核：

- host 参数校验保持 shape/dtype/chunk/varlen canonical metadata 和旧 Phase public contract；
- Phase 1/2/default 路由未删除或改指，Phase 3 独立枚举和 ABI；
- 新 kernel 机械复用 Phase 2 已验证的 score/A/solve workspace 公式、core-group 地址计算和
  paired AIV `CrossCoreSetFlag<0x2>` / AIC 单次 wait；
- 唯一数学替换是 AIV epilogue 使用最终 MTE3 staging `InitFusedCumsum`，该 helper 已由独立
  `80/80 exact`、core `8/8 + state 1/1`、完整包和 profiler 验证；
- 未发现新的严重或重要问题；没有裸 host allocation、越界公式变化、ABI 覆盖或未处理返回值。

审查发现并修复三处正式文档过期：`aclnnGdnCoreFwd.md`、`gdn_core_ablation.md` 和
`GDN_PHASE_VERSION_ARCHIVE_A2.md` 现已列出 Phase 3 累积路径；主计划归档状态也已同步。
为保持本地源码与最终构建包字节一致，没有修改已构建验证过的 Python/C++ 实现注释或行为。
本小步没有暂存、commit、tag、push、构建或运行 NPU；下一层只做 staged pre-commit 门禁。

### Phase 3 正式验收报告与归档前静态门禁（已完成）

已新增 `GDN_PHASE3_ACCEPTANCE_A2.md`，只使用最终 MTE3 staging 包证据，覆盖累积融合边界、
Phase 版本入口、产物身份、独立 `80/80 exact`、core `8/8 + state 1/1`、局部/core 生产性能、
workspace/peak、profiler/kernel `9 -> 8`、失败候选淘汰理由和明确未覆盖范围。

归档前本地门禁结果：

- ctypes ABI：`11/11 PASS`；
- CPU contract：独立进程 `2/2 PASS`，退出码 `0`；
- benchmark/runner `py_compile`：通过；
- 报告长十六进制字段全部为 64 位 SHA256；
- 报告 20 个唯一 SHA256 均能在本状态账本反查，缺失 `0`；
- 必需入口、包/库哈希、`80/80 exact`、`8/8` 和 `9 -> 8` 结论全部存在；
- C++ 最终路由包含 `l0op::ChunkCumsumKktSolveTri(`，不包含旧 split
  `l0op::ChunkCumsumKkt(`；
- `git diff --check` 通过，仅有既有 Windows line-ending 提示。

本小步没有修改 kernel、构建、运行 NPU、创建 commit/tag 或推送。下一层只审计正式提交清单，
不得把本地学习日志、临时脚本或原始结果提交进里程碑。

### Phase 3 最终独立局部生产性能矩阵（`8/8 PASS`）

在最终 MTE3 staging 安装包上重跑独立 `ChunkCumsumKkt`，覆盖 dense/varlen × FP16/BF16 ×
C64/C128。A2 device 1，同一进程逐轮 AB/BA 交替两小算子基线和单融合算子，关闭
`ASCEND_LAUNCH_BLOCKING`，warmup `10`、每个 variant 采集 `50` 个 NPU Event 样本。八例均满足：

- 公开 FP32 `g_cumsum` 和 `A_raw` 对两小算子 NPU 基线全张量 bit-exact，且全部有限；
- 对外 ACLNN 数 `2 -> 1`；
- fused median 和 P90 均改善；
- workspace max 持平，workspace sum 固定减少 `16,777,728 B`，peak allocated delta 固定减少
  `16,778,240 B`。

| case | baseline median | fused median | median 变化 | P90 变化 |
| --- | ---: | ---: | ---: | ---: |
| `D_FP16_C64` | `0.942500 ms` | `0.640060 ms` | `-32.089%` | `-47.936%` |
| `D_BF16_C64` | `1.071500 ms` | `0.728780 ms` | `-31.985%` | `-44.182%` |
| `D_FP16_C128` | `1.120740 ms` | `0.758620 ms` | `-32.311%` | `-23.374%` |
| `D_BF16_C128` | `1.062760 ms` | `0.715710 ms` | `-32.656%` | `-32.856%` |
| `V_FP16_C64` | `1.353440 ms` | `0.865230 ms` | `-36.072%` | `-36.080%` |
| `V_BF16_C64` | `1.261460 ms` | `0.839060 ms` | `-33.485%` | `-34.069%` |
| `V_FP16_C128` | `1.176600 ms` | `0.763910 ms` | `-35.075%` | `-36.463%` |
| `V_BF16_C128` | `1.177090 ms` | `0.764890 ms` | `-35.019%` | `-35.210%` |

本轮实际加载同一最终 vendor 目录中的兼容库名 `libcust_opapi.so` 和 `libopapi.so`；两者大小均为
`527,736 B`、SHA256 均为 `645336f9f65c522a06d74cf8b3df0ca6db2ae493fcd6fec980719eca7c8af07f`，
远端 `cmp` 逐字节通过，不是旧 OPP 路径污染。run 包仍为
`837742c6731143ec0ea55c517338e0daca3d3f295b9b7a71f079805b9b62bfdb`。benchmark SHA256 为
`8c3ccfd357251de86995259a01d5838aaa218fffa1f07c4d066971e32cbbc871`，矩阵 wrapper SHA256 为
`ccaa01f4e9d58fe356ff6d94d4ea8b5672705fafb825acb5df0c10e9d86b96e9`。

远端原始证据目录为 `/opt/chw/gdn-phase3-staging-local-perf-r2`，八份 JSON SHA256：

```text
D_BF16_C128  3c8a0efba9f1b2ee91e69c175081e4bcd32b009268c3923c7ada28d29337a34f
D_BF16_C64   4a51baec04675888ef2981c4dd119c8e4ad1c34d97661168381668b3c1c14adf
D_FP16_C128  70f90dc73d82643891542941834f33e2bddf1bf6b0ad6779d65936eddfb06207
D_FP16_C64   ac2058169fd621bef86a0cc5c6930187f16bcfb4a27b4cc80bcae7255e15e991
V_BF16_C128  2c0a0004580e564f6d555d7ed88a37eac20bb75dc34f09e2d75b7e14c0462b60
V_BF16_C64   30d4a46677a403c84c7255c97be717bb4c3fbd2ce0dc4b5d3a1ad3bf7cac3dd3
V_FP16_C128  65704f6b6e7071ad267a43d07f00fa3b24e93f39155fd26eda9a96266a5aa9af
V_FP16_C64   45b13dfc2232ab797e9d78cc92ba231487322f895cd59ecff308f72bd33b93ba
```

本地只读门禁摘要为 `8` case、`0` failure，SHA256
`a7665e632e6207db0bfd7b1a743b890cc7fc11b83291c46ef0e55b2b7a086a33`。本小步没有修改 kernel、
重跑 core 性能或 profiler。至此 Phase 3 收口证据审计中唯一缺失的最终独立局部性能已补齐，
下一层只生成正式验收报告。

### Phase 3 收口证据审计（已完成）

逐项审计冻结启动卡、当前实现、A2 运行证据和交付规范，现状如下：

| 项目 | 当前证据 | 判定 |
| --- | --- | --- |
| 累积融合实现 | `ChunkCumsumKktSolveTri` + 版本化 `aclnnGdnCoreFwdPhase3` | 已实现并由最终二进制验证 |
| 包/安装一致性 | run 包 `837742c6...bfdb`，host 库 `645336f9...f07f`，Phase 3 文件 `10/10` cmp | 通过 |
| ABI/静态测试 | A2 ctypes `11/11 PASS`，CPU contract `2/2 PASS` | 通过 |
| 独立精度 | 最终共享 helper 的 8 identities `80/80 exact` | 通过 |
| core 精度 | dense `4/4` + varlen `4/4` + state `1/1` bit-exact/有限 | 通过 |
| core 性能 | 最终包生产矩阵 `8/8` median 改善，workspace/peak 持平 | 通过 |
| core profiler | 实际 NPU kernel `9 -> 8`，目标段单 kernel 改善 `17.314%` | 通过 |
| 独立局部性能 | 旧实现曾 `8/8` 改善，但最终 MTE3 staging 修改后尚未重取 | **缺失** |
| 正式验收报告 | 尚无 `GDN_PHASE3_ACCEPTANCE_A2.md` | 缺失，待局部性能后补 |
| 版本归档 | Phase 3 commit/tag/archive 尚未创建 | 缺失，待报告和最终门禁后补 |

提交前 C++ 审查覆盖最终 kernel、tiling、L0 和版本化调度，未发现新的严重或重要问题；A2 再次
执行 ctypes `11/11` 和 CPU contract `2/2` 均通过。审计确认不能用完整 core 性能替代最终共享
helper 的独立局部性能，因此下一小步只补该 `8/8` 矩阵。

### Phase 3 cumsum MTE3 staging 最终 core profiler（已完成）

在当前已审计安装包和同一 `D_FP16_C64` contract 上，Phase 2/3 分别在干净进程采集 profiler：

- Phase 2 完整 core 为 `9` 个 NPU kernel，device duration 合计 `340.067 us`；
- Phase 3 完整 core 为 `8` 个 NPU kernel，device duration 合计 `319.486 us`；
- Phase 2 目标段为 `ChunkLocalCumsum 45.301 us + ChunkKktSolveTri 83.262 us = 128.563 us`；
- Phase 3 目标段为单个 `ChunkCumsumKktSolveTri 106.302 us`，相对改善 `22.260 us`（`17.314%`）；
- 实际 Phase 3 trace 不含独立 `ChunkLocalCumsum`、`ChunkCumsumKkt` 或 `SolveTri`，证明最终二进制
  执行的是冻结的累积融合路由，而不是早期拆分候选。

结构化证据 SHA256：

- `summary.json`：`ae726a91fb800de97e05e4632591b486c462abc74d1aa09d451a10abeffd27ac`；
- Phase 2 trace：`f8b6d42bcb3a8fef5da99b6ccbcd7996dfc5ff784eb3334ac8cf88af340f0077`；
- Phase 3 trace：`69e65a04d29fa2062bce6cef17dcf2016f84e7e48f0ba55724b662a808791fa3`。

profiler 证明完整 core kernel 数 `9 -> 8` 和目标段单 kernel 收益；单次 duration 只作为路由和
设备阶段证据，生产性能结论仍以此前 AB/BA 的 `8/8` median 为准。本小步未修改 kernel或重跑
性能矩阵。

### Phase 3 cumsum MTE3 staging varlen 生产性能矩阵（`4/4` median 改善）

varlen FP16/BF16 × C64/C128 四点使用同一生产口径完成：

| case | Phase 2 median | Phase 3 median | median 变化 | P90 变化 |
| --- | ---: | ---: | ---: | ---: |
| `V_FP16_C64` | `0.931980 ms` | `0.897620 ms` | `-3.687%` | `-4.371%` |
| `V_BF16_C64` | `0.950730 ms` | `0.932310 ms` | `-1.937%` | `+0.202%` |
| `V_FP16_C128` | `0.941230 ms` | `0.921650 ms` | `-2.080%` | `-2.607%` |
| `V_BF16_C128` | `0.939280 ms` | `0.920690 ms` | `-1.979%` | `-0.712%` |

四例主判据 median 全部改善；P90 三例改善，BF16/C64 仅回退 `0.202%`。所有运行 contract 和
有限性通过。每例 Phase 2/3 workspace max/sum 与 peak 完全相同：

- C64：`130229760 B` / `131035648 B`；
- C128：`143843840 B` / `145774080 B`。

结构化 `summary.json` SHA256 为
`203034e59d162c62b608a49be1caec5be5bc5eb06ebeaf95e489ede47ca22731`。连同 dense，Phase 3
生产性能主矩阵 `8/8` 的 median 均改善，workspace/peak 均不劣于 Phase 2；P90 中 dense 三点和
varlen BF16/C64 的回退继续保留为长尾证据，不据此改写主判据。下一层需用最终二进制 profiler
确认 kernel 路由与数量。

### Phase 3 cumsum MTE3 staging dense 生产性能矩阵（`4/4` median 改善）

其余三个 dense identity 使用与首点相同的生产口径完成。连同 `D_FP16_C64`，dense 四点结果为：

| case | Phase 2 median | Phase 3 median | median 变化 | P90 变化 |
| --- | ---: | ---: | ---: | ---: |
| `D_FP16_C64` | `1.007370 ms` | `0.973270 ms` | `-3.385%` | `-0.922%` |
| `D_BF16_C64` | `0.946660 ms` | `0.925350 ms` | `-2.251%` | `+5.305%` |
| `D_FP16_C128` | `0.959930 ms` | `0.934970 ms` | `-2.600%` | `+1.623%` |
| `D_BF16_C128` | `0.965750 ms` | `0.939910 ms` | `-2.676%` | `+2.336%` |

四例主判据 median 全部改善，满足启动卡的“不劣于 Phase 2”门槛；P90 只有 FP16/C64 改善，
其余三例回退 `1.623%–5.305%`，保留为最终 profiler/长尾审计项，不隐去或改写。所有运行的
contract 和有限性通过。各例 Phase 2/3 workspace max/sum 与 peak 完全相同：

- `D_BF16_C64`：`136917504 B` / `140098048 B`；
- `D_FP16_C128`：`152240640 B` / `157324800 B`；
- `D_BF16_C128`：`151708160 B` / `155937280 B`。

其余三个 dense identity 的结构化 `summary.json` SHA256 为
`33bc664e8c8bd7444749a773cabd51c555362a08eb593dc16f64ca4575380380`。本小步未运行 varlen 或
profiler；dense 结果不能替代 varlen 生产行为。

### Phase 3 cumsum MTE3 staging `D_FP16_C64` 生产性能首点（门禁通过）

A2 device 1、同一安装包和输入，关闭 `ASCEND_LAUNCH_BLOCKING`，按 4 轮 AB/BA、8 个干净
子进程运行；每个 variant warmup `10`、NPU Event 采样 `50` 次，聚合各 `200` 个样本：

| 指标 | Phase 2 | Phase 3 | Phase 3 变化 |
| --- | ---: | ---: | ---: |
| median | `1.007370 ms` | `0.973270 ms` | `-3.385%` |
| P90 | `1.047360 ms` | `1.037700 ms` | `-0.922%` |
| min | `0.864900 ms` | `0.371640 ms` | `-57.031%` |

Phase 2 四轮 batch median 为 `[1.009990,1.035690,0.993130,1.002620] ms`，Phase 3 为
`[0.970510,0.969410,1.032480,0.962830] ms`；前三轮中两轮 Phase 3 改善、一轮回退，第四轮改善，
聚合 median/P90 均满足不劣于 Phase 2 的门槛。异常低的单次 min 只原样记录，不作为主判据。
两 Phase workspace max/sum 均为 `137055744 B`，peak allocated delta 均为 `140239360 B`；
8 次运行全部通过有限性和 contract 检查。结构化 `summary.json` SHA256 为
`6ab6abb75e086da3af6d3b227fc4e6ad2deb461369f07d4fd0241141b293a5a6`。

生产性能首点门禁通过，可以扩其余 dense identity；本小步未运行其他性能 case 或 profiler。

### Phase 3 cumsum MTE3 staging core state accuracy（`1/1 PASS`）

A2 device 1 上 `STATE_D_BF16_C64` 同时提供非空 FP32 `initial_state` 并请求真实 `final_state`，
退出码为 `0`。Phase 3 对不可变 Phase 2 的 output、公开 `g_cumsum`、有效 solved A 和真实 final
state 均 bit-exact，两个 Phase 的全部输出/状态有限。结构化证据 SHA256：

- `STATE_D_BF16_C64.json`：`35d8d78b9d6e3ff127a36b29350a913a4fe299db263526b38b7cd861331891bf`；
- `summary.json`：`ecccd0ca365418bddb4423c8640193bd3d1d16183bcf1e973fcf19b13dc74fb0`。

至此本轮累积 Phase 3 core 精度门禁为 dense `4/4` + varlen `4/4` + state `1/1`，连同共享 helper
独立 `80/80 exact` 全部恢复，可以进入生产性能首点。本小步未运行性能或 profiler。

### Phase 3 cumsum MTE3 staging core varlen accuracy（`4/4 PASS`）

A2 device 1 上 `V_FP16_C64`、`V_BF16_C64`、`V_FP16_C128`、`V_BF16_C128` 全部通过。
每例 Phase 3 对不可变 Phase 2 的 output、公开 `g_cumsum`、有效 solved A 和 final-state presence
均 bit-exact，两个 Phase 的所有呈现组件有限，canonical varlen 序列边界通过。结构化证据 SHA256：

- `V_BF16_C128.json`：`f47d574bdcb4ce1009b3577b97f2ef9155f1246634569bf4c76b2f1fc05b79c5`；
- `V_BF16_C64.json`：`ce7576bb0e7907c0aeb93fda93425674dfdc9a1728904533dad21eecd7ea729b`；
- `V_FP16_C128.json`：`0804fda928078a19519bd3875517c1486c5449e8ba57707cba9217b128fbef5b`；
- `V_FP16_C64.json`：`57fc72580d7db4e81e4928cb79cc3239425b16ae67f20ea331fe19eb5faa27ee`；
- `summary.json`：`95d7095a217fe4cc5209b211ef8df5f211ed8604748e6934bd1729c87f13d6d5`。

至此累积 Phase 3 core 主路由 dense/varlen × FP16/BF16 × C64/C128 共 `8/8 PASS`；尚需单独
覆盖非空 initial/真实 final state，之后才能进入生产性能门禁。

### Phase 3 cumsum MTE3 staging core dense accuracy（`4/4 PASS`）

A2 device 1 上 `D_FP16_C64`、`D_BF16_C64`、`D_FP16_C128`、`D_BF16_C128` 全部通过。
每例 Phase 3 对不可变 Phase 2 的 output、公开 `g_cumsum`、有效 solved A 和 final-state presence
均 bit-exact，两个 Phase 的所有呈现组件有限。结构化证据 SHA256：

- `D_BF16_C128.json`：`111bb88ad18c328ae0d4999fa63c206aa5cd29cf9479ea35dcc70bd530c5b1fe`；
- `D_BF16_C64.json`：`c2d5d3b599aa3aa7b397198c6186e5966ba87f4751eb7f6defd0f1c605a9f375`；
- `D_FP16_C128.json`：`f605bf80e5ee9369b45a8eded1ea6a9e9dc0867657a3b4f8bb8d191ab64958b5`；
- `D_FP16_C64.json`：`01b9717e16bfd3bebdc021977c8fee95ced8d3139b5ab980fc35fe8c26f41bcd`；
- `summary.json`：`10ecd0b89ae8a2791cba09a8324b0f01973cd2e0c35eaa9d3d4cfcf26b02e521`。

本小步未运行 varlen/state、性能或 profiler。dense 结果不替代 varlen 序列边界验证。

### Phase 3 cumsum MTE3 staging 独立 varlen BF16/C128 exact（`10/10 PASS`）

复用 varlen FP16/C128 的 10 个 frozen contract，只改变 `k` dtype 为 BF16；A2 device 1 上每例融合
公开 FP32 `g_cumsum` 和 FP32 `A_raw` 均与两小算子 NPU 基线全张量逐位一致，输出有限，无效区
为零，退出码为 `0`。日志
`/opt/chw/gdn-phase3-staging-standalone-v-bf16-c128-exact10-r1.log` SHA256 为
`fcccf2efc527a329bd017a4ebfc5d120956520e12155a6c8f66fef9fb62e3cf3`。

至此共享 helper 修改后的独立 frozen 8 identities 共 `80/80 exact PASS`：

```text
dense/varlen × FP16/BF16 × C64/C128，每个身份 10 例
```

每例均验证公开 FP32 `g_cumsum`、FP32 `A_raw`、有限性和无效区清零。本结论恢复独立 value
evidence；尚未替代累积 Phase 3 core 的 dense/varlen/state 精度、性能或 profiler 门禁。

### Phase 3 cumsum MTE3 staging 独立 varlen FP16/C128 exact（`10/10 PASS`）

复用 varlen C64 的 10 个 frozen contract，只改变 `chunk_size=128`；A2 device 1 上每例融合公开
FP32 `g_cumsum` 和 FP32 `A_raw` 均与两小算子 NPU 基线全张量逐位一致，输出有限，无效区为零，
退出码为 `0`。日志 `/opt/chw/gdn-phase3-staging-standalone-v-fp16-c128-exact10-r1.log`
SHA256 为 `6ab30c7e817f64317cc83b2fe0fbe795349a54143a2286335ad1e4f838039f90`。

本小步未运行其他 identity、core 精度、性能或 profiler。独立 exact 累计 `70/80 PASS`。

### Phase 3 cumsum MTE3 staging 独立 varlen BF16/C64 exact（`10/10 PASS`）

复用 varlen FP16/C64 的 10 个 frozen contract，只改变 `k` dtype 为 BF16；A2 device 1 上每例融合
公开 FP32 `g_cumsum` 和 FP32 `A_raw` 均与两小算子 NPU 基线全张量逐位一致，输出有限，无效区
为零，退出码为 `0`。日志
`/opt/chw/gdn-phase3-staging-standalone-v-bf16-c64-exact10-r1.log` SHA256 为
`627a5f178ca7c326a58d0b552ec68244e15b0c48fc223c53ba8363d80d34c9f4`。

本小步未运行其他 identity、core 精度、性能或 profiler。独立 exact 累计 `60/80 PASS`。

### Phase 3 cumsum MTE3 staging 独立 varlen FP16/C64 exact（`10/10 PASS`）

A2 device 1 上运行 10 个 frozen varlen contract，覆盖单/多序列、1-token 序列、序列边界恰逢
chunk 边界、跨 chunk 尾块和多头。每例融合公开 FP32 `g_cumsum` 和 FP32 `A_raw` 均与各自
canonical metadata 的两小算子 NPU 基线全张量逐位一致，输出有限，无效区为零，退出码为 `0`。
日志 `/opt/chw/gdn-phase3-staging-standalone-v-fp16-c64-exact10-r1.log` SHA256 为
`a7ccc69b11ffc1b942147abb014c5465fb79aa4bbf10422e1e9280c8018b5198`。

本小步未运行其他 identity、core 精度、性能或 profiler。独立 exact 累计 `50/80 PASS`。

### Phase 3 cumsum MTE3 staging 独立 dense BF16/C128 exact（`10/10 PASS`）

复用 dense FP16/C128 的 10 个 frozen contract，只改变 `k` dtype 为 BF16；A2 device 1 上每例融合
公开 FP32 `g_cumsum` 和 FP32 `A_raw` 均与两小算子 NPU 基线全张量逐位一致，输出有限，无效区
为零，退出码为 `0`。日志
`/opt/chw/gdn-phase3-staging-standalone-d-bf16-c128-exact10-r1.log` SHA256 为
`9517bd38b40d889151e3a7bf9e3c523f4ca613ccec4e021b7b027369bc2e945a`。

至此独立 dense 四个身份 `40/40 exact PASS`；未由此推导 varlen，也未运行 core 精度、性能或
profiler。

### Phase 3 cumsum MTE3 staging 独立 dense FP16/C128 exact（`10/10 PASS`）

复用 dense C64 的 10 个 frozen contract，只改变 `chunk_size=128`；A2 device 1 上每例融合公开
FP32 `g_cumsum` 和 FP32 `A_raw` 均与两小算子 NPU 基线全张量逐位一致，输出有限，无效区为零，
退出码为 `0`。日志 `/opt/chw/gdn-phase3-staging-standalone-d-fp16-c128-exact10-r1.log`
SHA256 为 `0c27b35391842be73552a9309f6cda7e0fc58a530ae4e94bc437d7780f629cdc`。

本小步未运行其他 identity、core 精度、性能或 profiler。独立 exact 累计 `30/80 PASS`。

### Phase 3 cumsum MTE3 staging 独立 dense BF16/C64 exact（`10/10 PASS`）

复用 dense FP16/C64 的 10 个 frozen contract，只改变 `k` dtype 为 BF16；A2 device 1 上每例融合
公开 FP32 `g_cumsum` 和 FP32 `A_raw` 均与两小算子 NPU 基线全张量逐位一致，输出有限，无效区
为零，退出码为 `0`。日志
`/opt/chw/gdn-phase3-staging-standalone-d-bf16-c64-exact10-r1.log` SHA256 为
`0d98164c825d57ce96f63a04fc2552d886ff1e3b911d705bf26b4116e8c72f8c`。

本小步未运行其他 identity、core 精度、性能或 profiler。独立 exact 累计 `20/80 PASS`。

### Phase 3 cumsum MTE3 staging 独立 dense FP16/C64 exact（`10/10 PASS`）

在当前已审计安装包、A2 device 1 上运行 frozen dense FP16/C64 的 10 个 contract，覆盖
`T=1/2/63/64/65/127/128/129/193/257`、`B=1/2/3`、`H=1/2/3/4`。每例融合公开 FP32
`g_cumsum` 与独立 `ChunkLocalCumsum`、FP32 `A_raw` 与两小算子 NPU 基线全张量逐位一致，
输出有限，严格下三角外区域和尾部 padding 为零；固定 wrapper SHA256 为
`d075a420b1cea81d8f0e471f5a7e74601b0570be1c2329b4cb92feccd638ab74`，命令退出码为 `0`。

本小步未运行其他 identity、core 精度、性能或 profiler。独立 exact 累计 `10/80 PASS`。

### Phase 3 cumsum MTE3 staging 安装审计与原失败 smoke（已完成）

本轮 run 包 SHA256 复核为
`837742c6731143ec0ea55c517338e0daca3d3f295b9b7a71f079805b9b62bfdb`，完整安装到 A2
`chw-py11` 后，安装 `libcust_opapi.so` 与 CPack 逐字节一致，SHA256 为
`645336f9f65c522a06d74cf8b3df0ca6db2ae493fcd6fec980719eca7c8af07f`。Phase 1/2/3 与独立
`ChunkCumsumKkt` 的执行/GetWorkspaceSize 共 8 个符号齐全；独立/累积 Phase 3 的 4 个 device
object、4 个 per-kernel JSON 和 2 个 config JSON 共 `10/10` 与 CPack 逐字节一致。

只运行原失败 `D_FP16_C64`（`B=1,Hk/Hv=4/8,T=1025,C64,FP16`）accuracy smoke，退出码为
`0`。Phase 3 对不可变 Phase 2 的 output、公开 `g_cumsum`、有效 solved A 和 final-state presence
全部 bit-exact，output max-abs `0`；两条路径所有呈现组件非有限计数均为 `0`。结构化证据：

- `/opt/chw/gdn-phase3-staging-accuracy-smoke-r1/D_FP16_C64.json` SHA256：
  `01b9717e16bfd3bebdc021977c8fee95ced8d3139b5ab980fc35fe8c26f41bcd`；
- `/opt/chw/gdn-phase3-staging-accuracy-smoke-r1/summary.json` SHA256：
  `5582441015f8c7033e2b14737d0a94109ee9405cd2a9f0b5c1a1ca61eebe29f6`。

本小步未运行独立 `80/80`、core 全矩阵、性能或 profiler。由于共享 helper 的 MTE3 staging 已
改变 device 实现，旧独立 exact 证据不能直接沿用，下一步从 dense FP16/C64 单身份开始重取。

### Phase 3 cumsum MTE3 staging 首次非增量完整构建（门禁失败）

本地和 A2 共享 header SHA256 均为
`06256675c03916f51bb0df20bf80a7d4d137d01b6e08cf37cce59c631c47a22e`；构建与产物审计 wrapper
也逐字一致并通过远端 `bash -n`。A2 无本项目构建进程后执行一次非增量完整包构建；构建前 ABI
测试 `11/11 PASS`，且日志确认从空 `build` 重新配置并复制当前源码。

构建在独立 `ChunkCumsumKkt` 的 BF16/C128 路由（tiling key `788`）失败，OPC 最终错误为找不到
`ChunkCumsumKkt_*_mix_aiv_788.o`，`gmake` 退出码为 `2`，外层 wrapper 退出码为 `1`。失败发生在
package、构建拷贝/符号/产物审计之前；本步没有安装包、没有运行 Python/NPU，也没有修改本地源码。
当前末尾错误只是缺失产物症状，不能据此猜测根因；下一小步只提取该路由更早的编译错误。

首次失败后的只读产物审计确认：`ChunkCumsumKkt` index `1` 使用 BF16 参数 JSON，OPC 内部会同时
展开 tiling key `20/788`；失败时生成 wrapper 将 `asc_opc` stdout 捕获在 shell 变量中，且 OPC
清理了失败的 `kernel_meta_ChunkCumsumKkt_*` 目录，因此磁盘上没有可恢复的早期 compiler 日志或
相邻成功 object。现有证据仍不足以冻结源码修复。下一步只重放该一个 CMake target 并持久化日志，
不重复全量构建。

首次串行 target 重放只证明诊断封装缺少环境继承：独立 SSH 进程未 source CANN，生成 wrapper 在
调用前即报 `asc_opc: command not found`；没有进入 OPC 或编译 kernel，因此不计为 route `788` 重现，
也不产生源码结论。下一次使用固定 wrapper 显式加载 CANN/conda，并可靠记录退出码。

环境修正后的串行重放稳定复现，退出码文件为 `2`。首条真实错误位于共享 header staging 行：
`Copy(outputLocal, accLocal, 1)` 在 CANN `9.1.0.beta1` 下没有三参数重载；当前声明要求五参数。
后续所有 `mix_aic/mix_aiv *.o` 缺失均是该编译错误的派生症状。完整诊断证据为：

- `/opt/chw/gdn-phase3-staging-route788-diag.log`
- `/opt/chw/gdn-phase3-staging-route788-diag.rc`

因此最小修复变量冻结为只补齐该 vector `Copy` 的 mask、repeat 和 repeat params，不改变 staging
地址、event、累加表达式、GM 写回、tiling、workspace 或任何 Phase ABI。

### Phase 3 staging `Copy` 构建修复（本地门禁已完成）

仅将 staging 行改为 CANN 9.1 支持且仓内已有用例的五参数形式：
`Copy(outputLocal, accLocal, 1, 1, {1, 1, 8, 8})`。mask=`1`、repeat=`1`，只复制一个 FP32
元素；没有改变数学或同步边界。本地相关 Python `py_compile` 通过、ctypes ABI `11/11 PASS`、
`git diff --check` 通过（仅既有 Windows line-ending 提示）。修复后共享 header SHA256 为
`11a1399ef805ba69306fe9482e86ec202d742d433c2e63dbcd9a9a622f68872c`。本步未同步、构建、安装或
运行 NPU。

### Phase 3 staging `Copy` 修复单 target 构建门禁（已完成）

修复 header 已同步到 A2，本地、远端源码和强制刷新后的 `build/binary` 拷贝 SHA256 均为
`11a1399ef805ba69306fe9482e86ec202d742d433c2e63dbcd9a9a622f68872c`。显式加载 CANN/conda 后，
串行重放 BF16 `chunk_cumsum_kkt_ascend910b_1` target 成功，退出码文件为 `0`；生成聚合 object
`216,416 B` 与 JSON。JSON `kernelList` 明确包含 8 个 tiling key：
`10/266/522/778/20/276/532/788`，证明此前所有 route object 缺失均由三参数 `Copy` 编译失败导致，
五参数修复已越过该门禁。证据文件：

- `/opt/chw/gdn-phase3-staging-route788-diag.log`
- `/opt/chw/gdn-phase3-staging-route788-diag.rc`

本步未安装或运行 NPU。下一步必须从空 `build` 再做一次完整包构建，不能把现有单 target 增量结果
直接当验收包。

### Phase 3 staging 修复后非增量完整构建与产物审计（已完成）

固定 wrapper 从空 `build` 执行完整包构建并成功，退出码文件为 `0`。构建前 ctypes ABI
`11/11 PASS`；源码、`build/binary` 的独立/fused 两份共享 header 和 CPack header SHA256 均为
`11a1399ef805ba69306fe9482e86ec202d742d433c2e63dbcd9a9a622f68872c`。产物为：

```text
run package: 837742c6731143ec0ea55c517338e0daca3d3f295b9b7a71f079805b9b62bfdb
build libcust_opapi.so: 645336f9f65c522a06d74cf8b3df0ca6db2ae493fcd6fec980719eca7c8af07f
```

构建库同时导出 `aclnnChunkCumsumKkt`、`aclnnGdnCoreFwdPhase1/2/3` 及各自
`GetWorkspaceSize` 共 8 个必需符号。CPack 审计严格通过：Phase 3 两个算子各 FP16/BF16
device object 和 per-kernel JSON，共 `OBJECT_COUNT=4`、`KERNEL_JSON_COUNT=4`，另有两个 config
JSON，`CONFIG_JSON_COUNT=2`；所有文件 SHA256 已写入日志。证据：

- `/opt/chw/gdn-phase3-staging-full-build-r2.log`
- `/opt/chw/gdn-phase3-staging-full-build-r2.rc`

本步未安装或运行 NPU。下一步先做安装一致性和原失败单点，不能直接扩到完整 accuracy 矩阵。

### Phase 3 cumsum MTE3 staging 单变量优化（本地门禁已完成）

按冻结卡只修改共享 `ComputePrefixCumsumFromGm`：input ping/pong、`accLocal` 原地 FP32 顺序累加
保持不变；在既有 `rowBrcbBuf_` block 3/4 新增 output staging ping/pong，每次累加后以 vector
`Copy(..., 1)` 逐位复制到 staging，再由按槽 `V_MTE3/MTE3_V` 事件写公开 GM。同槽只在再次被
vector 覆写前等待上一笔 MTE3；任务结束等待 active 槽，最后一笔 MTE3 后只执行一次
`MTE3_MTE2` 闭环。GM、UB 总大小、公开写回顺序、tiling、workspace、KKT/solve、Phase ABI 与调度
均未修改。

本地门禁结果：

- ctypes ABI：`11/11 PASS`；
- 独立 exact、core accuracy/performance/profiler/诊断相关 Python `py_compile`：通过；
- `git diff --check`：通过，仅有既有 Windows line-ending 提示；
- 显式函数窗口确认 staging block 3/4 与 acc/input block 0/1/2 互不重叠，均位于既有 64-float
  buffer 内；row 0 `Adds`、row 1 起原地 `Add` 未变；
- 新共享 header SHA256：
  `06256675c03916f51bb0df20bf80a7d4d137d01b6e08cf37cce59c631c47a22e`。

本小步未同步远端、未构建、未安装或运行 NPU。

### Phase 3 cumsum MTE3 staging 单变量性能修复卡（已冻结）

只读 producer/consumer 审计确认 ping/pong 版本仍在每个 token 上执行：原地 FP32 累加到
`accLocal` -> `V_MTE3` -> MTE3 直接读取同一 `accLocal` -> `MTE3_MTE2` -> `MTE3_V`，下一 token
必须等 MTE3 消费完 `accLocal` 才能继续 vector 累加。该链是当前相对独立 `ChunkLocalCumsum`
新增的最窄逐 token 串行边界。

本轮只允许以下一个变量：

- `accLocal` 继续位于 `rowBrcbBuf_` block 0，raw-g input ping/pong 继续位于 block 1/2；
- 使用同一既有 64-float `rowBrcbBuf_` 的 block 3/4 作为 output staging ping/pong，不扩大 UB；
- row 0 仍为 `Adds(accLocal, inputLocal, 0.0f, 1)`，row 1 起仍为
  `Add(accLocal, accLocal, inputLocal, 1)`；每次累加后用 vector `Copy` 将 1 个 FP32 元素逐位复制
  到当前 staging 槽，不改变累加表达式、输入顺序或公开写回顺序；
- 每个 staging 槽使用独立 `V_MTE3/MTE3_V` event，只在同槽再次被 vector 覆写前等待其上一笔
  MTE3 消费完成；任务结束前等待所有 active staging 槽，保证后续 `ComputeGateBlock` 可安全覆写
  整个 `rowBrcbBuf_`；
- 只在最后一笔公开 MTE3 写回后 set/wait 一次 `MTE3_MTE2`。MTE3 队列有序，末笔事件同时封闭
  此前所有公开写回，之后既有 `CopyTaskVector(gCumsumGm, ...)` 才启动 MTE2；
- 保持 input ping/pong、GM 地址、公开 cumsum 再读作 KKT compute view、tiling、workspace、task/core
  调度、KKT/solve 和 Phase 1/2/3 ABI 不变；不删除其他 barrier，不改数学或融合边界。

仓内 fwd_o/fwd_h 已采用按 ping/pong 槽维护 `V_MTE3/MTE3_*` 并在槽复用/循环末尾等待的同类
A2 模式。若本地门禁通过，后续仍按完整包、原失败 accuracy smoke、共享 helper `80/80 exact`、
core `4+4+1` accuracy、生产性能首点逐层验证；任一 bit-exact/有限性门禁失败立即停止。

### Phase 3 cumsum ping/pong 独立设备性能复制（门禁失败，回退可复现）

在空闲 A2 device 2 使用同一包、同一输入、关闭 `ASCEND_LAUNCH_BLOCKING`，按 4 轮 AB/BA、
8 个干净子进程、每 variant warmup `10`/采样 `50` 独立复制。结果：

| 指标 | Phase 2 | Phase 3 | Phase 3 变化 |
| --- | ---: | ---: | ---: |
| median | `0.940080 ms` | `0.945290 ms` | `+0.554%` |
| P90 | `0.961500 ms` | `0.971940 ms` | `+1.086%` |
| min | `0.826000 ms` | `0.397600 ms` | `-51.864%` |

Phase 2 四轮 batch median 为 `[0.939030,0.933970,0.937470,0.950220] ms`，Phase 3 为
`[0.934610,0.939040,0.947460,0.961240] ms`；Phase 3 在首轮改善、后三轮回退。与 device 1 的
`+0.580%` median 同方向，因此严格门槛失败可复现，不能进入完整矩阵。异常低的单次 min 不作为
主判据；两 Phase workspace/peak 完全相同，各次有限性通过。

证据：

- `summary.json` SHA256：`85770dd7a0f41a7a389273ee622570cf872c459be337637a3c6ae83c3c2546da`；
- 日志 `/opt/chw/gdn-phase3-pingpong-perf-pilot-device2-r2.log` SHA256：
  `6b43219b3abd179326c1bc3fda3006ea8e7311da04e724099f50fdf5e3d86439`。

结束后 device 2 无进程；未修改代码或运行其他 case。下一步继续只优化已由 profiler 证明仍有
空间的融合 cumsum 数据通路，不改外部边界或下游阶段。

### Phase 3 cumsum ping/pong core profiler（已完成）

同一安装包、同一 `D_FP16_C64` contract 下，Phase 2/3 分别在干净进程采集单次 profiler：

- Phase 2 完整 core 为 `9` 个 NPU kernel，device 总时长 `334.667 us`；
- Phase 3 完整 core 为预期的 `8` 个 NPU kernel，device 总时长 `332.767 us`，改善 `1.900 us`；
- Phase 2 目标段 `ChunkLocalCumsum 48.721 us + ChunkKktSolveTri 81.382 us = 130.103 us`；
- Phase 3 `ChunkCumsumKktSolveTri = 119.322 us`，相对目标段改善 `10.780 us`（`8.286%`）；
- 其余 transpose/cast/recompute/fwd_h/fwd_o 路由一致，单次 fwd_h 波动抵消了部分目标段收益。

这证明 ping/pong 优化后的累积融合边界、`9 -> 8` kernel 数和目标段收益均已实现，生产首轮
`+0.580%` median 不能归因于融合 kernel 仍比 Phase 2 两个目标 kernel 合计更慢。单次 profiler
不能替代生产性能，因此下一步只做独立设备复制，不直接扩矩阵。

证据 SHA256：

- `summary.json`：`8f28a56f7a60bb5aaacc7ca570fabeec282fb31c08f159f710edf261aebb1f54`；
- Phase 2 trace：`f2ae3f4c979c90ea05528e6316db0314a383fd503acdc4ccbcb7a5123f59b6d6`；
- Phase 3 trace：`9dc272080b051fed907a3eb56bc7a1377c8cc0607f58f4790e0d0e36842a4407`；
- 日志 `/opt/chw/gdn-phase3-pingpong-core-profiler-r1.log`：
  `f88377515f9273aa20492102e0f3419f435fdf5410ebb6874756c9aefbebca28`。

结束后 device 1 无进程；未修改代码或运行其他性能 case。

### Phase 3 cumsum ping/pong 生产性能首点（门禁失败，已停止扩测）

正式计时关闭 `ASCEND_LAUNCH_BLOCKING`，A2 device 1 上使用同一包和输入，4 轮 AB/BA、8 个干净
子进程，每个 variant warmup `10`、NPU Event 采样 `50` 次，聚合各 `200` 个样本。结果：

| 指标 | Phase 2 | Phase 3 | Phase 3 变化 |
| --- | ---: | ---: | ---: |
| median | `0.941520 ms` | `0.946980 ms` | `+0.580%` |
| P90 | `0.963160 ms` | `0.969940 ms` | `+0.704%` |
| min | `0.829520 ms` | `0.830420 ms` | `+0.109%` |

Phase 2 四轮 batch median 为 `[0.940790,0.938090,0.937030,0.946800] ms`，Phase 3 为
`[0.958890,0.937360,0.937840,0.949370] ms`，方向两胜两负；相比优化前 `+3.470%` median 回退已
显著收窄，但严格“median 不劣于 Phase 2”门槛仍失败，不能以噪声名义直接收口。Phase 2/3 的
workspace max/sum 均为 `137055744 B`，peak allocated delta 均为 `140239360 B`，各次运行有限性
均通过。

证据：

- `summary.json` SHA256：`20c0906f4329ff96fe28f329d18a5ca56360862326a2d7b27cf50a3781320d56`；
- 日志 `/opt/chw/gdn-phase3-pingpong-perf-pilot-r1.log` SHA256：
  `e4a49ee14c8905152ad4c8eb257ad3661c7742fc66fd2ba9d17a2e42a8c4b670`。

按启动卡停止其他性能 case，下一步只做 profiler；结束后 device 1 无进程。

### Phase 3 cumsum ping/pong core state accuracy（`1/1 PASS`）

A2 device 1 上 `STATE_D_BF16_C64` 同时提供 initial state 并请求 final state，退出码为 `0`。
Phase 3 对不可变 Phase 2 的 output、公开 `g_cumsum`、有效 solved A 和 final state 均 bit-exact，
两个 Phase 的全部输出/状态有限。结构化证据 SHA256：

- `STATE_D_BF16_C64.json`：`35d8d78b9d6e3ff127a36b29350a913a4fe299db263526b38b7cd861331891bf`；
- `summary.json`：`ecccd0ca365418bddb4423c8640193bd3d1d16183bcf1e973fcf19b13dc74fb0`；
- 日志 `/opt/chw/gdn-phase3-pingpong-state-accuracy-r1.log`：
  `b64b81a8e76e4829efb31b1689b684081a100999fbeb9706c49e166ab0f0cfce`。

至此本轮累积 Phase 3 core 精度门禁为 dense `4/4` + varlen `4/4` + state `1/1`，可以进入生产
性能首点。结束后 device 1 无进程；本小步未运行性能或 profiler。

### Phase 3 cumsum ping/pong core varlen accuracy（`4/4 PASS`）

A2 device 1 上 `V_FP16_C64`、`V_BF16_C64`、`V_FP16_C128`、`V_BF16_C128` 全部通过。
每例 Phase 3 对不可变 Phase 2 的 output、公开 `g_cumsum`、有效 solved A 和 final-state presence
均 bit-exact，两个 Phase 的所有输出/状态有限性门禁通过。结构化证据 SHA256：

- `V_BF16_C128.json`：`f47d574bdcb4ce1009b3577b97f2ef9155f1246634569bf4c76b2f1fc05b79c5`；
- `V_BF16_C64.json`：`ce7576bb0e7907c0aeb93fda93425674dfdc9a1728904533dad21eecd7ea729b`；
- `V_FP16_C128.json`：`0804fda928078a19519bd3875517c1486c5449e8ba57707cba9217b128fbef5b`；
- `V_FP16_C64.json`：`57fc72580d7db4e81e4928cb79cc3239425b16ae67f20ea331fe19eb5faa27ee`；
- `summary.json`：`95d7095a217fe4cc5209b211ef8df5f211ed8604748e6934bd1729c87f13d6d5`；
- 日志 `/opt/chw/gdn-phase3-pingpong-varlen-accuracy-r1.log`：
  `b5301e80b77b7927e3d8ac3a06d7857efb4afce8efcccf375a7fbe25ced6abff`。

结束后 device 1 无进程。本小步未运行 state、性能或 profiler。

### Phase 3 cumsum ping/pong core dense accuracy（`4/4 PASS`）

A2 device 1 上 `D_FP16_C64`、`D_BF16_C64`、`D_FP16_C128`、`D_BF16_C128` 全部通过。
每例 Phase 3 对不可变 Phase 2 的 output、公开 `g_cumsum`、有效 solved A 和 final-state presence
均 bit-exact，两个 Phase 的所有输出/状态有限性门禁通过。结构化证据 SHA256：

- `D_BF16_C128.json`：`111bb88ad18c328ae0d4999fa63c206aa5cd29cf9479ea35dcc70bd530c5b1fe`；
- `D_BF16_C64.json`：`c2d5d3b599aa3aa7b397198c6186e5966ba87f4751eb7f6defd0f1c605a9f375`；
- `D_FP16_C128.json`：`f605bf80e5ee9369b45a8eded1ea6a9e9dc0867657a3b4f8bb8d191ab64958b5`；
- `D_FP16_C64.json`：`01b9717e16bfd3bebdc021977c8fee95ced8d3139b5ab980fc35fe8c26f41bcd`；
- `summary.json`：`10ecd0b89ae8a2791cba09a8324b0f01973cd2e0c35eaa9d3d4cfcf26b02e521`；
- 日志 `/opt/chw/gdn-phase3-pingpong-dense-accuracy-r1.log`：
  `305c2d0731640732aa6901dac919b00e643330ffba21df2a7e9616a387d0e20d`。

结束后 device 1 无进程。本小步未运行 varlen/state、性能或 profiler。

### Phase 3 cumsum ping/pong 独立 varlen BF16/C128 exact（`10/10 PASS`）

复用 varlen FP16/C128 的 10 个冻结 contract，只改变 `k` dtype 为 BF16；A2 device 1 上融合公开
`g_cumsum` 和 `A_raw` 均与两小算子 NPU 基线全张量逐位一致，输出有限，无效区为零，退出码为
`0`。日志 `/opt/chw/gdn-phase3-pingpong-standalone-v-bf16-c128-exact10-r1.log` SHA256 为
`8f690a8652e4c70e22d965dc0400244e442d031c85bf1364ca29534e9a555bef`，结束后 device 1 无进程。

至此共享 helper 修改后的独立 frozen 8 identities 共 `80/80 exact PASS`：

```text
dense/varlen × FP16/BF16 × C64/C128，每个身份 10 例
```

每例均验证公开 FP32 `g_cumsum`、FP32 `A_raw`、有限性和无效区清零。该结论恢复了共享 helper 的
独立 value evidence；尚未替代累积 Phase 3 core 的完整 accuracy/state/性能/profiler 门禁。

### Phase 3 cumsum ping/pong 独立 varlen FP16/C128 exact（`10/10 PASS`）

复用 varlen C64 的 10 个冻结 contract，只改变 `chunk_size=128`；A2 device 1 上融合公开
`g_cumsum` 和 `A_raw` 均与两小算子 NPU 基线全张量逐位一致，输出有限，无效区为零，退出码为
`0`。日志 `/opt/chw/gdn-phase3-pingpong-standalone-v-fp16-c128-exact10-r1.log` SHA256 为
`04dc6daa1297c9f83349b15c289d516a162c10e8daacbd14e61bf25b20acb88a`，结束后 device 1 无进程。

本小步未运行其他 identity、core 性能或 profiler。

### Phase 3 cumsum ping/pong 独立 varlen BF16/C64 exact（`10/10 PASS`）

复用 varlen FP16/C64 的 10 个冻结 contract，只改变 `k` dtype 为 BF16；A2 device 1 上融合公开
`g_cumsum` 和 `A_raw` 均与两小算子 NPU 基线全张量逐位一致，输出有限，无效区为零，退出码为
`0`。日志 `/opt/chw/gdn-phase3-pingpong-standalone-v-bf16-c64-exact10-r1.log` SHA256 为
`a1756b6d7819e7348464b13a3b9bf2f42c6bc6ee286ca3857bb423b134d79c65`，结束后 device 1 无进程。

本小步未运行其他 identity、core 性能或 profiler。

### Phase 3 cumsum ping/pong 独立 varlen FP16/C64 exact（`10/10 PASS`）

A2 device 1 上重跑冻结的 10 个 varlen contract，覆盖单/多序列、1-token 序列、序列边界恰逢
chunk 边界、跨 chunk 尾块和多头。每例融合公开 `g_cumsum` 和 `A_raw` 均与各自正确元数据粒度
的两小算子 NPU 基线全张量逐位一致，输出有限，无效区为零，退出码为 `0`。日志
`/opt/chw/gdn-phase3-pingpong-standalone-v-fp16-c64-exact10-r1.log` SHA256 为
`8798ce9194f6df4cda6fa517df6dcb943ec6568f806409168de834fd43219b0d`，结束后 device 1 无进程。

本小步未运行其他 identity、core 性能或 profiler。

### Phase 3 cumsum ping/pong 独立 dense BF16/C128 exact（`10/10 PASS`）

复用 dense FP16/C128 的 10 个冻结 contract，只改变 `k` dtype 为 BF16；A2 device 1 上融合公开
`g_cumsum` 和 `A_raw` 均与两小算子 NPU 基线全张量逐位一致，输出有限，无效区为零，退出码为
`0`。日志 `/opt/chw/gdn-phase3-pingpong-standalone-d-bf16-c128-exact10-r1.log` SHA256 为
`4a892fd7828c54fa99e3505675a4ba54a818d8d21a93a269622cd28645464236`，结束后 device 1 无进程。

至此独立 dense 四个完整身份共 `40/40 exact PASS`；尚未由此推导 varlen。未运行 core 性能或
profiler。

### Phase 3 cumsum ping/pong 独立 dense FP16/C128 exact（`10/10 PASS`）

复用 dense C64 的 10 个冻结 contract，只改变 `chunk_size=128`；A2 device 1 上融合公开
`g_cumsum` 和 `A_raw` 均与两小算子 NPU 基线全张量逐位一致，输出有限，无效区为零，退出码为
`0`。日志 `/opt/chw/gdn-phase3-pingpong-standalone-d-fp16-c128-exact10-r1.log` SHA256 为
`4ee9824f78cd5a83fa6538453008d821f93514479d21a42a2d44eaadc9613db3`，结束后 device 1 无进程。

本小步未运行其他 identity、core 性能或 profiler。

### Phase 3 cumsum ping/pong 独立 dense BF16/C64 exact（`10/10 PASS`）

复用 dense FP16/C64 的 10 个冻结 contract，只改变 `k` dtype 为 BF16；A2 device 1 上融合公开
`g_cumsum` 和 `A_raw` 均与两小算子 NPU 基线全张量逐位一致，输出有限，无效区为零，退出码为
`0`。日志 `/opt/chw/gdn-phase3-pingpong-standalone-d-bf16-c64-exact10-r1.log` SHA256 为
`1ea7e44a265276e22b8614601f5b8a58dc181fd3800349ff23109378c7f2306b`，结束后 device 1 无进程。

本小步未运行其他 identity、core 性能或 profiler。

### Phase 3 cumsum ping/pong 独立 dense FP16/C64 exact（`10/10 PASS`）

同一安装包、A2 device 1 上重跑独立 `ChunkCumsumKkt` dense FP16/C64 的 10 个冻结 contract，
覆盖 `T=1/2/63/64/65/127/128/129/193/257`、`B=1/2/3`、`H=1/2/3/4`。每例融合公开
`g_cumsum` 与独立 `ChunkLocalCumsum`、融合 `A_raw` 与两小算子 NPU 基线均全张量逐位一致，
输出有限，严格下三角外区域和尾部 padding 为零；退出码为 `0`。日志
`/opt/chw/gdn-phase3-pingpong-standalone-d-fp16-c64-exact10-r1.log` SHA256 为
`8744aca6fef4390573f79a9f1ce7b7d92999939cace7e3cc6cb1ee2038e1124f`，结束后 device 1 无进程。

本小步未运行其他 identity、core 性能或 profiler。

### Phase 3 cumsum ping/pong 安装与 core accuracy smoke（已完成）

本轮完整包 SHA256 门禁通过后成功安装。安装后 host `libcust_opapi.so` 与包内逐字节一致，SHA256
为 `645336f9f65c522a06d74cf8b3df0ca6db2ae493fcd6fec980719eca7c8af07f`；独立/累积 Phase 3 的
四个 device object、四个 per-kernel JSON 和两个 config JSON 共 `10/10` 与包内文件逐字节一致，
Phase 1/2/3 与独立 `ChunkCumsumKkt` 的 8 个执行/GetWorkspaceSize 符号齐全。

A2 device 1 只运行原失败 `D_FP16_C64` (`B=1,Hk/Hv=4/8,T=1025,C64,FP16`) accuracy smoke，
退出码为 `0`：Phase 3 对不可变 Phase 2 的 output、公开 `g_cumsum`、有效 solved A 和 final-state
presence 全部 bit-exact，output max-abs `0`；两条路径各组件非有限计数均为 `0`。结构化证据：

- `D_FP16_C64.json` SHA256：
  `01b9717e16bfd3bebdc021977c8fee95ced8d3139b5ab980fc35fe8c26f41bcd`；
- `summary.json` SHA256：
  `5582441015f8c7033e2b14737d0a94109ee9405cd2a9f0b5c1a1ca61eebe29f6`；
- 日志 `/opt/chw/gdn-phase3-pingpong-install-smoke-r1.log` SHA256：
  `960625a40e3f2b8a77e9ab281a3b6aeabd2bd10abb527a9bb9f589283884eeb5`。

运行结束后 device 1 无进程；本小步未扩矩阵或运行性能。由于共享 helper 同时进入独立
`ChunkCumsumKkt`，必须重新取得独立 exact 证据，旧 `80/80` 不能直接沿用。

### Phase 3 cumsum ping/pong 完整包构建（已完成）

A2 远端共享 header 与本地 SHA256 一致，固定 wrapper `bash -n` 通过；随后在 `chw-py11`、CANN
`9.1.0.beta1` 下执行一次非增量完整包构建，退出码为 `0`。源码、两份 `build/binary` 拷贝和 CPack
源码拷贝的共享 header SHA256 均为
`317cf0558f6e6ffc8fefd0cd18064af5e96886f82a089afd06e12cb4c794014a`。

构建证据：

- run 包 SHA256：`3b581d2594dbf2f7d6a54a368b52cf7447fcb3f732b7f166d58e870a510e2be5`；
- host `libcust_opapi.so` SHA256：
  `645336f9f65c522a06d74cf8b3df0ca6db2ae493fcd6fec980719eca7c8af07f`；
- 包内包含独立/累积 Phase 3 各 FP16/BF16 两个 device object、四个对应 per-kernel JSON 和两个
  config JSON；
- `aclnnChunkCumsumKkt`、`aclnnGdnCoreFwdPhase1/2/3` 的执行与 `GetWorkspaceSize` 共 8 个 host
  符号齐全；
- 构建日志 `/opt/chw/gdn-phase3-pingpong-build-r1.log` SHA256：
  `6010f5f79962db460e0091a9d24bd680166b63e3738dfd948cea15a1e686afa3`。

构建期间发现另一用户在不同工作目录构建独立算子；未干预该进程，本项目使用独立源码和构建
目录，不构成产物混用。A2 device 1 无运行进程。本小步未安装或运行 NPU。

### Phase 3 cumsum ping/pong 单变量优化（本地门禁已完成）

按冻结卡只修改共享 `ComputePrefixCumsumFromGm`：`accLocal` 仍位于 `rowBrcbBuf_` block 0，raw-g
输入改用既有 block 1/2 两个 32-byte 对齐 ping/pong 槽，并分别维护 `V_MTE2/MTE2_V` event；任务
入口先封闭上一 task 对共享 buffer 的 gate/vector 使用，循环中预取下一 token，并只在同槽再次被
MTE2 覆写前等待上一轮 vector 消费完成。row 0 `Adds`、row 1 起原地 `Add`、逐 token FP32
公开写回和 `V_MTE3/MTE3_MTE2/MTE3_V` 顺序未变；GM、UB 总大小、tiling、workspace、KKT/solve、
Phase 1/2/3 ABI 与调度均未修改。

本地门禁结果：

- ctypes ABI：`11/11 PASS`；
- 相关独立测试、accuracy/performance/profiler/诊断 Python 脚本 `py_compile`：通过；
- `git diff --check`：通过，仅有既有 Windows line-ending 提示；
- 显式函数窗口核对：两个输入槽互不重叠且均在既有 64-float `rowBrcbBuf_` 内，数学与 GM 写回
  顺序保持不变；
- 新共享 header SHA256：
  `317cf0558f6e6ffc8fefd0cd18064af5e96886f82a089afd06e12cb4c794014a`。

本小步未同步远端、未构建、未安装或运行 NPU。

### Phase 3 cumsum MTE2/vector 单变量性能修复卡（已冻结）

只读审计确认 `ComputePrefixCumsumFromGm` 当前将 `accLocal` 和唯一 `inputLocal` 分别放在
`rowBrcbBuf_` 的前两个 32-byte block。每个 token 都在写该 `inputLocal` 前同步执行一组
`V_MTE2` set/wait，再执行 MTE2、`MTE2_V` 和顺序 FP32 vector 累加。该依赖修复了已证明的
stale-beta/WAW 问题，但 profiler 中 Phase 3 累积 kernel 比 Phase 2 两个目标 kernel 合计慢
`23.560 us`（`18.46%`），且生产首点 median 回退 `3.470%`；当前最明确的新增串行边界正是
逐 token 复用同一输入 block。

本轮只允许以下一个变量：

- `accLocal` 仍使用 `rowBrcbBuf_` block 0；raw-g 输入使用 block 1/2 两个既有、互不重叠且
  32-byte 对齐的 ping/pong 槽，不扩大任何 UB buffer；
- 每个槽使用独立 `V_MTE2` 与 `MTE2_V` event。任务开始时先封闭上一 task 对共享
  `rowBrcbBuf_` 的 gate/vector 写入；循环中只在同一槽被下一次 MTE2 覆写前等待该槽上一轮
  vector 消费完成，并预取下一 token 到另一槽；
- 数学顺序严格保持 row 0 `Adds(accLocal, inputLocal, 0.0f, 1)`，row 1 起
  `Add(accLocal, accLocal, inputLocal, 1)`，每次累加后的公开 FP32 写回顺序不变；
- 保留已有 `V_MTE3/MTE3_MTE2/MTE3_V`、公开 `g_cumsum` 再读作 KKT compute view、GM 地址、
  tiling key、task/core 调度、workspace、KKT/solve 和 Phase 1/2/3 ABI；不顺手优化其他 barrier。

若本地门禁通过，下一层只做源码三份 SHA256、完整包构建、符号和原失败点 smoke；若任一精度
签名不再 bit-exact，立即回退到本卡重新审计，不通过放宽阈值或缩小 case 制造通过结论。

### Phase 3 `D_FP16_C64` profiler 根因（已完成）

同一安装包、同一 contract 下，Phase 2/3 分别在干净进程采集单次 profiler。离线
解析证明：

- Phase 2 完整 core 为 `9` 个 NPU kernel，device 总时长 `338.527 us`；
- Phase 3 完整 core 为 `8` 个 NPU kernel，device 总时长 `379.288 us`；
- Phase 2 `ChunkLocalCumsum` 为 `44.281 us`，`ChunkKktSolveTri` 为 `83.382 us`，合计
  `127.663 us`；
- Phase 3 `ChunkCumsumKktSolveTri` 为 `151.223 us`，比上述两 kernel 合计慢
  `23.560 us` (`18.46%`)；
- 因此 `local_cumsum + KKT + solve_tri` 的累积融合边界和 `9 -> 8` kernel 数均已实现，
  性能回退集中在新融合 kernel，不是调度仍调用旧 `ChunkLocalCumsum` 或 `ChunkKktSolveTri`。

结构化 profiler `summary.json` SHA256 为
`3c1c2642cd18dff1c12171fefe63075144b8c04a5cce2681b60f2e34cf251367`；日志
`/opt/chw/gdn-phase3-v-mte2-core-profiler-r1.log` SHA256 为
`288428e8530f3a1ffd1e693c16029885c0f8630c3fc1211b969b6cbc5bb9ffb0`。结束后 device 1 无本项目
残留进程。

当前高概率瓶颈是正确性修复将 `V_MTE2` set/wait 放在每个 token 的顺序循环中，使
MTE2 和 vector 读取全部串行化；该结论尚需通过源码对照冻结最小修复，本小步不直接删除依赖。

### Phase 3 stale-beta `V_MTE2` 修复生产性能首点（门禁失败，已停止扩测）

运行前只读确认 A2 device 1 无运行进程、AIC/AIV 利用率为 `0%`。正式计时关闭
`ASCEND_LAUNCH_BLOCKING`，warmup `10`、每个干净子进程 `50` 轮 NPU Event 计时，
4 轮 AB/BA 平衡聚合，每个 Phase 共 `200` 个样本。

`D_FP16_C64` 结果：

| 指标 | Phase 2 | Phase 3 | Phase 3 变化 |
| --- | ---: | ---: | ---: |
| median | `0.953260 ms` | `0.986340 ms` | `+3.470%` |
| P90 | `1.027260 ms` | `1.043020 ms` | `+1.534%` |
| min | `0.814660 ms` | `0.850360 ms` | `+4.382%` |

Phase 2 四轮 batch median 为
`[0.953710,1.023720,0.949380,0.944600] ms`，Phase 3 为
`[1.038430,0.984020,0.979850,0.976760] ms`，3/4 轮方向为 Phase 3 更慢。Phase 2/3 的
workspace max/sum 均为 `137055744 B`，peak allocated delta 均为 `140239360 B`，因此显存无回退，
但 median 不劣于 Phase 2 的收口门槛失败。

证据：

- `summary.json` SHA256：`b5ebae3baff9788511b50de1b0c7b202dfbfe997cfb95084c2231a89a8e36420`；
- 日志 `/opt/chw/gdn-phase3-v-mte2-perf-pilot-r1.log` SHA256：
  `fa9ad105cd8d9477f9df8b22764c4b094cb59fa1cddf9d730c6d8bb8b359b055`。

本轮退出码为 `0`，说明 runner 与数据生成完整，不代表性能门禁通过。按启动卡停止
其他性能 case，下一步只做单点 profiler 根因定位。结束后 device 1 无本项目残留进程；
远端其他用户的 `evalscope` 进程未被修改或终止。

### Phase 3 stale-beta `V_MTE2` 修复 state accuracy（`1/1` 已完成）

A2 device 1 上 `STATE_D_BF16_C64` (`B=2,H=4,T=1024,C=64`) 同时提供 initial state
并请求 final state，退出码为 `0`。Phase 3 对不可变 Phase 2 的 output、公开
`g_cumsum`、有效 solved A 和 final state 均 bit-exact，两个 Phase 的全部输出/状态均有限。

证据 SHA256：

- `STATE_D_BF16_C64.json`：`35d8d78b9d6e3ff127a36b29350a913a4fe299db263526b38b7cd861331891bf`；
- `summary.json`：`ecccd0ca365418bddb4423c8640193bd3d1d16183bcf1e973fcf19b13dc74fb0`；
- 日志 `/opt/chw/gdn-phase3-v-mte2-state-accuracy-r1.log`：
  `db08393c38931ede415656e5a72db5589b7c10c7ade7618c3a8b2bb5265fa193`。

至此 Phase 3 累积融合 core 精度门禁为 dense `4/4` + varlen `4/4` + state `1/1`，
可进入生产性能首点。结束后 device 1 无本项目残留进程。

### Phase 3 stale-beta `V_MTE2` 修复 varlen accuracy 矩阵（`4/4` 已完成）

A2 device 1 上 `V_FP16_C64`、`V_BF16_C64`、`V_FP16_C128`、`V_BF16_C128`
全部通过。每例 Phase 3 对不可变 Phase 2 的 output、公开 `g_cumsum`、有效 solved A
和 final state 均 bit-exact，两个 Phase 的输出/状态有限性门禁全部通过。

结构化证据 SHA256：

- `V_BF16_C64.json`：`ce7576bb0e7907c0aeb93fda93425674dfdc9a1728904533dad21eecd7ea729b`；
- `V_FP16_C64.json`：`57fc72580d7db4e81e4928cb79cc3239425b16ae67f20ea331fe19eb5faa27ee`；
- `V_BF16_C128.json`：`f47d574bdcb4ce1009b3577b97f2ef9155f1246634569bf4c76b2f1fc05b79c5`；
- `V_FP16_C128.json`：`0804fda928078a19519bd3875517c1486c5449e8ba57707cba9217b128fbef5b`；
- `summary.json`：`95d7095a217fe4cc5209b211ef8df5f211ed8604748e6934bd1729c87f13d6d5`；
- 日志 `/opt/chw/gdn-phase3-v-mte2-varlen-accuracy-r1.log`：
  `f3da8b1d849690a0ebad8f1908af1b965b7f7a40d954c9508762e8d7b2a0c9aa`。

矩阵退出码为 `0`，结束后 device 1 无本项目残留进程。远端同时存在不属于本项目的
`evalscope` 进程，本轮未终止、修改或利用该进程。

### Phase 3 stale-beta `V_MTE2` 修复 dense accuracy 矩阵（`4/4` 已完成）

A2 device 1 上 `D_FP16_C64`、`D_BF16_C64`、`D_FP16_C128`、`D_BF16_C128`
全部通过。每例 Phase 3 对不可变 Phase 2 的 output、公开 `g_cumsum`、有效 solved A
和 final state 均 bit-exact，两个 Phase 的输出/状态有限性门禁全部通过。

结构化证据 SHA256：

- `D_BF16_C64.json`：`c2d5d3b599aa3aa7b397198c6186e5966ba87f4751eb7f6defd0f1c605a9f375`；
- `D_FP16_C64.json`：`01b9717e16bfd3bebdc021977c8fee95ced8d3139b5ab980fc35fe8c26f41bcd`；
- `D_BF16_C128.json`：`111bb88ad18c328ae0d4999fa63c206aa5cd29cf9479ea35dcc70bd530c5b1fe`；
- `D_FP16_C128.json`：`f605bf80e5ee9369b45a8eded1ea6a9e9dc0867657a3b4f8bb8d191ab64958b5`；
- `summary.json`：`10ecd0b89ae8a2791cba09a8324b0f01973cd2e0c35eaa9d3d4cfcf26b02e521`；
- 日志 `/opt/chw/gdn-phase3-v-mte2-dense-accuracy-r1.log`：
  `780d579456d544c1945a3c4cef95151c5f666d874f901ad8a09fe273a59fb928`。

矩阵退出码为 `0`，结束后 device 1 无本项目残留进程。随后的只读内联 Python 摘要
命令被 PowerShell/SSH 剥离字符串引号而失败；它不重跑 NPU、不修改 JSON，已将
`summary.json` 拉回本地用 PowerShell 结构化解析，再次确认四例的五个 bit-exact 分量均为 `true`。

### Phase 3 stale-beta `V_MTE2` 修复 `D_FP16_C64` accuracy pilot（已完成）

A2 device 1 上用固定 seed 生成一份 CPU 输入，Phase 2/3 在两个干净 Python 进程中
分别执行并同步成功，退出码为 `0`。原失败
`D_FP16_C64` (`B=1,Hk/Hv=4/8,T=1025,C=64`) 结果为：

- 独立 `ChunkLocalCumsum` 对 Phase 2 和 Phase 3 公开 `g_cumsum` 均全张量 bit-exact，
  差异元素数从修复前 `320` 降为 `0`；
- Phase 2/3 `g_cumsum` 逐位一致，max-abs `0`，全部有限；
- solved A 全张量 `524800/524800` 和有效区域 `258048/258048` 逐位一致，
  max-abs `0`，全部有限；
- output 逐位一致，max-abs `0`，全部有限；final-state presence 一致；
- `g_difference_groups` 为空，证明前一 task tail beta 污染签名已消失。

证据 SHA256：

- `inputs.pt`：`0ed4e745831dcff5b6a083e72c9fe83b51a062a09e8fc9a1092aa2bf4cbc8b73`；
- `phase2.pt`：`f2d4e1412dcaca67368bf1156a66b4138e709fbd16f37b3d2f880dad3418da59`；
- `phase3.pt`：`8b23cd03b78a802c271ed09a499e5406e041ce0bf7cf3e8b582818ec5f47455d`；
- `summary.json`：`f251c3b5dfa5474ce355f90a6211323bfde1c55c5bf4b31e64aeddb8dc81235a`；
- 日志 `/opt/chw/gdn-phase3-v-mte2-accuracy-pilot-r1.log`：
  `9903f3fe30cac7e4c2a7d1fe0c47fa7a850f12fec978d77968dc663830f58305`。

本小步未运行其他 identity、性能或 profiler，结束后 device 1 无本项目残留进程。

### Phase 3 stale-beta `V_MTE2` 修复包安装与逐文件审计（已完成）

完整 run 包
`3c85fd1f39aa1631cf14e1c91c483bf847d389cc9804270ac0ec53bc78e50f04` 已成功合并安装到
A2 `chw-py11` wheel OPP 目录。安装后门禁结果：

- 安装后 `libcust_opapi.so` 与包内文件 `cmp` 一致，SHA256 为
  `645336f9f65c522a06d74cf8b3df0ca6db2ae493fcd6fec980719eca7c8af07f`；
- Phase 1/2/3 与独立 `ChunkCumsumKkt` 共 8 个执行/`GetWorkspaceSize` 符号齐全；
- `ChunkCumsumKkt`/`ChunkCumsumKktSolveTri` 的 4 个 device object、4 个 kernel JSON 和
  2 个 config JSON 共 10 个文件全部存在，并与包内文件逐个 `cmp` 一致；
- 安装器对内部 `chunk_cumsum_kkt_solve_tri` 提示未找到独立 ACLNN ABI header；该算子是
  Phase 3 core 的 L0 内部调用，本轮 host 符号与包内/安装后 device 文件门禁均通过；
- 审计 wrapper 本地/远程 SHA256 一致，远程 `bash -n` 通过；
- 日志 `/opt/chw/gdn-phase3-v-mte2-install-audit-r2.log` SHA256 为
  `715de2506bf54ef5ccb0b737216e969dcb68c26606255e97b48e6fdbcd7a8b87`。

本小步未启动 Python 算子或 NPU，结束后无本项目残留进程。

首次安装尝试的 run 包 SHA256 门禁通过，但固定 wrapper 未在安装前激活
`chw-py11`，安装器因无法定位已安装 wheel 的 vendor 目录而退出码 `1`。该次未进入包内/
安装后文件比较，未启动 Python 算子或 NPU。证据日志
`/opt/chw/gdn-phase3-v-mte2-install-audit-r1.log` SHA256 为
`86daf9994097929c261d31e35991f2a0064f5e10acb2a208d1df72624aa71392`。下一次仍只重试同一安装门禁，
唯一变量是在 wrapper 中先 source CANN 并激活 `chw-py11`。

### Phase 3 stale-beta `V_MTE2` 修复完整包构建（已完成）

本地/远程共享 header 和两个固定 wrapper SHA256 一致，远程 `bash -n`、
Python 语法和 ctypes ABI `11/11 PASS` 后，A2 上一次非增量完整包构建成功，退出码为 `0`。

产物门禁结果：

- 本地、远程源码、`build/binary` 的 `ChunkScaledDotKkt`/`ChunkCumsumKkt` 两份
  共享 header 和 CPack 源码拷贝 SHA256 全部为
  `8f656a7aab953bd0e496b86fda9bd7003d6eb38ca84bd52fd23ebf192c23503e`；
- 包内 `ChunkCumsumKkt` 与 `ChunkCumsumKktSolveTri` 各有 FP16/BF16 两个 device object
  和两个 kernel JSON，另有两个顶层 config JSON；计数门禁为 `4/4/2 PASS`；
- 四个 device object SHA256 为
  `9b36e830fc909d572f78003be597b43e7e5a93f5a09a84f621af40cad1c5b26c`、
  `1d17407bf179c85f153e8f85733c1cedf55a0145e925d60879f130aa8454ed13`、
  `e2ab749803838459547d3d8fa4be75bae2aa83782e1d846b6d4e25c27dda9e33`、
  `cae49996f749f3676d9732351c29d7ffdb5d6538ab8646f2d6529eacb05e1e57`；
- Phase 1/2/3 与独立 `ChunkCumsumKkt` 共 8 个执行/`GetWorkspaceSize` host 符号齐全；
- run 包 SHA256：
  `3c85fd1f39aa1631cf14e1c91c483bf847d389cc9804270ac0ec53bc78e50f04`；
- host `libcust_opapi.so` SHA256：
  `645336f9f65c522a06d74cf8b3df0ca6db2ae493fcd6fec980719eca7c8af07f`；
- 构建日志 `/opt/chw/gdn-phase3-v-mte2-build-r1.log` SHA256：
  `d67cc6ce378e55c9d077087e0cc90207b012f4afdbfafe09f8d67b130455b39a`。

首次启动封装因 PowerShell/SSH 对远程变量引号处理不完整，远程 Bash 在解析阶段
退出，未进入固定 wrapper；确认无构建进程后改为直接执行 wrapper，上述正式构建成功。
本小步未安装新包或运行 NPU。

### Phase 3 stale-beta `V_MTE2` WAW 修复（本地门禁已完成）

共享 `ComputePrefixCumsumFromGm` 在每轮 `DataCopyPad(inputLocal, ...)` 之前新增
`V_MTE2` set/wait，使首轮封闭上一 task 的 gate `Brcb(beta)` 对同一
`rowBrcbBuf_` 的写入，后续每轮封闭上一轮 vector 对 `inputLocal` 的读取。
cumsum 数学、GM 地址、已有 `MTE2_V/V_MTE3/MTE3_MTE2/MTE3_V` 顺序、KKT/solve、
tiling、workspace 和 Phase 1/2/3 调度均未修改。

本地门禁结果：

- ctypes ABI：`11/11 PASS`；
- 累积精度诊断、profiler 诊断、smoke、正式 benchmark/matrix runner 和独立算子测试
  `py_compile` 通过；
- `git diff --check` 通过，仅有既有 Windows line-ending 提示；
- 显式函数行窗口静态核对确认 `V_MTE2` wait 在 `DataCopyPad(inputLocal, ...)` 前，row 0
  仍使用 `Adds(accLocal, inputLocal, 0.0f, 1)`，row 1 起仍使用原地 `Add`；
- 新共享 header SHA256：
  `8f656a7aab953bd0e496b86fda9bd7003d6eb38ca84bd52fd23ebf192c23503e`。

首次自动静态核对因用 LF 正则提取 CRLF 函数区段失败而误报 `false`；该命令没有
修改文件，随后改用显式行窗口复核全部通过。本小步未同步远程、未构建、未安装或运行 NPU。

### Phase 3 stale-beta 静态/二进制路由审计（已完成）

新包的两份累积 device `.o` SHA256 为
`ae38813981376f7e1a29bbb0c99aec87c23804c583f81ac3fa4aa2cc8e6e564d` 和
`a5c925e23f10ccedd3901f6c574e5720c24cecd1e81df45447f07e13227e4e28`；上一版对应 `.o` 为
`cc1cd6f14ef52f0899e6457fb55d95a491e56bd620d73b3b661c9c2f4f782faf` 和
`362d43899709ab31e83db5712f66164e5a50cd0c230669f80c01c90acc6ea6c1`。本轮非增量构建只修改
共享 header，生成日志明确重新编译 `ChunkCumsumKktSolveTri` 两个二进制，因此可以排除“新源码未进入
累积二进制”。JSON 绑定两份 object 的 tiling key `10/266/20/276`、`MIX_AIC 1:2` 路由，包内与
安装后 `.o`/JSON 已逐文件一致；这也排除错包或 host dispatch 到旧 object。

静态 producer/consumer 审计得到第一处仍未闭合的依赖：

1. `rowBrcbBuf_` 的前两个 32-byte block 在 cumsum 中分别作为 `accLocal` 和 `inputLocal`；
2. 同一 buffer 随后在 `ComputeGateBlock` 被 vector `Brcb(g)`、最终被 `Brcb(beta)` 覆写；
3. 下一 task 的 cumsum 立即由 MTE2 `DataCopyPad` 重写 `inputLocal`，每一 row 也重复重写该地址；
4. 当前只有写后读 `MTE2_V`，没有读/写后写 `V_MTE2`。因此它能保证新 MTE2 数据在后续 vector
   使用前就绪，却不能保证上一 row 的 `Adds/Add` 已读完 `inputLocal`，也不能保证上一 task 的
   `Brcb(beta)` 已写完同一 UB 后才开始新 MTE2；
5. 仓内 KDA 的同类 `CopyVectorIn -> vector op -> 复用 UB` 路径在 `MTE2_V` 后显式成对使用
   `V_MTE2`，证实该 WAW/复用依赖是已采用的 A2 模式。

该缺口与动态签名完全吻合：只有同一 AIV 从前一 head 的 1-token tail task 切到下一 head C64 task
时暴露，坏 task 首值逐位等于前一 task 最后 beta。结论从“stale beta 来源相关”推进为：实际二进制
执行了新首行赋值，但共享 UB 在 gate/vector 与下一 task/row MTE2 之间缺少 `V_MTE2`，使首行源在
MTE2/vector WAW 竞态下仍可读到 stale beta。下一步只补这一条已由源码和动态值共同支持的依赖。

### Phase 3 stale-beta 新包分量级定点诊断（已完成）

诊断 Python 本地 `py_compile` 和 wrapper `git diff --check` 通过；本地/远端 Python 与 wrapper
SHA256 分别一致，远端 `bash -n` 通过。首次预检在 conda 加载前调用登录环境不存在的 `python`，
因 `&&` 提前终止，未生成输入、未运行 NPU；随后只保留 `bash -n`/哈希预检，由固定 wrapper 加载
`chw-py11` 后执行。A2 device 1 上 Phase 2/3 分别在干净 Python 进程中同步成功，退出码为 `0`，
结束后 device 1 无残留进程。

新包差异签名与上一轮逐文件一致：

- Phase 2 `g_cumsum` 与独立 `ChunkLocalCumsum` 全张量逐位一致；
- Phase 3 `g_cumsum` 仍有 `320/8200` 个差异，首个 `[0,64,1]`，max-abs
  `0.6076866388320923`，两侧全部有限；
- 差异仍仅为 head `1/3/4/5/6` 的完整 C64 chunk 1，共 5 组，每组 64 个元素；五组 Phase 3
  首值都逐位等于同一 AIV 前一 head 1-token tail task 的最后 beta；
- Phase 2/3 solved A 全张量 `524800/524800` bit-exact，有效区域 `258048/258048` bit-exact，
  max-abs `0`，全部有限；
- output 仍有 `95,049` 个差异，首个 `[0,1,64,0]`，max-abs
  `0.0005898475646972656`，两侧全部有限；final-state presence 一致。

`phase2.pt`、`phase3.pt` 和 `summary.json` SHA256 分别仍为
`f2d4e1412dcaca67368bf1156a66b4138e709fbd16f37b3d2f880dad3418da59`、
`785a09d8e3cb3f446c8dc02d08964479fce8a1f138e852d3a996ca4862636104`、
`1ede56963c4e94e01cf3f278bb6a5c6bbc572fe9dc7a1cbfcd8be98eca38b52a`，与旧诊断完全相同。
本轮输入 SHA256 为 `0ed4e745831dcff5b6a083e72c9fe83b51a062a09e8fc9a1092aa2bf4cbc8b73`，日志
`/opt/chw/gdn-phase3-cumulative-core-accuracy-diag-r3.log` SHA256 为
`d45ce3312eadaed72faa74222bcf8e9ea90ef1d4f01257018c8feb287a95904f`。

因此源码级“row 0 改用 `Adds`”没有改变实际累积路径数值签名。下一步必须先把新源码绑定到实际
device 二进制/执行路径，并检查跨 task 的 pipeline 依赖；在这层证据完成前不做第三次猜测性修复。

### Phase 3 stale-beta 首行赋值修复安装与 accuracy pilot（门禁失败，已停止扩测）

安装/accuracy wrapper 本地静态检查、runner `py_compile`、ctypes ABI `11/11 PASS`，本地/远端
wrapper SHA256 一致且远端 `bash -n` 通过。A2 device 1 启动前空闲；完整 run 包
`8d87c2cc4871c87cc2fe847a5158db025a24f68281bc6ad8144310694cd52a59` 安装成功。安装后的
host 库 SHA256 为 `645336f9f65c522a06d74cf8b3df0ca6db2ae493fcd6fec980719eca7c8af07f`，与包内逐文件
`cmp` 一致；Phase 1/2/3 与独立 `ChunkCumsumKkt` 共 8 个 host 符号齐全，两类 Phase 3 kernel
的 4 个 device `.o` 和 4 个 per-kernel JSON 均与包内逐文件一致。

随后只运行原失败 `D_FP16_C64` accuracy pilot。两条 Phase 路径均能调用和同步，但门禁仍失败：
final-state presence 一致，output、`g_cumsum` 和有效 solved A 均不 bit-exact，output max-abs 为
`0.0006310939788818359`，与上一版失败 pilot 相同。runner 在首个 case 抛出断言并退出 `1`，
未生成 case JSON/summary，未扩其他 identity、性能或 profiler；device 1 结束后无残留进程。

证据日志 `/opt/chw/gdn-phase3-stale-beta-first-row-install-pilot-r2.log` SHA256 为
`08e20e7b9c6b19d5870b78bdf03d396679bb733107fd56c84b615958a0d1ccde`，退出码文件内容为 `1`。
该结果证明 task 首行改用 `Adds` 仍不足以修复累积 Phase 3；下一步必须重新取得新包的分量级差异
签名，不能沿用旧诊断或继续猜测同步/UB 根因。

### Phase 3 stale-beta 首行赋值修复完整包构建（已完成）

只同步共享 `chunk_scaled_dot_kkt.h` 后，本地与 A2 隔离源码树 SHA256 均为
`b05c2428b455678ff91b39f96a9fcc76d30231d6b00a4f459f1faa2ea39cded9`。远端完整构建 wrapper
`bash -n` 通过，本地/远端 wrapper SHA256 一致；本地 ctypes ABI `11/11 PASS`、`git diff --check`
通过。A2 执行一次非增量完整包构建，SSH/构建退出码均为 `0`，本步未安装、未运行 NPU。

构建后源码、`build/binary` 中 `ChunkScaledDotKkt` 与 `ChunkCumsumKkt` 两份 header，以及 CPack
源码拷贝 SHA256 均为上述 `b05c...ded9`。run 包 SHA256 为
`8d87c2cc4871c87cc2fe847a5158db025a24f68281bc6ad8144310694cd52a59`；host
`libcust_opapi.so` SHA256 为
`645336f9f65c522a06d74cf8b3df0ca6db2ae493fcd6fec980719eca7c8af07f`。包内精确包含：

- `ChunkCumsumKkt` FP16/BF16 两个 device `.o` 和两个 per-kernel JSON；
- `ChunkCumsumKktSolveTri` FP16/BF16 两个 device `.o` 和两个 per-kernel JSON；
- 两类算子各一个顶层 config JSON；
- Phase 1/2/3 与独立 `ChunkCumsumKkt` 共 8 个执行/GetWorkspaceSize host 符号。

构建日志 `/opt/chw/gdn-phase3-stale-beta-first-row-build-r2.log` SHA256 为
`088027f766db96f0edf82919e52ba9a60b50528d1ea6e0bce92747ca31231db7`，退出码文件内容为 `0`。
只读产物审计脚本在本地/远端 SHA256 一致且远端 `bash -n` 通过后执行，计数门禁
`OBJECT_COUNT=4`、`KERNEL_JSON_COUNT=4`、`CONFIG_JSON_COUNT=2` 全部通过。

### Phase 3 stale-beta 首行赋值最小修复（本地门禁已完成）

共享 `ComputePrefixCumsumFromGm` 已删除未奏效的任务前 `Duplicate(accLocal, 0)`；每个 task 的 row 0
改为 `Adds(accLocal, inputLocal, 0.0f, 1)`，row 1 起仍为原地 `Add`。这严格复用已验证
`ChunkLocalCumsum::ProcessSequenceChunk` 的首元素初始化语义。现有 MTE2/V/MTE3 事件、GM 地址、
UB buffer、tiling、KKT/solve、workspace 和 Phase 1/2/3 调度均未修改。

本地门禁结果：

- ctypes ABI：`11/11 PASS`；
- 诊断、正式 benchmark 和 Phase 3 matrix runner `py_compile`：通过；
- `git diff --check`：通过，仅有既有 Windows line-ending 提示；
- 静态核对确认 fused 分支 row 0 走 `Adds`、后续 row 走 `Add`，非 fused 分支不变；
- 新共享 header SHA256：
  `b05c2428b455678ff91b39f96a9fcc76d30231d6b00a4f459f1faa2ea39cded9`。

该哈希与此前独立 `ChunkCumsumKkt` exact 通过时的共享 header 相同，但累积 Phase 3 使用不同 task
调度，旧证据不能替代重新构建后的累积 core pilot。本小步未同步、构建、安装或运行 NPU。

### Phase 3 stale-beta 修复后分量级定点诊断（已完成）

使用已安装修复包、固定 seed 和同一 CPU 输入，在两个干净 Python 进程中分别运行 Phase 2/3；
两条路径均成功同步退出，device 1 运行后无残留进程。修复后差异签名与修复前完全一致：

- Phase 3 公开 `g_cumsum` 仍有 `320/8200` 个元素不同，首个差异 `[0,64,1]`，max-abs
  `0.6076866388320923`，双方全部有限；
- 差异仍严格集中在 head `1/3/4/5/6` 的完整 C64 chunk 1，共 5 组，每组 64 个元素；
- 五个坏块首值仍逐位等于前一 head 的 1-token tail task 最后一个 beta；
- solved A 全张量和有效区域均 bit-exact，max-abs `0`，双方全部有限；
- output 有 `95,049` 个元素不同，max-abs `5.898475646972656e-4`，双方全部有限；
- final-state presence 一致。

这证明预先 `Duplicate(accLocal, 0)` 加统一原地 `Add` 没有消除任务首行的 stale source。由于 solved
A 对块内常量偏移不敏感，它恢复 exact 不代表公开 cumsum 正确。仓内已验证
`ChunkLocalCumsum::ProcessSequenceChunk` 不依赖预先清零：首行直接以
`Adds(accLocal, rowLocal, 0.0f, count)` 赋值，后续行才原地累加。下一次最小修改只恢复这条首行
赋值语义，不新增猜测性 barrier。

证据文件及 SHA256：

- `/opt/chw/gdn-phase3-cumulative-core-accuracy-diag-r2/summary.json`：
  `1ede56963c4e94e01cf3f278bb6a5c6bbc572fe9dc7a1cbfcd8be98eca38b52a`；
- `/opt/chw/gdn-phase3-cumulative-core-accuracy-diag-r2/phase2.pt`：
  `f2d4e1412dcaca67368bf1156a66b4138e709fbd16f37b3d2f880dad3418da59`；
- `/opt/chw/gdn-phase3-cumulative-core-accuracy-diag-r2/phase3.pt`：
  `785a09d8e3cb3f446c8dc02d08964479fce8a1f138e852d3a996ca4862636104`；
- 诊断日志 `/opt/chw/gdn-phase3-cumulative-core-accuracy-diag-r2.log`：
  `7ba6f22eda7d4353221801e85bb73ad474e4126a9d141729e8850df668ac18f6`，退出码 `0`。

### Phase 3 stale-beta 修复包安装与 D_FP16_C64 accuracy pilot（门禁失败，已停止扩测）

完整 run 包
`a243127a8d2228b68621a7220012191e6cf0d8d6be5a8f2b9c3ba9d6e573fac5` 已成功安装到 A2
`chw-py11` 环境。安装 wrapper 逐文件 `cmp` 证明已安装 `libcust_opapi.so`、
`ChunkCumsumKkt`/`ChunkCumsumKktSolveTri` 的 FP16/BF16 四个 device `.o` 和四个 kernel JSON
均与包内文件一致；host 库 SHA256 为
`645336f9f65c522a06d74cf8b3df0ca6db2ae493fcd6fec980719eca7c8af07f`。Phase 1/2/3 和独立
`ChunkCumsumKkt` 共 8 个执行/GetWorkspaceSize 符号齐全。

随后只运行原失败 `D_FP16_C64` (`B=1,Hk/Hv=4/8,T=1025,C=64`) accuracy pilot。门禁仍失败并
立即停止，未生成 case JSON、未扩其他 identity、未运行性能或 profiler：final-state presence
一致，但 output、`g_cumsum` 和有效 solved A 均未与 Phase 2 bit-exact，output max-abs 为
`6.310939788818359e-4`。device 1 运行后无残留进程。

证据文件：

- `/opt/chw/gdn-phase3-stale-beta-install-pilot-r1.log`，SHA256
  `86a1cb7a74b5e217fe76940714a8a4e35530c7554cfefd642a7d745bfbf16047`；
- `/opt/chw/gdn-phase3-stale-beta-install-pilot-r1.rc`，内容 `1`，SHA256
  `741d14df730e53a5a019a710116f696db4ec23a132b74cf6fbb3cf7617e68313`。

该结果证明上一小步的 32-byte `accLocal` 清零修复尚不足以恢复累积 Phase 3 bit-exact，不能据此
扩精度矩阵或进入性能门禁；下一步先重新取得修复后的差异签名，不从旧诊断直接猜第二处根因。

### Phase 3 累积融合 stale-beta 修复完整包构建（已完成）

A2 远端 wrapper `bash -n`、Python/ABI `11/11 PASS` 后完成一次非增量完整包构建，退出码为 `0`。
本地、远端源码和 `build/binary` 的 standalone 两份共享 header，以及 CPack 源码拷贝 SHA256
均为 `72501296e8cfb6126364913e2983131a8f68eca5d670c8023282b20f8d53eef1`。

新包同时包含 `ChunkCumsumKkt` 与 `ChunkCumsumKktSolveTri` 的 FP16/BF16 两组 device `.o` 和
per-kernel JSON。与当前旧安装包逐文件比较，四个 `.o` 和四个 JSON 哈希均已变化，证明共享
helper 修复被两类 kernel 实际编译；host `libcust_opapi.so` 哈希仍为
`645336f9f65c522a06d74cf8b3df0ca6db2ae493fcd6fec980719eca7c8af07f`，符合只修改 device helper。
Phase 1/2/3 和独立 `ChunkCumsumKkt` 共 8 个执行/GetWorkspaceSize 符号齐全。

产物证据：

- run 包 SHA256：`a243127a8d2228b68621a7220012191e6cf0d8d6be5a8f2b9c3ba9d6e573fac5`；
- 构建日志：`/opt/chw/gdn-phase3-stale-beta-fix-build-r1.log`；
- 构建日志 SHA256：`34947106a135619a084e7149f11cc01b5ee84ccdb5bccb823a1ae97a38310331`。

本小步未安装或运行 NPU。

### Phase 3 累积融合 stale-beta 最小修复（本地门禁已完成）

共享 `ComputePrefixCumsumFromGm` 仅做任务级状态初始化修复：row 循环前以
`Duplicate(accLocal, 0.0f, 8)` 清零完整 32-byte block并执行 `PIPE_V` barrier；所有 row 统一
`Add(accLocal, accLocal, inputLocal, 1)`。接口、tiling、GM 地址、event 顺序、KKT/solve、workspace
和 Phase 1/2/3 调度均未修改。

本地门禁结果：

- `git diff --check`：通过，仅有既有 Windows line-ending 提示；
- ctypes ABI：`11/11 PASS`；
- 诊断、benchmark、matrix runner Python 语法：通过；
- 静态路由核对：Phase 1、Phase 2 和累积 Phase 3 分支仍并存；
- 修复后共享 header SHA256：
  `72501296e8cfb6126364913e2983131a8f68eca5d670c8023282b20f8d53eef1`。

本小步未同步远端、未构建、未安装或运行 NPU。

### Phase 3 累积融合 D_FP16_C64 根因与最小修复卡（已冻结）

对 `320` 个 `g_cumsum` 差异按 head/chunk/task/AIV 分组后，全部是完整的 C64 chunk 1，分别为
head `1/3/4/5/6`；它们横跨两个 AIV sub-block 和多个 core group，不符合单个 AIV 未执行。
五个坏 task 的共同前驱关系是：同一 AIV sub-block 刚处理完前一个 head 的 1-token 尾块。

进一步做值来源 exact 匹配，五个坏块的 Phase 3 首值均逐位等于该前驱 tail task 最后写入
`rowBrcbBuf_` 的 beta：

```text
head 1 <- previous head 0 tail beta: 0.5517578125
head 3 <- previous head 2 tail beta: 0.236328125
head 4 <- previous head 3 tail beta: 0.380615234375
head 5 <- previous head 4 tail beta: 0.496337890625
head 6 <- previous head 5 tail beta: 0.25341796875
```

坏块后续 63 个 token 相对 Phase 2 保持几乎恒定偏移，说明 raw g 的后续顺序累加正确，仅任务
首值继承了 stale beta。源码唯一对应是 `ComputePrefixCumsumFromGm` 将 `accLocal` 置于
`rowBrcbBuf_` 起始 block，而 `ComputeGateBlock` 在任务末尾以 `Brcb(beta)` 覆写同一区域；下一任务
只靠 row 0 的单元素 `Adds(accLocal, inputLocal, 0.0f, 1)` 隐式重置。在累积 kernel 的连续
core-group 调度中该覆盖不可靠。独立 `ChunkCumsumKkt` 的跨核步进调度没有暴露这一前驱组合，
因此此前 `80/80 exact` 不能排除该状态依赖。

最小修复固定为任务级显式零初始化加统一 `Add`，不新增同步协议、不扩大 buffer、不改 GM 地址。
因为修改共享 helper，修复包除重跑累积 core pilot 外，还必须回归独立 `ChunkCumsumKkt` 的
8 identities exact 门禁，不能用旧 `80/80` 证据替代。

分组证据：

- `/opt/chw/gdn-phase3-cumulative-core-accuracy-diag-r1/summary_root.json`
- SHA256：`1ede56963c4e94e01cf3f278bb6a5c6bbc572fe9dc7a1cbfcd8be98eca38b52a`

### Phase 3 累积融合 D_FP16_C64 分量定位（已完成）

使用同一份 CPU 输入文件，在两个干净 Python 进程中分别运行 Phase 2 和 Phase 3；Phase 2
进程额外运行独立 `ChunkLocalCumsum`。两条路径均调用、同步并退出成功，结构化结果为：

- Phase 2 `g_cumsum` 与独立 `ChunkLocalCumsum` 全张量逐位一致；
- Phase 3 公开 `g_cumsum` 有 `320/8200` 个元素不同，首个差异为 `[B=0,T=64,H=1]`，
  max-abs `0.6076866388320923`，两侧均无 NaN/Inf；
- Phase 2/3 solved A **全张量**逐位一致，`524800/524800` 个元素 max-abs `0`，均有限；
- Phase 2/3 output 有 `95049` 个元素不同，首个差异为 `[B=0,H=1,T=64,D=0]`，
  max-abs `0.0005898475646972656`；final-state presence 一致。

因此 KKT/solve 的内部 compute view 和 solved A 已排除；故障确定在新累积 kernel 的公开
`g_cumsum` GM 写回或多 AIV 输出所有权。后续 core 读取错误的公开 `g_cumsum`，所以 output
随同一 token/head 边界产生偏差。当前不能把该结果泛化到其他 identity，也尚未修改实现。

证据文件：

- `/opt/chw/gdn-phase3-cumulative-core-accuracy-diag-r1/summary.json`
- summary SHA256：`299ea5665770e8ee3f38f514761b4704ec7ec09f0cf3b879859749c2b5c100b0`
- `/opt/chw/gdn-phase3-cumulative-core-accuracy-diag-r1.log`
- log SHA256：`1c038326d5e49fa325caa4a8bd9bb857d6bbe252d05f0c3803c5be6c5b450cdf`

### Phase 3 累积融合 D_FP16_C64 accuracy pilot（门禁失败，已停止扩测）

本地与 A2 远端 matrix wrapper、runner、benchmark SHA256 一致；实际安装
`libcust_opapi.so` SHA256 仍为
`645336f9f65c522a06d74cf8b3df0ca6db2ae493fcd6fec980719eca7c8af07f`，device 1 运行前空闲。
未重新构建或安装，只运行
`B=1,Hk/Hv=4/8,T=1025,FP16,C=64` 一个 Phase 2/3 core accuracy case。

Phase 2/3 的 final-state presence 一致，但 `g_cumsum`、有效 A 和 output 均未逐位一致；output
max-abs 为 `0.0006310939788818359`。benchmark 在首个 assertion 按协议退出，matrix runner
返回码为 `1`，未生成 case JSON，也未扩其他 dtype/chunk/varlen identity。两条 Phase 均完成
调用并同步，故本次是累积融合精度门禁失败，不是命令封装、符号缺失或安装环境错误。

证据文件：

- `/opt/chw/gdn-phase3-cumulative-core-accuracy-pilot-r1.log`
- 日志 SHA256：`4e6ae7ebf0308ab60b7b759a99682c0896afef192ec8f488988be2bb2c388a2f`

### Phase 3 累积融合安装后最小 smoke（已完成）

安装前 run 包 SHA256 重查为
`f705efeb687f1d2264c5a0e296bb987ad748f8d690233bfdee845a94189f8323`，新 op 两个 kernel
binary/config 哈希可追溯，device 1 无进程。完整包安装成功，实际安装 `libcust_opapi.so` SHA256
为 `645336f9f65c522a06d74cf8b3df0ca6db2ae493fcd6fec980719eca7c8af07f`，与 build 库一致；
Phase 1/2/3 和 `ChunkCumsumKkt` 必需符号齐全，新 op config、源码和 proto 均存在。

`B=1,H=2,T=128,FP16,C=64,output_final_state=True` 的 Phase 2/3 最小 smoke 通过：output、真实
final state、FP32 `g_cumsum` 和 solved A 均全部有限且逐位一致，四项 max-abs 均为 `0`，退出码
为 `0`。安装器因 Phase 3 版本头相对旧安装文件发生预期修改而提示 ABI warning，但 C API 函数
签名未变、8 个版本化执行/GetWorkspaceSize 符号存在且实际调用通过。当前只证明该最小身份。

### Phase 3 累积融合完整包构建（已完成）

A2 完整包构建成功。新 `ChunkCumsumKktSolveTri` 两个 dtype/chunk 编译组完成，包内包含两个 `.o`
和 `.json`、kernel config、proto 与 L0 object；新 kernel 源码、两份 build/binary 拷贝和 CPack 源码
拷贝 SHA256 均为 `d3647e9d3476b216d630357646f7a6fa51c9b0f9cf1c4b53849f0b54d5cbd5f3`。
共享 fused-cumsum helper 保持已验收哈希
`b05c2428b455678ff91b39f96a9fcc76d30231d6b00a4f459f1faa2ea39cded9`。

run 包 SHA256 为 `f705efeb687f1d2264c5a0e296bb987ad748f8d690233bfdee845a94189f8323`；
build `libcust_opapi.so` SHA256 为
`645336f9f65c522a06d74cf8b3df0ca6db2ae493fcd6fec980719eca7c8af07f`，Phase 1/2/3 和
`ChunkCumsumKkt` 必需符号齐全。构建日志：`/opt/chw/gdn-phase3-cumulative-core-build-r1.log`。

外层 PowerShell/SSH 命令错误转义 `$rc`，没有生成 sidecar `.rc`；未据此接受结果，而是额外核对
run 包时间/哈希、日志成功尾标、两个新 kernel 编译 target、包内二进制、源码拷贝哈希、符号和
无构建进程后才判定构建本体成功。尚未安装或运行 NPU。

### Phase 3 core 改接累积融合 op（已完成）

`aclnnGdnCoreFwdPhase3` 已从 `ChunkCumsumKkt -> Cast -> SolveTri` 改接为单次
`l0op::ChunkCumsumKktSolveTri`，直接取得 FP32 `g_cumsum` 和 solved A。Phase 3 不再分配 FP32
raw A；Phase 1 继续独立 KKT/Cast/SolveTri，Phase 2/default 继续
`ChunkLocalCumsum -> ChunkKktSolveTri`，均未改路由。版本头注释和 ABI 静态断言已同步，并明确
禁止 Phase 3 core 回退到 `l0op::ChunkCumsumKkt(`。

本地 `git diff --check` 通过，ctypes ABI `11/11 PASS`。尚未构建或运行 NPU。

### Phase 3 累积融合 op 静态骨架（已完成）

新增 `ChunkCumsumKktSolveTri` 的 kernel entry、tiling-key、OpDef、InferShape、L0 launcher，并在
现有 Phase 2 目录复用 `ChunkKktSolveTriTilingData`、tiling 函数、workspace 布局和编译 include。
kernel entry 保持 Phase 2 的 score/KKT/solve barrier，只将 AIV 初始化切到已验证的
`InitFusedCumsum`，输出 FP32 `g_cumsum` 和与 `k` 同 dtype 的 solved A。未修改 Phase 3 core
调度、公开 standalone ACLNN/Python API、旧算子或 Phase 1/2。

本地 `git diff --check` 通过；既有 ctypes ABI `11/11 PASS`。当前只是静态骨架，尚未构建，不能
宣称新 op 可编译或运行。

### Phase 3 core 累积融合方案（已冻结）

只读审计确认最小方案可以机械复用两条已验证路径，而不新造同步协议：新增
`ChunkCumsumKktSolveTri`，以 Phase 2 `ChunkKktSolveTri` 的 tiling、workspace 和 MIX barrier 为
骨架，将 AIV KKT epilogue从 `Init` 切到独立 Phase 3 已 `80/80 exact` 的 `InitFusedCumsum`。

新 kernel 输入为 `k` FP16/BF16、raw `g/beta` FP32 和可选 canonical varlen metadata；输出公开
FP32 `g_cumsum` 与同 `k` dtype 的 solved A。workspace 仍依次为 FP32 score、低精度 A hand-off、
per-core solve workspace。AIC 生成 score 后发 `SCORE_READY`；配对 AIV 完成 cumsum、KKT 和所有
MTE3 写回后发 `KKT_READY`；配对 AIC 只等待一次并运行 solve。dense/varlen、外部 GVA 扩头、
transpose 和 `K==V==128` 边界不变，旧 `ChunkCumsumKkt`、`ChunkKktSolveTri` 和 Phase 1/2 均不改。

Phase 3 core 最终直接消费新 kernel 两个输出，不再分派 Cast/SolveTri；预期完整 core NPU kernel
数由 Phase 2 的 `9` 降为 `8`。详细修正卡已写入 `GDN_FUSION_PLAN_A2.md`。

### Phase 3 core D_FP16_C64 profiler 根因（已完成）

分别在两个干净进程 profile 完整版本化 ACLNN。Phase 2 实际为 9 个 NPU kernel、设备 duration
合计 `330.527 us`；关键阶段是 `ChunkLocalCumsum 48.461 us + ChunkKktSolveTri 82.082 us`。
Phase 3 实际为 10 个 NPU kernel、合计 `380.248 us`；关键阶段是
`ChunkCumsumKkt 108.762 us + Cast 6.120 us + SolveTri 60.281 us`。其余 3 个 transpose、
recompute_w_u、fwd_h 和 fwd_o 路由一致。

因此 Phase 3 虽将 local_cumsum + KKT 从两个局部 kernel 减为一个，却在 core 路径重新拆出了
Phase 2 已融合的 solve_tri，并额外恢复一次 cast；core kernel 数 `9 -> 10`，阶段设备时间约
`130.543 -> 175.163 us`。这足以解释生产 median 回退，不应继续扩大旧实现的性能矩阵。

结构化 profiler 证据：`/opt/chw/gdn-phase3-core-profiler-d-fp16-c64-r1/summary.json`，SHA256
`089d817c3deb263b9d535adf6475694bad51a24d0491b269042a199a9a42df72`；Phase 2/3 trace SHA256
分别为 `821fd89f3cea39b9c7486bfc00449b3180f84e460219baa800c9ae1124e5d49e` 和
`b9a853edfdd1ee668ec6a751d8d9cbc5881367d9ee664eea4c56548652bac1c6`。运行结束后 device 1 无进程。

### Phase 3 core D_FP16_C64 生产性能 pilot（门禁失败，已停止扩测）

A2 device 1 在关闭 `ASCEND_LAUNCH_BLOCKING` 后，按 4 轮 AB/BA、8 个干净子进程运行；每个
variant 每轮 warmup 10、采集 50 个 NPU Event 样本，合计各 `200` 个样本。8 次运行的 output、
`g_cumsum`、有效 A 和 final-state presence 均有限；每次均为 1 ACLNN，workspace/peak 签名一致。

Phase 2/3 median 分别为 `1.041130/1.079460 ms`，Phase 3 回退 `3.682%`；P90 分别为
`1.087500/1.150200 ms`，Phase 3 回退 `5.766%`。四轮 Phase 2 batch median 为
`[1.044460, 1.058280, 1.048410, 1.007610] ms`，Phase 3 为
`[1.129620, 1.075830, 1.060660, 1.077250] ms`，每轮方向均为 Phase 3 更慢，因此不是单个长尾
样本造成。Phase 2/3 workspace max 为 `137055744/137544192 B`，Phase 3 增加 `488448 B`；
peak delta 为 `140239360/141595136 B`，增加 `1355776 B`。

结构化证据为 `/opt/chw/gdn-phase3-core-perf-d-fp16-c64-r1/summary.json`，SHA256
`df845f85fefbf0d45b4ba6a2af58898ed56b7e6b4f8025e4b49a661a947c0bbf`；实际加载
`libcust_opapi.so` SHA256 为
`4fa4be65c5b365a772f30eb086d9fcaaeccbd21deb51d2d2a1dc12379441d7df`。本地/远端 benchmark、
matrix runner 和远程 wrapper SHA256 一致，运行结束后 device 1 无 NPU 进程。按启动卡门禁，
当前不扩其他 dtype/chunk/varlen 身份，先用 profiler 解释回退。

### Phase 3 core state accuracy（已完成）

`STATE_D_BF16_C64` 使用非空 FP32 `initial_state` 且 `output_final_state=True`，Phase 2/3 的
output、`g_cumsum`、有效 A 和真实 final state 全部逐位一致，所有组件有限。证据：
`/opt/chw/gdn-phase3-core-accuracy-state/summary.json`。

Phase 3 core 功能/精度门禁至此为主路由 `8/8 PASS` 加 state `1/1 PASS`，可以进入生产性能；
该结论仍不包含性能、core profiler 或最终归档。

### Phase 3 core varlen accuracy 矩阵（4/4 完成）

`V_BF16_C64/V_FP16_C128/V_BF16_C128` 均通过；加上 pilot，varlen FP16/BF16 × C64/C128
共 `4/4 PASS`。每例 Phase 2/3 的 output、`g_cumsum`、有效 A 和 final-state presence 逐位一致，
所有呈现组件有限。证据：`/opt/chw/gdn-phase3-core-accuracy-varlen/summary.json`。

至此主路由 dense/varlen × FP16/BF16 × C64/C128 共 `8/8 PASS`；尚需单独覆盖非空 initial/
final state，之后才进入生产性能。

### Phase 3 core varlen FP16/C64 accuracy pilot（已完成）

正式 runner 使用 `cu_seqlens=[0,1,66,259]` 运行 `V_FP16_C64`，Phase 2/3 的 output、
`g_cumsum`、有效 A 和 final-state presence 逐位一致，所有呈现组件有限。证据：
`/opt/chw/gdn-phase3-core-accuracy-varlen-pilot/summary.json`。该结果只覆盖 varlen FP16/C64。

### Phase 3 core dense accuracy 矩阵（4/4 完成）

`D_BF16_C64/D_FP16_C128/D_BF16_C128` 均在正式 runner 中通过；加上 pilot，dense
FP16/BF16 × C64/C128 共 `4/4 PASS`。每例 Phase 2/3 的 output、`g_cumsum`、有效 A 和
final-state presence 均逐位一致，所有呈现组件有限。证据：
`/opt/chw/gdn-phase3-core-accuracy-dense/summary.json`。dense 结果不能替代 varlen 序列边界验证。

### Phase 3 core dense FP16/C64 accuracy pilot（已完成）

正式 runner 在 A2 device 1 运行 `D_FP16_C64`（物理 `B=1,H=8,T=1025,K=V=128,C=64`）。
Phase 2/3 的 output、`g_cumsum`、有效 A 和 final-state presence 均逐位一致，output max-abs `0`；
两个版本所有已呈现组件的非有限计数均为 `0`。结构化证据：
`/opt/chw/gdn-phase3-core-accuracy-pilot/summary.json`，本地只读镜像
`.phase3_core_accuracy_pilot_summary.json`。该 pilot 不代表其余 7 个路由身份已完成。

### Phase 3 正式 benchmark/runner 接入（已完成）

正式 `benchmark_gdn_core_ablation.py` 新增命名 variant
`phase3_one_aclnn_fused_cumsum_kkt`，覆盖独立 accuracy、1 ACLNN/workspace 统计、standalone
生产计时和 `core_phase2_vs_phase3` 成对对照；既有 Phase 1/2、六调用和 KKT+solve 结果字段保留。
新增 `run_gdn_phase3_performance_matrix.py`，accuracy 模式只比较不可变 Phase 2/3，performance
模式使用干净子进程 AB/BA 平衡聚合，避免把 Phase 1 旧问题混入 Phase 3 门禁。

本地 `py_compile`、AST 检查、runner `--help` 与 `git diff --check` 通过。真实 benchmark 在
Windows import torch 时仍被已归档的双 OpenMP runtime 拦住，发生在参数解析前；未设置
`KMP_DUPLICATE_LIB_OK`，下一小步在 A2 干净 conda 环境验证。本小步未运行 NPU。

### Phase 3 core 安装后最小 smoke（已完成）

完整包安装成功，实际加载 `libcust_opapi.so` SHA256 为
`4fa4be65c5b365a772f30eb086d9fcaaeccbd21deb51d2d2a1dc12379441d7df`，与 build 库一致；
Phase 1/2/3 与 `ChunkCumsumKkt` 共 8 个必需符号全部存在。安装器提示版本头增加 Phase 3 导致
ABI 文件变化；本次是完整包同步替换，并已用符号和实际调用验证，不存在旧库混载。

A2 device 1 dense FP16/C64 `B=1,H=2,T=128` smoke 中，Phase 2/3 的 output、final state、
`g_cumsum` 和 A 全张量逐位一致，最大绝对误差 `0`，且全部有限，脚本退出码 `0`。这只证明最小
dense 可调用，不替代 8 身份端到端矩阵或生产性能门禁。

### Phase 3 core 首次完整包构建（已完成）

本地与 A2 远端 6 个同步文件 SHA256 一致；远端 `bash -n`、Python 语法和 descriptor ABI
`11/11 PASS`。完整包构建成功：

- run 包：`build_out/fla-npu-fla_npu_linux-aarch64.run`，SHA256
  `ce9b5106edbc11b5299d3b63c73be9cd9dec57b7ec5f40e8cea5c2a80887a5e6`；
- build `libcust_opapi.so` SHA256
  `4fa4be65c5b365a772f30eb086d9fcaaeccbd21deb51d2d2a1dc12379441d7df`；
- build 库同时导出 `ChunkCumsumKkt` 和 `GdnCoreFwdPhase1/2/3` 的执行与
  `GetWorkspaceSize` 共 8 个必需符号；
- Phase 3 版本头的打包拷贝 SHA256 与源码一致；`ChunkCumsumKkt` 三份 kernel 构建拷贝 SHA256
  均为 `a168288d476921a508a4b99f1c90ff0019664f9ca3b32da1864023f03dbbcd1d`。

外层 SSH 日志捕获命令因 PowerShell 引号处理未生成 `.rc` 文件，但构建脚本运行到最后的符号检查，
shell 退出码为 `0`，产物/哈希/符号均已直接复核，且无残留构建进程。该缺失不伪装为 rc 证据；
构建日志位于 `/opt/chw/gdn-phase3-core-build.log`。本小步未安装新包或运行 NPU。

### Phase 3 core Python/ABI 小步（已完成）

新增稳定 Python 入口 `npu_gdn_core_fwd_phase3`，固定映射
`aclnnGdnCoreFwdPhase3`，并复用 Phase 1/2 完全相同的 15 参数 descriptor contract。默认入口仍
映射 `aclnnGdnCoreFwd`，Phase 1/2 的固定符号未改变。

本地 `py_compile`、`git diff --check` 通过；使用 `python -S` 运行 descriptor ABI 单测
`11/11 PASS`。测试同时断言 Phase 3 C++ 路由包含 `ChunkCumsumKkt`，Phase 1 仍包含独立 KKT/
SolveTri，Phase 2 仍包含 `ChunkKktSolveTri`。本小步未构建、安装或运行 NPU。

### Phase 3 core C++ 版本化调度小步（已完成）

新增 `GdnCorePhase::PHASE_3_FUSED_CUMSUM_KKT` 和导出声明
`aclnnGdnCoreFwdPhase3/GetWorkspaceSize`。Phase 3 路径固定为：

```text
transpose raw g/beta
  -> ChunkCumsumKkt (FP32 g_cumsum + A_raw)
  -> cast A_raw
  -> SolveTri (沿用 dense/varlen 既有 layout 适配)
  -> recompute_w_u -> fwd_h -> fwd_o
```

Phase 1 的 `local_cumsum -> KKT -> solve`、Phase 2/default 的
`local_cumsum -> ChunkKktSolveTri` 路由均未改变。本小步只修改版本化 C++ 调度和头文件，
`git diff --check` 通过；尚未构建或运行 NPU，不能据此宣称 Phase 3 core 可调用。

### Phase 3 core 调度 contract 只读核对（已完成）

- `ChunkCumsumKkt` 接受 head-first `k/g/beta`，输出 FP32 `g_cumsum[B,H,T]` 和
  `A_raw[B,H,T,C]`；与 core 现有 transpose 后的 `gBht/betaBht` 以及内部张量完全匹配。
- `SolveTri` 仍需低精度 A；Phase 3 复用 Phase 1 已验证的 FP32 A cast 和 dense/varlen layout
  适配，不改变数值顺序或输出 contract。
- 最小 Phase 3 core 路径是两个数学 kernel：`ChunkCumsumKkt` 与独立 `SolveTri`。它是验证
  端到端收益的版本化候选，不宣称 `cumsum+KKT+solve` 已单 kernel 融合。
- Phase 1/2/default 的函数名、枚举路由和调度保持不变；新增路径需要独立
  `aclnnGdnCoreFwdPhase3/GetWorkspaceSize`，后续再接 Python/ABI 和 A2 构建门禁。

### Phase 3 varlen 局部生产性能矩阵（4/4 完成）

固定不等长序列 `cu_seqlens=[0,1,65,257,514,769,1025]`，物理 `B=1,H=8,T=1025`，四个
dtype/chunk 身份均在相同 A2 device 1、安装库和生产计时口径下通过：

| dtype/chunk | baseline median | fused median | median 变化 | P90 变化 |
| --- | ---: | ---: | ---: | ---: |
| FP16/C64 | `1.15526 ms` | `0.76100 ms` | `-34.127%` | `-33.503%` |
| BF16/C64 | `1.10812 ms` | `0.72897 ms` | `-34.216%` | `-33.786%` |
| FP16/C128 | `1.21163 ms` | `0.80327 ms` | `-33.703%` | `-32.442%` |
| BF16/C128 | `1.23595 ms` | `0.83134 ms` | `-32.737%` | `-33.253%` |

每例融合 `g_cumsum/A_raw` 均对两小算子 NPU 基线全张量逐位一致且有限，ACLNN 数 `2 -> 1`，
workspace max 持平，workspace sum 和 peak delta 均减少 `16,777,728 B`。结构化证据为
`/opt/chw/gdn-phase3-local-perf-varlen-{fp16,bf16}-c{64,128}.json`。

至此局部生产性能矩阵 `8/8 PASS`：dense/varlen × FP16/BF16 × C64/C128 的 median/P90 全部
改善，满足启动卡中“局部有收益才接入 core”的前置条件。下一阶段必须新增 Phase 3 版本化入口，
不能原地修改 Phase 1/2。

### Phase 3 dense 局部生产性能矩阵（4/4 完成）

其余三个 dense dtype/chunk 身份均在相同 A2 device 1、安装库和生产计时口径下通过：

| dtype/chunk | baseline median | fused median | median 变化 | P90 变化 |
| --- | ---: | ---: | ---: | ---: |
| BF16/C64 | `0.99332 ms` | `0.66336 ms` | `-33.218%` | `-32.434%` |
| FP16/C128 | `1.05761 ms` | `0.72259 ms` | `-31.677%` | `-31.342%` |
| BF16/C128 | `1.04949 ms` | `0.70823 ms` | `-32.517%` | `-32.852%` |

加上首个 FP16/C64，4/4 dense 身份的融合 `g_cumsum/A_raw` 均全张量逐位一致且有限，median/P90
均改善。每例 ACLNN 数均 `2 -> 1`，workspace max 与 KKT 基线持平，融合消除了独立 cumsum 的
`16,777,728 B` workspace，因此 workspace sum 和 peak delta 均减少同等规模。结构化证据为
`/opt/chw/gdn-phase3-local-perf-dense-{fp16,bf16}-c{64,128}.json`。

### Phase 3 dense FP16/C64 局部生产性能首点（已完成）

A2 device 1、Ascend910B3、CANN `9.1.0.beta1`，关闭 `ASCEND_LAUNCH_BLOCKING`，使用
`B=1,H=8,T=1025,K=128,C=64`。同一进程交替 AB/BA，warmup `10`、每个 variant `50` 个
NPU Event 样本。安装库 SHA256 为
`eabc87564a5967513c9a4954a18ce26e25318e248c7a38acbe07b018e8a84a43`。

- 融合 `g_cumsum`/`A_raw` 对两小算子 NPU 基线全张量逐位一致，最大绝对误差 `0`，全部有限；
- baseline/fused median 为 `1.09235/0.73096 ms`，融合改善 `33.084%`；
- baseline/fused P90 为 `1.11572/0.76754 ms`，融合改善 `31.207%`；
- ACLNN 数 `2 -> 1`；workspace sum `35,783,680 -> 19,005,952 B`；
- workspace max 持平为 `19,005,952 B`；peak delta `37,917,696 -> 21,139,456 B`。

结构化证据：`/opt/chw/gdn-phase3-local-perf-dense-fp16-c64.json`。该点证明当前融合方向有明确
局部收益，但尚未覆盖其余 dtype/chunk/varlen 身份，也不代表 Phase 3 core 已接入。

### Phase 3 dense FP16/C64 fused profiler smoke（已完成）

A2 device 1 空闲时只重新采集 fused trace，未重复采集 baseline。脚本退出码为 `0`，实际设备
kernel 为：

```text
aclnnChunkCumsumKkt_ChunkCumsumKkt_ChunkCumsumKkt: KERNEL_MIX_AIC
```

fused 路径设备 kernel 数为 `1`；与已离线验证的两小算子 baseline `2` 个 kernel 对照，局部
NPU kernel 数确认 `2 -> 1`，ACLNN 数按入口 contract 同样为 `2 -> 1`。两份 Chrome trace 的
kernel 事件均不含 tiling key 或 block dim，因此只归档实际 kernel 名和 task type，不从源码值
猜测运行时元数据。profiler 单次 duration 不进入性能验收结论。

### Phase 3 dense FP16/C64 baseline trace 离线解析（已完成）

修复后的兼容摘要器已在 A2 上离线解析现有 `baseline.json`，退出码为 `0`，没有重新启动 NPU
采集。设备事件筛选得到恰好两个 kernel：

```text
aclnnChunkLocalCumsum_ChunkLocalCumsum_ChunkLocalCumsum: KERNEL_AIVEC
aclnnChunkScaledDotKkt_ChunkScaledDotKkt_ChunkScaledDotKkt: KERNEL_MIX_AIC
```

因此 dense FP16/C64 局部两算子基线的实际 NPU kernel 数为 `2`。当前 Chrome trace 的 kernel
事件只包含 task/stream/connection 元数据，不含 tiling key 或 block dim；本小步不从源码 key
推断运行时路由。下一小步只补 fused trace，并继续遵守“缺失即记录缺失、不猜测”的证据规则。

### Phase 3 dense FP16/C64 profiler smoke 尝试 1（摘要器失败）

A2 device 1 预检时曾短暂出现外部 `python3`，未启动 profiler、未干预该进程；它自然退出且设备
利用率恢复 0 后才开始采集。baseline profiler 成功采集并导出
`/opt/chw/gdn-phase3-profiler-smoke-dense-fp16-c64/baseline.json`，但本地摘要器假设 trace 顶层为
dict；当前 torch_npu 导出的是 list，触发 `AttributeError`。失败发生在 baseline 离线摘要阶段，
fused profiler 尚未启动，因此没有 kernel 数或路由结论。

证据文件：

- `/opt/chw/gdn-phase3-profiler-smoke-dense-fp16-c64.log`
- `/opt/chw/gdn-phase3-profiler-smoke-dense-fp16-c64.rc`

摘要器现已兼容 dict/list 两种 Chrome trace 结构；下一步先离线验证已有 trace，不直接重采。

首次离线解析启动命令又在 PowerShell/SSH 嵌套引号处失败，远端 shell 报 `unexpected EOF`；该次
未进入 Python、未读取或修改 trace、未触发 NPU。按开发手册改为由已上传 Bash wrapper 透传
`--summarize-existing-baseline`，不再拼接远端多步命令。

### Phase 3 varlen BF16/C128 exact matrix（已完成）

复用 varlen FP16/C128 的 10 个 contract，只改变 `k` dtype 为 BF16；A2 device 1 运行
`10/10 PASS`。每例的融合 `g_cumsum` 和 `A_raw` 均与两小算子 NPU 基线全张量逐位一致，输出
有限，严格下三角外区域和尾部 padding 为零；退出码为 `0`。

至此冻结的 8 个完整身份共 `80/80 exact PASS`：

```text
dense/varlen × FP16/BF16 × C64/C128，每个身份 10 例
```

每例同时验证公开 FP32 `g_cumsum`、内部 KKT 最终物化的 FP32 `A_raw`、有限性和无效区清零。
该结论是 value evidence；profiler route、性能、workspace 和端到端 Phase 3 core 尚未执行。

证据文件：

- `/opt/chw/gdn-phase3-varlen-bf16-c128-exact10.log`
- `/opt/chw/gdn-phase3-varlen-bf16-c128-exact10.rc`

### Phase 3 varlen FP16/C128 exact matrix（已完成）

复用 varlen C64 的 10 个序列 contract，只改变 `chunk_size=128`；A2 device 1 运行
`10/10 PASS`。每例的融合 `g_cumsum` 和 `A_raw` 均与两小算子 NPU 基线全张量逐位一致，输出
有限，严格下三角外区域和尾部 padding 为零；退出码为 `0`。

证据文件：

- `/opt/chw/gdn-phase3-varlen-fp16-c128-exact10.log`
- `/opt/chw/gdn-phase3-varlen-fp16-c128-exact10.rc`

### Phase 3 varlen BF16/C64 exact matrix（已完成）

复用 varlen FP16/C64 的 10 个序列 contract，只改变 `k` dtype 为 BF16；A2 device 1 运行
`10/10 PASS`。每例的融合 `g_cumsum` 和 `A_raw` 均与两小算子 NPU 基线全张量逐位一致，输出
有限，严格下三角外区域和尾部 padding 为零；退出码为 `0`。

证据文件：

- `/opt/chw/gdn-phase3-varlen-bf16-c64-exact10.log`
- `/opt/chw/gdn-phase3-varlen-bf16-c64-exact10.rc`

### Phase 3 varlen FP16/C64 exact matrix（已完成）

A2 device 1 运行 `10/10 PASS`，覆盖 `T=1/2/63/64/65/127/128/129/193/257`，包括单序列、
2/3 序列、1-token 序列、序列边界恰逢 chunk 边界、跨 chunk 尾块和多头。每例的融合
`g_cumsum` 和 `A_raw` 均与使用各自正确元数据粒度的两小算子 NPU 基线全张量逐位一致；输出
有限，严格下三角外区域和尾部 padding 为零。测试进程退出码为 `0`，结束后 device 1 无进程。

证据文件：

- `/opt/chw/gdn-phase3-varlen-fp16-c64-exact10.log`
- `/opt/chw/gdn-phase3-varlen-fp16-c64-exact10.rc`

### Phase 3 dense BF16/C128 exact matrix（已完成）

复用 dense FP16/C128 的 10 个 contract，只改变 `k` dtype 为 BF16；A2 device 1 运行
`10/10 PASS`。每例的融合 `g_cumsum` 和 `A_raw` 均与两小算子 NPU 基线全张量逐位一致，输出
有限，严格下三角外区域和尾部 padding 为零。

至此 4 个 dense 完整身份共 `40/40 PASS`：FP16/BF16 × C64/C128，每个身份 10 个 exact case。
这不代表 4 个 varlen 身份已通过。

证据文件：

- `/opt/chw/gdn-phase3-dense-bf16-c128-exact10.log`
- `/opt/chw/gdn-phase3-dense-bf16-c128-exact10.rc`

### Phase 3 dense FP16/C128 exact matrix（已完成）

复用 dense C64 的 10 个 shape/batch/head contract，只改变 `chunk_size=128`；A2 device 1 运行
`10/10 PASS`。每例的融合 `g_cumsum` 和 `A_raw` 均与两小算子 NPU 基线全张量逐位一致，输出
有限，严格下三角外区域和尾部 padding 为零。覆盖 `T<128`、`T=128`、`T=129/193/257` 的
整块、尾块和多 chunk 边界。该结论只覆盖 dense FP16/C128 身份。

证据文件：

- `/opt/chw/gdn-phase3-dense-fp16-c128-exact10.log`
- `/opt/chw/gdn-phase3-dense-fp16-c128-exact10.rc`

### Phase 3 dense BF16/C64 exact matrix（已完成）

复用 dense FP16/C64 的 10 个 shape/batch/head contract，只改变 `k` dtype 为 BF16；A2 device 1
运行 `10/10 PASS`。每例的融合 `g_cumsum` 和 `A_raw` 均与两小算子 NPU 基线全张量逐位一致，
输出有限，严格下三角外区域和尾部 padding 为零。该结论只覆盖 dense BF16/C64 身份；未用它
推导 C128 或 varlen。

证据文件：

- `/opt/chw/gdn-phase3-dense-bf16-c64-exact10.log`
- `/opt/chw/gdn-phase3-dense-bf16-c64-exact10.rc`

### Phase 3 dense FP16/C64 exact matrix（已完成）

在同一修复包、A2 device 1 上运行 `10/10 PASS`。覆盖 `T=1/2/63/64/65/127/128/129/193/257`、
`B=1/2/3`、`H=1/2/3/4`，包含 tiny、整 chunk、尾块、多 chunk、多 batch 和多头。每例均满足：

- 融合 `g_cumsum` 与独立 NPU `ChunkLocalCumsum` 全张量逐位一致；
- 融合 `A_raw` 与两小算子 NPU 基线全张量逐位一致；
- 两个融合输出全部有限；
- `A_raw` 对角、上三角和尾部 padding 为零。

该结论只覆盖完整身份
`(run package 9dfbce..., tiling key 10 source route, fused cumsum subpath, k=FP16/g=FP32/beta=FP32/out=FP32, dense optional-input state)`；
实际 profiler route/tiling key 仍待性能前的独立取证，不能由源码 key 单独宣称已确认。

证据文件：

- `/opt/chw/gdn-phase3-dense-fp16-c64-exact10.log`
- `/opt/chw/gdn-phase3-dense-fp16-c64-exact10.rc`

### Phase 3 cumsum 修复尝试 2（dense FP16/C64 单例通过）

完整包构建、安装、源码/三份构建拷贝 SHA256 和 6 个必需 ACLNN 符号检查均通过。修复 header
SHA256 为 `b05c2428b455678ff91b39f96a9fcc76d30231d6b00a4f459f1faa2ea39cded9`，run 包 SHA256 为
`9dfbce90986faf2140f82c90dfb6a7aafe042e7819021231a8bbbbaafb027fa0`，安装库 SHA256 为
`eabc87564a5967513c9a4954a18ce26e25318e248c7a38acbe07b018e8a84a43`。

同一个 dense FP16/C64 定点诊断通过：

```text
fused g_cumsum vs NPU ChunkLocalCumsum: 256/256 exact, max-abs 0
fused A_raw vs two-op NPU baseline:       16384/16384 exact, max-abs 0
fused A_raw valid region:                  8064/8064 exact, max-abs 0
fused/baseline invalid nonzero:            0/0
```

融合与独立 NPU cumsum 对 CPU `torch.cumsum` 的差异签名也相同：均为 `62/256` 逐位相等、max-abs
`8.94e-08`，证明融合路径复现了实际小算子 FP32 累加顺序，而不是只贴近数学公式。

根因是尝试 1 缺少每次 GM 写回后的 `MTE3_MTE2` 依赖。尝试 2 对齐仓内已验证
`kda_gate_cumsum` 模式：每类 hard event 在循环前获取一次，并在每轮执行
`MTE2_V -> V_MTE3 -> MTE3_MTE2 -> MTE3_V`。这同时保证公开 FP32 `g_cumsum` 写回完成后，KKT
compute view 才从 GM 读回。

证据文件：

- `/opt/chw/gdn-phase3-prefix-fix2-build.log`
- `/opt/chw/gdn-phase3-prefix-fix2-build.rc`

### Phase 3 cumsum 修复尝试 1（失败）

完整包构建、安装、源码/构建拷贝 SHA256 和 6 个必需 ACLNN 符号检查均通过。源码与三份构建拷贝
`chunk_scaled_dot_kkt.h` SHA256 均为
`6557c2f9677d78ea4e0266a477d8d1e81e220b8d203e9652eb1bfe74c423d3b0`，run 包 SHA256 为
`cefc39cfc41dde4085397ea0ed94f823754e78b0ede3b557f3b3e74a603a6fae`。

同一个 dense FP16/C64 诊断未通过：融合 `g_cumsum` 对 NPU 基线仅 `5/256` 相等，max-abs
`0.2436266541`；融合 `A_raw` 有效区仅 `5/8064` 相等，max-abs `0.2949646711`，无效区仍为零且
输出有限。首个 chunk 前两个 token 一度与基线一致，随后漂移；说明 raw scalar UB 问题虽被替换，
但新 helper 的异步事件序列仍不正确。

证据文件：

- `/opt/chw/gdn-phase3-prefix-fix-build.log`
- `/opt/chw/gdn-phase3-prefix-fix-build.rc`

与仓内已验证 `kda_gate_cumsum` 对比，尝试 1 在每次 MTE3 写回后没有立即等待
`MTE3_MTE2`，且循环内 helper 反复获取 event ID。尝试 2 只修这两点：每类 hard event 在循环前
获取一次，并严格执行 `MTE2_V -> V_MTE3 -> MTE3_MTE2 -> MTE3_V`；不改变数学、布局或接口。

### Phase 3 dense FP16/C64 定点诊断（已完成）

同一个 `B=1,H=2,T=128,K=128,C=64` case 的融合、两小算子 NPU 基线和 CPU 参考均在同一进程、
同一安装包、A2 device 1 上运行。结果：

```text
fused g_cumsum vs raw g:             31/256 equal, max-abs 0.2596395314
fused g_cumsum vs CPU cumsum:         4/256 equal, max-abs 0.0580495745
NPU ChunkLocalCumsum vs CPU cumsum:  62/256 equal, max-abs 8.94e-08
fused g_cumsum vs NPU baseline:       4/256 equal, max-abs 0.0580495745
fused A_raw valid vs NPU baseline: 5517/8064 equal, max-abs 0.0512158275
fused/baseline A_raw invalid nonzero: 0/0
```

融合 `g_cumsum` 每个 chunk 从第二个 token 开始即与 NPU 基线分歧，且个别值出现 `4.2e-45`；
因此不是 CPU `torch.cumsum` 的归约顺序误差，也不是输出交换。第一处源级分歧定位到探索 kernel
`ComputePrefixCumsum`：它对已进入 vector pipeline 的 UB tensor 使用 raw scalar pointer 原地循环，
没有已证明的 scalar/vector 同步语义。该实现不能作为 Phase 3 正确性基线。

证据文件：

- `/opt/chw/gdn-phase3-dense-fp16-c64-diag.log`
- `/opt/chw/gdn-phase3-dense-fp16-c64-diag.rc`

最小修复只修改该 helper：复用 `ChunkLocalCumsum` 的顺序 FP32 vector-add 语义，输入和累加值放在
独立 32-byte 对齐 UB 地址，显式执行 MTE2/V/MTE3 事件；公开 FP32 `g_cumsum` 写回后再作为 KKT
compute view 读回。未修改 ACLNN、tiling、KKT 数学、Phase 1/2 或其他融合边界。

### Phase 3 安装后符号与最小 smoke（精度门禁失败）

修复环境加载后，安装库 SHA256 为
`eabc87564a5967513c9a4954a18ce26e25318e248c7a38acbe07b018e8a84a43`。以下 6 个符号均存在：

```text
aclnnChunkCumsumKktGetWorkspaceSize / aclnnChunkCumsumKkt
aclnnGdnCoreFwdPhase1GetWorkspaceSize / aclnnGdnCoreFwdPhase1
aclnnGdnCoreFwdPhase2GetWorkspaceSize / aclnnGdnCoreFwdPhase2
```

只运行首个 dense FP16/C64 case `B=1,H=2,T=128,K=128,C=64`。融合调用成功，两个输出由稳定
wrapper 按 contract 分配为 FP32，有限性检查通过；随后测试脚本额外执行 CPU exact/精度比较时，
`g_cumsum` 在 `256` 个元素中有 `252` 个不一致，max-abs 为 `0.05804957449436188`。该计数正好只
保留 2 heads × 2 chunks 的 4 个 chunk 首元素相等，优先怀疑融合输出仍是 raw `g` 或 prefix
cumsum 未生效；这只是待验证假设。脚本在首个断言停止，未比较 `A_raw`，未启动其他 7 个 case。

证据文件：

- `/opt/chw/gdn-phase3-postinstall-smoke.log`
- `/opt/chw/gdn-phase3-postinstall-smoke.rc`

静态复核确认 ACLNN/L0/OpDef/kernel 的两个输出顺序均为 `g_cumsum, A`，未发现输出交换；下一步
用两小算子 NPU 基线做单例定点诊断，不能用 CPU 并行 cumsum 的数值顺序直接判定 kernel 根因。

### Phase 3 首次完整包构建（安装后门禁待续）

本地与 A2 远端 7 个关键源码/测试文件 SHA256 一致，远端 `bash -n`、Python 语法、ABI `10/10`
和 CPU 参考 `2/2` 均通过。A2 device 1 启动前无运行进程。完整包构建成功，且三份
`chunk_cumsum_kkt.cpp` 构建拷贝均与源码 SHA256
`a168288d476921a508a4b99f1c90ff0019664f9ca3b32da1864023f03dbbcd1d` 一致；run 包 SHA256 为
`82a8f17e054c04e4918dace8ac653f92304b047c9167f5d35ee88ac0783d809f`。完整包安装成功并列出
`chunk_cumsum_kkt`、`gdn_core_fwd_phase_versions` 和 Phase 1/2 所需算子。

安装后的 wrapper 在 `set -u` 下 source vendor `set_env.bash` 时，因为
`ASCEND_CUSTOM_OPP_PATH` 未定义而停止，退出码为 `1`。失败发生在安装库符号检查和 Python/NPU
smoke 之前，因此本小步尚无符号或算子运行结论。证据文件：

- `/opt/chw/gdn-phase3-build-smoke.log`
- `/opt/chw/gdn-phase3-build-smoke.rc`

已在本地 wrapper 中初始化 `ASCEND_CUSTOM_OPP_PATH`、`LD_LIBRARY_PATH`，并为尚未安装进 wheel
的 Phase 3 Python wrapper 设置源码 `PYTHONPATH`；下一小步只同步该 wrapper 并续跑安装后门禁。

### Phase 3 独立可调用层小步（已完成）

本小步未修改现有 `ChunkCumsumKkt` OpDef、tiling、kernel、L0 launcher 或 Phase 1/2 调度。
新增正式 `aclnnChunkCumsumKkt/GetWorkspaceSize`，其首版 contract 固定为
`k=[B,H,T,128]` FP16/BF16、raw `g/beta=[B,H,T]` FP32、`chunkSize=64/128`，输出
`g_cumsum=[B,H,T]` 与 `A_raw=[B,H,T,C]` 均为 FP32；varlen 只接受物理 `B=1` 和 canonical
sequence-major 元数据。接口已接入 `npu_custom.yaml`、legacy `FLANpuOpApi.cpp`、默认解耦
ctypes/Python 稳定入口、`test.sh`、ABI 测试、CPU 参考测试和 ACLNN 文档。

本地门禁结果：

- `python -m py_compile`：新增/修改的 Python wrapper、ABI 测试和参考测试全部通过；
- `python -m unittest torch_custom/fla_npu/test/test_gdn_core_fwd_ctypes_abi.py -v`：`10/10 PASS`；
- `python torch_custom/fla_npu/test/test_npu_chunk_cumsum_kkt.py --cpu-only`：`2/2 PASS`，覆盖 dense、
  varlen 边界重置、FP32 cumsum、严格下三角及尾部清零 contract；
- `git diff --check`：通过，仅有既有 Windows line-ending 提示；
- 本小步未构建、安装或运行 NPU，符合启动卡边界。

## 7. Phase 2 收口判据（已完成）

Phase 2 只有在以下证据均写入正式验收报告后才能关闭：

- dtype/chunk/layout 交叉组合通过；
- batch/head 和长序列代表点通过；
- launch blocking 关闭后，Phase 1/2 在匹配的干净进程中分别测量，Phase 2 相对基线无性能回退；
- Phase 2 单路径重复稳定，所有输出/state 有独立 NaN/Inf 检查，旧基线非有限用例改用 CPU FP64
  或其他已验证高精度参考；
- 精度、state、workspace/peak memory 和 profiler 结论归档；
- 正式测试脚本、报告、commit SHA、tag 和产物 SHA256 可追溯。
