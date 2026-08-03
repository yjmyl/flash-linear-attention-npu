# A2 GDN Phase 6 验收报告

> 最后更新：2026-08-03
> 状态：功能、精度、性能、clean 包与隔离 wheel 回归均已关闭；Git 里程碑和不可变 annotated tag `gdn-a2-phase6` 已创建。
> 基线：不可变 `aclnnGdnCoreFwdPhase5` / `gdn_core_fwd_phase5`。

## 1. 阶段边界

```text
Phase 5: (A+B+C) + (D+E+F) + public g_cumsum transpose
Phase 6: (A+B+C+D+E+F)，public g_cumsum BHT->BTH 写回内置
```

`A=ChunkLocalCumsum`、`B=ChunkScaledDotKkt`、`C=SolveTri`、`D=RecomputeW/U`、
`E=FwdH`、`F=FwdO`。Phase 6 新增版本化 `aclnnGdnCoreFwdPhase6` 和单个 MIX kernel
`ChunkGdnCoreFwd`；默认兼容入口仍保持 Phase 2，未自动切换。

公开 `g_cumsum` 仍落 GM，但不再单独启动 ABC 与 DEF 之间的 transpose。owner AIV 在 UB 内完成
BHT 到 BTH 的整行组织和写回；因此不能把公开输出的 GM 字节错误记作“消除”。

## 2. 最终调度与同步

- `kAbcTaskOrder=true` 使 recompute 沿用 ABC 的连续核内任务分片；同一 AIC 生产并消费自己的
  `A` tile，最终实现没有 ABC/DEF 边界全核同步。
- 初始 `T=1025` 探针发现 `A` 的跨核生产/消费竞争。临时 `SyncAll<false>()` 仅用于证明因果，
  最终被同核 producer affinity 替换。
- varlen 的独立重复探针发现共享 H 到 O 边界可见性问题，而非 ABC 数学或公开 transpose 问题。
  `GDNFwdHKernel::Process()` 仅在 `isVariedLen` 时增加末尾 `SyncAll<false>()`；Phase 4、5、6
  各八次重复均得到唯一 `o` 哈希，且固定 DEF 回放与 Phase 5 一致。
- 这两个同步结论不能混淆：Phase 6 没有保留 ABC/DEF 全核 barrier；varlen H/O 的保护属于既有
  后缀边界的活性/可见性合同。

## 3. 功能与精度

全部以同一安装包内的 Phase 5 为基线，比较 `o`、公开 `g_cumsum`、有效 `A` 和存在时的
`final_state`。通过条件为逐位一致、`output_max_abs=0` 和全部有限。

| 范围 | 覆盖 | 结果 |
| --- | --- | --- |
| dense C64 | FP16 `T=128/1025`，BF16 `T=128/1024` | `4/4` 无 state bit-exact / finite |
| dense C128 | FP16 `T=128/1025`，BF16 `T=128/1024` | `4/4` bit-exact / finite |
| varlen | FP16/BF16 x C64/C128，`T=259`，canonical `cu_seqlens` | `4/4` bit-exact / finite |
| state 主合同 | varlen FP16/BF16 x C64/C128，`initial_state + final_state` | `4/4` bit-exact / finite |
| state 边界 | zero initial + final；initial only | `2/2` bit-exact / finite |

state 主合同的 final-state 元素数为 `393216`。最终隔离 wheel runtime smoke 再次覆盖 varlen
BF16/C64 `T=259, cu_seqlens=[0,1,66,259]`，所有公开输出 bit-exact、最大绝对误差为 `0`、
全部有限。

## 4. 正式完整 ACLNN 性能

测试在 A2 device 2 进行；每个 identity 从同进程 AB/BA NPU Event 轮次中选择三轮稳定样本，
每轮 `20 warmup + 200` 个样本，汇总为每 identity `600` 个样本。稳定门槛为 Phase 5 和
Phase 6 均满足 `median/P90 >= 0.8`。负值表示 Phase 6 延迟下降。

| identity | Phase 5 median | Phase 6 median | 变化 | workspace Phase 5 -> Phase 6 |
| --- | ---: | ---: | ---: | ---: |
| dense FP16/C64, T1025 | 1.02078 ms | 0.99109 ms | -2.909% | 79,010,304 -> 83,140,608 B |
| dense BF16/C64, T1024 | 1.03793 ms | 1.01446 ms | -2.261% | 78,746,112 -> 82,743,808 B |
| dense FP16/C128, T1025 | 0.99956 ms | 0.98088 ms | -1.869% | 82,169,856 -> 92,297,728 B |
| dense BF16/C128, T1024 | 1.12351 ms | 1.09342 ms | -2.678% | 81,905,664 -> 91,506,688 B |
| varlen FP16/C64, T259 | 0.92256 ms | 0.88286 ms | -4.303% | 74,746,368 -> 76,756,992 B |
| varlen BF16/C64, T259 | 0.92037 ms | 0.90188 ms | -2.009% | 74,746,368 -> 76,756,992 B |
| varlen FP16/C128, T259 | 0.97502 ms | 0.94035 ms | -3.556% | 79,478,784 -> 85,916,160 B |
| varlen BF16/C128, T259 | 0.96772 ms | 0.94425 ms | -2.425% | 79,478,784 -> 85,916,160 B |

八个 identity 全部改善，矩阵 pairwise median 为 `-2.733%`。Phase 6 workspace 全部上升；
这是一次性能优先的接受结果，不为压低 workspace 新增 dtype、shape 或 layout 运行时分支。

## 5. Profiler 机理证据

最终安装包在 A2 device 2 的 dense FP16/C64/T128 trace 中复验完整 ACLNN 设备任务数：

```text
Phase 5: Transpose + Cast + Transpose + ChunkCumsumKktSolveTri + Transpose + ChunkRecomputeWUFwdHO = 6
Phase 6: Transpose + Cast + Transpose + ChunkGdnCoreFwd                                             = 4
```

P0 机理 trace 中，目标段 `ABC + public transpose + DEF` 为 `102.262 us`，
`ChunkGdnCoreFwd` 为 `84.162 us`。该 trace 只说明融合机理；正式性能结论以上节的同进程
AB/BA 矩阵为准。

## 6. Clean 产物与交付回归

全新 clean 源树通过两次 `git diff --check` 和源 ABI 测试后完整构建。最终身份如下：

| 产物 | SHA256 |
| --- | --- |
| run 包 | `91b903dea03ff83431589d3188f5c16b90303fade4387dbe9c463e9c1f4f2897` |
| wheel | `a785f4b81272f0ec0487a8cbcf0f570e9f6859a239c77a7902ef0855c67856be` |
| `libcust_opmaster_rt2.0.so` | `90271992010e8c67ede9b116df392c6d8622850a104c1efbca7f17bee4d6347a` |
| 安装态 `libcust_opapi.so` | `8d9a92ffb20db33aac62755b1eccdb0fc5a1bf190409c01921fcb7d211d7dc1e` |

- 安装态 `ChunkGdnCoreFwd` 精确包含 `4 .o + 4 .json`。
- 源树和隔离 wheel 的 ctypes ABI 都是 `11 passed, 6 subtests passed`。
- wheel 内 `libcust_opapi.so` 与隔离安装态文件经 `cmp` 字节一致；`PYTHONPATH` 解析到
  `/opt/chw/phase6_final_wheel_site_20260801_r2/fla_npu/__init__.py`。
- `gdn_core_fwd_phase6` import 通过；wheel runtime varlen smoke 通过；
  `test_npu_gdn_demo_composite.py` 在 `GDN_TEST_TORCH_L2NORM=1` 下 `2/2 PASS`。

## 7. 证据位置

- 正式性能汇总：`/opt/chw/gdn-phase6-full-perf-summary-d2-r1/summary.json`
- varlen 精度：`/opt/chw/gdn-phase6-varlen-ho-sync-accuracy-d2-r1`
- varlen/state 合同：`/opt/chw/gdn-phase6-varlen-ho-sync-state-accuracy-d2-r1`
- 最终 profiler：`/opt/chw/gdn-phase6-final-profile-d2-r1`
- clean 构建、隔离安装与 wheel 回归：`/opt/chw/gdn-phase6-final-clean-build-d2-r2`
- 隔离 OPP：`/opt/chw/phase6_final_vendor_20260801_r2`
- 隔离 wheel site：`/opt/chw/phase6_final_wheel_site_20260801_r2`

## 8. 结论与边界

Phase 6 已满足冻结的 A2 `K==V==128`、外部 GVA、FP16/BF16、C64/C128、dense/varlen、
canonical metadata、initial/final state 合同，以及完整 core Phase 5/6 性能比较。验收和 Git
归档均已关闭；默认入口继续保持 Phase 2，不因本 Phase 自动切换。

本 Phase 明确不覆盖 `V=256`、原生 GVA、causal_conv1d、RMSNorm/gate、backward 或完整模型性能；
它们必须作为后续独立规格，而不是反向扩大本次验收范围。
