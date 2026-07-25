# A2 GDN Phase 2 验收报告

## 1. 结论

Phase 2 `chunk_scaled_dot_kkt + solve_tri` 于 2026-07-25 在 A2（Ascend 910B）完成实现、功能/精度验收和代表性性能筛查。已测 4 个正式性能 case 均无回退，但尚不能声明完成“全场景性能验收”：规格笛卡积、长序列/多头扩展性以及生产性能模式仍待补测。

已验证的融合边界为：

```text
k + g_cumsum + beta
    -> KKT
    -> solve_tri
    -> A
```

融合实现使用单个 `ChunkKktSolveTri` MIX kernel，对外 ACLNN 调用数由 2 降为 1。当前保留独立的私有 KKT workspace；在没有全局阶段 barrier 或可证明正确的 tile-local producer/consumer 设计前，不允许将该 workspace 与输出 `A` 原地别名。

## 2. 构建与部署

- 源码目录：A2 隔离验收工作树
- 完整验收包：`build_out/fla-npu-fla_npu_linux-aarch64.run`
- 包 SHA256：`ebcc8ecf20b8030acf51f75e0f62fe155f28d33379686cce647b6ecdd6e7c93c`
- 包大小：`4,505,980` bytes
- 融合 kernel 源码与构建拷贝 SHA256 一致：`629a52ff9e9eb2655d4bb56c0367d8b6858c682994c8b330657dc1b4ae32bc06`
- 同一安装包已确认包含 `GdnCoreFwd`、`ChunkScaledDotKkt`、`SolveTri` 和 `ChunkKktSolveTri` 的 ACLNN 符号。

使用完整包是验收前提。仅安装单个融合算子的局部 OPP 包会覆盖共享 `libcust_opapi.so`，并丢失基线符号。

## 3. 功能与精度

独立 Phase 2 A/B 用例 `6/6 PASS`，覆盖：

| 维度 | 覆盖 |
| --- | --- |
| layout | dense、varlen |
| dtype | BF16、FP16 |
| chunk | 64、128 |
| shape | `B=1/2/3`、`H=1/4/8`、整 chunk 和尾块 |
| varlen | 等价 TND 物理 `B=1`，多序列 `cu_seqlens` |

端到端消融矩阵 4 个场景均通过，融合路径与六算子基线的 GDN core 输出、`g_cumsum` 和有效 `A` 区域二进制一致：

| 场景 | 结果 |
| --- | --- |
| dense, BF16, C64, GVA 外部扩头, `B=1,T=1024` | PASS |
| dense, FP16, C128, 尾块, `B=2,T=257` | PASS |
| varlen, BF16, C64, `[1,65,193]` token 分布 | PASS |
| varlen, FP16, C128, `[65,194]` token 分布 | PASS |

varlen BF16/C64 的独立两算子 `solve_tri` 基线在有效区域产生 8 个非有限值，因此该阶段使用 CPU FP64 逆矩阵参考验证融合输出：最大绝对误差 `2.81e-4`，cosine `1.000028`，融合输出全部有限。独立 6 例回归中的 varlen FP16/C128 也观察到基线非有限，融合输出对 CPU 参考最大绝对误差 `4.16e-5`。这是旧基线的稳定性问题，不是 Phase 2 融合回归。

## 4. 性能与显存详细结果

### 4.1 测量口径

- 硬件：A2（Ascend 910B），device 2。
- CANN：`9.1.0.beta1`。
- 安装包 SHA256：`ebcc8ecf20b8030acf51f75e0f62fe155f28d33379686cce647b6ecdd6e7c93c`。
- 计时：`torch.npu.Event` NPU elapsed time，warmup 10 次，正式迭代 30 次，每个 variant 前后同步。
- 执行顺序：先基线，再统一 ACLNN，最后融合 variant；本轮未做交替/随机顺序测量。
- 调试环境：正式 4 case 运行时启用了 `ASCEND_LAUNCH_BLOCKING=1`。因此当前数据适合阶段性回归筛查，不能代替关闭 launch blocking 后的生产性能报告。

本报告区分 3 类 variant：

| variant | 计算路径 | 对外 ACLNN 数 |
| --- | --- | ---: |
| `legacy_kkt_then_solve_tri` | 独立 KKT + 独立 solve_tri | 2 |
| `fused_kkt_solve_tri` | Phase 2 单融合 kernel | 1 |
| `legacy_six_aclnn` | Python 串联六算子基线 | 6 |
| `fused_kkt_solve` | Python 串联，KKT+solve 替换为融合算子 | 5 |
| `composite_one_aclnn` | 统一 `aclnnGdnCoreFwd`，内部已调用融合 KKT+solve | 1 |

`composite_one_aclnn` 没有“同一 ACLNN 内部仍使用独立 KKT+solve”的 Phase 2 前置对照，所以只列绝对值，不把它与 Python 六调用之间的全部差异归因于 Phase 2 kernel 融合。

### 4.2 正式性能 case

| ID | layout/dtype/chunk | B | 逻辑 Hk/Hv | 物理 q/k 头 | T / 序列长度 | 尾块 |
| --- | --- | ---: | ---: | ---: | --- | --- |
| P1 | dense BF16 C64 | 1 | 4/8 | 8 | 1024 | 无 |
| P2 | dense FP16 C128 | 2 | 4/8 | 8 | 257 | 有 |
| P3 | varlen BF16 C64 | 1 | 4/8 | 8 | 259 / `[1,65,193]` | 有 |
| P4 | varlen FP16 C128 | 1 | 4/8 | 8 | 259 / `[65,194]` | 有 |

这里的 GVA 是入口外部将 q/k 从 Hk=4 扩展到 Hv=8，扩头发生在计时区间之外。它不能代表“原生 Hk != Hv kernel”的性能。

### 4.3 Phase 2 局部阶段 latency

绝对时间（ms）：

| ID | 基线 mean | 融合 mean | 基线 median | 融合 median | 基线 P90 | 融合 P90 | 基线 min | 融合 min |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| P1 | 1.434 | 0.759 | 1.406 | 0.751 | 1.499 | 0.773 | 1.352 | 0.728 |
| P2 | 1.308 | 0.711 | 1.287 | 0.706 | 1.374 | 0.754 | 1.229 | 0.661 |
| P3 | 1.848 | 0.786 | 1.830 | 0.781 | 1.955 | 0.813 | 1.722 | 0.761 |
| P4 | 1.885 | 0.909 | 1.863 | 0.901 | 1.974 | 0.959 | 1.760 | 0.839 |

降低比例：

| ID | mean | median | P90 | min |
| --- | ---: | ---: | ---: | ---: |
| P1 | 47.1% | 46.6% | 48.4% | 46.1% |
| P2 | 45.6% | 45.2% | 45.1% | 46.2% |
| P3 | 57.5% | 57.3% | 58.4% | 55.8% |
| P4 | 51.8% | 51.6% | 51.4% | 52.3% |

局部阶段样本稳定，4 case 的 mean/median/P90/min 都改善，因此“KKT+solve 融合本身有收益”的结论在已测场景上是明确的。

### 4.4 回放到完整 GDN core

Python 直接串联路径：六 ACLNN 基线 vs 五 ACLNN Phase 2（ms）。

| ID | 基线 mean | 融合 mean | 基线 median | 融合 median | 基线 P90 | 融合 P90 | 基线 min | 融合 min |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| P1 | 9.248 | 4.126 | 4.520 | 4.230 | 7.765 | 4.469 | 4.328 | 3.732 |
| P2 | 8.550 | 3.793 | 4.040 | 3.881 | 6.884 | 4.026 | 3.926 | 3.517 |
| P3 | 7.791 | 4.630 | 4.824 | 4.769 | 5.514 | 5.006 | 4.655 | 4.183 |
| P4 | 9.729 | 4.594 | 6.519 | 4.749 | 10.767 | 5.020 | 5.514 | 4.015 |

降低比例：

| ID | mean | median | P90 | min |
| --- | ---: | ---: | ---: | ---: |
| P1 | 55.4% | 6.4% | 42.5% | 13.8% |
| P2 | 55.6% | 3.9% | 41.5% | 10.4% |
| P3 | 40.6% | 1.2% | 9.2% | 10.1% |
| P4 | 52.8% | 27.2% | 53.4% | 27.2% |

六 ACLNN 基线的 30 次样本中存在约 `47–50 ms` 长尾，因此 mean 改善幅度被显著放大。本报告以 median 作为主判据：4 case 均未回退，但 P2/P3 只有 `3.9%/1.2%` 改善，需要在关闭 launch blocking、交替测量后确认是否高于噪声。

统一 `aclnnGdnCoreFwd` 绝对 latency（内部已使用 Phase 2 融合 kernel）：

| ID | mean | median | P90 | min |
| --- | ---: | ---: | ---: | ---: |
| P1 | 2.599 | 1.188 | 1.272 | 1.142 |
| P2 | 2.530 | 1.076 | 1.127 | 1.036 |
| P3 | 2.570 | 1.115 | 1.285 | 1.074 |
| P4 | 3.931 | 1.209 | 1.412 | 1.129 |

这一表证明融合实现已经接入统一 ACLNN 且可运行，但由于缺少“统一 ACLNN + 未融合 KKT/solve”的同环境对照，不用该表计算 Phase 2 加速比。

### 4.5 Workspace 与峰值显存

Phase 2 局部阶段（MiB）：

| ID | 基线 workspace max | 融合 workspace max | max 变化 | 基线 workspace sum | 融合 workspace sum | sum 下降 | 基线 peak delta | 融合 peak delta | peak 下降 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| P1 | 18.00 | 19.78 | +9.9% | 34.32 | 19.78 | 42.4% | 39.32 | 21.00 | 46.6% |
| P2 | 19.00 | 23.13 | +21.7% | 36.09 | 23.13 | 35.9% | 42.02 | 25.00 | 40.5% |
| P3 | 16.88 | 17.91 | +6.1% | 33.20 | 17.91 | 46.0% | 34.72 | 18.25 | 47.4% |
| P4 | 17.50 | 21.13 | +20.7% | 34.34 | 21.13 | 38.5% | 37.86 | 22.51 | 40.5% |

融合后的单 ACLNN workspace max 反而增加 `6.1%–21.7%`，因为正确性要求保留私有 KKT hand-off buffer；但两个独立 workspace 合并后，workspace sum 和峰值分配明显下降。不能只报“最大 workspace 低于 50 MB”而隐去 workspace max 增加这一事实。

完整 GDN core Python 串联路径（MiB）：

| ID | ACLNN 数 | 融合 ACLNN 数 | workspace max 基线/融合 | workspace sum 基线/融合 | sum 下降 | peak delta 基线/融合 | peak 下降 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| P1 | 6 | 5 | 53.00 / 53.00 | 173.08 / 158.54 | 8.4% | 190.42 / 174.10 | 8.6% |
| P2 | 6 | 5 | 58.02 / 58.02 | 182.62 / 169.66 | 7.1% | 195.59 / 179.58 | 8.2% |
| P3 | 6 | 5 | 53.00 / 53.00 | 170.46 / 155.18 | 9.0% | 176.77 / 160.81 | 9.0% |
| P4 | 6 | 5 | 58.02 / 58.02 | 180.37 / 167.16 | 7.3% | 187.19 / 171.84 | 8.2% |

统一 `aclnnGdnCoreFwd` 的绝对 workspace 为 `124.20–139.68 MiB`，大于 50 MB；但它将六个阶段的内部中间区统一放入一个 executor workspace，而其峰值分配仍低于 Python 六算子基线。因此：

- 如果“额外显存不超过 50 MB”指融合相对基线的增量，当前为负增量，满足要求；
- 如果指单个 ACLNN 绝对 workspace，当前统一 ACLNN 不满足，需要先向需求方确认口径。

### 4.6 Profiler

Profiler 只在 P1 上执行：

| 范围 | 基线 ACLNN | 融合 ACLNN | 基线 NPU kernel | 融合 NPU kernel |
| --- | ---: | ---: | ---: | ---: |
| KKT + solve_tri 局部阶段 | 2 | 1 | 3 | 1 |
| Python 完整 core 路径 | 6 | 5 | 12 | 10 |
| 统一 `aclnnGdnCoreFwd` | 不适用 | 1 | 不适用 | 9 |

### 4.7 其他已运行但不纳入正式结论的 case

| case | 结果 | 排除理由 |
| --- | --- | --- |
| dense BF16 C128, `B=1,Hk/Hv=4/8,T=1024` | 六调用 / 统一 ACLNN / 五调用 median = `4.683/1.213/4.255 ms` | 在完整验收包之前运行，旧脚本未输出局部阶段结果 |
| dense BF16 C64 早期运行 | 五调用 median 异常为 `15.004 ms` | 当时局部 OPP 包/共享符号和构建版本问题尚未收敛，已被最终同合包 P1 结果取代 |
| varlen FP16 C64 早期运行 | 未产生性能数据 | benchmark 将 Python list 错传给 `chunk_local_cumsum`，用例在计时前失败 |

同一 contract 的 `phase2_accept_*.json` 和 `phase2_final_*.json` 是修复过程和最终完整包的重复运行，不是新的 shape case；本报告的正式表格使用 `phase2_final_*.json`。

### 4.8 待补的性能矩阵

| 维度 | 当前状态 | 待补内容 |
| --- | --- | --- |
| dtype × chunk × layout | 4 个对角组合 | 至少补 dense FP16 C64、dense BF16 C128、varlen FP16 C64、varlen BF16 C128 |
| batch/head 扩展 | 性能 case 主要是物理 H=8，dense B=1/2 | 补 H=1/4/16/32 代表点及更大 dense B |
| 序列长度 | `T=257/259/1024` | 补 `T=4096/32768` 等实际长序列和更多 chunk 数 |
| GVA | 外部扩头，且扩头不在计时区间 | 原生 `Hk != Hv` 实现后重新验收，并计入布局/扩头成本 |
| V 维度 | 仅 `V=128` | `V=256` 在后续规格扩展完成后验收 |
| state/完整 Demo | 未做性能矩阵 | 补 initial/final state，以及 causal conv + RMSNorm/gate 的完整 Demo |
| 测量方法 | launch blocking，固定 variant 顺序 | 关闭 launch blocking，使用成对交替/随机顺序、更多迭代和独立 profiler |

因此当前可以归档的严格结论是：**Phase 2 局部融合在 4 个代表 case 上有明确收益，回放到 core 后未观察到回退；需求中的“全场景性能不劣化”尚未被完整证明。**

## 5. 范围边界

- 当前仅验收 `K == V == 128`；`K=128,V=256` 留待后续阶段。
- Phase 2 保持 Phase 1 的物理头契约：GVA 由外部将 q/k 扩展到 `Hv`；尚不是融合 kernel 内部原生 `Hk != Hv`。
- transpose/layout 位置未在 Phase 2 同时修改。
- 当前私有 KKT hand-off workspace 是正确性所需，后续只能通过全局阶段 barrier 或 tile-local 流水设计去除。

## 6. 下一步

先补齐 4.8 中的 Phase 2 性能门禁，至少完成 dtype/chunk/layout 交叉组合、头数/长序列代表点和关闭 launch blocking 的交替测量。在此期间可以做 Phase 3 的方案分析，但在 Phase 2 全性能门禁收口前，不开始不可逆的 Phase 3 kernel 替换。
