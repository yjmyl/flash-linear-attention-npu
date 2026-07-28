# A2 GDN Phase 版本归档

本文档只记录可直接构建和 A/B 的 GDN core Phase 身份、commit/tag 与归档规则，
不承载当前进度。主原则是：一个 Phase 一个版本化 ACLNN，新 Phase 不覆盖旧 Phase。

## Git 归档规则

1. GDN A2 的 Phase 快照统一保存在个人 remote `chw` 的专用归档分支上；当前分支为 `gdn-a2-phase-archive`。
2. 每个 Phase 完成实现和验收后必须新增一个里程碑 commit。新 Phase 只能追加新 commit，禁止 amend、rebase、force-push 或改写已归档 Phase。
3. 每个里程碑 commit 必须保留该 Phase 的版本化 ACLNN、Python wrapper、测试、接口文档和验收报告，并确保旧 Phase 仍可从同一源码和同一 wheel 构建、调用和 A/B。
4. 验收报告必须记录 Phase、commit SHA、基线 commit、wheel/OPP SHA256、A2 环境、精度矩阵、性能矩阵、kernel 数和 workspace/显存结果。
5. 机器相关临时脚本、构建目录、原始 profiler 目录和临时 JSON 不进入 Phase commit；可复现命令和汇总结论应整理到正式测试脚本或验收文档。
6. 已归档 Phase 如需修复，只允许追加明确标注的修复 commit，并记录原因和回归证据；不得修改历史 commit。新融合边界始终进入下一个 Phase。

Phase 1/2 在本规则建立前位于同一个未提交工作区，因此首个 Git 归档点包含可并存的 Phase 1 和 Phase 2。它作为 Phase 2 里程碑，同时也是 Phase 1 的首个可追溯恢复点。Phase 3 起严格执行“一 Phase 一新增 commit”。

### Git 里程碑

| Phase | 分支 | Git 里程碑 | 说明 |
| --- | --- | --- | --- |
| Phase 1 | `gdn-a2-phase-archive` | `gdn-a2-phase2^{commit}` | 历史恢复点；版本化 ACLNN 已与 Phase 2 并存 |
| Phase 2 | `gdn-a2-phase-archive` | `gdn-a2-phase2^{commit}` | 不可变代码快照；生产性能证据由后续 commit `2b8161d` 追加，不移动该 tag |
| Phase 3 | `gdn-a2-phase-archive` | `gdn-a2-phase3^{commit}` = `7fb8f05b59ab56a8392e0f6c9bef071714894826` | 不可变 annotated tag 已远端推送并逐 SHA 回查 |
| Phase 4 | `gdn-a2-phase-archive` | `gdn-a2-phase4^{commit}` = `9719f2701f62ec7ef3d67751af52d1a1ea3c9435` | 不可变 annotated tag 已推送并完成远端逐 SHA 回查 |

`gdn-a2-phase2` 是只指向本次里程碑 commit 的不可变 tag；可用 `git rev-parse gdn-a2-phase2^{commit}` 获取精确 commit SHA。后续 Phase 使用新的 `gdn-a2-phaseN` tag，禁止移动已有 tag。

## 固定版本

| Phase | ACLNN / Python | 内部路径 | 状态 |
| --- | --- | --- | --- |
| Phase 1 | `aclnnGdnCoreFwdPhase1` / `gdn_core_fwd_phase1` | `local_cumsum -> KKT -> cast -> solve_tri -> recompute_w_u -> fwd_h -> fwd_o` | 已恢复并通过 A2 smoke |
| Phase 2 | `aclnnGdnCoreFwdPhase2` / `gdn_core_fwd_phase2` | `local_cumsum -> ChunkKktSolveTri -> recompute_w_u -> fwd_h -> fwd_o` | 已固化，并按冻结范围完成 A2 功能/精度/生产性能收口 |
| Phase 3 | `aclnnGdnCoreFwdPhase3` / `gdn_core_fwd_phase3` | `ChunkCumsumKktSolveTri -> recompute_w_u -> fwd_h -> fwd_o` | 已按冻结范围完成 A2 功能/精度/生产性能/profiler 和 Git 归档收口 |
| Phase 4 | `aclnnGdnCoreFwdPhase4` / `gdn_core_fwd_phase4` | `ChunkCumsumKktSolveTri -> recompute_w_u -> ChunkGatedDeltaRuleFwdHO` | 已按冻结范围完成 A2 功能/精度/生产性能/profiler 和 Git 归档收口 |
| 默认入口 | `aclnnGdnCoreFwd` / `gdn_core_fwd` | 当前与 Phase 2 相同 | 兼容入口，不作为永久 Phase 快照 |

Phase 1 的原始统一 ACLNN 曾被 Phase 2 原地切换到融合 KKT + solve_tri，导致同一 ACLNN 内的 Phase 1 对照消失。2026-07-25 起通过版本化入口纠正：Phase 1、Phase 2 和 Phase 3 均以独立版本化入口在同一包内并存。

## 代码与验证入口

- 默认 C API：`chunk_gated_delta_rule_fwd_h/op_host/op_api/aclnn_gdn_core_fwd.h/.cpp`
- Phase 固定 C API：`chunk_gated_delta_rule_fwd_h/op_host/op_api/aclnn_gdn_core_fwd_phase_versions.h`
- Python ctypes：`torch_custom/fla_npu/fla_npu/ops/ascendc/_aclnn_ctypes.py`
- ABI 测试：`torch_custom/fla_npu/test/test_gdn_core_fwd_ctypes_abi.py`
- 统一消融：`torch_custom/fla_npu/test/benchmark_gdn_core_ablation.py`

消融脚本中的固定 variant：

```text
phase1_one_aclnn_six_kernels
phase2_one_aclnn_fused_kkt_solve
phase3_one_aclnn_fused_cumsum_kkt
phase4_one_aclnn_fused_fwd_ho
```

Phase 3 variant 名为兼容既有结构化报告而保留；最终内部路径是累积单 kernel
`ChunkCumsumKktSolveTri`，不再是早期 `ChunkCumsumKkt -> Cast -> SolveTri` 拆分候选。

## 归档门禁

每个 Phase 必须保留以下证据：

1. 版本化 ACLNN 和 Python 符号可从同一个 wheel 调用。
2. 与前一 Phase 在同输入、同安装包、同 device 上交替测量。
3. 精度、latency、kernel 数、workspace/峰值显存结果归档。
4. 记录源码提交或明确的归档点，以及 wheel SHA256。
5. 下一 Phase 开始后仍可复跑旧 Phase，不需要重新安装旧包。

## 2026-07-25 恢复验证

- A2/CANN `9.1.0.beta1` 完整算子包构建成功。
- 同一 `libcust_opapi.so` 同时导出默认、Phase 1、Phase 2 共 6 个执行和 `GetWorkspaceSize` 符号。
- 安装后的 `libcust_opapi.so` SHA256：`87c387757fcc19227a166aa90be149b4ce02bc386f8982ac731fbb8741a3349e`。
- `aclnn_gdn_core_fwd.cpp` 源码 SHA256：`f027b30e09abee351efaa2e839a4a46a80f3cb614006281a231de3f964d38c0c`。
- `aclnn_gdn_core_fwd_phase_versions.h` 源码 SHA256：`a0b68515e9ebb3b51841e0cb1fb35dabb5dd8d8dfa63165015848bd67a0b8c36`。
- dense BF16：`B=1,H=2,T=128,C=64`；Phase 1/2 的输出、`g_cumsum`、有效 `A` 对六调用基线逐位一致，最大输出绝对误差 `0`。
- varlen FP16 尾块：`cu_seqlens=[0,65,130],H=2,C=64`；Phase 1/2 同样逐位一致，最大输出绝对误差 `0`。
- 本地和 A2 ctypes ABI 单测均 `10/10 PASS`；安装包 API 检查通过。

这两例是恢复完整性 smoke，不替代 Phase 1 原 `6/6` 验收和 Phase 2 正式性能报告。上述源码哈希和动态库哈希与 Git 里程碑共同构成本次恢复点的可追溯证据。

## 2026-07-25 Phase 2 性能收口

- 生产性能关闭 `ASCEND_LAUNCH_BLOCKING`；dense/varlen 的 FP16/BF16 × C64/C128 交叉点和 `B=4,H=4,T=4096` 扩展点的主判据 median 均无可复现回退。
- varlen 使用干净进程 AB/BA 平衡测量，规避已证明的 Phase 1 workspace 初值依赖；Phase 2 的 varlen standalone 生产运行均通过独立有限性检查。
- `T=32768` varlen 上 Phase 2 单路径通过，median `6.911 ms`；Phase 1 首次同步 MTE 越界，因此不产生相对基线结论。
- 实际加载的 `libcust_opapi.so` SHA256 为 `87c387757fcc19227a166aa90be149b4ce02bc386f8982ac731fbb8741a3349e`，与恢复验证一致。
- 详细矩阵、workspace/peak、异常复测和剩余范围见 `GDN_PHASE2_ACCEPTANCE_A2.md`。

性能收口只追加测试、报告和归档元数据，不改变 `gdn-a2-phase2` 指向的 `f2a4b46` 代码快照。应在当前归档分支新增普通 commit 保存正式 benchmark/runner、报告和文档，禁止 amend 旧里程碑或移动 tag。

## 2026-07-26 Phase 3 验收收口

- 最终边界为单 `ChunkCumsumKktSolveTri`：raw FP32 cumsum + KKT + low-precision hand-off + solve_tri；
- 完整 run 包 SHA256：`837742c6731143ec0ea55c517338e0daca3d3f295b9b7a71f079805b9b62bfdb`；
- 安装 host 库 SHA256：`645336f9f65c522a06d74cf8b3df0ca6db2ae493fcd6fec980719eca7c8af07f`；
- 独立 `ChunkCumsumKkt` 最终共享 helper `80/80 exact`，core dense/varlen `8/8` 加 state `1/1` bit-exact/有限；
- 共享 helper 局部微基准 `8/8` median 改善；作为最终结论的完整 GDN core Phase 2/3 生产性能也是 `8/8` median 改善，core workspace/peak 对 Phase 2 持平；
- profiler 证明完整 core NPU kernel 数 `9 -> 8`，目标段由两个 kernel 合并为一个；
- 详细证据与范围边界见 `GDN_PHASE3_ACCEPTANCE_A2.md`。

Phase 3 实现与验收里程碑 commit 为 `7fb8f05b59ab56a8392e0f6c9bef071714894826`，parent 为
`c76ab5a15b28ea5d890cbf7939ad340ce6b875be`，严格包含 31 个已审计正式文件。本地 annotated tag
`gdn-a2-phase3` object 为 `b26159171f8aa0b1340f3f927412795853dc72e9`，peeled commit 正是
`7fb8f05...`，message 为 `A2 GDN Phase 3 cumulative fusion`。远端 branch 已 fast-forward 到
`1595456dabbae42e70912c4b8981bbbdaace3279`，随后 Phase 3 tag 单独推送；最终 `git ls-remote`
逐 SHA 回查确认 branch、Phase 2 tag 和 Phase 3 tag 均与冻结记录一致。推送过程未使用 force，
也未移动 `gdn-a2-phase2`。

## 2026-07-28 Phase 4 验收收口

- 最终边界为 `(A+B+C) + D + (E+F)`，`E+F` 由单 `ChunkGatedDeltaRuleFwdHO` MIX kernel 实现；
- 完整 run 包 SHA256：`a297168f3b5d14a09afd23acd060d3ab546bab9cbd71e7c8d301ebe3ce9b9206`；
- 安装 host 库 SHA256：`6f67757282030b90f95f92f403beaf1a3bdeeee9cd52ccf9de87f58226c5a23d`；
- core dense/varlen `8/8` 加 state `1/1` 对 Phase 3 bit-exact/有限；
- dense/varlen `8/8` 生产性能点无可复现实质回退，workspace 全部下降 `24.11%~27.89%`；
- profiler 证明完整 core NPU 任务数 `8 -> 7`，目标段从两个 kernel 合并为一个；
- 详细证据、噪声判定和范围边界见 `GDN_PHASE4_ACCEPTANCE_A2.md`。

Phase 4 实现与验收里程碑 commit 为 `9719f2701f62ec7ef3d67751af52d1a1ea3c9435`，parent 为
`f336dbdb7d13ba30a7c41ccc046c4bfe428858cf`。本地 annotated tag `gdn-a2-phase4` object 为
`dd79407814abfea81e964e207c721fe4e0c9c360`，peeled commit 为上述里程碑。归档只允许快进追加
新 commit/tag，不得移动 `gdn-a2-phase2` / `gdn-a2-phase3` / `gdn-a2-phase4`。
远端回查确认 `refs/tags/gdn-a2-phase4^{}` 为
`9719f2701f62ec7ef3d67751af52d1a1ea3c9435`，归档分支包含该里程碑并只向前追加归档元数据；
Phase 2/3 tag 也保持原 SHA，推送未使用 force。
