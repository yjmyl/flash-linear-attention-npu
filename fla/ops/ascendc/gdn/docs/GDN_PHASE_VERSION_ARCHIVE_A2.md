# A2 GDN Phase 版本归档

本文档记录可直接构建和 A/B 的 GDN core Phase 快照。主原则是：一个 Phase 一个版本化 ACLNN，新 Phase 不覆盖旧 Phase。

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
| Phase 2 | `gdn-a2-phase-archive` | `gdn-a2-phase2^{commit}` | 不可变代码快照；生产性能门禁已收口，待追加性能证据 commit，不移动该 tag |

`gdn-a2-phase2` 是只指向本次里程碑 commit 的不可变 tag；可用 `git rev-parse gdn-a2-phase2^{commit}` 获取精确 commit SHA。后续 Phase 使用新的 `gdn-a2-phaseN` tag，禁止移动已有 tag。

## 固定版本

| Phase | ACLNN / Python | 内部路径 | 状态 |
| --- | --- | --- | --- |
| Phase 1 | `aclnnGdnCoreFwdPhase1` / `gdn_core_fwd_phase1` | `local_cumsum -> KKT -> cast -> solve_tri -> recompute_w_u -> fwd_h -> fwd_o` | 已恢复并通过 A2 smoke |
| Phase 2 | `aclnnGdnCoreFwdPhase2` / `gdn_core_fwd_phase2` | `local_cumsum -> ChunkKktSolveTri -> recompute_w_u -> fwd_h -> fwd_o` | 已固化，并按冻结范围完成 A2 功能/精度/生产性能收口 |
| 默认入口 | `aclnnGdnCoreFwd` / `gdn_core_fwd` | 当前与 Phase 2 相同 | 兼容入口，不作为永久 Phase 快照 |

Phase 1 的原始统一 ACLNN 曾被 Phase 2 原地切换到融合 KKT + solve_tri，导致同一 ACLNN 内的 Phase 1 对照消失。2026-07-25 起通过版本化入口纠正：Phase 1 和 Phase 2 在一个包内并存，后续 Phase 3 必须新增独立入口。

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
```

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
