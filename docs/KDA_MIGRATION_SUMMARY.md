# KDA Mega-Kernel 迁移全流程总结

> 本文档记录将 KDA (Kimi Delta Attention) 融合算子从 `megagdn-pto` 仓库迁移到 `flash-linear-attention-npu` (fla_npu) 框架的完整过程，以便后续复现。

---

## 1. 背景与目标

### 1.1 需求
- **源算子**: KDA 融合算子，位于 `D:\workspace\kdatest\megagdn-pto\kernels\pto`
- **目标仓**: `flash-linear-attention-npu` (fla_npu 框架)
- **最终要求**: 在被迁移到的仓，按照其 README 能正常调用，功能、精度正常
- **验证方式**: 用户内网 NPU 机器 (910B) 上 `git clone` + 一键脚本完成构建、安装、精度测试

### 1.2 参与仓库

| 角色 | 仓库 | 路径 |
|------|------|------|
| **源仓** | megagdn-pto (含 KDA kernel 原始实现) | `D:\workspace\kdatest\megagdn-pto` |
| **参考仓** | xllm_ops (已成功迁移 mega_chunk_gdn) | `D:\workspace\kdatest\xllm_ops` |
| **目标仓** | flash-linear-attention-npu | `D:\workspace\kdatest\flash-linear-attention-npu` |
| **子模块** | pto-isa (PTO 指令集库) | `gitcode.com/cann/pto-isa.git` |

### 1.3 环境

**公网 41 机器 (开发验证)**:
- CANN 9.1.0.beta1
- conda env `chw`, Python 3.13
- NPU 910B3 x4

**用户内网机器 (最终验证)**:
- CANN 9.1.0 (`/opt/c00808382/zhengbao/0623_9.1.0/ascend-toolkit`)
- conda env `c00808382`, Python 3.9.23
- torch 2.8.0+cpu, torch_npu 2.8.0.post4
- NPU 910B
- PT 文件: `/opt/c00808382/selfatk/kda/kda_debug_input_tensors_rank0.pt` + `kda_debug_output_tensors_rank0.pt`

---

## 2. Phase 1 — 分析源算子 + 参考仓模式

### 2.1 源算子分析 (megagdn-pto/kernels/pto)

KDA 是一个 mega-kernel，单次 NPU launch 完成以下 6 个子阶段：
1. **gate_cumsum** — 门控累积和
2. **kkt** — KKT 矩阵构造
3. **tri_inverse** — 三角矩阵求逆 (WY 表示)
4. **wy** — WY 分解
5. **chunk_h** — chunk 隐状态计算
6. **chunk_o** — chunk 输出计算

**关键参数**:
- 算子名: `MegaChunkKda` (PascalCase)
- head_dim (K): 128
- chunk_size (C): 128
- 数据类型: half (fp16)
- 输入: 9 个 (q, k, v, g, beta, A_log, dt_bias, cu_seqlens, state)
- 输出: 9 个 (o, recurrent_state, + 7 个中间结果)

### 2.2 参考仓分析 (xllm_ops/mega_chunk_gdn)

参考仓已成功迁移了 `MegaChunkGdn` 算子，其完整结构如下：

```
fla/ops/ascendc/gdn/{op_name}/
├── CMakeLists.txt              # 顶层：遍历子目录
├── op_host/
│   ├── CMakeLists.txt          # 编译选项 + include 路径
│   ├── {op_name}_def.cpp       # OpDef：输入/输出/属性声明
│   ├── {op_name}_tiling.h      # TilingData 结构体
│   ├── {op_name}_tiling.cpp    # TilingFunc：blockDim/workspace/tiling data
│   ├── {op_name}_infershape.cpp # InferShapeFunc：输出 shape 推导
│   └── op_api/
│       ├── {op_name}_l0.h      # l0 接口声明
│       ├── {op_name}_l0.cpp    # l0 接口实现（调用 kernel entry）
│       ├── aclnn_{op_name}.h   # aclnn 接口声明
│       └── aclnn_{op_name}.cpp # aclnn 接口实现（workspace + l0 调用）
├── op_kernel/
│   ├── {op_name}.cpp           # kernel entry（GetTilingData + 子阶段调用）
│   ├── {sub_kernel_1}.cpp      # 子阶段实现（从源仓复制）
│   └── include/
│       └── kernel_utils.h      # 共享工具函数
```

### 2.3 目标仓适配层分析

fla_npu 框架的算子注册机制：
- `torch_custom/fla_npu/npu_custom.yaml` — 算子签名声明
- `torch_custom/fla_npu/op_plugin/ops/opapi/FLANpuOpApi.cpp` — C++ op_api 适配实现
- `torch_custom/fla_npu/fla_npu/ops/ascendc/__init__.py` — Python wrapper

### 2.4 KDA vs GDN 差异

| 维度 | GDN | KDA |
|------|-----|-----|
| 输入 | 7 个 | 9 个 (多 A_log, dt_bias) |
| 输出 | 7 个 | 9 个 |
| 属性 | GDN_D, GDN_C | KDA_KERNEL_NAME, GDN_D, GDN_C |
| 子模块 | 无 | 需要 pto-isa |
| 常量 | SYNC_* | SYNC_* + SHIFT_* |

---

## 3. Phase 2-6 — 迁移实施

### 3.1 创建目录结构

在目标仓创建 `fla/ops/ascendc/gdn/mega_chunk_kda/`，完全参照 `mega_chunk_gdn` 的目录结构。

### 3.2 文件清单 (24 个文件, 6545 行新增代码)

#### op_kernel (8 个文件, 从源仓复制 + 适配)
| 文件 | 行数 | 说明 |
|------|------|------|
| `mega_chunk_kda.cpp` | 448 | kernel entry, 删除与 pto 重复的 4 个 SYNC_* 常量 |
| `gate_cumsum_kda.cpp` | 368 | gate 累积和, fp16 输入, fp32 中间计算 |
| `kkt_kda.cpp` | 466 | KKT 矩阵, 含数值稳定: `exp(g_cs[r]-g_cs[c])` 差值形式 |
| `tri_inverse_impl.cpp` | 824 | 三角矩阵求逆 |
| `wy_kda.cpp` | 1043 | WY 分解 |
| `chunk_h_kda.cpp` | 843 | chunk 隐状态 |
| `chunk_o_kda.cpp` | 842 | chunk 输出 |
| `include/kernel_utils.h` | 49 | 共享工具函数 |

**关键修改**:
- 添加 `inline` 到 `mega_solve_tril` 和 `mega_kernel_kda_impl`，解决 AIC/AIV 双编译重复符号问题
- 删除 `mega_chunk_kda.cpp` 中与 pto 重复的 4 个 `SYNC_*` 常量，保留 pto 未提供的 2 个 `SHIFT_*` 常量

#### op_host (6 个文件)
| 文件 | 行数 | 说明 |
|------|------|------|
| `CMakeLists.txt` | 36 | 编译选项 + include 路径 |
| `mega_chunk_kda_def.cpp` | 77 | OpDef: 9 输入 + 9 输出 + 属性 |
| `mega_chunk_kda_tiling.h` | 23 | TilingData 结构体 |
| `mega_chunk_kda_tiling.cpp` | 189 | TilingFunc: blockDim/workspace/tiling data |
| `mega_chunk_kda_infershape.cpp` | 118 | InferShapeFunc: 输出 shape 推导 |

#### op_api (4 个文件)
| 文件 | 行数 | 说明 |
|------|------|------|
| `mega_chunk_kda_l0.h` | 23 | l0 接口声明 |
| `mega_chunk_kda_l0.cpp` | 69 | l0 接口实现 |
| `aclnn_mega_chunk_kda.h` | 27 | aclnn 接口声明 |
| `aclnn_mega_chunk_kda.cpp` | 75 | aclnn 接口: 9 输入 + 9 输出 + numMatrices |

#### CMakeLists (2 个文件)
- `mega_chunk_kda/CMakeLists.txt` (19 行) — 顶层, `-DKDA_KERNEL_NAME=mega_chunk_kda` + 保留 `GDN_D`/`GDN_C` 宏名 + pto-isa include + `--cce-auto-sync=on`
- `mega_chunk_kda/op_host/CMakeLists.txt` (36 行)

#### torch_npu 适配层 (4 个文件)
| 文件 | 说明 |
|------|------|
| `npu_custom.yaml` | 算子签名声明 (+2 行) |
| `FLANpuOpApi.cpp` | C++ op_api 适配 (+45 行) |
| `fla_npu/ops/ascendc/__init__.py` | Python wrapper (+128 行) |
| `npu_custom.yaml` | 算子签名 (+2 行) |

#### 测试脚本 (2 个文件)
| 文件 | 行数 | 说明 |
|------|------|------|
| `test_npu_mega_chunk_kda.py` | 276 | 8 个基础精度测试 |
| `test_pt_kda.py` | 340 | pt 端到端测试 (含 debug 打印) |

#### 一键脚本
| 文件 | 行数 | 说明 |
|------|------|------|
| `build_and_test.sh` | 217 | 一键构建 + 安装 + 测试 |

### 3.3 关键技术决策

1. **pto-isa 作为 git 子模块** — 遵循 xllm_ops 模式
   ```
   git submodule add https://gitcode.com/cann/pto-isa.git third_party/pto-isa
   ```

2. **Op 名称**: `MegaChunkKda`，遵循 `MegaChunkGdn` 模式

3. **CMakeLists 配置**:
   - `-DKDA_KERNEL_NAME=mega_chunk_kda` (区分 GDN)
   - 保留 `GDN_D`/`GDN_C` 宏名 (复用 pto-isa 头文件)
   - pto-isa include 路径
   - `--cce-auto-sync=on` 编译选项

4. **aclnn 接口**: 9 输入 + 9 输出 + numMatrices

5. **Python 包装器** 放在 `fla_npu/ops/ascendc/__init__.py` 中:
   - 使用 `lru_cache` 缓存 mask/minus_identity
   - `_custom_fns` dict 在循环前保存, 循环后恢复
   - `assert q.dtype == torch.float16` (kernel 仅支持 fp16)

6. **L 矩阵稳定 split 形式**: `(kc * g_cs.exp()) @ (kc * (-g_cs).exp()).T`

7. **kernel 全程用 `half` (fp16)**, 不支持 bf16

8. **`kkt_kda.cpp` 内部数值稳定**:
   - 使用 `exp(g_cs[r]-g_cs[c])` 差值形式, 避免 `exp(+500)` 溢出
   - 注释: "Kimi KDA gates (g = -exp(A_log)*softplus(...)) are unbounded"

### 3.4 Bug 修复 (迁移过程中)

| Bug | 根因 | 修复 |
|-----|------|------|
| L 矩阵符号反转 | split 形式符号错误 | 修正为 `(kc * g_cs.exp()) @ (kc * (-g_cs).exp()).T` |
| `_stats_ok` `.all()` 误判 | 应该用 `.any()` | 改为 `.any()` |
| `frob_rel` 未定义 | 变量名拼写错误 | 修正变量名 |
| AIC/AIV 双编译重复符号 | 函数未 inline | 添加 `inline` 到 `mega_solve_tril` 和 `mega_kernel_kda_impl` |

---

## 4. Phase 7 — 构建 + 部署 + 测试

### 4.1 公网 41 机器验证 (通过)

- **8/8 基础精度测试 PASSED** (43.57s)
- wheel 构建成功

### 4.2 内网机器验证

#### 4.2.1 一键脚本 `build_and_test.sh`

脚本 7 步流程:
1. CANN env (source set_env.sh)
2. conda env activate
3. build wheel
4. install wheel
5. source vendor set_env.bash (设置 ASCEND_CUSTOM_OPP_PATH)
6. base tests (8 个基础精度测试)
7. pt test (端到端测试)

#### 4.2.2 遇到的问题及修复 (6 个 commit)

| # | Commit | 问题 | 根因 | 修复 |
|---|--------|------|------|------|
| 1 | `bd3a8f1` | `custom_aclnn_extension_lib*.so not found` | 未设置 `FLA_NPU_BUILD_LEGACY_EXTENSION=1` | build_and_test.sh 中添加该环境变量 |
| 2 | `d7c2279` | torch_npu 版本检查失败 + triton 缺失 | 内网环境版本与公网不同 | 添加 `FLA_NPU_SKIP_ENV_CHECK=1` |
| 3 | `0557a88` | `FLA_NPU_SKIP_ENV_CHECK=1` 不生效 | setup.py 有 merge 产生的重复 env check 块 | 删除重复块 |
| 4 | `e0b0910` | `aclnnMegaChunkKda not in libopapi.so` | vendor `set_env.bash` 仅在 pt 测试前 source | 基础测试前也 source |
| 5 | `ca5aff9` | `ASCEND_CUSTOM_OPP_PATH: unbound variable` | `set -u` 与 `set_env.bash` 冲突 | source 前加 `set +u`, 后恢复 `set -u` |
| 6 | `9c38894` | pt 测试误报 PASSED | `test_pt_kda.py` 缺 `sys.exit(1)` | 加 `sys.exit(0/1)` + debug 打印 + `--clamp-gate` |

#### 4.2.3 内网验证结果

- ✅ **wheel 构建成功** (~7 分钟)
- ✅ **8/8 基础精度测试 PASSED**
- ❌ **pt 端到端测试**: NaN + 87% 误差 (进行中)

### 4.3 pt 端到端测试问题分析

#### 根因诊断

1. **bf16→fp16 转换导致 gate 溢出**:
   - pt 输入: q/k/v/g dtype=bf16, beta=fp32, A_log=bf16, dt_bias=bf16
   - 脚本 `_to_fp16()` 转 bf16→fp16
   - 若 gate `g` 有正值 >11, `exp(g)` 溢出 fp16 (fp16 max ~65504, exp(11.1) 即溢出)

2. **kernel 内部已做数值稳定**:
   - `kkt_kda.cpp` 使用 `exp(g_cs[r]-g_cs[c])` 差值形式 + clamp
   - 但输入 fp16 本身可能已 inf/NaN

3. **pt 元数据**: `safe_gate=True`, `lower_bound=-5.0` — 原始模型中可能用于 clamp gate

4. **基础测试不溢出**: `test_npu_mega_chunk_kda.py` line 212: `g_log = -rand * g_scale` (全负, 不溢出)

#### 修复方案

重写 `test_pt_kda.py`:
- 加 `_print_range()`: 打印 min/max/mean/nan/inf
- 加 per-head gate 分析: 统计 g>11 (fp16 溢出) 和 g<-11 (underflow)
- 加 `--clamp-gate` 参数: clamp g 到下界 (对应 pt 元数据 `lower_bound=-5.0`)
- 加 `sys.exit(0)` PASS / `sys.exit(1)` MARGINAL+FAIL

### 4.4 下一步操作

用户需要在内网机器执行:
```bash
# 1. 拉取最新代码
cd /opt/c00808382/selfatk/kda/flash-linear-attention-npu
git pull chw main_kda

# 2. 只跑 pt 测试 (已构建安装过)
bash build_and_test.sh \
    --skip-build --skip-install --skip-base-test \
    --input  /opt/c00808382/selfatk/kda/kda_debug_input_tensors_rank0.pt \
    --output /opt/c00808382/selfatk/kda/kda_debug_output_tensors_rank0.pt

# 3. 若 gate 溢出, 加 clamp
bash build_and_test.sh \
    --skip-build --skip-install --skip-base-test \
    --clamp-gate -5.0 \
    --input  /opt/c00808382/selfatk/kda/kda_debug_input_tensors_rank0.pt \
    --output /opt/c00808382/selfatk/kda/kda_debug_output_tensors_rank0.pt
```

观察 debug 输出:
- gate 值域 (是否有正值 >11)
- per-head gate stats
- 转换后是否有 inf/NaN
- NPU 输出 + 参考输出 range

---

## 5. Git 信息

### 5.1 分支
- `chw/main_kda` — 迁移代码所在分支
- `origin/main` — 上游主干

### 5.2 Commit 历史 (我们的 8 个 commit)

```
9c38894 Fix pt test: add debug prints, fix exit code, add --clamp-gate option
ca5aff9 Fix: temporarily disable set -u when sourcing vendor set_env.bash
e0b0910 Fix: source vendor set_env.bash BEFORE base tests (was only before pt test)
0557a88 Fix: remove duplicate env check block that bypassed FLA_NPU_SKIP_ENV_CHECK
d7c2279 Fix: add FLA_NPU_SKIP_ENV_CHECK=1 to skip torch_npu/triton version checks
bd3a8f1 Fix: build with FLA_NPU_BUILD_LEGACY_EXTENSION=1 to produce custom_aclnn_extension_lib.so
d2d1eed Add build_and_test.sh: one-command build + install + test for mega_chunk_kda
dd5e46c kdatest  (初始迁移 commit: 全部 24 个文件)
```

### 5.3 远程
- `chw` = `https://github.com/yjmyl/flash-linear-attention-npu.git`
- 子模块 `pto-isa` = `https://gitcode.com/cann/pto-isa.git` (内网可达)

---

## 6. 关键文件索引

### 6.1 算子代码

| 路径 | 说明 |
|------|------|
| `fla/ops/ascendc/gdn/mega_chunk_kda/` | 算子根目录 |
| `fla/ops/ascendc/gdn/mega_chunk_kda/CMakeLists.txt` | 顶层 CMake |
| `fla/ops/ascendc/gdn/mega_chunk_kda/op_host/` | op_host 代码 |
| `fla/ops/ascendc/gdn/mega_chunk_kda/op_kernel/` | kernel 代码 |
| `third_party/pto-isa/` | pto-isa 子模块 |

### 6.2 适配层

| 路径 | 说明 |
|------|------|
| `torch_custom/fla_npu/npu_custom.yaml` | 算子签名 |
| `torch_custom/fla_npu/op_plugin/ops/opapi/FLANpuOpApi.cpp` | C++ op_api |
| `torch_custom/fla_npu/fla_npu/ops/ascendc/__init__.py` | Python wrapper |
| `torch_custom/fla_npu/fla_npu/__init__.py` | 模块初始化 |

### 6.3 测试脚本

| 路径 | 说明 |
|------|------|
| `torch_custom/fla_npu/test/test_npu_mega_chunk_kda.py` | 8 个基础精度测试 |
| `torch_custom/fla_npu/test/test_pt_kda.py` | pt 端到端测试 |

### 6.4 构建脚本

| 路径 | 说明 |
|------|------|
| `build_and_test.sh` | 一键构建 + 安装 + 测试 |
| `setup.py` | wheel 构建逻辑 |

---

## 7. 如何复现全流程

### 7.1 从零开始 (新机器)

```bash
# 1. Clone 仓库 (含子模块)
git clone --recursive https://github.com/yjmyl/flash-linear-attention-npu.git
cd flash-linear-attention-npu
git checkout main_kda

# 2. 准备 PT 文件 (用户内网机器)
#    kda_debug_input_tensors_rank0.pt
#    kda_debug_output_tensors_rank0.pt

# 3. 一键构建 + 测试
bash build_and_test.sh \
    --cann   /opt/c00808382/zhengbao/0623_9.1.0/ascend-toolkit \
    --conda  /opt/anaconda3/bin/conda \
    --env    c00808382 \
    --soc    ascend910b \
    --input  /path/to/kda_debug_input_tensors_rank0.pt \
    --output /path/to/kda_debug_output_tensors_rank0.pt
```

### 7.2 已构建过, 只跑测试

```bash
bash build_and_test.sh \
    --skip-build --skip-install \
    --cann   /opt/c00808382/zhengbao/0623_9.1.0/ascend-toolkit \
    --conda  /opt/anaconda3/bin/conda \
    --env    c00808382 \
    --soc    ascend910b \
    --input  /path/to/input.pt \
    --output /path/to/output.pt
```

### 7.3 只跑 pt 端到端测试 (带 gate clamp)

```bash
bash build_and_test.sh \
    --skip-build --skip-install --skip-base-test \
    --clamp-gate -5.0 \
    --cann   /opt/c00808382/zhengbao/0623_9.1.0/ascend-toolkit \
    --conda  /opt/anaconda3/bin/conda \
    --env    c00808382 \
    --soc    ascend910b \
    --input  /path/to/input.pt \
    --output /path/to/output.pt
```

### 7.4 手动调用算子 (Python)

```python
import torch
import torch_npu
from fla_npu.ops.ascendc import mega_chunk_kda

# 准备输入 (全部 fp16, NPU device)
q = torch.randn(B, T, H, K, dtype=torch.float16, device="npu")
k = torch.randn(B, T, H, K, dtype=torch.float16, device="npu")
v = torch.randn(B, T, H, V, dtype=torch.float16, device="npu")
g = torch.randn(B, T, H, dtype=torch.float16, device="npu")
beta = torch.randn(B, T, H, dtype=torch.float16, device="npu")
A_log = torch.randn(B, H, V, V, dtype=torch.float16, device="npu")
dt_bias = torch.randn(H, dtype=torch.float16, device="npu")
cu_seqlens = torch.tensor([0, T], dtype=torch.int32, device="npu")
state = torch.zeros(B, H, K, V, dtype=torch.float16, device="npu")

# 调用
o, recurrent_state, *rest = mega_chunk_kda(
    q, k, v, g, beta, A_log, dt_bias, cu_seqlens, state,
    chunk_size=128, scale=1.0 / (K ** 0.5)
)
```

---

## 8. 迁移 Skill 文档

完整的迁移方法论已沉淀为 Skill 文档:
- 路径: `.agents/skills/cann-op-migration/SKILL.md`
- 适用场景: 任何 CANN 自定义融合算子迁移到 fla_npu 框架
- 包含: 7 个 Phase 的详细步骤 + 代码模板 + 常见问题

---

## 9. 未完成事项

| # | 事项 | 状态 | 说明 |
|---|------|------|------|
| 1 | pt 端到端测试通过 | ❌ 进行中 | NaN 问题, 需用户拉取最新 test_pt_kda.py + 加 `--clamp-gate` 测试 |
| 2 | 迁移流程总结文档 | ✅ 本文档 | |
| 3 | (可选) 清理 stash | 待定 | 本地可能有 stash 残留 |

---

## 10. 附录: pt 文件分析

### 10.1 输入张量

| 键 | shape | dtype | 说明 |
|----|-------|-------|------|
| `q` | (1, 131072, 2, 128) | bf16 | query |
| `k` | (1, 131072, 2, 128) | bf16 | key |
| `v` | (1, 131072, 2, 128) | bf16 | value |
| `g` | (1, 131072, 2) | bf16 | gate |
| `beta` | (1, 131072, 2) | fp32 | beta |
| `A_log` | (1, 2, 128, 128) | bf16 | A 矩阵 log |
| `dt_bias` | (2,) | bf16 | dt 偏置 |
| `cu_seqlens` | (2,) | int32 | 序列边界 |
| `state` | (1, 2, 128, 128) | fp32 | 初始隐状态 |

### 10.2 输出张量

| 键 | shape | dtype | 说明 |
|----|-------|-------|------|
| `o` | (1, 131072, 2, 128) | bf16 | 输出 |
| `recurrent_state` | (1, 2, 128, 128) | fp32 | 最终隐状态 |

### 10.3 元数据

- `safe_gate=True`
- `lower_bound=-5.0`
- `mode='chunk'`
- `step=1`
- `layer=0`
- T=131072, HV=2, K=128, V=128, chunk_size=128, tc=2077, dtype=BF16

---

*文档生成时间: 2026-07-15*
*迁移执行者: Sisyphus (OhMyOpenCode)*
