#!/usr/bin/env bash
###############################################################################
# build_and_test.sh — 一键构建 + 安装 + 测试 mega_chunk_kda 算子
#
# 用法:
#   bash build_and_test.sh --input /path/to/input.pt --output /path/to/output.pt
#
# 可选环境变量 / 参数 (见 --help):
#   --cann    CANN set_env.sh 所在目录   (默认 /usr/local/Ascend/ascend-toolkit)
#   --conda   conda 可执行文件路径       (默认 ~/miniconda3/bin/conda)
#   --env     conda 环境名               (默认 chw)
#   --soc     目标芯片                   (默认 ascend910b; A3=ascend910_93, A5=ascend950)
#   --chunk-size  传给 test_pt_kda.py    (默认 128, 与算子内置一致)
#   --scale       传给 test_pt_kda.py    (默认不传, 脚本自动 1/sqrt(K))
#   --skip-build      跳过 wheel 编译 (已编译过时复用 dist/)
#   --skip-install    跳过 wheel 安装
#   --skip-base-test  跳过 8 个基础精度测试 (只想跑 pt 测试时)
###############################################################################
set -euo pipefail

# ---------- colors ----------
C_R='\033[0;31m'; C_G='\033[0;32m'; C_Y='\033[1;33m'; C_B='\033[0;34m'; C_N='\033[0m'
log()  { echo -e "${C_B}[$(date +%H:%M:%S)]${C_N} $*"; }
ok()   { echo -e "${C_G}[$(date +%H:%M:%S)] OK   ${C_N} $*"; }
warn() { echo -e "${C_Y}[$(date +%H:%M:%S)] WARN ${C_N} $*"; }
err()  { echo -e "${C_R}[$(date +%H:%M:%S)] ERROR${C_N} $*" >&2; }

# ---------- defaults ----------
CANN_PATH="/usr/local/Ascend/ascend-toolkit"
CONDA_BIN="$HOME/miniconda3/bin/conda"
CONDA_ENV="chw"
SOC="ascend910b"
CHUNK_SIZE=""
SCALE=""
SKIP_BUILD=0
SKIP_INSTALL=0
SKIP_BASE_TEST=0
INPUT=""
OUTPUT=""

# ---------- usage ----------
usage() {
  cat <<'EOF'
用法: bash build_and_test.sh --input <pt> --output <pt> [选项]

必填:
  --input  PATH    kda_debug_input_tensors_rank0.pt 路径
  --output PATH    kda_debug_output_tensors_rank0.pt 路径

可选:
  --cann    PATH   CANN set_env.sh 所在目录   (默认 /usr/local/Ascend/ascend-toolkit)
  --conda   PATH   conda 可执行文件路径       (默认 ~/miniconda3/bin/conda)
  --env     NAME   conda 环境名               (默认 chw)
  --soc     NAME   目标芯片                   (默认 ascend910b)
                   910B 系列 = ascend910b
                   910A3     = ascend910_93
                   910A5     = ascend950
  --chunk-size N   传给 test_pt_kda.py        (默认 128, 不传则用算子默认)
  --scale    F     传给 test_pt_kda.py        (默认不传, 脚本自动 1/sqrt(K))
  --skip-build     跳过 wheel 编译
  --skip-install   跳过 wheel 安装
  --skip-base-test 跳过 8 个基础精度测试
  -h, --help       显示本帮助

示例:
  # 全流程 (首次)
  bash build_and_test.sh \
      --input  /data/kda_debug_input_tensors_rank0.pt \
      --output /data/kda_debug_output_tensors_rank0.pt

  # 自定义 CANN 路径 + 芯片 + conda 环境
  bash build_and_test.sh \
      --cann /opt/chw/zhengbao/9.1.0.beta1/ascend-toolkit \
      --conda ~/anaconda3/bin/conda --env myenv \
      --soc ascend910_93 \
      --input  /data/in.pt --output /data/out.pt

  # 已编译安装过, 只跑 pt 测试
  bash build_and_test.sh \
      --skip-build --skip-install --skip-base-test \
      --input /data/in.pt --output /data/out.pt
EOF
}

# ---------- parse args ----------
while [[ $# -gt 0 ]]; do
  case "$1" in
    --input)        INPUT="$2"; shift 2;;
    --output)       OUTPUT="$2"; shift 2;;
    --cann)         CANN_PATH="$2"; shift 2;;
    --conda)        CONDA_BIN="$2"; shift 2;;
    --env)          CONDA_ENV="$2"; shift 2;;
    --soc)          SOC="$2"; shift 2;;
    --chunk-size)   CHUNK_SIZE="$2"; shift 2;;
    --scale)        SCALE="$2"; shift 2;;
    --skip-build)       SKIP_BUILD=1; shift;;
    --skip-install)     SKIP_INSTALL=1; shift;;
    --skip-base-test)   SKIP_BASE_TEST=1; shift;;
    -h|--help)      usage; exit 0;;
    *) err "未知参数: $1"; usage; exit 1;;
  esac
done

# ---------- validate ----------
[[ -z "$INPUT" ]]  && { err "--input 必填"; usage; exit 1; }
[[ -z "$OUTPUT" ]] && { err "--output 必填"; usage; exit 1; }
[[ ! -f "$INPUT" ]]  && { err "input 文件不存在: $INPUT"; exit 1; }
[[ ! -f "$OUTPUT" ]] && { err "output 文件不存在: $OUTPUT"; exit 1; }
[[ ! -f "$CANN_PATH/set_env.sh" ]] && { err "CANN set_env.sh 不存在: $CANN_PATH/set_env.sh"; \
  err "请用 --cann 指定实际路径, 例如: --cann /opt/chw/zhengbao/9.1.0.beta1/ascend-toolkit"; exit 1; }
[[ ! -f "$CONDA_BIN" ]] && { err "conda 不存在: $CONDA_BIN"; \
  err "请用 --conda 指定, 例如: --conda ~/anaconda3/bin/conda"; exit 1; }

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$REPO_ROOT"
log "仓库根目录: $REPO_ROOT"

# ---------- Step 1: CANN env ----------
log "========== Step 1/6: 设置 CANN 环境 =========="
source "$CANN_PATH/set_env.sh"
ok "CANN env 已加载 (ASCEND_HOME_PATH=$ASCEND_HOME_PATH)"

# ---------- Step 2: conda env ----------
log "========== Step 2/6: 激活 conda 环境 [$CONDA_ENV] =========="
eval "$("$CONDA_BIN" shell.bash hook)"
conda activate "$CONDA_ENV"
ok "Python: $(python --version 2>&1) @ $(which python)"
ok "torch_npu: $(python -c 'import torch_npu; print(torch_npu.__version__)' 2>&1)"

# ---------- Step 3: build wheel ----------
if [[ $SKIP_BUILD -eq 1 ]]; then
  warn "跳过 wheel 编译 (--skip-build)"
else
  log "========== Step 3/6: 编译 wheel (SOC=$SOC, LEGACY_EXTENSION=1) =========="
  log "这通常需要 10-20 分钟 (含 C++ 扩展编译), 请耐心等待..."
  FLA_NPU_SOC="$SOC" FLA_NPU_BUILD_LEGACY_EXTENSION=1 \
    python -m pip wheel --no-build-isolation --no-deps . -w dist
  ok "wheel 编译完成"
fi

WHEEL=$(ls dist/flash_linear_attention_npu-*.whl 2>/dev/null | head -1 || true)
if [[ -z "$WHEEL" ]]; then
  err "dist/ 下没有 wheel 文件, 请去掉 --skip-build 先编译一次"
  exit 1
fi
ok "wheel: $WHEEL"

# ---------- Step 4: install wheel ----------
if [[ $SKIP_INSTALL -eq 1 ]]; then
  warn "跳过 wheel 安装 (--skip-install)"
else
  log "========== Step 4/6: 安装 wheel =========="
  python -m pip install --force-reinstall --no-deps "$WHEEL"
  ok "wheel 安装完成"
fi

# ---------- Step 5: base precision test ----------
if [[ $SKIP_BASE_TEST -eq 1 ]]; then
  warn "跳过基础精度测试 (--skip-base-test)"
else
  log "========== Step 5/6: 8 个基础精度测试 =========="
  cd torch_custom/fla_npu/test
  python test_npu_mega_chunk_kda.py
  cd "$REPO_ROOT"
  ok "基础精度测试完成"
fi

# ---------- Step 6: pt end-to-end test ----------
log "========== Step 6/6: pt 端到端精度测试 =========="

# 6a. set vendor env (每次必须 source, 否则 aclnnMegaChunkKda not found)
log "设置算子运行环境 (ASCEND_CUSTOM_OPP_PATH)..."
VENDOR_DIR=$(python -c "import fla_npu, os; print(os.path.join(os.path.dirname(fla_npu.__file__), 'opp', 'vendors', 'fla_npu_transformer'))")
if [[ ! -f "$VENDOR_DIR/bin/set_env.bash" ]]; then
  err "vendor set_env.bash 不存在: $VENDOR_DIR/bin/set_env.bash"
  err "可能 wheel 未正确安装, 请去掉 --skip-install 重试"
  exit 1
fi
source "$VENDOR_DIR/bin/set_env.bash"
ok "算子运行环境已加载 (ASCEND_CUSTOM_OPP_PATH=$ASCEND_CUSTOM_OPP_PATH)"

# 6b. run pt test
log "运行 test_pt_kda.py..."
log "  input  = $INPUT"
log "  output = $OUTPUT"
PT_ARGS=(--input "$INPUT" --output "$OUTPUT")
[[ -n "$CHUNK_SIZE" ]] && PT_ARGS+=(--chunk-size "$CHUNK_SIZE")
[[ -n "$SCALE" ]]      && PT_ARGS+=(--scale "$SCALE")

cd torch_custom/fla_npu/test
python test_pt_kda.py "${PT_ARGS[@]}"
RESULT=$?
cd "$REPO_ROOT"

echo ""
if [[ $RESULT -eq 0 ]]; then
  ok "=========================================="
  ok "  全部完成! pt 端到端测试 PASSED"
  ok "=========================================="
else
  err "=========================================="
  err "  pt 端到端测试 FAILED (exit=$RESULT)"
  err "=========================================="
  err "常见排查:"
  err "  1. aclnnMegaChunkKda not found → vendor set_env.bash 未 source"
  err "  2. Unable to initialize ... op_api → CANN set_env.sh 未 source"
  err "  3. max_rel > 5% → 检查 pt 文件输入输出 key 是否与脚本预期一致"
fi

exit $RESULT
