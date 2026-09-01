"""Static contracts for the experimental A2 FP32 SolveTri integration."""

from pathlib import Path


ROOT = Path(__file__).resolve().parents[5]
SOLVE_DIR = ROOT / "fla/ops/ascendc/gdn/chunk_gdn_fwd/solve_tri"
FUSED_SOLVE = ROOT / (
    "fla/ops/ascendc/gdn/chunk_gdn_fwd/chunk_kkt_solve_tri/"
    "op_kernel/chunk_cumsum_kkt_solve_tri.cpp"
)
FUSED_TILING = ROOT / (
    "fla/ops/ascendc/gdn/chunk_gdn_fwd/chunk_kkt_solve_tri/"
    "op_host/chunk_kkt_solve_tri_tiling.cpp"
)
PHASE6_TILING = ROOT / (
    "fla/ops/ascendc/gdn/chunk_gdn_fwd/chunk_gdn_core_fwd/"
    "op_host/chunk_gdn_core_fwd_tiling.cpp"
)


def test_standalone_fp32_solver_is_scoped_to_dense_chunk64():
    kernel = (SOLVE_DIR / "op_kernel/solve_tri.cpp").read_text(encoding="utf-8")

    assert '#include "solve_tri_fp32.h"' in kernel
    assert "useFp32Solve = ms == 64" in kernel
    assert "tilingData.layoutMode == 0 || tilingData.layoutMode == 1" in kernel
    assert "SolveTriCubeFp32<half>" in kernel
    assert "SolveTriCubeFp32<bfloat16_t>" in kernel
    assert "SolveTriVectorFp32<half>" in kernel
    assert "SolveTriVectorFp32<bfloat16_t>" in kernel
    assert "SolveTriCube<128, bfloat16_t>" in kernel


def test_fp32_solver_preserves_fp32_workspace_and_full_tail_rows():
    header = (SOLVE_DIR / "op_kernel/solve_tri_fp32.h").read_text(encoding="utf-8")

    assert "SetHF32Mode(false);" in header
    assert "GlobalTensor<float> workspaceGM_;" in header
    assert "LocalTensor<float> fp32LocalA_;" in header
    assert "bool perCoreWorkspace = false" in header
    assert header.count("if (perCoreWorkspace)") == 2
    assert header.count("coreWorkspaceGM_ = workspaceGM_;") == 2
    assert "FP32_MATRIX_SIZE * sizeof(T)" in header
    assert "Cast(inputLocal_, fp32LocalA_, RoundMode::CAST_RINT" in header


def test_fused_dense_chunk64_routes_to_fp32_with_old_fallback_retained():
    kernel = FUSED_SOLVE.read_text(encoding="utf-8")

    assert '#include "solve_tri_fp32.h"' in kernel
    assert "if constexpr (MATRIX_SIZE == 64)" in kernel
    assert "if (tilingData->layoutMode == 0)" in kernel
    assert "SolveTriCubeFp32<T>" in kernel
    assert "SolveTriVectorFp32<T>" in kernel
    assert "tilingData, true" in kernel
    assert "SolveTriCube<MATRIX_SIZE, T>" in kernel
    assert "SolveTriVector<MATRIX_SIZE, T>" in kernel


def test_all_dense_chunk64_callers_allocate_four_fp32_slots_per_core():
    standalone = (SOLVE_DIR / "op_host/solve_tri_tiling.cpp").read_text(encoding="utf-8")
    fused = FUSED_TILING.read_text(encoding="utf-8")
    phase6 = PHASE6_TILING.read_text(encoding="utf-8")

    for source in (standalone, fused, phase6):
        assert "fp32WorkspaceSlots = 4" in source
        assert "sizeof(float)" in source
    assert "chunkSize == 64 && (layoutMode == 0 || layoutMode == 1)" in standalone
    assert "bt == 64 && isVarlen == 0" in fused
    assert "abc.BT == 64 && !isVarlen" in phase6
