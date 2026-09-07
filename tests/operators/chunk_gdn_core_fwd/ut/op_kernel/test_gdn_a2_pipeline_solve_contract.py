"""A2 Solve 内部接入约束；不依赖 NPU，不代替完整精度门禁。"""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[5]
OP = ROOT / 'fla/ops/ascendc/gdn/chunk_gdn_fwd/chunk_gdn_core_fwd'


def test_component_is_device_inline_not_nested_launch():
    text = (OP / 'op_kernel/solve_tri_pipeline_a2.h').read_text(encoding='utf-8')
    assert '__global__' not in text
    assert '<<<' not in text
    assert 'SyncAll<false>();' in text
    assert 'start+=GetBlockNum()*2*E' in text
    assert 'SetHF32Mode(false);' in text


def test_bnsd_and_varlen_information_remains_private():
    kernel = (OP / 'op_kernel/chunk_gdn_core_fwd.cpp').read_text(encoding='utf-8')
    tiling = (OP / 'op_host/chunk_gdn_core_fwd_tiling.cpp').read_text(encoding='utf-8')
    assert 'GdnTritonSolve::Run<InputT, InputT>' in kernel
    assert 'solveSequenceCount' in kernel
    assert 'solveCu.GetValue(sequence + 1) - solveCu.GetValue(sequence)' in kernel
    assert 'cuShape->GetStorageShape().GetDim(0) - 1' in tiling
    assert 'isAscend910B && npuArch == NpuArch::DAV_2201 ? 1 : 0' in tiling
    assert 'abc.layoutMode = isVarlen ? 4 : 0' in tiling


def test_workspace_retains_separate_fp32_versions():
    source = (OP / 'op_host/chunk_gdn_core_fwd_tiling.cpp').read_text(encoding='utf-8')
    assert '48 * 32 * 32' in source
    assert '34 * 64 * 64' in source
    for name in ('solveFp32InputOffset', 'solveD16Offset', 'solveD32Offset', 'solveD64Offset'):
        assert name in source
