"""Static output-subblock join contract for the fused GDN FwdO kernel."""

from pathlib import Path


ROOT = Path(__file__).resolve().parents[5]
FWD_O_KERNEL = (
    ROOT
    / "fla/ops/ascendc/gdn/chunk_gdn_fwd/chunk_fwd_o/op_kernel/gemm/kernel"
    / "gdn_fwd_o_kernel.hpp"
)
FWD_O_SCHEDULER = (
    ROOT
    / "fla/ops/ascendc/gdn/chunk_gdn_fwd/chunk_fwd_o/op_kernel/gemm/block"
    / "block_scheduler_gdn_fwd_o.hpp"
)


def test_fwd_o_joins_aiv_subblocks_before_publishing_output_completion():
    kernel = FWD_O_KERNEL.read_text(encoding="utf-8")
    scheduler = FWD_O_SCHEDULER.read_text(encoding="utf-8")

    assert "GDN_FWD_O_AIV_FLAG_STRIDE" not in scheduler
    assert "CrossCoreSetFlag<0x4" not in kernel
    assert "CrossCoreWaitFlag<0x4" not in kernel
    assert "PublishAivCompletion" not in kernel
    output_completion = (
        "Catlass::Arch::CrossCoreBarrier<0x1, PIPE_MTE3>();\n"
        "                    Arch::CrossCoreSetFlag<0x2, PIPE_MTE3>("
        "vecBlockScheduler.vec2Done[streamId]);"
    )
    assert output_completion in kernel
    assert kernel.count("CrossCoreBarrier<0x1, PIPE_MTE3>();") == 1
    assert "AscendC::PipeBarrier<PIPE_MTE3>();" not in kernel
    assert kernel.count("CrossCoreSetFlag<0x2, PIPE_MTE3>(") == 4
    assert kernel.count("Arch::CrossCoreWaitFlag(cubeBlockScheduler.vec1Done[streamId]);") == 1
    assert kernel.count("Arch::CrossCoreWaitFlag(cubeBlockScheduler.vec2Done[streamId]);") == 1
    assert kernel.count("Arch::CrossCoreWaitFlag(cubeBlockScheduler.vec2Done[0]);") == 1
    assert kernel.count("Arch::CrossCoreWaitFlag(cubeBlockScheduler.vec2Done[1]);") == 1
