"""Static completion-credit contract for the fused GDN FwdO kernel."""

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


def test_fwd_o_publishes_one_credit_after_both_aiv_subblocks_finish():
    kernel = FWD_O_KERNEL.read_text(encoding="utf-8")
    scheduler = FWD_O_SCHEDULER.read_text(encoding="utf-8")

    assert "GDN_FWD_O_AIV_FLAG_STRIDE" not in scheduler
    assert "CrossCoreSetFlag<0x4" not in kernel
    assert "CrossCoreWaitFlag<0x4" not in kernel
    assert "CrossCoreBarrier<0x1, PIPE_MTE3>()" in kernel
    assert "if (subBlockIdx == 0)" in kernel
    assert kernel.count("JoinAivSubblocks(subBlockNum);") == 3
    assert kernel.count("PublishAivCompletion(") == 5
    assert kernel.count("Arch::CrossCoreWaitFlag(cubeBlockScheduler.vec1Done[streamId]);") == 1
    assert kernel.count("Arch::CrossCoreWaitFlag(cubeBlockScheduler.vec2Done[streamId]);") == 2
