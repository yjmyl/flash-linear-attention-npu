# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Tianjin University, Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Ascend C backed FLA NPU operators.

This module provides stable Python import paths backed by Python ctypes aclnn
calls.  Compatibility helpers for legacy torch_npu/torch.ops.npu call sites are
kept opt-in and are not installed during normal import.
"""

from __future__ import annotations

import functools
import types
import warnings
from typing import Callable, Optional

from ._aclnn_ctypes import ASCENDC_CTYPES_OPS

_ASCENDC_OPS = (
    "npu_fast_gelu_custom",
    "npu_fast_gelu_custom_backward",
    "npu_causal_conv1d",
    "npu_causal_conv1d_bwd",
    "npu_prepare_wy_repr_bwd_full",
    "npu_chunk_gated_delta_rule_bwd_dhu",
    "npu_chunk_bwd_dv_local",
    "npu_prepare_wy_repr_bwd_da",
    "npu_chunk_bwd_dqkwg",
    "npu_chunk_fwd_o",
    "npu_chunk_gated_delta_rule_fwd_h",
    "npu_recompute_w_u_fwd",
    "npu_chunk_scaled_dot_kkt",
    "npu_solve_tri",
    "npu_mega_chunk_kda",
)

BACKWARD_OPS = {
    "fast_gelu_custom": "fast_gelu_custom_backward",
    "npu_fast_gelu_custom": "npu_fast_gelu_custom_backward",
    "causal_conv1d": "causal_conv1d_bwd",
    "npu_causal_conv1d": "npu_causal_conv1d_bwd",
}

_LEGACY_TORCH_OPS_WARNING = (
    "torch.ops.npu.{name} is a legacy FLA NPU compatibility API. This call path "
    "depends on the PyTorch/torch_npu dispatcher ABI and will not be supported "
    "in a future fla_npu release. Use fla_npu.ops.ascendc.{public_name}(...) "
    "or the decoupled Ascend C API instead."
)

_DIRECT_RUNTIME_READY = False
_DIRECT_RUNTIME_ERROR: Optional[Exception] = None


def _prepare_direct_runtime(*, raise_on_error: bool = True) -> None:
    """Prepare embedded OPP paths and load custom op_api libraries."""

    global _DIRECT_RUNTIME_ERROR, _DIRECT_RUNTIME_READY
    if _DIRECT_RUNTIME_READY:
        return

    try:
        import fla_npu

        fla_npu.load_ascendc_opapi_libraries()
    except Exception as exc:
        _DIRECT_RUNTIME_ERROR = exc
        if raise_on_error:
            raise RuntimeError(
                "Unable to initialize fla_npu Ascend C op_api libraries. "
                "Please source the CANN set_env.sh before importing "
                "fla_npu.ops.ascendc or calling Ascend C operators."
            ) from exc
    else:
        _DIRECT_RUNTIME_ERROR = None
        _DIRECT_RUNTIME_READY = True


def _torch_npu_namespace():
    import torch

    return torch.ops.npu


def _ensure_legacy_torch_ops_loaded() -> None:
    import fla_npu

    is_loaded = getattr(fla_npu, "is_legacy_torch_ops_loaded", lambda: False)
    if not is_loaded():
        fla_npu.load_legacy_torch_ops()


def _get_torch_op(name: str):
    namespace = _torch_npu_namespace()
    if not hasattr(namespace, name):
        _ensure_legacy_torch_ops_loaded()
        namespace = _torch_npu_namespace()
    if not hasattr(namespace, name):
        raise AttributeError(
            f"torch.ops.npu.{name} is not registered. Call "
            "fla_npu.load_legacy_torch_ops() first if you need the legacy "
            "torch.ops.npu compatibility path."
        )
    return _unwrap_legacy_torch_op(getattr(namespace, name))


@functools.lru_cache(maxsize=None)
def _get_direct_op(name: str):
    _prepare_direct_runtime()
    try:
        return ASCENDC_CTYPES_OPS[name]
    except KeyError as exc:
        raise AttributeError(f"fla_npu.ops.ascendc has no ctypes Ascend C op {name}.") from exc


def _warn_legacy_torch_op(name: str) -> None:
    warnings.warn(
        _LEGACY_TORCH_OPS_WARNING.format(
            name=name,
            public_name=_strip_npu_prefix(name),
        ),
        FutureWarning,
        stacklevel=3,
    )


def _unwrap_legacy_torch_op(op):
    return getattr(op, "_fla_npu_original_op", op)


class _LegacyTorchOpOverloadWarningWrapper:
    _fla_npu_legacy_warning_wrapper = True

    def __init__(self, name: str, overload):
        self._fla_npu_name = name
        self._fla_npu_original_op = overload

    def __call__(self, *args, **kwargs):
        _warn_legacy_torch_op(self._fla_npu_name)
        return self._fla_npu_original_op(*args, **kwargs)

    def __getattr__(self, name: str):
        return getattr(self._fla_npu_original_op, name)

    def __repr__(self) -> str:
        return repr(self._fla_npu_original_op)


class _LegacyTorchOpWarningWrapper:
    _fla_npu_legacy_warning_wrapper = True

    def __init__(self, name: str, op):
        self._fla_npu_name = name
        self._fla_npu_original_op = op
        self.__name__ = name
        self.__qualname__ = name
        self.__doc__ = getattr(op, "__doc__", None)

    def __call__(self, *args, **kwargs):
        _warn_legacy_torch_op(self._fla_npu_name)
        return self._fla_npu_original_op(*args, **kwargs)

    def __getattr__(self, name: str):
        value = getattr(self._fla_npu_original_op, name)
        if callable(value):
            return _LegacyTorchOpOverloadWarningWrapper(self._fla_npu_name, value)
        return value

    def __repr__(self) -> str:
        return repr(self._fla_npu_original_op)


def _make_raw_wrapper(name: str) -> Callable:
    @functools.wraps(_get_direct_op)
    def wrapper(*args, **kwargs):
        return _get_direct_op(name)(*args, **kwargs)

    wrapper.__name__ = name
    wrapper.__qualname__ = name
    wrapper.__doc__ = f"Call the direct Ascend C binding for {name}."
    return wrapper


def _strip_npu_prefix(name: str) -> str:
    return name[4:] if name.startswith("npu_") else name


def _has_tensor_requiring_grad(*values) -> bool:
    try:
        import torch
    except Exception:
        return False

    for value in values:
        if isinstance(value, torch.Tensor) and value.requires_grad:
            return True
    return False


class _FastGeluCustomFunction:
    @staticmethod
    def apply(input_tensor):
        import torch

        class Function(torch.autograd.Function):
            @staticmethod
            def forward(ctx, self):
                ctx.save_for_backward(self)
                return _get_direct_op("npu_fast_gelu_custom")(self)

            @staticmethod
            def backward(ctx, grad):
                (self,) = ctx.saved_tensors
                return _get_direct_op("npu_fast_gelu_custom_backward")(grad, self)

        return Function.apply(input_tensor)


def fast_gelu_custom(input_tensor):
    """FastGELU with automatic binding to its custom backward operator."""

    if _has_tensor_requiring_grad(input_tensor):
        return _FastGeluCustomFunction.apply(input_tensor)
    return _get_direct_op("npu_fast_gelu_custom")(input_tensor)


def causal_conv1d(
    x,
    weight,
    bias=None,
    conv_states=None,
    *,
    query_start_loc=None,
    cache_indices=None,
    initial_state_mode=None,
    num_accepted_tokens=None,
    activation_mode=0,
    pad_slot_id=-1,
    run_mode=0,
    head_num=0,
):
    """Causal conv1d with automatic backward binding for prefill mode.

    Decode/speculative modes mutate cache state and are left on the raw op path.
    """

    can_bind_backward = (
        run_mode == 0
        and activation_mode == 0
        and query_start_loc is None
        and cache_indices is None
        and initial_state_mode is None
        and num_accepted_tokens is None
        and _has_tensor_requiring_grad(x, weight, bias)
    )
    if not can_bind_backward:
        return _get_direct_op("npu_causal_conv1d")(
            x=x,
            weight=weight,
            bias=bias,
            conv_states=conv_states,
            query_start_loc=query_start_loc,
            cache_indices=cache_indices,
            initial_state_mode=initial_state_mode,
            num_accepted_tokens=num_accepted_tokens,
            activation_mode=activation_mode,
            pad_slot_id=pad_slot_id,
            run_mode=run_mode,
            head_num=head_num,
        )

    import torch

    class Function(torch.autograd.Function):
        @staticmethod
        def forward(ctx, x_, weight_, bias_, conv_states_):
            y = _get_direct_op("npu_causal_conv1d")(
                x=x_,
                weight=weight_,
                bias=bias_,
                conv_states=conv_states_,
                query_start_loc=query_start_loc,
                cache_indices=None,
                initial_state_mode=None,
                num_accepted_tokens=None,
                activation_mode=activation_mode,
                pad_slot_id=pad_slot_id,
                run_mode=run_mode,
                head_num=head_num,
            )
            tensors = [x_, weight_]
            ctx.has_bias = bias_ is not None
            if bias_ is not None:
                tensors.append(bias_)
            ctx.activation_mode = activation_mode
            ctx.save_for_backward(*tensors)
            return y

        @staticmethod
        def backward(ctx, grad):
            saved = list(ctx.saved_tensors)
            x_ = saved.pop(0)
            weight_ = saved.pop(0)
            bias_ = saved.pop(0) if ctx.has_bias else None
            dx, dw, db, _ = _get_direct_op("npu_causal_conv1d_bwd")(
                x=x_,
                y=None if ctx.activation_mode == 0 else None,
                weight=weight_,
                dy=grad,
                initial_state=None,
                dht=None,
                query_start_loc=None,
                activation=0,
                input_layout="BSH",
            )
            return dx, dw, (db if bias_ is not None else None), None

    return Function.apply(x, weight, bias, conv_states)


# ---------------------------------------------------------------------------
# KDA mega-kernel helpers
# ---------------------------------------------------------------------------

@functools.lru_cache(maxsize=48)
def _mega_chunk_kda_minus_identity(device_ty: str, device_index: int, chunk_size: int):
    """Shared ``[C,C] fp16`` buffer with diagonal ``-1`` for tri_inverse."""
    import torch
    idx = device_index if device_index >= 0 else 0
    dev = torch.device(device_ty, idx) if device_ty != "cpu" else torch.device("cpu")
    t = torch.zeros(chunk_size, chunk_size, device=dev, dtype=torch.float16)
    t.fill_diagonal_(-1)
    return t


@functools.lru_cache(maxsize=48)
def _mega_chunk_kda_causal_masks(device_ty: str, device_index: int, chunk_size: int):
    """Lower-triangle masks for intra-chunk KKT attention (rows>cols, rows>=cols)."""
    import torch
    idx = device_index if device_index >= 0 else 0
    dev = torch.device(device_ty, idx) if device_ty != "cpu" else torch.device("cpu")
    m_lower = torch.tril(torch.ones(chunk_size, chunk_size, device=dev), diagonal=-1).float()
    m_full = torch.tril(torch.ones(chunk_size, chunk_size, device=dev), diagonal=0).float()
    return m_lower, m_full


def mega_chunk_kda(
    q,
    k,
    v,
    g,
    beta,
    cu_seqlens=None,
    *,
    chunk_size: int = 128,
    return_final_state: bool = False,
    return_all: bool = False,
):
    """Fused KDA mega-kernel: all six KDA stages in a single NPU launch.

    Fuses gate_cumsum -> kkt -> solve_tril -> wy -> chunk_h -> chunk_o.

    Args:
        q, k:       ``[1, T, HV, K]`` fp16, GQA-expanded; ``q`` pre-scaled.
        v:          ``[1, T, HV, V]`` fp16.
        g:          ``[1, T, HV, K]`` fp16, raw per-dimension gate logits.
        beta:       ``[1, T, HV]``    fp16, post-sigmoid beta.
        cu_seqlens: ``int32`` cumulative sequence lengths, or ``None`` for one
                    sequence of length ``T``.
        chunk_size: Chunk size C (default 128, must match compiled GDN_C).
        return_final_state: If True, also return ``[N_seq, HV, K, V]`` final states.
        return_all: If True, return all 9 raw outputs as a tuple.

    Returns:
        ``out`` of shape ``[1, T, HV, V]`` fp16 by default.
        ``(out, final_state)`` if ``return_final_state=True``.
        9-tuple of all outputs if ``return_all=True``.
    """
    import torch

    assert q.dtype == torch.float16 and v.dtype == torch.float16
    dev = q.device
    HV, K = q.shape[2], q.shape[3]
    C = chunk_size
    T = q.shape[1]

    # Synthesise cu_seqlens for the single-sequence case
    if cu_seqlens is None:
        cu_seqlens = torch.tensor([0, T], dtype=torch.int32, device=dev)
    elif cu_seqlens.dtype != torch.int32:
        cu_seqlens = cu_seqlens.to(torch.int32)

    N_seq = int(cu_seqlens.numel()) - 1

    # Precomputed masks (cached per device/chunk_size)
    dt = dev.type
    di = dev.index if dev.index is not None else -1
    mask_strict, mask_incl = _mega_chunk_kda_causal_masks(dt, di, C)
    minus_id = _mega_chunk_kda_minus_identity(dt, di, C)

    # Total chunks and num_matrices
    cu_list = cu_seqlens.cpu().tolist()
    tc = sum(
        (cu_list[i + 1] - cu_list[i] + C - 1) // C
        for i in range(len(cu_list) - 1)
    )
    num_matrices = tc * HV

    # Head-major permutes (match the kernel's expected layout)
    q_hm = q.permute(0, 2, 1, 3).contiguous()    # [1, HV, T, K]
    k_hm = k.permute(0, 2, 1, 3).contiguous()    # [1, HV, T, K]
    beta_hm = beta.permute(0, 2, 1).contiguous() # [1, HV, T]
    v_c = v.contiguous()
    g_c = g.contiguous()

    out, g_sum, g_cs, L, A_inv, u, w, s, v_corr = _get_torch_op("npu_mega_chunk_kda")(
        q_hm, k_hm, v_c, g_c, beta_hm,
        mask_strict, mask_incl, minus_id, cu_seqlens,
        num_matrices,
    )

    if return_all:
        return out, g_sum, g_cs, L, A_inv, u, w, s, v_corr

    if return_final_state:
        seq_lens = cu_seqlens[1:] - cu_seqlens[:-1]
        chunks_per_seq = (seq_lens.long() + C - 1) // C
        last_chunk_idx = chunks_per_seq.cumsum(0) - 1
        final_state = s[last_chunk_idx]  # [N_seq, HV, K, V]
        return out, final_state

    return out


def install_torch_npu_ops_compat() -> None:
    """Expose wrappers through the legacy ``torch_npu.ops`` namespace."""

    try:
        import torch_npu
    except Exception:
        return

    ops = getattr(torch_npu, "ops", None)
    if ops is None:
        ops = types.SimpleNamespace()
        setattr(torch_npu, "ops", ops)

    for name in _ASCENDC_OPS:
        setattr(ops, name, globals()[name])
        setattr(ops, _strip_npu_prefix(name), globals()[_strip_npu_prefix(name)])


def install_legacy_torch_ops_warning() -> None:
    """Warn when users call legacy ``torch.ops.npu`` FLA NPU operators."""

    namespace = _torch_npu_namespace()
    for name in _ASCENDC_OPS:
        if not hasattr(namespace, name):
            continue
        current = getattr(namespace, name)
        if getattr(current, "_fla_npu_legacy_warning_wrapper", False):
            continue
        setattr(namespace, name, _LegacyTorchOpWarningWrapper(name, current))


# Save custom wrapper implementations before the loop below overwrites them.
_custom_fns = {
    "fast_gelu_custom": fast_gelu_custom,
    "causal_conv1d": causal_conv1d,
    "mega_chunk_kda": mega_chunk_kda,
}

for _name in _ASCENDC_OPS:
    globals()[_name] = _make_raw_wrapper(_name)
    globals()[_strip_npu_prefix(_name)] = globals()[_name]

# Restore custom wrappers (they add argument preprocessing that the raw
# torch.ops.npu binding does not provide).
for _name, _fn in _custom_fns.items():
    globals()[_name] = _fn

_prepare_direct_runtime(raise_on_error=False)

__all__ = [
    "BACKWARD_OPS",
    "install_legacy_torch_ops_warning",
    "install_torch_npu_ops_compat",
    *sorted(set(_ASCENDC_OPS)),
    *sorted({_strip_npu_prefix(name) for name in _ASCENDC_OPS}),
]
