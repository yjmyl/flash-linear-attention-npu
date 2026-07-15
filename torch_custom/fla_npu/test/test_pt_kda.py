"""End-to-end KDA mega-kernel test using real model pt tensors.

Usage on NPU:
    source <site-packages>/fla_npu/opp/vendors/fla_npu_transformer/bin/set_env.bash
    python test_pt_kda.py --input /path/to/kda_debug_input_tensors_rank0.pt \
                          --output /path/to/kda_debug_output_tensors_rank0.pt

The script:
  1. Loads the input/output pt dumped from the real model
  2. Preprocesses (dtype → fp16, q *= scale, GQA expand, cu_seqlens → int32)
  3. Calls fla_npu mega_chunk_kda
  4. Compares against the reference output `o` and `recurrent_state`
"""

import argparse
import os
import sys

import torch
import torch.nn.functional as F
import torch_npu  # noqa: F401  (registers npu device)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _tensor_info(name: str, t):
    if isinstance(t, torch.Tensor):
        print(f"    {name:20s} shape={tuple(t.shape)} dtype={t.dtype}")
    else:
        print(f"    {name:20s} = {t!r}")


def _print_range(name: str, t: torch.Tensor):
    """Print min/max/mean/nan/inf for a tensor."""
    if not isinstance(t, torch.Tensor) or t.numel() == 0:
        print(f"    {name:12s}: <empty or non-tensor>")
        return
    has_nan = torch.isnan(t).any().item()
    has_inf = torch.isinf(t).any().item()
    finite = t.isfinite()
    if finite.any():
        mn = t[finite].min().item()
        mx = t[finite].max().item()
        mn_val = t.float().mean().item() if t.dtype != torch.int32 else 0.0
        print(f"    {name:12s}: min={mn:+.6f}  max={mx:+.6f}  mean={mn_val:+.6f}  "
              f"nan={has_nan}  inf={has_inf}")
    else:
        print(f"    {name:12s}: ALL non-finite  nan={has_nan}  inf={has_inf}")


def _to_fp16(t: torch.Tensor) -> torch.Tensor:
    if t.dtype == torch.bfloat16:
        return t.to(torch.float16)
    if t.dtype == torch.float32:
        return t.to(torch.float16)
    if t.dtype == torch.float16:
        return t
    return t.to(torch.float16)


def _maybe_expand_gqa(q: torch.Tensor, k: torch.Tensor):
    """If q has fewer heads than k, GQA-expand q to match k's head count."""
    if q.shape[2] == k.shape[2]:
        return q
    if q.shape[2] < k.shape[2]:
        G = k.shape[2] // q.shape[2]
        assert q.shape[2] * G == k.shape[2], f"GQA ratio mismatch: q={q.shape[2]}, k={k.shape[2]}"
        print(f"    GQA expand: q heads {q.shape[2]} -> {k.shape[2]} (G={G})")
        return q.repeat_interleave(G, dim=2)
    # k has fewer heads (unusual) — expand k instead
    G = q.shape[2] // k.shape[2]
    assert k.shape[2] * G == q.shape[2]
    print(f"    GQA expand: k heads {k.shape[2]} -> {q.shape[2]} (G={G})")
    raise RuntimeError("q has more heads than k — please check your pt file")


def _stats(name: str, diff: torch.Tensor, ref: torch.Tensor):
    """Print max/mean relative error stats, masking inf/nan."""
    valid = torch.isfinite(diff) & torch.isfinite(ref) & (ref.abs() > 1e-6)
    if not valid.any():
        print(f"    {name}: no valid elements for stats")
        return
    rel = (diff[valid].abs() / ref[valid].abs())
    print(f"    {name}: max_rel={rel.max().item():.6f}  "
          f"mean_rel={rel.mean().item():.6f}  "
          f"valid={valid.sum().item()}/{valid.numel()}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="path to kda_debug_input_tensors_rank0.pt")
    parser.add_argument("--output", required=True, help="path to kda_debug_output_tensors_rank0.pt")
    parser.add_argument("--chunk-size", type=int, default=128)
    parser.add_argument("--scale", type=float, default=None,
                        help="q scale; default = 1/sqrt(K)")
    parser.add_argument("--clamp-gate", type=float, default=None,
                        help="clamp gate g to [lower_bound, upper_bound]; "
                             "e.g. --clamp-gate -5.0 clamps g >= -5.0")
    parser.add_argument("--no-gate-transform", action="store_true",
                        help="treat pt 'g' as final gate (skip A_log/dt_bias transform). "
                             "By default we apply g = -A_log.exp()*softplus(a+dt_bias) since "
                             "the pt dumps raw gate logits, not the final (always-negative) gate.")
    args = parser.parse_args()

    # ---- 1. Load pt ----
    print("=" * 60)
    print("1. Loading pt files")
    print("=" * 60)
    inp = torch.load(args.input, map_location="cpu", weights_only=False)
    out_ref = torch.load(args.output, map_location="cpu", weights_only=False)

    print("  Input keys:")
    for k, v in inp.items():
        _tensor_info(k, v)
    print("  Output keys:")
    for k, v in out_ref.items():
        _tensor_info(k, v)

    # Print metadata
    print("\n  Metadata from pt:")
    print(f"    safe_gate   = {inp.get('safe_gate')}")
    print(f"    lower_bound = {inp.get('lower_bound')}")
    print(f"    mode        = {inp.get('mode')}")
    print(f"    step        = {inp.get('step')}")
    print(f"    layer       = {inp.get('layer')}")

    # ---- 2. Extract & preprocess ----
    print("\n" + "=" * 60)
    print("2. Preprocessing")
    print("=" * 60)

    q = inp["q"]
    k = inp["k"]
    v = inp["v"]
    g = inp["g"]
    beta = inp["beta"]
    cu_seqlens = inp.get("cu_seqlens")

    # Pull gate-transform params if present (pt dumps raw gate logits `a`,
    # not the final always-negative gate `g = -A_log.exp()*softplus(a+dt_bias)`).
    A_log = inp.get("A_log")
    dt_bias = inp.get("dt_bias")
    safe_gate = inp.get("safe_gate", False)
    lower_bound = inp.get("lower_bound", None)

    # Print value ranges BEFORE conversion
    print("\n  --- Value ranges (BEFORE dtype conversion) ---")
    _print_range("q", q)
    _print_range("k", k)
    _print_range("v", v)
    _print_range("g", g)
    _print_range("beta", beta)

    # Gate-specific analysis
    if isinstance(g, torch.Tensor):
        print("\n  --- Gate analysis (raw pt 'g') ---")
        g_finite = g.float()
        n_pos = (g_finite > 0).sum().item()
        n_gt11 = (g_finite > 11.0).sum().item()
        n_lt_neg11 = (g_finite < -11.0).sum().item()
        n_total = g.numel()
        print(f"    g > 0:    {n_pos}/{n_total} ({100*n_pos/n_total:.2f}%)")
        print(f"    g > 11:   {n_gt11}/{n_total} (exp would overflow fp16)")
        print(f"    g < -11:  {n_lt_neg11}/{n_total} (exp → 0 in fp16, underflow)")
        # Check per-head gate stats
        for h in range(g.shape[2]):
            gh = g[0, :, h, :].float()
            print(f"    head {h}: g min={gh.min().item():+.4f}  max={gh.max().item():+.4f}  "
                  f"mean={gh.mean().item():+.4f}")

    # ---- Gate transformation: raw logit `a` → final gate `g` ----
    # Model formula (flash_gated_delta_rule.py:1102):
    #   g = -A_log.float().exp() * softplus(a.float() + dt_bias)
    # The pt file's `g` field is the raw per-dimension gate logit `a` (can be
    # positive), NOT the final gate (which is always ≤ 0).  KDA kernels and the
    # reference implementation both assume gate ≤ 0 (they compute exp(g_cs)
    # directly, which overflows for positive g).  We must apply the transform
    # here so the kernel receives the same always-negative gate the model
    # produces at inference time.
    if not args.no_gate_transform and isinstance(A_log, torch.Tensor) and isinstance(dt_bias, torch.Tensor):
        print("\n  --- Gate transformation: a → g = -A_log.exp()*softplus(a+dt_bias) ---")
        a = g.float()                                   # [1, T, HV, K]
        A = A_log.float()                               # [HV]
        db = dt_bias.float()                            # [HV*K] or [HV]

        HV, K = a.shape[2], a.shape[3]
        print(f"    a        shape={tuple(a.shape)}")
        print(f"    A_log    shape={tuple(A.shape)}  values={A.tolist()}")
        print(f"    dt_bias  shape={tuple(db.shape)}")

        # Reshape dt_bias to broadcast against a [1, T, HV, K].
        if db.numel() == HV * K:
            db = db.view(1, 1, HV, K)
        elif db.numel() == HV:
            db = db.view(1, 1, HV, 1)
        else:
            print(f"    ⚠️  dt_bias numel={db.numel()} does not match HV={HV} or HV*K={HV*K}; "
                  "skipping transform")
            db = None

        if db is not None:
            A_b = A.view(1, 1, HV, 1)                   # [1, 1, HV, 1]
            g = -A_b.exp() * F.softplus(a + db)         # [1, T, HV, K], always ≤ 0
            print(f"    transformed g shape={tuple(g.shape)}")
            _print_range("g (transformed)", g)

            # safe_gate clamp (matches model's safe_gate=True, lower_bound=-5.0)
            if safe_gate and lower_bound is not None:
                n_clamped = (g < lower_bound).sum().item()
                g = g.clamp(min=float(lower_bound))
                print(f"    safe_gate clamp to [{lower_bound}, +inf): "
                      f"{n_clamped} elements clamped")
                _print_range("g (clamped)", g)

            # Re-run per-head stats on the final gate
            print("\n  --- Gate analysis (after transform) ---")
            g_finite = g.float()
            n_pos = (g_finite > 0).sum().item()
            n_total = g.numel()
            print(f"    g > 0:    {n_pos}/{n_total} ({100*n_pos/n_total:.2f}%)  (should be 0%)")
            for h in range(g.shape[2]):
                gh = g[0, :, h, :].float()
                print(f"    head {h}: g min={gh.min().item():+.4f}  max={gh.max().item():+.4f}  "
                      f"mean={gh.mean().item():+.4f}")
    elif not args.no_gate_transform:
        print("\n  ⚠️  pt file missing A_log/dt_bias — cannot apply gate transform. "
              "Treating 'g' as final gate (may cause NaN if g contains positive values).")

    # dtype → fp16
    q = _to_fp16(q)
    k = _to_fp16(k)
    v = _to_fp16(v)
    g = _to_fp16(g)
    beta = _to_fp16(beta)
    print("\n  dtype → fp16 done")

    # Print value ranges AFTER conversion
    print("\n  --- Value ranges (AFTER fp16 conversion) ---")
    _print_range("q", q)
    _print_range("k", k)
    _print_range("v", v)
    _print_range("g", g)
    _print_range("beta", beta)

    # Optional gate clamping
    if args.clamp_gate is not None:
        lb = args.clamp_gate
        n_clamped = (g < lb).sum().item()
        g = g.clamp(min=lb)
        print(f"\n  Gate clamped to [{lb}, +inf): {n_clamped} elements affected")
        _print_range("g (clamped)", g)

    # scale
    K = q.shape[-1]
    scale = args.scale if args.scale is not None else float(K ** -0.5)
    print(f"\n  scale = {scale:.6f}  (K={K})")
    q = q * scale

    # GQA expand
    q = _maybe_expand_gqa(q, k)

    # cu_seqlens
    if cu_seqlens is None:
        T = q.shape[1]
        cu_seqlens = torch.tensor([0, T], dtype=torch.int32)
        print(f"  cu_seqlens = None → synthesised [0, {T}]")
    else:
        cu_seqlens = cu_seqlens.to(torch.int32).cpu()
        print(f"  cu_seqlens = {cu_seqlens.tolist()}")

    # initial_state check
    is0 = inp.get("initial_state")
    if isinstance(is0, torch.Tensor):
        all_zero = bool((is0 == 0).all())
        print(f"  initial_state: shape={tuple(is0.shape)} all_zero={all_zero}")
        if not all_zero:
            print("  ⚠️  WARNING: initial_state is non-zero — mega-kernel does NOT support")
            print("     non-zero initial_state. Results will not match reference.")
            print("     (Original KDA mega-kernel also falls back to Triton in this case.)")

    # Move to NPU
    dev = torch.npu.current_device()
    q = q.to(dev)
    k = k.to(dev)
    v = v.to(dev)
    g = g.to(dev)
    beta = beta.to(dev)
    cu_seqlens = cu_seqlens.to(dev)
    print(f"  moved to {dev}")

    print(f"\n  Final shapes:")
    print(f"    q    {tuple(q.shape)} {q.dtype}")
    print(f"    k    {tuple(k.shape)} {k.dtype}")
    print(f"    v    {tuple(v.shape)} {v.dtype}")
    print(f"    g    {tuple(g.shape)} {g.dtype}")
    print(f"    beta {tuple(beta.shape)} {beta.dtype}")

    # ---- 3. Call mega_chunk_kda ----
    print("\n" + "=" * 60)
    print("3. Calling mega_chunk_kda")
    print("=" * 60)

    from fla_npu.ops.ascendc import mega_chunk_kda

    torch.npu.synchronize()
    out, final_state = mega_chunk_kda(
        q, k, v, g, beta, cu_seqlens,
        chunk_size=args.chunk_size,
        return_final_state=True,
    )
    torch.npu.synchronize()
    print(f"  out          shape={tuple(out.shape)} dtype={out.dtype}")
    print(f"  final_state  shape={tuple(final_state.shape)} dtype={final_state.dtype}")

    # Print NPU output ranges
    print("\n  --- NPU output ranges ---")
    _print_range("out", out)
    _print_range("final_state", final_state)

    # ---- 4. Compare with reference ----
    print("\n" + "=" * 60)
    print("4. Comparing with reference")
    print("=" * 60)

    o_ref = out_ref["o"]
    if o_ref.dtype != torch.float16:
        o_ref = o_ref.to(torch.float16)
    o_ref = o_ref.to(dev)

    # Print reference output ranges
    print("  --- Reference output ranges ---")
    _print_range("o_ref", o_ref)

    # Shape check
    if out.shape != o_ref.shape:
        print(f"  ⚠️  Shape mismatch! out={tuple(out.shape)} vs ref={tuple(o_ref.shape)}")
        # Try to broadcast/reshape
        if out.numel() == o_ref.numel():
            out_cmp = out.reshape(o_ref.shape)
            print(f"     reshaped out to {tuple(out_cmp.shape)}")
        else:
            print("     Cannot compare — different element counts")
            sys.exit(1)
    else:
        out_cmp = out

    diff = (out_cmp - o_ref).float()
    abs_diff = diff.abs()
    print(f"\n  o: abs_diff max={abs_diff.max().item():.6f}  mean={abs_diff.mean().item():.6f}")
    _stats("o_rel", diff, o_ref.float())

    # recurrent_state
    rs_ref = out_ref.get("recurrent_state")
    if isinstance(rs_ref, torch.Tensor):
        rs_ref = rs_ref.to(dev).float()
        fs = final_state.float()
        print(f"\n  --- recurrent_state comparison ---")
        _print_range("rs_ref", rs_ref)
        _print_range("final_state", fs)
        if fs.shape == rs_ref.shape:
            rs_diff = (fs - rs_ref).abs()
            print(f"  recurrent_state: abs_diff max={rs_diff.max().item():.6f}  "
                  f"mean={rs_diff.mean().item():.6f}")
            _stats("rs_rel", fs - rs_ref, rs_ref)

            # Per-sequence comparison
            n_seq = fs.shape[0]
            for s_idx in range(min(n_seq, 4)):
                seq_diff = (fs[s_idx] - rs_ref[s_idx]).abs()
                seq_valid = torch.isfinite(fs[s_idx]) & torch.isfinite(rs_ref[s_idx])
                if seq_valid.any():
                    print(f"    seq {s_idx}: max_abs_diff={seq_diff[seq_valid].max().item():.6f}  "
                          f"mean_abs_diff={seq_diff[seq_valid].mean().item():.6f}")
        else:
            print(f"  recurrent_state shape mismatch: out={tuple(fs.shape)} vs ref={tuple(rs_ref.shape)}")
            print(f"  (final_state dump for manual inspection)")
            print(f"    final_state[0,0,:3,:3] = \n{fs[0,0,:3,:3]}")
            print(f"    ref[0,0,:3,:3] = \n{rs_ref[0,0,:3,:3]}")

    # ---- 5. Verdict ----
    print("\n" + "=" * 60)
    print("5. Verdict")
    print("=" * 60)

    # Explicit NaN/Inf fraction check — if output is mostly non-finite, FAIL hard.
    n_total = out_cmp.numel()
    n_nan = torch.isnan(out_cmp.float()).sum().item()
    n_inf = torch.isinf(out_cmp.float()).sum().item()
    nan_frac = n_nan / n_total if n_total > 0 else 1.0
    print(f"  out: {n_nan}/{n_total} NaN ({100*nan_frac:.2f}%), {n_inf}/{n_total} Inf")
    if nan_frac > 0.01:
        print(f"  ❌ FAIL  (output {100*nan_frac:.2f}% NaN — gate transform may be missing or kernel overflow)")
        sys.exit(1)

    valid = torch.isfinite(diff) & torch.isfinite(o_ref.float()) & (o_ref.float().abs() > 1e-6)
    if valid.any():
        rel = (diff[valid].abs() / o_ref.float()[valid].abs())
        max_rel = rel.max().item()
        mean_rel = rel.mean().item()
        if max_rel < 0.05:
            print(f"  ✅ PASS  (max_rel={max_rel:.6f}  mean_rel={mean_rel:.6f})")
            sys.exit(0)
        elif max_rel < 0.15:
            print(f"  ⚠️  MARGINAL  (max_rel={max_rel:.6f}  mean_rel={mean_rel:.6f})")
            sys.exit(1)
        else:
            print(f"  ❌ FAIL  (max_rel={max_rel:.6f}  mean_rel={mean_rel:.6f})")
            sys.exit(1)
    else:
        print("  ❌ Cannot evaluate — no valid elements")
        sys.exit(1)


if __name__ == "__main__":
    main()
