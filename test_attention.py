"""test_attention.py — shape / causality / gradient / truncation checks for
MultiHeadAttention, GroupedQueryAttention, and CompressedConvolutionalAttention."""

import torch
from attention import MultiHeadAttention, GroupedQueryAttention, CompressedConvolutionalAttention

torch.manual_seed(0)


def check_causality(module, x, name):
    """Perturbing token t+1..T-1 must not change output at token t (causal mask check)."""
    B, T, E = x.shape
    out1 = module(x)
    x2 = x.clone()
    x2[:, T // 2:, :] += torch.randn_like(x2[:, T // 2:, :]) * 10.0  # big perturbation, second half
    out2 = module(x2)
    early_diff = (out1[:, :T // 2] - out2[:, :T // 2]).abs().max().item()
    late_diff = (out1[:, T // 2:] - out2[:, T // 2:]).abs().max().item()
    status = "OK" if early_diff < 1e-4 else "FAIL"
    print(f"  [{name}] causality: early-token max diff={early_diff:.2e} ({status}), "
          f"late-token max diff={late_diff:.2e} (should be large)")
    assert early_diff < 1e-4, f"{name} leaks future information!"


def check_shape_and_grad(module, x, name):
    out = module(x)
    assert out.shape == x.shape, f"{name} shape mismatch: {out.shape} vs {x.shape}"
    loss = out.pow(2).mean()
    loss.backward()
    grads_ok = all(
        p.grad is not None and torch.isfinite(p.grad).all()
        for p in module.parameters() if p.requires_grad
    )
    print(f"  [{name}] shape OK {tuple(out.shape)}, gradients finite: {grads_ok}")
    assert grads_ok, f"{name} produced NaN/inf gradients"
    module.zero_grad()


def main():
    B, T, E = 2, 32, 256
    x = torch.randn(B, T, E)

    print("=== MultiHeadAttention (baseline) ===")
    mha = MultiHeadAttention(dim=E, n_heads=8)
    check_shape_and_grad(mha, x.clone().requires_grad_(True), "MHA")
    check_causality(mha, x, "MHA")
    print(f"  params: {mha.param_count():,}")

    print("\n=== GroupedQueryAttention (baseline) ===")
    gqa = GroupedQueryAttention(dim=E, n_heads=8, n_kv_heads=2)
    check_shape_and_grad(gqa, x.clone().requires_grad_(True), "GQA")
    check_causality(gqa, x, "GQA")
    print(f"  params: {gqa.param_count():,}, kv-cache dim/token: {gqa.kv_cache_dim_per_token()}")

    print("\n=== CompressedConvolutionalAttention (target) ===")
    cca = CompressedConvolutionalAttention(
        dim=E, n_heads_latent=8, n_kv_heads_latent=8,
        compression_q=4, compression_kv=4,
    )
    check_shape_and_grad(cca, x.clone().requires_grad_(True), "CCA (full rank)")
    check_causality(cca, x, "CCA (full rank)")
    print(f"  params: {cca.param_count():,}, "
          f"kv-cache dim/token: {cca.kv_cache_dim_per_token()} "
          f"(vs GQA's {gqa.kv_cache_dim_per_token()} at comparable compression)")

    print("\n=== CCA Matryoshka truncation (the MatCCA hook) ===")
    max_r_kv = cca.latent_dim_kv_max
    max_r_q = cca.latent_dim_q_max
    for frac in [1.0, 0.5, 0.25]:
        r_q = int(max_r_q * frac)
        r_kv = int(max_r_kv * frac)
        r_q -= r_q % cca.head_dim_q  # keep multiple of head_dim
        r_kv -= r_kv % cca.head_dim_kv
        r_q, r_kv = max(r_q, cca.head_dim_q), max(r_kv, cca.head_dim_kv)
        out = cca(x, active_rank_q=r_q, active_rank_kv=r_kv)
        cache_dim = cca.kv_cache_dim_per_token(active_rank_kv=r_kv)
        print(f"  truncated to r_q={r_q}/{max_r_q}, r_kv={r_kv}/{max_r_kv} -> "
              f"out shape {tuple(out.shape)}, kv-cache dim/token={cache_dim}")
        assert out.shape == x.shape, "truncated forward changed output shape!"
        loss = out.pow(2).mean()
        loss.backward()
        cca.zero_grad()

    print("\nAll checks passed.")


if __name__ == "__main__":
    main()
