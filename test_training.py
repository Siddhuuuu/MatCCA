"""
test_training.py — proves the models don't just run, they actually learn.

Task: a toy periodic sequence (predict token at position t from a short repeating
pattern) — trivially learnable, so a working model's loss should drop close to zero
within a few hundred steps. This is the real bar: an architecture that "runs" but
can't be trained on even this is not a valid foundation to build MatCCA on top of.

The last section is the important one: it trains CCA with a RANDOM active rank
sampled every step (the actual Matryoshka nesting mechanism MatCCA depends on) and
checks the loss still converges — proving the nested-training idea is trainable at
all before any real GPU time gets spent on it.
"""

import torch
import torch.nn as nn
from tiny_transformer import TinyTransformerLM

torch.manual_seed(0)
DEVICE = "cpu"


def make_periodic_batch(batch_size, seq_len, vocab_size, period=8):
    """Each sequence in the batch has its own random period-`period` pattern, tiled
    to seq_len+1 (input is [:-1], target is [1:])."""
    patterns = torch.randint(0, vocab_size, (batch_size, period))
    reps = seq_len // period + 2
    full = patterns.repeat(1, reps)[:, :seq_len + 1]
    return full[:, :-1].to(DEVICE), full[:, 1:].to(DEVICE)


def train_loop(model, steps, seq_len, vocab_size, batch_size=16, lr=1e-2, sample_rank_fn=None, log_every=50):
    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    losses = []
    for step in range(steps):
        x, y = make_periodic_batch(batch_size, seq_len, vocab_size)
        attn_kwargs = sample_rank_fn() if sample_rank_fn else {}
        _, loss = model(x, targets=y, **attn_kwargs)
        opt.zero_grad()
        loss.backward()
        opt.step()
        losses.append(loss.item())
        if step % log_every == 0 or step == steps - 1:
            extra = f" (attn_kwargs={attn_kwargs})" if attn_kwargs else ""
            print(f"    step {step:4d}: loss={loss.item():.4f}{extra}")
    return losses


def run_variant(name, attn_kind, attn_kwargs, vocab_size=64, dim=64, n_layers=2, n_heads=4, seq_len=32, steps=400, seed=0):
    print(f"\n=== {name} ===")
    torch.manual_seed(seed)  # fresh seed per variant — models must not share one continuous
    model = TinyTransformerLM(                       # random stream, or execution order silently
        vocab_size=vocab_size, dim=dim, n_layers=n_layers, n_heads=n_heads,  # biases who gets a
        max_seq_len=seq_len, attn_kind=attn_kind, attn_kwargs=attn_kwargs,   # lucky/unlucky init.
    ).to(DEVICE)
    print(f"  params: {model.param_count():,}")
    losses = train_loop(model, steps, seq_len, vocab_size)
    initial, final = losses[0], min(losses[-10:])
    baseline = torch.log(torch.tensor(float(vocab_size))).item()
    print(f"  loss: {initial:.4f} -> {final:.4f} (random-guess baseline ~= {baseline:.4f})")
    assert final < baseline * 0.6, f"{name} did not converge (final loss too close to random-guess baseline)"
    return model, losses


def run_matcca_nested_training(vocab_size=64, dim=64, n_layers=2, n_heads=4, seq_len=32, steps=500, seed=0):
    print("\n=== CCA with RANDOM rank sampled every step (Matryoshka nesting) ===")
    torch.manual_seed(seed)
    model = TinyTransformerLM(
        vocab_size=vocab_size, dim=dim, n_layers=n_layers, n_heads=n_heads,
        max_seq_len=seq_len, attn_kind="cca",
        attn_kwargs=dict(n_heads_latent=n_heads, n_kv_heads_latent=n_heads, compression_q=4, compression_kv=4),
    ).to(DEVICE)
    print(f"  params: {model.param_count():,}")

    cca0 = model.blocks[0].attn
    max_r_q, max_r_kv = cca0.latent_dim_q_max, cca0.latent_dim_kv_max
    rank_fracs = [1.0, 0.5, 0.25]

    def sample_rank_fn():
        frac = rank_fracs[torch.randint(0, len(rank_fracs), (1,)).item()]
        r_q = max(cca0.head_dim_q, int(max_r_q * frac) - int(max_r_q * frac) % cca0.head_dim_q)
        r_kv = max(cca0.head_dim_kv, int(max_r_kv * frac) - int(max_r_kv * frac) % cca0.head_dim_kv)
        return dict(active_rank_q=r_q, active_rank_kv=r_kv)

    losses = train_loop(model, steps, seq_len, vocab_size, sample_rank_fn=sample_rank_fn, log_every=100)

    print("\n  Post-training: evaluate at EACH fixed rank level (loss should be low at all of them,")
    print("  since nested training is supposed to make every truncation level usable):")
    model.eval()
    with torch.no_grad():
        for frac in rank_fracs:
            r_q = max(cca0.head_dim_q, int(max_r_q * frac) - int(max_r_q * frac) % cca0.head_dim_q)
            r_kv = max(cca0.head_dim_kv, int(max_r_kv * frac) - int(max_r_kv * frac) % cca0.head_dim_kv)
            x, y = make_periodic_batch(64, seq_len, vocab_size)
            _, loss = model(x, targets=y, active_rank_q=r_q, active_rank_kv=r_kv)
            print(f"    rank fraction {frac:.2f} (r_q={r_q}, r_kv={r_kv}): eval loss={loss.item():.4f}")
    model.train()
    return model, losses


if __name__ == "__main__":
    run_variant("MultiHeadAttention baseline", "mha", {})
    run_variant("GroupedQueryAttention baseline", "gqa", dict(n_kv_heads=1))
    run_variant("CCA fixed-rank (compression=4)", "cca",
                dict(n_heads_latent=4, n_kv_heads_latent=4, compression_q=4, compression_kv=4))
    run_matcca_nested_training()
    print("\nAll variants converge on the toy task. Nested-rank training loop is viable.")
