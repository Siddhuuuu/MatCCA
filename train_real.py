"""
train_real.py — the real thing: trains on actual text (tiny-Shakespeare by default,
point DATA_PATH at any .txt file — TinyStories or a C4 subset work fine on Kaggle
with internet enabled, just download to a .txt and pass the path), at a scale meant
to be run on a GPU, with checkpointing so a Kaggle session timeout doesn't lose
progress, and a text-generation sanity check at the end so you can *read* whether
the model learned something, not just watch a loss number.

Bumped from the toy config: dim 64->256, seq_len 32->256, real steps, real val split.
This is still a first real-data pass, not the full 50-150M param experiment from the
scoping doc's §7 — think of this as "does the whole pipeline hold up on real text and
at 10x the toy scale" before committing a full Kaggle GPU-hours budget to the complete
comparison.
"""

import os
import time
import torch

from tiny_transformer import TinyTransformerLM
from real_data import prepare_dataset, get_batch, estimate_loss

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DATA_PATH = "shakespeare.txt"          # point this at TinyStories/C4/anything else
CHECKPOINT_DIR = "checkpoints"
os.makedirs(CHECKPOINT_DIR, exist_ok=True)


def train_one_variant(
    name, attn_kind, attn_kwargs, train_data, val_data, vocab_size,
    dim=256, n_layers=6, n_heads=8, seq_len=256, batch_size=32,
    steps=2000, lr=3e-4, eval_every=200, eval_batches=20,
    checkpoint_every=500, sample_rank_fn=None, resume=True, seed=0,
):
    print(f"\n{'=' * 60}\n{name} (seed={seed})\n{'=' * 60}")
    ckpt_path = os.path.join(CHECKPOINT_DIR, f"{name.replace(' ', '_')}_seed{seed}.pt")

    torch.manual_seed(seed)  # seed BEFORE model creation — controls init, not just data order
    model = TinyTransformerLM(
        vocab_size=vocab_size, dim=dim, n_layers=n_layers, n_heads=n_heads,
        max_seq_len=seq_len, attn_kind=attn_kind, attn_kwargs=attn_kwargs,
    ).to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    start_step = 0

    if resume and os.path.exists(ckpt_path):
        ckpt = torch.load(ckpt_path, map_location=DEVICE)
        model.load_state_dict(ckpt["model"])
        opt.load_state_dict(ckpt["optimizer"])
        start_step = ckpt["step"]
        print(f"  resumed from checkpoint at step {start_step}")

    print(f"  params: {model.param_count():,}  |  device: {DEVICE}")
    t0 = time.time()

    for step in range(start_step, steps):
        x, y = get_batch(train_data, batch_size, seq_len, DEVICE)
        attn_kwargs_step = sample_rank_fn() if sample_rank_fn else {}
        _, loss = model(x, targets=y, **attn_kwargs_step)
        opt.zero_grad()
        loss.backward()
        opt.step()

        if step % eval_every == 0 or step == steps - 1:
            val_loss = estimate_loss(model, val_data, batch_size, seq_len, DEVICE, eval_batches, **attn_kwargs_step)
            elapsed = time.time() - t0
            print(f"  step {step:5d} | train loss {loss.item():.4f} | val loss {val_loss:.4f} "
                  f"| {elapsed:.0f}s elapsed")

        if step % checkpoint_every == 0 and step > start_step:
            torch.save({"model": model.state_dict(), "optimizer": opt.state_dict(), "step": step}, ckpt_path)

    torch.save({"model": model.state_dict(), "optimizer": opt.state_dict(), "step": steps}, ckpt_path)
    final_val = estimate_loss(model, val_data, batch_size, seq_len, DEVICE, eval_batches)
    print(f"  FINAL val loss: {final_val:.4f}  (checkpoint saved to {ckpt_path})")
    return model, final_val


@torch.no_grad()
def generate(model, tok, prompt, max_new_tokens=200, seq_len=256, **attn_kwargs):
    model.eval()
    ids = tok.encode(prompt).unsqueeze(0).to(DEVICE)
    for _ in range(max_new_tokens):
        ids_cond = ids[:, -seq_len:]
        logits, _ = model(ids_cond, **attn_kwargs)
        next_logits = logits[:, -1, :]
        probs = torch.softmax(next_logits, dim=-1)
        next_id = torch.multinomial(probs, num_samples=1)
        ids = torch.cat([ids, next_id], dim=1)
    model.train()
    return tok.decode(ids[0])


def main():
    print(f"Loading real text data from '{DATA_PATH}'...")
    train_data, val_data, tok = prepare_dataset(DATA_PATH)
    train_data, val_data = train_data.to(DEVICE), val_data.to(DEVICE)
    print(f"vocab_size={tok.vocab_size}, train={len(train_data):,} chars, val={len(val_data):,} chars")
    print(f"device: {DEVICE}\n")

    common = dict(train_data=train_data, val_data=val_data, vocab_size=tok.vocab_size)

    mha_model, _ = train_one_variant("MHA baseline", "mha", {}, **common)
    print("\nsample generation:")
    print(generate(mha_model, tok, prompt="ROMEO:", max_new_tokens=200))

    gqa_model, _ = train_one_variant("GQA baseline", "gqa", dict(n_kv_heads=2), **common)

    cca_model, _ = train_one_variant(
        "CCA fixed-rank (compression=4)", "cca",
        dict(n_heads_latent=8, n_kv_heads_latent=8, compression_q=4, compression_kv=4),
        **common,
    )
    print("\nsample generation:")
    print(generate(cca_model, tok, prompt="ROMEO:", max_new_tokens=200))

    # MatCCA: nested training with a random rank sampled each step
    matcca_kwargs = dict(n_heads_latent=8, n_kv_heads_latent=8, compression_q=4, compression_kv=4)
    max_r = 256 // 4  # dim // compression
    head_dim = max_r // 8
    rank_fracs = [1.0, 0.5, 0.25]

    def sample_rank_fn():
        import random
        frac = random.choice(rank_fracs)
        r = max(head_dim, int(max_r * frac) - int(max_r * frac) % head_dim)
        return dict(active_rank_q=r, active_rank_kv=r)

    matcca_model, _ = train_one_variant(
        "MatCCA (nested Matryoshka rank)", "cca", matcca_kwargs,
        sample_rank_fn=sample_rank_fn, **common,
    )
    print("\nMatCCA evaluated at each fixed rank level:")
    for frac in rank_fracs:
        r = max(head_dim, int(max_r * frac) - int(max_r * frac) % head_dim)
        val_loss = estimate_loss(matcca_model, val_data, 32, 256, DEVICE, 20, active_rank_q=r, active_rank_kv=r)
        print(f"  rank fraction {frac:.2f} (r={r}/{max_r}): val loss={val_loss:.4f}")

    print("\nDone. Compare the FINAL val loss lines above across all four variants —")
    print("that's your first real, non-toy signal on whether MatCCA holds up.")


if __name__ == "__main__":
    main()
