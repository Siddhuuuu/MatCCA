"""ablate.py — turn each RECONSTRUCTION component off one at a time to find out
which one is actually breaking learning on the toy periodic task."""

import torch
from tiny_transformer import TinyTransformerLM
from test_training import train_loop

torch.manual_seed(0)
vocab_size, dim, n_layers, n_heads, seq_len, steps = 64, 64, 2, 4, 32, 250

configs = {
    "all ON (current default)":        dict(use_conv_mix=True,  use_qk_mean=True,  use_v_shift=True),
    "conv_mix OFF":                    dict(use_conv_mix=False, use_qk_mean=True,  use_v_shift=True),
    "qk_mean OFF":                     dict(use_conv_mix=True,  use_qk_mean=False, use_v_shift=True),
    "v_shift OFF":                     dict(use_conv_mix=True,  use_qk_mean=True,  use_v_shift=False),
    "ALL reconstructions OFF (bare)":  dict(use_conv_mix=False, use_qk_mean=False, use_v_shift=False),
}

baseline = torch.log(torch.tensor(float(vocab_size))).item()
print(f"random-guess baseline: {baseline:.4f}\n")

for name, flags in configs.items():
    torch.manual_seed(0)
    model = TinyTransformerLM(
        vocab_size=vocab_size, dim=dim, n_layers=n_layers, n_heads=n_heads,
        max_seq_len=seq_len, attn_kind="cca",
        attn_kwargs=dict(n_heads_latent=n_heads, n_kv_heads_latent=n_heads,
                          compression_q=4, compression_kv=4, **flags),
    )
    losses = train_loop(model, steps, seq_len, vocab_size, log_every=steps)  # only log last step
    final = min(losses[-10:])
    verdict = "LEARNED" if final < baseline * 0.7 else "stuck at baseline"
    print(f"[{name}] final loss={final:.4f}  -> {verdict}")
