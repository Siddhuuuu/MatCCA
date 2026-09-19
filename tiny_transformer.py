"""
tiny_transformer.py — a small decoder-only LM wrapping MHA / GQA / CCA interchangeably.

Default config is deliberately tiny (a few million params) so this trains fast on CPU
for correctness checks here; bump dim/n_layers/vocab up for a real Kaggle T4 run (see
the scoping doc's suggested plan: 50-150M params, C4-scale data, matched token budget
across conditions).

Positional encoding: plain learned absolute embeddings, for simplicity in this first
scaffold. Real CCA/MLA-style models use RoPE; integrating RoPE into the compressed
latent space is its own subtlety (where exactly it composes with the down-projection)
that isn't nailed down here — flagged as a follow-up, not done yet.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from attention import MultiHeadAttention, GroupedQueryAttention, CompressedConvolutionalAttention


class MLP(nn.Module):
    def __init__(self, dim, expansion=4):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, dim * expansion),
            nn.GELU(),
            nn.Linear(dim * expansion, dim),
        )

    def forward(self, x):
        return self.net(x)


class Block(nn.Module):
    def __init__(self, dim, attn_module):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = attn_module
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = MLP(dim)

    def forward(self, x, **attn_kwargs):
        x = x + self.attn(self.norm1(x), **attn_kwargs)
        x = x + self.mlp(self.norm2(x))
        return x


def make_attention(kind, dim, n_heads, **kwargs):
    if kind == "mha":
        return MultiHeadAttention(dim, n_heads)
    if kind == "gqa":
        return GroupedQueryAttention(dim, n_heads, n_kv_heads=kwargs.get("n_kv_heads", n_heads // 4))
    if kind == "cca":
        return CompressedConvolutionalAttention(
            dim,
            n_heads_latent=kwargs.get("n_heads_latent", n_heads),
            n_kv_heads_latent=kwargs.get("n_kv_heads_latent", n_heads),
            compression_q=kwargs.get("compression_q", 4),
            compression_kv=kwargs.get("compression_kv", 4),
            conv_kernel=kwargs.get("conv_kernel", 4),
            use_conv_mix=kwargs.get("use_conv_mix", True),
            use_qk_mean=kwargs.get("use_qk_mean", True),
            use_v_shift=kwargs.get("use_v_shift", True),
        )
    raise ValueError(f"unknown attention kind: {kind}")


class TinyTransformerLM(nn.Module):
    def __init__(
        self, vocab_size, dim=128, n_layers=4, n_heads=8, max_seq_len=256,
        attn_kind="cca", attn_kwargs=None,
    ):
        super().__init__()
        attn_kwargs = attn_kwargs or {}
        self.max_seq_len = max_seq_len
        self.attn_kind = attn_kind
        self.tok_emb = nn.Embedding(vocab_size, dim)
        self.pos_emb = nn.Embedding(max_seq_len, dim)
        self.blocks = nn.ModuleList([
            Block(dim, make_attention(attn_kind, dim, n_heads, **attn_kwargs))
            for _ in range(n_layers)
        ])
        self.norm_f = nn.LayerNorm(dim)
        self.lm_head = nn.Linear(dim, vocab_size, bias=False)
        self.lm_head.weight = self.tok_emb.weight  # weight tying

    def forward(self, idx, targets=None, **attn_kwargs):
        B, T = idx.shape
        assert T <= self.max_seq_len
        pos = torch.arange(T, device=idx.device)
        x = self.tok_emb(idx) + self.pos_emb(pos)[None, :, :]
        for block in self.blocks:
            x = block(x, **attn_kwargs)
        x = self.norm_f(x)
        logits = self.lm_head(x)
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), targets.reshape(-1))
        return logits, loss

    def param_count(self):
        return sum(p.numel() for p in self.parameters())
