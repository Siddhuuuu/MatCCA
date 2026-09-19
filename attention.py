"""
attention.py - MHA / GQA baselines plus Compressed Convolutional Attention (CCA/CCGQA),
with a Matryoshka-style truncation hook for the MatCCA experiment.

UPDATE: this now follows the paper's actual Listing 1 code and equations (8)-(11)
(Figliolia et al., arXiv:2510.04476), not a prose-based reconstruction. Three things
the earlier version got wrong, now fixed:
- QK-mean is a same-position average of PRE-conv q and k, added as a residual bias to
  POST-conv q/k -- NOT a causal running mean over time (that was a real conceptual bug,
  not just an approximation).
- "Channel mixing" is a second convolution, grouped by attention head (mixes channels
  within a head plus a small sequence window) -- NOT a dense LayerNorm+MLP block. The
  MLP version was almost certainly the main cause of CCA benchmarking ~1.5x SLOWER than
  MHA in this codebase, the opposite of the paper's claimed 1.7x speedup; the paper
  itself notes naive (non-fused) implementations of these ops carry real overhead, but
  a grouped conv should be far cheaper than a full MLP.
- Value-shift uses TWO SEPARATELY-LEARNED projections (one sees the current token, one
  sees the previous token), each producing half the heads -- NOT one projection reused
  with its output shifted in time.
- Normalization is L2-normalize-and-rescale with a learnable exponential key
  temperature -- NOT RMSNorm (close in spirit, not identical).
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiHeadAttention(nn.Module):
    """Standard causal MHA -- baseline #1."""

    def __init__(self, dim: int, n_heads: int):
        super().__init__()
        assert dim % n_heads == 0
        self.n_heads = n_heads
        self.head_dim = dim // n_heads
        self.qkv = nn.Linear(dim, 3 * dim, bias=False)
        self.out_proj = nn.Linear(dim, dim, bias=False)

    def forward(self, x, **kwargs):
        B, T, E = x.shape
        qkv = self.qkv(x).view(B, T, 3, self.n_heads, self.head_dim)
        q, k, v = qkv.unbind(dim=2)
        q, k, v = (t.transpose(1, 2) for t in (q, k, v))
        out = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        out = out.transpose(1, 2).reshape(B, T, E)
        return self.out_proj(out)

    def param_count(self):
        return sum(p.numel() for p in self.parameters())


class GroupedQueryAttention(nn.Module):
    """Standard causal GQA -- baseline #2."""

    def __init__(self, dim: int, n_heads: int, n_kv_heads: int):
        super().__init__()
        assert dim % n_heads == 0 and n_heads % n_kv_heads == 0
        self.n_heads = n_heads
        self.n_kv_heads = n_kv_heads
        self.head_dim = dim // n_heads
        self.q_proj = nn.Linear(dim, n_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(dim, n_kv_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(dim, n_kv_heads * self.head_dim, bias=False)
        self.out_proj = nn.Linear(n_heads * self.head_dim, dim, bias=False)

    def forward(self, x, **kwargs):
        B, T, E = x.shape
        q = self.q_proj(x).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(B, T, self.n_kv_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, T, self.n_kv_heads, self.head_dim).transpose(1, 2)
        rep = self.n_heads // self.n_kv_heads
        k = k.repeat_interleave(rep, dim=1)
        v = v.repeat_interleave(rep, dim=1)
        out = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        out = out.transpose(1, 2).reshape(B, T, self.n_heads * self.head_dim)
        return self.out_proj(out)

    def kv_cache_dim_per_token(self):
        return 2 * self.n_kv_heads * self.head_dim

    def param_count(self):
        return sum(p.numel() for p in self.parameters())


def _causal_conv1d(x, weight, bias, groups, kernel_size):
    x_t = F.pad(x.transpose(1, 2), (kernel_size - 1, 0))
    out = F.conv1d(x_t, weight, bias, groups=groups)
    return out.transpose(1, 2)


class CCAConvMix(nn.Module):
    """Eq. (8): qtilde = conv2_seq+ch(conv1_seq(qtilde)). Two sequential causal convs:
    conv0 is fully depthwise (sequence-only mixing, no channel mixing at all).
    conv1 is grouped by attention head (mixes channels WITHIN a head, plus another
    small sequence window) -- replaces the dense MLP the earlier version used."""

    def __init__(self, latent_dim: int, n_heads: int, kernel_size: int = 4):
        super().__init__()
        assert latent_dim % n_heads == 0
        self.latent_dim = latent_dim
        self.n_heads = n_heads
        self.head_dim = latent_dim // n_heads
        self.kernel_size = kernel_size
        self.conv0 = nn.Conv1d(latent_dim, latent_dim, kernel_size, groups=latent_dim, bias=True)
        self.conv1 = nn.Conv1d(latent_dim, latent_dim, kernel_size, groups=n_heads, bias=True)

    def forward(self, x, active_dim: int = None):
        dim = active_dim or self.latent_dim
        active_heads = dim // self.head_dim

        w0 = self.conv0.weight[:dim]
        b0 = self.conv0.bias[:dim] if self.conv0.bias is not None else None
        out = _causal_conv1d(x, w0, b0, groups=dim, kernel_size=self.kernel_size)

        w1 = self.conv1.weight[:dim]
        b1 = self.conv1.bias[:dim] if self.conv1.bias is not None else None
        out = _causal_conv1d(out, w1, b1, groups=active_heads, kernel_size=self.kernel_size)
        return out

    def param_count(self):
        return sum(p.numel() for p in self.parameters())


def qk_mean_couple(q_pre, k_pre, q_post, k_post, n_q_heads, n_kv_heads):
    """Eq. (9): average the PRE-conv q and k at the same position (broadcasting k up
    to q's head count for GQA groups), then add that average as a residual bias to
    the POST-conv q and k. No time dimension involved."""
    rep = n_q_heads // n_kv_heads
    k_pre_b = k_pre.repeat_interleave(rep, dim=2) if rep > 1 else k_pre
    qk_mean_q = (q_pre + k_pre_b) / 2
    qk_mean_k = qk_mean_q.view(*qk_mean_q.shape[:2], n_kv_heads, rep, -1).mean(dim=3) if rep > 1 else qk_mean_q
    return q_post + qk_mean_q, k_post + qk_mean_k


def l2_norm_rescale(x, eps: float = 1e-6):
    head_dim = x.shape[-1]
    norm = x.norm(p=2, dim=-1, keepdim=True).clamp(min=eps)
    return x * (head_dim ** 0.5) / norm


class CompressedConvolutionalAttention(nn.Module):
    """CCA / CCGQA -- down-project Q, K, V into a shared compressed latent space and
    run the entire attention operation there. Supports Matryoshka-style truncation via
    active_rank_q / active_rank_kv for the MatCCA experiment."""

    def __init__(
        self, dim: int, n_heads_latent: int, n_kv_heads_latent: int = None,
        compression_q: int = 4, compression_kv: int = 4, conv_kernel: int = 4,
        use_conv_mix: bool = True, use_qk_mean: bool = True, use_v_shift: bool = True,
    ):
        super().__init__()
        n_kv_heads_latent = n_kv_heads_latent or n_heads_latent
        assert n_heads_latent % n_kv_heads_latent == 0
        self.dim = dim
        self.n_heads_latent = n_heads_latent
        self.n_kv_heads_latent = n_kv_heads_latent
        self.compression_q = compression_q
        self.compression_kv = compression_kv
        self.use_conv_mix = use_conv_mix
        self.use_qk_mean = use_qk_mean
        self.use_v_shift = use_v_shift

        self.latent_dim_q_max = dim // compression_q
        self.latent_dim_kv_max = dim // compression_kv
        assert self.latent_dim_q_max % n_heads_latent == 0
        assert self.latent_dim_kv_max % n_kv_heads_latent == 0
        assert self.latent_dim_kv_max % 2 == 0, "latent_dim_kv must be even (v-shift splits it in half)"
        self.head_dim_q = self.latent_dim_q_max // n_heads_latent
        self.head_dim_kv = self.latent_dim_kv_max // n_kv_heads_latent

        self.down_q = nn.Linear(dim, self.latent_dim_q_max, bias=False)
        self.down_k = nn.Linear(dim, self.latent_dim_kv_max, bias=False)
        half = self.latent_dim_kv_max // 2
        self.down_v_now = nn.Linear(dim, half, bias=False)
        self.down_v_prev = nn.Linear(dim, self.latent_dim_kv_max - half, bias=False)

        self.conv_q = CCAConvMix(self.latent_dim_q_max, n_heads_latent, conv_kernel)
        self.conv_k = CCAConvMix(self.latent_dim_kv_max, n_kv_heads_latent, conv_kernel)

        self.key_temp = nn.Parameter(torch.zeros(n_kv_heads_latent))
        self.up_proj = nn.Linear(self.latent_dim_q_max, dim, bias=False)

        self.capture_activations = False
        self._captured_k_lat = []
        self._captured_q_lat = []

    def forward(self, x, active_rank_q: int = None, active_rank_kv: int = None):
        B, T, E = x.shape

        r_q = active_rank_q or self.latent_dim_q_max
        r_kv = active_rank_kv or self.latent_dim_kv_max
        assert r_q % self.head_dim_q == 0, "active_rank_q must be a multiple of head_dim_q"
        assert r_kv % self.head_dim_kv == 0, "active_rank_kv must be a multiple of head_dim_kv"
        assert r_kv % 2 == 0, "active_rank_kv must stay even (v-shift splits it in half)"
        h_q = r_q // self.head_dim_q
        h_kv = r_kv // self.head_dim_kv

        q_pre = F.linear(x, self.down_q.weight[:r_q, :])
        k_pre = F.linear(x, self.down_k.weight[:r_kv, :])

        if self.capture_activations:
            self._captured_q_lat.append(q_pre.detach())
            self._captured_k_lat.append(k_pre.detach())

        if self.use_conv_mix:
            q_post = self.conv_q(q_pre, active_dim=r_q)
            k_post = self.conv_k(k_pre, active_dim=r_kv)
        else:
            q_post, k_post = q_pre, k_pre

        q_pre_h = q_pre.view(B, T, h_q, self.head_dim_q)
        k_pre_h = k_pre.view(B, T, h_kv, self.head_dim_kv)
        q_post_h = q_post.view(B, T, h_q, self.head_dim_q)
        k_post_h = k_post.view(B, T, h_kv, self.head_dim_kv)

        if self.use_qk_mean:
            q, k = qk_mean_couple(q_pre_h, k_pre_h, q_post_h, k_post_h, h_q, h_kv)
        else:
            q, k = q_post_h, k_post_h

        q = l2_norm_rescale(q)
        k = l2_norm_rescale(k) * torch.exp(self.key_temp[:h_kv]).view(1, 1, h_kv, 1)

        r_half = r_kv // 2
        if self.use_v_shift:
            x_prev = F.pad(x[:, :-1, :], (0, 0, 1, 0))
            v_now = F.linear(x, self.down_v_now.weight[:r_half, :])
            v_prev = F.linear(x_prev, self.down_v_prev.weight[:r_half, :])
            v_lat = torch.cat([v_now, v_prev], dim=-1)
        else:
            v_now = F.linear(x, self.down_v_now.weight[:r_half, :])
            v_prev = F.linear(x, self.down_v_prev.weight[:r_half, :])
            v_lat = torch.cat([v_now, v_prev], dim=-1)
        v = v_lat.view(B, T, h_kv, self.head_dim_kv)

        q, k, v = (t.transpose(1, 2) for t in (q, k, v))
        if h_q != h_kv:
            assert h_q % h_kv == 0, "active latent query/kv heads must be GQA-compatible"
            rep = h_q // h_kv
            k = k.repeat_interleave(rep, dim=1)
            v = v.repeat_interleave(rep, dim=1)

        out = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        out = out.transpose(1, 2).reshape(B, T, r_q)
        out = F.linear(out, self.up_proj.weight[:, :r_q])
        return out

    def kv_cache_dim_per_token(self, active_rank_kv: int = None):
        r_kv = active_rank_kv or self.latent_dim_kv_max
        return 2 * r_kv

    def param_count(self):
        return sum(p.numel() for p in self.parameters())
