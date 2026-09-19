"""
run_multiseed.py — the missing rigor: repeats each attention variant across multiple
seeds and reports mean +/- std, instead of trusting single runs. Everything so far
(CCA beating MHA, MatCCA's degradation curve) has been n=1 per condition — this is
what actually tells you whether those findings are real or noise.

Reuses train_one_variant from train_real.py unchanged; just loops seeds around it.
Checkpoints are seed-aware (train_real.py's train_one_variant now names them
{variant}_seed{N}.pt), so runs across seeds don't collide with each other.
"""

import random
import statistics
import torch

import train_real as tr
from real_data import prepare_dataset, estimate_loss

SEEDS = [0, 1, 2]
STEPS = 2000  # keep matched to your single-seed run for comparability
RANK_FRACS = [1.0, 0.5, 0.25]


def run_all_seeds(name, attn_kind, attn_kwargs, common, sample_rank_fn=None, **kwargs):
    results = []
    for seed in SEEDS:
        _, val_loss = tr.train_one_variant(
            name, attn_kind, attn_kwargs, seed=seed, steps=STEPS,
            sample_rank_fn=sample_rank_fn, resume=True, **common, **kwargs,
        )
        results.append(val_loss)
    mean = statistics.mean(results)
    std = statistics.stdev(results) if len(results) > 1 else 0.0
    print(f"\n>>> {name}: {results} -> mean={mean:.4f}, std={std:.4f}")
    return results, mean, std


def run_matcca_all_seeds(name, attn_kwargs, common, dim=256, compression=4, n_heads=8):
    """MatCCA needs its own runner: after each seed trains, evaluate at every rank
    level (not just the training-time loss), then aggregate mean+/-std PER LEVEL
    across seeds — this is what actually tells you whether the weird non-monotonic
    curve from the single-seed run was real or noise."""
    max_r = dim // compression
    head_dim = max_r // n_heads

    def sample_rank_fn():
        frac = random.choice(RANK_FRACS)
        r = max(head_dim, int(max_r * frac) - int(max_r * frac) % head_dim)
        return dict(active_rank_q=r, active_rank_kv=r)

    per_level_results = {frac: [] for frac in RANK_FRACS}
    for seed in SEEDS:
        random.seed(seed)  # sample_rank_fn uses the global random module, not torch
        model, _ = tr.train_one_variant(
            name, "cca", attn_kwargs, seed=seed, steps=STEPS, dim=dim, n_heads=n_heads,
            sample_rank_fn=sample_rank_fn, resume=True, **common,
        )
        for frac in RANK_FRACS:
            r = max(head_dim, int(max_r * frac) - int(max_r * frac) % head_dim)
            val_loss = estimate_loss(
                model, common["val_data"], 32, 256, tr.DEVICE, 20,
                active_rank_q=r, active_rank_kv=r,
            )
            per_level_results[frac].append(val_loss)
            print(f"    seed={seed}, rank {frac:.2f} (r={r}/{max_r}): val loss={val_loss:.4f}")

    summary = {}
    print(f"\n>>> {name}, per rank level across {len(SEEDS)} seeds:")
    for frac, results in per_level_results.items():
        mean = statistics.mean(results)
        std = statistics.stdev(results) if len(results) > 1 else 0.0
        summary[frac] = (results, mean, std)
        print(f"    rank {frac:.2f}: {results} -> mean={mean:.4f}, std={std:.4f}")
    return summary


def main():
    print(f"Loading real text data from '{tr.DATA_PATH}'...")
    train_data, val_data, tok = prepare_dataset(tr.DATA_PATH)
    train_data, val_data = train_data.to(tr.DEVICE), val_data.to(tr.DEVICE)
    common = dict(train_data=train_data, val_data=val_data, vocab_size=tok.vocab_size)

    summary = {}

    summary["MHA"] = run_all_seeds("MHA baseline", "mha", {}, common)
    summary["GQA"] = run_all_seeds("GQA baseline", "gqa", dict(n_kv_heads=2), common)
    summary["CCA fixed-rank"] = run_all_seeds(
        "CCA fixed-rank (compression=4)", "cca",
        dict(n_heads_latent=8, n_kv_heads_latent=8, compression_q=4, compression_kv=4),
        common,
    )

    matcca_summary = run_matcca_all_seeds(
        "MatCCA (nested Matryoshka rank)",
        dict(n_heads_latent=8, n_kv_heads_latent=8, compression_q=4, compression_kv=4),
        common,
    )

    print(f"\n{'=' * 60}\nSUMMARY across {len(SEEDS)} seeds\n{'=' * 60}")
    for name, (results, mean, std) in summary.items():
        print(f"  {name:20s}: {mean:.4f} +/- {std:.4f}   (runs: {[f'{r:.4f}' for r in results]})")
    print(f"\n  MatCCA, by rank level (compare rank=1.00 row against CCA fixed-rank above —")
    print(f"  that's the direct 'does nesting cost you anything at matched compression' check):")
    for frac, (results, mean, std) in matcca_summary.items():
        print(f"    rank {frac:.2f}: {mean:.4f} +/- {std:.4f}   (runs: {[f'{r:.4f}' for r in results]})")

    print("\nRead this as: if a variant's mean +/- std range overlaps with another's,")
    print("you cannot yet claim one beats the other — that's the actual bar for")
    print("'this finding is real' rather than 'this run got lucky'.")


if __name__ == "__main__":
    main()
