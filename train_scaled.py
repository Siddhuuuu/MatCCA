"""
train_scaled.py — the real-scale run: real BPE-tokenized data (not char-level), a
config sized for a Kaggle T4 (16GB), full loss-curve tracking, matplotlib plots, and
cleanly formatted printed tables.

Default config here targets ~25-35M non-embedding params (dim=512, 8 layers, 8 heads,
seq_len=512) — an intermediate step up from the toy 4-5M scale, chosen to be safely
within a single Kaggle T4 session before pushing toward the full 50-150M target. If
you hit CUDA out-of-memory, first thing to reduce is BATCH_SIZE, not model size.

Runs single-seed by default (SEEDS = [0]) — get this working and timed at real scale
first, THEN decide if you want to spend the GPU-hours on multi-seed at this scale too
(edit SEEDS to add more once you've seen how long one pass takes).
"""

import os
import time
import json
import statistics
import torch
import matplotlib
matplotlib.use("Agg")  # no display needed, just save PNGs
import matplotlib.pyplot as plt

from tiny_transformer import TinyTransformerLM
from bpe_data import prepare_dataset, get_batch, estimate_loss

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
CORPUS_PATH = "tinystories_train.txt"       # from download_tinystories.py
TOKENIZER_PATH = "tokenizer.json"           # from train_tokenizer.py
CHECKPOINT_DIR = "checkpoints_scaled"
PLOTS_DIR = "plots"
os.makedirs(CHECKPOINT_DIR, exist_ok=True)
os.makedirs(PLOTS_DIR, exist_ok=True)

# ---- scaled-up config, sized for a Kaggle T4 (16GB) ----
DIM = 512
N_LAYERS = 8
N_HEADS = 8
SEQ_LEN = 512
BATCH_SIZE = 32          # reduce this first if you hit OOM, not DIM/N_LAYERS
STEPS = 5000
LR = 3e-4
EVAL_EVERY = 250
EVAL_BATCHES = 30
CHECKPOINT_EVERY = 1000
SEEDS = [0]              # add more seeds here once you've timed one pass


def train_one_variant(
    name, attn_kind, attn_kwargs, train_data, val_data, vocab_size,
    seed=0, sample_rank_fn=None, resume=True,
):
    print(f"\n{'=' * 70}\n{name} (seed={seed})\n{'=' * 70}")
    ckpt_path = os.path.join(CHECKPOINT_DIR, f"{name.replace(' ', '_')}_seed{seed}.pt")

    torch.manual_seed(seed)
    model = TinyTransformerLM(
        vocab_size=vocab_size, dim=DIM, n_layers=N_LAYERS, n_heads=N_HEADS,
        max_seq_len=SEQ_LEN, attn_kind=attn_kind, attn_kwargs=attn_kwargs,
    ).to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=LR)
    start_step = 0
    history = {"step": [], "train_loss": [], "val_loss": []}

    if resume and os.path.exists(ckpt_path):
        ckpt = torch.load(ckpt_path, map_location=DEVICE)
        model.load_state_dict(ckpt["model"])
        opt.load_state_dict(ckpt["optimizer"])
        start_step = ckpt["step"]
        history = ckpt.get("history", history)
        print(f"  resumed from checkpoint at step {start_step}")

    n_params = model.param_count()
    print(f"  params: {n_params:,}  |  device: {DEVICE}  |  seq_len: {SEQ_LEN}  |  batch: {BATCH_SIZE}")
    t0 = time.time()

    for step in range(start_step, STEPS):
        x, y = get_batch(train_data, BATCH_SIZE, SEQ_LEN, DEVICE)
        attn_kwargs_step = sample_rank_fn() if sample_rank_fn else {}
        _, loss = model(x, targets=y, **attn_kwargs_step)
        opt.zero_grad()
        loss.backward()
        opt.step()

        if step % EVAL_EVERY == 0 or step == STEPS - 1:
            val_loss = estimate_loss(model, val_data, BATCH_SIZE, SEQ_LEN, DEVICE, EVAL_BATCHES, **attn_kwargs_step)
            elapsed = time.time() - t0
            history["step"].append(step)
            history["train_loss"].append(loss.item())
            history["val_loss"].append(val_loss)
            print(f"  step {step:5d} | train {loss.item():.4f} | val {val_loss:.4f} | {elapsed:.0f}s elapsed")

        if step % CHECKPOINT_EVERY == 0 and step > start_step:
            torch.save({"model": model.state_dict(), "optimizer": opt.state_dict(),
                        "step": step, "history": history}, ckpt_path)

    torch.save({"model": model.state_dict(), "optimizer": opt.state_dict(),
                "step": STEPS, "history": history}, ckpt_path)
    final_val = estimate_loss(model, val_data, BATCH_SIZE, SEQ_LEN, DEVICE, EVAL_BATCHES)
    total_time = time.time() - t0
    print(f"  FINAL val loss: {final_val:.4f}  |  total time: {total_time / 60:.1f} min")
    return model, final_val, history, n_params


def print_table(rows, headers):
    """Manually formatted table — aligned columns, no extra dependency."""
    widths = [max(len(str(h)), max((len(str(r[i])) for r in rows), default=0)) for i, h in enumerate(headers)]
    line = " | ".join(h.ljust(w) for h, w in zip(headers, widths))
    print(line)
    print("-" * len(line))
    for r in rows:
        print(" | ".join(str(c).ljust(w) for c, w in zip(r, widths)))


def plot_training_curves(all_histories, save_path):
    fig, ax = plt.subplots(figsize=(9, 6))
    colors = {"MHA": "#4C72B0", "GQA": "#DD8452", "CCA fixed-rank": "#55A868", "MatCCA": "#C44E52"}
    for name, hist in all_histories.items():
        color = colors.get(name, None)
        ax.plot(hist["step"], hist["val_loss"], label=f"{name} (val)", color=color, linewidth=2)
    ax.set_xlabel("training step")
    ax.set_ylabel("validation loss")
    ax.set_title("Validation loss during training — real BPE-tokenized data")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"  saved plot: {save_path}")


def plot_final_comparison(summary, save_path):
    names = list(summary.keys())
    means = [summary[n][1] for n in names]
    stds = [summary[n][2] for n in names]
    fig, ax = plt.subplots(figsize=(7, 5))
    bars = ax.bar(names, means, yerr=stds, capsize=6,
                   color=["#4C72B0", "#DD8452", "#55A868", "#C44E52"][:len(names)])
    ax.set_ylabel("final validation loss (lower is better)")
    ax.set_title("Final val loss by architecture, real data")
    for bar, mean in zip(bars, means):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(), f"{mean:.3f}",
                ha="center", va="bottom")
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"  saved plot: {save_path}")


def plot_matcca_curve(rank_results, save_path):
    fracs = sorted(rank_results.keys())
    means = [rank_results[f][1] for f in fracs]
    stds = [rank_results[f][2] for f in fracs]
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.errorbar(fracs, means, yerr=stds, marker="o", capsize=6, linewidth=2, color="#C44E52")
    ax.set_xlabel("active rank fraction (1.0 = uncompressed)")
    ax.set_ylabel("validation loss")
    ax.set_title("MatCCA: quality vs. compression level (one trained model)")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"  saved plot: {save_path}")


def main():
    print(f"Loading real BPE-tokenized data from '{CORPUS_PATH}' using '{TOKENIZER_PATH}'...")
    train_data, val_data, tok = prepare_dataset(CORPUS_PATH, TOKENIZER_PATH)
    train_data, val_data = train_data.to(DEVICE), val_data.to(DEVICE)
    print(f"vocab_size={tok.vocab_size}, train={len(train_data):,} tokens, "
          f"val={len(val_data):,} tokens, device={DEVICE}")

    common = dict(train_data=train_data, val_data=val_data, vocab_size=tok.vocab_size)
    summary, histories, rank_results = {}, {}, {}

    for seed in SEEDS:
        model, val_loss, hist, n_params = train_one_variant("MHA", "mha", {}, seed=seed, **common)
        summary.setdefault("MHA", []).append(val_loss)
        histories["MHA"] = hist

        _, val_loss, hist, _ = train_one_variant("GQA", "gqa", dict(n_kv_heads=2), seed=seed, **common)
        summary.setdefault("GQA", []).append(val_loss)
        histories["GQA"] = hist

        _, val_loss, hist, _ = train_one_variant(
            "CCA fixed-rank", "cca",
            dict(n_heads_latent=8, n_kv_heads_latent=8, compression_q=4, compression_kv=4),
            seed=seed, **common,
        )
        summary.setdefault("CCA fixed-rank", []).append(val_loss)
        histories["CCA fixed-rank"] = hist

        max_r, head_dim = DIM // 4, (DIM // 4) // 8
        rank_fracs = [1.0, 0.5, 0.25]

        def sample_rank_fn():
            import random
            frac = random.choice(rank_fracs)
            r = max(head_dim, int(max_r * frac) - int(max_r * frac) % head_dim)
            return dict(active_rank_q=r, active_rank_kv=r)

        matcca_model, val_loss, hist, _ = train_one_variant(
            "MatCCA", "cca",
            dict(n_heads_latent=8, n_kv_heads_latent=8, compression_q=4, compression_kv=4),
            seed=seed, sample_rank_fn=sample_rank_fn, **common,
        )
        summary.setdefault("MatCCA", []).append(val_loss)
        histories["MatCCA"] = hist

        for frac in rank_fracs:
            r = max(head_dim, int(max_r * frac) - int(max_r * frac) % head_dim)
            rl = estimate_loss(matcca_model, val_data, BATCH_SIZE, SEQ_LEN, DEVICE, EVAL_BATCHES,
                                active_rank_q=r, active_rank_kv=r)
            rank_results.setdefault(frac, []).append(rl)

    # ---- printed summary table ----
    print(f"\n{'=' * 70}\nSUMMARY\n{'=' * 70}")
    rows = []
    stats = {}
    for name, vals in summary.items():
        mean = statistics.mean(vals)
        std = statistics.stdev(vals) if len(vals) > 1 else 0.0
        stats[name] = (vals, mean, std)
        rows.append([name, f"{mean:.4f}", f"{std:.4f}", str(len(vals))])
    print_table(rows, ["variant", "mean val loss", "std", "n seeds"])

    print("\nMatCCA by rank level:")
    rank_stats = {}
    rows2 = []
    for frac, vals in rank_results.items():
        mean = statistics.mean(vals)
        std = statistics.stdev(vals) if len(vals) > 1 else 0.0
        rank_stats[frac] = (vals, mean, std)
        rows2.append([f"{frac:.2f}", f"{mean:.4f}", f"{std:.4f}"])
    print_table(rows2, ["rank fraction", "mean val loss", "std"])

    # ---- plots ----
    print("\nGenerating plots...")
    plot_training_curves(histories, os.path.join(PLOTS_DIR, "training_curves.png"))
    plot_final_comparison(stats, os.path.join(PLOTS_DIR, "final_comparison.png"))
    plot_matcca_curve(rank_stats, os.path.join(PLOTS_DIR, "matcca_rank_curve.png"))

    with open(os.path.join(PLOTS_DIR, "results_summary.json"), "w") as f:
        json.dump({
            "summary": {k: {"values": v[0], "mean": v[1], "std": v[2]} for k, v in stats.items()},
            "matcca_by_rank": {str(k): {"values": v[0], "mean": v[1], "std": v[2]} for k, v in rank_stats.items()},
        }, f, indent=2)
    print(f"\nAll done. Plots in '{PLOTS_DIR}/', raw numbers in '{PLOTS_DIR}/results_summary.json'.")


if __name__ == "__main__":
    main()
