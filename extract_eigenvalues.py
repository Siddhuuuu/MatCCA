"""
extract_eigenvalues.py — the missing link: train a real CCA model, hook its ACTUAL
down-projected K-latent activations during a forward pass, compute the REAL empirical
covariance/eigenvalue spectrum per layer, and feed that into reverse_waterfilling_allocation().

No synthetic np.random eigenvalues anywhere in this file — everything downstream of
training comes from measuring what the model actually does on real forward passes.

Honest limitation: the underlying task is still the toy periodic pattern from
test_training.py, not real language data — this sandbox has no route to Hugging
Face/Common Crawl to pull real text in. Nothing else changes when you swap in real
data on Kaggle: same hooks, same covariance computation, same allocator call.
"""

import torch
from tiny_transformer import TinyTransformerLM
from test_training import train_loop, make_periodic_batch
from water_filling import reverse_waterfilling_allocation, greedy_distortion_allocation

torch.manual_seed(0)


def extract_eigenvalues_per_layer(model, calib_batches, seq_len, vocab_size, source="k"):
    """Enables activation capture on each CCA layer, runs calibration batches through
    the model in eval/no_grad mode, and returns the REAL empirical covariance
    eigenvalues per layer — i.e. exactly what MatryoshkaKV's PCA-init step computes,
    just applied to CCA's down-projected K latents instead.

    NOTE: this uses the model's internal capture_activations flag, not
    nn.Module.register_forward_hook — see the comment in attention.py's
    CompressedConvolutionalAttention.__init__ for why standard hooks don't work here."""
    ccas = [block.attn for block in model.blocks]
    for cca in ccas:
        cca.capture_activations = True
        cca._captured_k_lat.clear()
        cca._captured_q_lat.clear()

    model.eval()
    with torch.no_grad():
        for _ in range(calib_batches):
            x, _ = make_periodic_batch(64, seq_len, vocab_size)
            model(x)
    model.train()

    eigenvalues_per_layer = {}
    for i, cca in enumerate(ccas):
        captured = cca._captured_k_lat if source == "k" else cca._captured_q_lat
        A = torch.cat(captured, dim=0).reshape(-1, captured[0].shape[-1])  # (N_tokens, latent_dim)
        A = A - A.mean(dim=0, keepdim=True)
        cov = (A.T @ A) / A.shape[0]
        eigvals = torch.linalg.eigvalsh(cov)            # ascending, real (cov is symmetric PSD)
        eigvals = eigvals.flip(0).clamp(min=0).numpy()  # descending, clip tiny negative numerical noise
        eigenvalues_per_layer[f"layer_{i}"] = eigvals
        cca.capture_activations = False

    return eigenvalues_per_layer


def main():
    vocab_size, dim, n_layers, n_heads, seq_len = 64, 64, 6, 4, 32

    print("Step 1: train a real CCA model on the toy task (need trained weights before")
    print("the activations mean anything)...")
    model = TinyTransformerLM(
        vocab_size=vocab_size, dim=dim, n_layers=n_layers, n_heads=n_heads,
        max_seq_len=seq_len, attn_kind="cca",
        attn_kwargs=dict(n_heads_latent=n_heads, n_kv_heads_latent=n_heads,
                          compression_q=4, compression_kv=4),
    )
    losses = train_loop(model, 400, seq_len, vocab_size, lr=1e-2, log_every=100)
    print(f"  trained: loss {losses[0]:.3f} -> {min(losses[-10:]):.3f}\n")

    print("Step 2: hook the REAL down_k activations, run 20 calibration batches,")
    print("compute the REAL empirical covariance eigenvalues per layer (no np.random")
    print("anywhere below this line)...")
    eigenvalues_per_layer = extract_eigenvalues_per_layer(
        model, calib_batches=20, seq_len=seq_len, vocab_size=vocab_size,
    )
    for k, v in eigenvalues_per_layer.items():
        print(f"  {k}: {v.round(4)}")

    total_dims = sum(len(v) for v in eigenvalues_per_layer.values())
    print(f"\nStep 3: run the water-filling allocator (§5.1) on these REAL eigenvalues,")
    print(f"total dims available across all {n_layers} layers: {total_dims}\n")

    for compression_factor in [2, 4, 8]:
        budget = total_dims // compression_factor
        wf_alloc, theta, distortion = reverse_waterfilling_allocation(eigenvalues_per_layer, budget)
        greedy_alloc = greedy_distortion_allocation(eigenvalues_per_layer, budget)
        agree = wf_alloc == greedy_alloc
        print(f"--- compression {compression_factor}x (budget={budget}/{total_dims}) ---")
        print(f"  water level (theta): {theta:.4f}, discarded variance: {distortion:.4f}")
        print(f"  allocation: {wf_alloc}")
        print(f"  matches greedy baseline on REAL data: {agree}")

    print("\nDone — this is the actual §5.1 mechanism running on a real trained model's")
    print("real activations, not a synthetic placeholder. Compare the per-layer pattern")
    print("above against MatryoshkaKV's reported 'shallow layers need more budget'")
    print("finding — on THIS toy task, that pattern may or may not hold; that's a real")
    print("empirical question now, not an assumption.")


if __name__ == "__main__":
    main()
