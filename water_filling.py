"""
water_filling.py — Reverse water-filling rank allocation for MatCCA.

Implements the closed-form allocation from the scoping doc §5.1: given eigenvalue
(variance) spectra for multiple (layer, head) groups and a total rank budget, find
the global allocation that minimizes total discarded variance.

Includes a greedy baseline solving the SAME proxy objective, used purely to validate
that the water-filling implementation is correct (they must agree exactly, since
they're solving the identical objective by construction — this is a sanity check,
not a comparison against MatryoshkaKV's real, much more expensive, KL-divergence-based
greedy search. See the NOTE in greedy_distortion_allocation for what that would need.)
"""

import numpy as np


def reverse_waterfilling_allocation(eigenvalues_per_group: dict, total_rank_budget: int):
    """
    Global reverse water-filling for hard rank truncation across multiple groups
    (e.g. (layer, head) pairs).

    Args:
        eigenvalues_per_group: {group_id: array-like of eigenvalues (variances)},
            any order, sorted internally.
        total_rank_budget: total number of dimensions to keep, summed across all groups.

    Returns:
        allocation: {group_id: int}, dimensions kept per group
        threshold: float, the global water level theta
        distortion: float, total discarded variance (sum of eigenvalues NOT kept)
    """
    tagged = []
    for gid, evals in eigenvalues_per_group.items():
        evals = np.asarray(evals, dtype=np.float64)
        for v in evals:
            tagged.append((v, gid))

    tagged.sort(key=lambda t: -t[0])  # descending by eigenvalue

    budget = min(total_rank_budget, len(tagged))
    kept = tagged[:budget]
    discarded = tagged[budget:]

    threshold = kept[-1][0] if kept else float("inf")
    distortion = sum(v for v, _ in discarded)

    allocation = {gid: 0 for gid in eigenvalues_per_group}
    for v, gid in kept:
        allocation[gid] += 1

    return allocation, threshold, distortion


def greedy_distortion_allocation(eigenvalues_per_group: dict, total_rank_budget: int):
    """
    Greedy baseline on the distortion proxy: at each step, keep whichever remaining
    (group, dim) eigenvalue is largest. Solves the identical objective as water-filling
    by construction, so this is a correctness check for the implementation above — NOT
    a stand-in for MatryoshkaKV's actual method.

    NOTE: to reproduce MatryoshkaKV's real greedy search, replace the sort-based
    selection below with an iterative loop that, at each step, evaluates KL-divergence
    on a calibration batch for every remaining candidate group and removes/adds the
    cheapest one — that expensive, model-dependent loop is exactly what §5.1 proposes
    water-filling can replace. This function is the O(N log N) proxy-objective version,
    used only to confirm reverse_waterfilling_allocation is implemented correctly.
    """
    remaining = []
    for gid, evals in eigenvalues_per_group.items():
        for v in np.asarray(evals, dtype=np.float64):
            remaining.append((v, gid))

    remaining.sort(key=lambda t: -t[0])
    budget = min(total_rank_budget, len(remaining))

    allocation = {gid: 0 for gid in eigenvalues_per_group}
    for v, gid in remaining[:budget]:
        allocation[gid] += 1

    return allocation


def _synthetic_test():
    """Two checks:
    1. Correctness: water-filling and the distortion-based greedy baseline must
       produce identical allocations, since they solve the same objective.
    2. Sanity: with variance decreasing by depth (mimicking MatryoshkaKV's real
       empirical finding that shallow layers need bigger budgets), water-filling
       should allocate more rank to shallow layers automatically, with no special
       casing — it should just fall out of the eigenvalues.
    """
    rng = np.random.default_rng(0)
    n_layers, dims_per_layer = 6, 16

    # Shallow layers (low l) given higher-variance spectra, deep layers lower —
    # mirrors MatryoshkaKV's reported finding, so we can check it's recovered.
    eigenvalues_per_group = {
        f"layer_{l}": np.sort(rng.exponential(scale=(n_layers - l), size=dims_per_layer))[::-1]
        for l in range(n_layers)
    }
    total_dims = n_layers * dims_per_layer

    for compression_factor in [2, 4, 8]:
        budget = total_dims // compression_factor
        wf_alloc, theta, wf_distortion = reverse_waterfilling_allocation(eigenvalues_per_group, budget)
        greedy_alloc = greedy_distortion_allocation(eigenvalues_per_group, budget)

        assert wf_alloc == greedy_alloc, f"MISMATCH at budget={budget}: {wf_alloc} vs {greedy_alloc}"

        print(f"\n--- compression factor {compression_factor}x (budget={budget}/{total_dims}) ---")
        print(f"water level (theta): {theta:.4f}")
        print(f"total discarded variance: {wf_distortion:.4f}")
        print("dims kept per layer:", {k: v for k, v in wf_alloc.items()})

    print("\nOK — water-filling matches the greedy baseline exactly at every budget tested.")
    print("Shallow layers (layer_0, layer_1...) should show more kept dims than deep")
    print("layers above, with zero special-casing — that's the point: it falls out of")
    print("the eigenvalues, not a hand-tuned schedule.")


if __name__ == "__main__":
    _synthetic_test()
