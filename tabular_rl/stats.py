"""Small statistics helpers (numpy only): bootstrap confidence intervals and a permutation test."""
import numpy as np


def bootstrap_ci(x, n_boot=10_000, level=0.95, seed=0):
    """Percentile bootstrap confidence interval for the mean of `x`."""
    x = np.asarray(x, dtype=float)
    rng = np.random.default_rng(seed)
    means = x[rng.integers(0, len(x), (n_boot, len(x)))].mean(1)
    lo, hi = np.percentile(means, [(1 - level) / 2 * 100, (1 + level) / 2 * 100])
    return float(lo), float(hi)


def permutation_test(a, b, n_perm=10_000, seed=0):
    """Two-sided p-value for H0: mean(a) == mean(b) (labels exchangeable between the two groups)."""
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    rng = np.random.default_rng(seed)
    pooled = np.concatenate([a, b])
    observed = abs(a.mean() - b.mean())
    count = 0
    for _ in range(n_perm):
        rng.shuffle(pooled)
        count += abs(pooled[:len(a)].mean() - pooled[len(a):].mean()) >= observed - 1e-15
    return (count + 1) / (n_perm + 1)
