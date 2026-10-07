"""Two small statistics tools to tell real differences from luck, with only 20 seeds per configuration."""
import numpy as np


def bootstrap_ci(x, n_boot=10_000, level=0.95, seed=0):
    """95% confidence interval for the mean of `x`.

    Idea: pretend our 20 results are the whole world, draw 20 of them again (with repetition)
    many times, and see how much the mean moves. The middle 95% of those means is the interval.
    """
    x = np.asarray(x, dtype=float)
    rng = np.random.default_rng(seed)
    means = x[rng.integers(0, len(x), (n_boot, len(x)))].mean(1)
    lo, hi = np.percentile(means, [(1 - level) / 2 * 100, (1 + level) / 2 * 100])
    return float(lo), float(hi)


def permutation_test(a, b, n_perm=10_000, seed=0):
    """p-value for "a and b have the same mean".

    If the two configurations were equally good, the labels "a" and "b" would not matter. So we
    shuffle the labels many times and count how often a difference at least as big as the real one
    appears just by chance. Rare means the difference is real.
    """
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    rng = np.random.default_rng(seed)
    pooled = np.concatenate([a, b])
    observed = abs(a.mean() - b.mean())
    count = 0
    for _ in range(n_perm):
        rng.shuffle(pooled)
        count += abs(pooled[:len(a)].mean() - pooled[len(a):].mean()) >= observed - 1e-15
    return (count + 1) / (n_perm + 1)
