"""One look for every figure, plus the two plots we draw everywhere: a value/policy grid and a
learning curve over seeds. Figures are saved to results/figures/."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from .envs import ARROWS
from .paths import FIGURES

# Okabe-Ito palette: stays readable for colour-blind people and in black and white
COLORS = {
    "Monte Carlo": "#7a7a7a",
    "MC constant-α": "#CC79A7",
    "MC Exploring Starts": "#000000",
    "SARSA": "#0072B2",
    "n-step SARSA": "#56B4E9",
    "Expected SARSA": "#009E73",
    "Q-learning": "#D55E00",
    "Double Q-learning": "#E69F00",
}
ENV_NAMES = {
    "gridworld_deterministic": "Gridworld (deterministic)",
    "gridworld_slippery": "Gridworld (slippery 80/10/10)",
    "cliff_walking": "Cliff Walking",
}

plt.rcParams.update({
    "figure.dpi": 110,
    "savefig.dpi": 160,
    "font.size": 11,
    "axes.titlesize": 12,
    "axes.titleweight": "bold",
    "axes.labelsize": 11,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.alpha": 0.3,
    "legend.fontsize": 9,
    "legend.frameon": False,
    "figure.titlesize": 14,
    "figure.titleweight": "bold",
})


def save(fig, name):
    FIGURES.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGURES / name, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  [figure] results/figures/{name}")


def plot_value_policy(ax, spec, Q, title="", path=None):
    """The grid coloured by V(s) = max_a Q(s,a), with the greedy action(s) as arrows.
    When several actions are equally good, all of them are drawn."""
    nr, nc = spec.shape
    V = Q.max(axis=1)
    grid = np.full((nr, nc), np.nan)
    for s in spec.valid_states:
        grid[s // nc, s % nc] = V[s]
    lo, hi = np.nanmin(grid), np.nanmax(grid)
    ax.imshow(grid, cmap="viridis", vmin=lo, vmax=hi)
    small = nc > 6
    for s in range(spec.n_states):
        r, c = divmod(s, nc)
        if s in spec.walls:
            ax.add_patch(plt.Rectangle((c - .5, r - .5), 1, 1, color="#222222"))
        elif s in spec.cliffs:
            ax.add_patch(plt.Rectangle((c - .5, r - .5), 1, 1, color="#f4a6a0"))
            ax.text(c, r, "cliff", ha="center", va="center", fontsize=6, color="#7a1f1a")
        elif s in spec.terminals:
            goal = spec.terminals[s] > 0 or spec.name == "cliff_walking"
            ax.add_patch(plt.Rectangle((c - .5, r - .5), 1, 1, color="#ffd54f" if goal else "#e57373"))
            ax.text(c, r, "GOAL" if goal else "PIT", ha="center", va="center", weight="bold",
                    fontsize=7 if small else 10)
        else:
            light = (V[s] - lo) / (hi - lo + 1e-12) > 0.6
            col = "black" if light else "white"
            ax.text(c, r - .25, f"{V[s]:.2f}", ha="center", va="center", fontsize=6 if small else 10, color=col)
            best = np.flatnonzero(Q[s] >= Q[s].max() - 1e-6 * max(1.0, abs(Q[s].max())))  # show ties
            ax.text(c, r + .15, "".join(ARROWS[int(a)] for a in best), ha="center", va="center",
                    fontsize=(11 if small else 18) - 3 * (len(best) > 1), color=col, weight="bold")
    if path:
        ax.plot([p % nc for p in path], [p // nc for p in path], color="white", ls="--", lw=2, alpha=.9)
    sr, sc = divmod(spec.start_state, nc)
    ax.text(sc - .42, sr - .32, "S", color="white", fontsize=8, weight="bold",
            bbox=dict(boxstyle="round,pad=0.1", fc="black", ec="none", alpha=.6))
    ax.set_title(title, fontsize=10)
    ax.set_xticks([]); ax.set_yticks([]); ax.grid(False)
    for sp in ax.spines.values():
        sp.set_visible(True)


def plot_band(ax, y, label=None, color=None, x=None, robust=False, floor=None):
    """A curve summarising many seeds. y has one row per seed.

    By default: mean with a +-1 std band. robust=True: median with the 25-75% band instead, so a
    couple of runaway seeds do not hide what the typical run does. `floor` stops the band from going
    below a value that makes no sense (regret can't be negative).
    """
    y = np.asarray(y, dtype=float)
    if robust:
        m, lo, hi = np.median(y, 0), np.percentile(y, 25, 0), np.percentile(y, 75, 0)
    else:
        m = y.mean(0)
        lo, hi = m - y.std(0), m + y.std(0)
    if floor is not None:
        m, lo, hi = np.maximum(m, floor), np.maximum(lo, floor), np.maximum(hi, floor)
    x = np.arange(len(m)) if x is None else x
    ax.plot(x, m, label=label, color=color, lw=1.8)
    ax.fill_between(x, lo, hi, color=color, alpha=.15, lw=0)


def snap_x(out, log_every=10):
    """The episode number of every Q snapshot (one every `log_every` episodes)."""
    return np.arange(1, out["regret"].shape[1] + 1) * log_every


def smooth(y, w=10):
    """Moving average that keeps the same length (the window is just shorter at the start)."""
    y = np.asarray(y, dtype=float)
    c = np.cumsum(y, axis=-1)
    out = c.copy()
    out[..., w:] = c[..., w:] - c[..., :-w]
    n = np.minimum(np.arange(1, y.shape[-1] + 1), w)
    return out / n
