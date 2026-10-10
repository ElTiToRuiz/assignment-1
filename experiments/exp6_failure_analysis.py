"""Experiment 6: why Monte Carlo and Double Q-learning fail in Cliff Walking, and how to fix them.

Every step costs -1, so with gamma = 0.99 a policy that never reaches the goal is worth
-1 / (1 - gamma) = -100. The first episodes are very long and full of falls. If the estimates sink
to that -100 level before the goal has been found, every action looks equally bad, the agent wanders
in circles and the news of the goal never travels back.

  - Monte Carlo with the class step 1/N keeps an average of every return it has ever seen, so the
    awful returns of the first episodes never leave the average. A constant alpha forgets them.
  - Double Q-learning with alpha = 0.5 collapses in a few seeds, because each table only gets half
    of the updates. A smaller alpha avoids it.
"""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from tabular_rl.envs import make_spec
from tabular_rl.training import train
from tabular_rl.plotting import COLORS, save, snap_x

from .common import CONFIG, N_JOBS, SEEDS, save_table

ENV = "cliff_walking"
BASE = CONFIG[ENV]  # 1000 episodes, alpha 0.5, constant eps 0.1
VARIANTS = [  # (label, algorithm, overrides, colour)
    ("Monte Carlo (1/N, class)", "Monte Carlo", {}, COLORS["Monte Carlo"]),
    ("MC constant α=0.1", "MC constant-α", dict(alpha=0.1), COLORS["MC constant-α"]),
    ("MC constant α=0.1 + exploring starts", "MC Exploring Starts", dict(alpha=0.1), COLORS["MC Exploring Starts"]),
    ("Double Q α=0.5", "Double Q-learning", {}, COLORS["Double Q-learning"]),
    ("Double Q α=0.1", "Double Q-learning", dict(alpha=0.1), "#8c510a"),
    ("SARSA α=0.5 (reference)", "SARSA", {}, COLORS["SARSA"]),
    ("Q-learning α=0.5 (reference)", "Q-learning", {}, COLORS["Q-learning"]),
]


def stuck(o):
    """Seeds whose last 100 episodes mostly hit the 500-step limit (lost in loops)."""
    return o["lengths"][:, -100:].mean(1) > 0.5 * make_spec(ENV).max_steps


def main():
    print("exp6: failure analysis in Cliff Walking")
    spec = make_spec(ENV)
    outs = {label: train(algo, ENV, SEEDS, n_jobs=N_JOBS, **{**BASE, **kw}) for label, algo, kw, _ in VARIANTS}
    plateau = -1 / (1 - spec.gamma)

    fig, ax = plt.subplots(1, 2, figsize=(17, 5.6), layout="constrained")
    rng = np.random.default_rng(0)
    rows = []
    for k, (label, _, _, color) in enumerate(VARIANTS):
        o = outs[label]
        ret = o["returns"][:, -100:].mean(1)
        bad = stuck(o)
        x = k + rng.uniform(-0.18, 0.18, len(ret))
        ax[0].scatter(x[~bad], -ret[~bad], color=color, s=28, alpha=.85)
        ax[0].scatter(x[bad], -ret[bad], color=color, s=60, marker="X", edgecolor="black", lw=.5)
        ax[0].hlines(-np.median(ret), k - .3, k + .3, color="black", lw=2)
        rows.append({"variant": label, "stuck seeds": f"{int(bad.sum())}/{len(bad)}",
                     "return (last 100 ep) median": round(float(np.median(ret)), 1),
                     "return mean": round(float(ret.mean()), 1),
                     "greedy regret (median seed)": round(abs(float(np.median(o["regret"][:, -1]))), 2),
                     "seeds with optimal greedy policy": f"{int((o['regret'][:, -1] < 1e-6).sum())}/{len(bad)}"})
    ax[0].set_yscale("log")
    ticks = ["Monte Carlo\n$1/N$ (class)", "MC\nconstant $\\alpha=0.1$", "MC $\\alpha=0.1$\n+ exploring starts",
             "Double Q\n$\\alpha=0.5$", "Double Q\n$\\alpha=0.1$", "SARSA $\\alpha=0.5$\n(reference)",
             "Q-learning $\\alpha=0.5$\n(reference)"]
    ax[0].set_xticks(range(len(VARIANTS)), ticks, fontsize=9)
    ax[0].set(ylabel="cost = −return (log, lower is better)",
              title="Per-seed online return, last 100 episodes  (✕ = stuck in loops, bar = median)")
    ax[0].grid(axis="x", visible=False)

    o = outs["Double Q α=0.5"]
    bad = stuck(o)
    for q, b in zip(o["q_start_opt"], bad):
        ax[1].plot(snap_x(o), q, color="#b2182b" if b else COLORS["Double Q-learning"], lw=1.6 if b else .8,
                   alpha=.95 if b else .5)
    ax[1].axhline(plateau, ls="--", color="black", lw=1)
    ax[1].text(snap_x(o)[-1], plateau + 3, r"$-1/(1-\gamma) = -100$: value of never reaching the goal", ha="right", fontsize=9)
    ax[1].axhline(o["Q_star"][spec.start_state].max(), ls=":", color="green", lw=1.2)
    ax[1].text(snap_x(o)[-1], o["Q_star"][spec.start_state].max() + 3, r"$Q^*(s_0, a^*)$", ha="right", fontsize=9, color="green")
    ax[1].set(xlabel="episode", ylabel=r"$Q(s_0, a^*)$", ylim=(-130, 0),
              title=rf"Double Q $\alpha=0.5$: $Q$ at the start state per seed (red = the {int(bad.sum())} stuck seeds)")
    fig.suptitle(r"Cliff Walking failure analysis: estimates collapse to the $-100$ 'never arrive' plateau")
    save(fig, "exp6_failure_analysis.png")
    save_table(pd.DataFrame(rows), "exp6_failure_analysis")


if __name__ == "__main__":
    main()
