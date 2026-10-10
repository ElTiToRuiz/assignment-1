"""Experiment 3: on-policy vs off-policy, the classic Cliff Walking example.

SARSA learns a safe path away from the edge, because it knows it will sometimes take a random step.
Q-learning learns the shortest path along the edge, because it learns about the greedy policy. So
while exploring, Q-learning falls more often, even though what it learned is the true optimum.
"""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from tabular_rl.envs import greedy_path, make_spec
from tabular_rl.metrics import moving_average
from tabular_rl.plotting import COLORS, plot_band, plot_value_policy, save, snap_x

from .common import CONFIG, save_table, train_default


def main():
    print("exp3: cliff walking, SARSA vs Q-learning")
    spec = make_spec("cliff_walking")
    outs = {a: train_default(a, "cliff_walking") for a in ["SARSA", "Q-learning"]}
    fig = plt.figure(figsize=(15, 8.5), layout="constrained")
    gs = fig.add_gridspec(2, 2, height_ratios=[1, 1.1])
    rows = []
    for j, (a, o) in enumerate(outs.items()):
        Q = o["Q"].mean(0)
        path = greedy_path(spec, Q)
        plot_value_policy(fig.add_subplot(gs[0, j]), spec, Q, f"{a}: greedy path ({len(path) - 1} steps)", path=path)
        rows.append({"algorithm": a, "greedy path length": len(path) - 1,
                     "online return (last 100 ep)": round(float(o["returns"][:, -100:].mean()), 2),
                     "greedy-policy regret (median seed)": round(abs(float(np.median(o["regret"][:, -1]))), 3),
                     "seeds with optimal greedy policy": f"{int((o['regret'][:, -1] < 1e-6).sum())}/{len(o['regret'])}"})
    ax1, ax2 = fig.add_subplot(gs[1, 0]), fig.add_subplot(gs[1, 1])
    for a, o in outs.items():
        plot_band(ax1, moving_average(o["returns"], 20), a, COLORS[a], robust=True)
        plot_band(ax2, o["regret"], a, COLORS[a], x=snap_x(o), robust=True, floor=1e-3)
    ax1.set(ylim=(-120, 0), xlabel="episode", ylabel="return",
            title=rf"Online return while exploring ($\varepsilon={CONFIG['cliff_walking']['eps'][0]}$)" + "\nSARSA is better: it avoids the edge")
    ax2.set(yscale="log", xlabel="episode", ylabel=r"regret (0 drawn at $10^{-3}$)",
            title=r"Regret of the greedy ($\varepsilon=0$) policy" + "\nQ-learning is better: it learns the optimal path")
    ax1.legend(); ax2.legend()
    fig.suptitle("Cliff Walking: on-policy (SARSA) vs off-policy (Q-learning)")
    save(fig, "exp3_cliff_walking.png")
    save_table(pd.DataFrame(rows), "exp3_cliff")


if __name__ == "__main__":
    main()
