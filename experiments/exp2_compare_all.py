"""Experiment 2: every algorithm on every environment, with the same settings, so they can be compared.

It also saves the final gridworld Q-tables in results/models/ (the "trained models" the demo plays).
"""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from tabular_rl.models import save_model
from tabular_rl.plotting import COLORS, ENV_NAMES, save

from .common import ALGOS, ENVS, save_table, summary_row, train_default


def bars_figure(all_outs):
    """Where every algorithm ends after training, all environments side by side."""
    metrics = [("match", "Optimal greedy actions (%)", 100, False),
               ("regret", r"Regret of the greedy policy at $s_0$", 1, True),
               ("rmse_opt", "RMSE of Q on optimal actions", 1, True)]
    fig, axes = plt.subplots(1, 3, figsize=(18, 5.2), layout="constrained")
    w = 0.8 / len(ALGOS)
    for ax, (key, title, scale, log) in zip(axes, metrics):
        for k, a in enumerate(ALGOS):
            vals = np.array([np.abs(all_outs[e][a][key][:, -1]) * scale for e in ENVS])
            m, sd = vals.mean(1), vals.std(1)
            xs = np.arange(len(ENVS)) + (k - (len(ALGOS) - 1) / 2) * w
            ax.bar(xs, np.where(m < 1e-6, np.nan, m) if log else m, w, yerr=None if log else sd,
                   color=COLORS[a], label=a, capsize=2)
            if log:  # a zero cannot be drawn on a log axis: mark it explicitly
                for xx, mm in zip(xs, m):
                    if mm < 1e-6:
                        ax.text(xx, 1.3e-4, "0", ha="center", va="bottom", fontsize=9, weight="bold", color=COLORS[a])
        ax.set_xticks(range(len(ENVS)), [ENV_NAMES[e] for e in ENVS], fontsize=9)
        ax.set_title(title + (" (log, lower is better)" if log else " (higher is better)"))
        if log:
            ax.set_yscale("log")
            ax.set_ylim(bottom=1e-4)
        ax.grid(axis="x", visible=False)
    fig.legend(*axes[0].get_legend_handles_labels(), loc="outside lower center", ncol=len(ALGOS), fontsize=10)
    fig.suptitle("Final performance after training (mean over 20 seeds)")
    save(fig, "exp2_summary_bars.png")


def main():
    print("exp2: all algorithms x all environments")
    rows, all_outs = [], {}
    for env_name in ENVS:
        outs = {a: train_default(a, env_name) for a in ALGOS}
        all_outs[env_name] = outs
        for a, o in outs.items():
            if env_name.startswith("gridworld"):  # the demo plays the class gridworld only
                save_model(env_name, a, o["Q"].mean(0))
            rows.append({"env": ENV_NAMES[env_name], "algorithm": a, **summary_row(o, env_name)})
    bars_figure(all_outs)
    save_table(pd.DataFrame(rows), "exp2_all_algorithms")


if __name__ == "__main__":
    main()
