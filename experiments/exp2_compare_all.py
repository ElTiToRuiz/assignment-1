"""Experiment 2: every algorithm on every environment, with the same settings, so they can be compared.

It also saves the final Q-tables in results/models/ (the "trained models" the demo plays).
"""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from tabular_rl.envs import make_spec
from tabular_rl.metrics import moving_average
from tabular_rl.models import save_model
from tabular_rl.plotting import COLORS, ENV_NAMES, plot_band, plot_value_policy, save, snap_x

from .common import ALGOS, ENVS, save_table, summary_row, train_default


def curves_figure(env_name, outs):
    """Learning curves of every algorithm on one environment."""
    fig, ax = plt.subplots(1, 3, figsize=(17, 4.6), layout="constrained")
    cliff = env_name == "cliff_walking"
    for a, o in outs.items():
        plot_band(ax[0], moving_average(o["returns"], 50), a, COLORS[a], robust=True)
        plot_band(ax[1], o["rmse_opt"], a, COLORS[a], x=snap_x(o))
        plot_band(ax[2], o["regret"], a, COLORS[a], x=snap_x(o), robust=True, floor=1e-3 if cliff else 0)
    ax[0].set(title="Training return (moving avg 50, median + IQR)", ylabel="return")
    ax[1].set(title="RMSE of Q on optimal actions (mean ± std)", yscale="log")
    ax[2].set(title="Regret of greedy policy at s0 (median + IQR)")
    if cliff:
        ax[0].set_ylim(-200, 0)
        ax[2].set(yscale="log", ylabel="regret (0 drawn at 1e-3)")
    for a_ in ax:
        a_.set_xlabel("episode")
    ax[0].legend(ncol=2)
    fig.suptitle(f"{ENV_NAMES[env_name]}: all algorithms (20 seeds)")
    save(fig, f"exp2_curves_{env_name}.png")


def policies_figure(env_name, outs):
    """What each algorithm ends up believing: its values and greedy policy, next to the optimum."""
    spec = make_spec(env_name)
    cliff = env_name == "cliff_walking"
    fig, axes = plt.subplots(*(4, 2) if cliff else (2, 4), figsize=(14, 9) if cliff else (16, 6.5), layout="constrained")
    axes = axes.ravel()
    plot_value_policy(axes[0], spec, next(iter(outs.values()))["Q_star"], "Optimal Q* (value iteration)")
    for ax, (a, o) in zip(axes[1:], outs.items()):
        plot_value_policy(ax, spec, o["Q"].mean(0), a)
    axes[-1].axis("off")
    fig.suptitle(f"{ENV_NAMES[env_name]}: learned V(s)=max Q and greedy policy (mean Q over seeds)")
    save(fig, f"exp2_policies_{env_name}.png")


def bars_figure(all_outs):
    """Where every algorithm ends after training, all environments side by side."""
    metrics = [("match", "Optimal greedy actions (%)", 100, False),
               ("regret", "Regret of greedy policy at s0", 1, True),
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
            save_model(env_name, a, o["Q"].mean(0))
            rows.append({"env": ENV_NAMES[env_name], "algorithm": a, **summary_row(o, env_name)})
        curves_figure(env_name, outs)
        policies_figure(env_name, outs)
    bars_figure(all_outs)
    save_table(pd.DataFrame(rows), "exp2_all_algorithms")


if __name__ == "__main__":
    main()
