"""Experiment 1, the minimum requirement: SARSA and Q-learning on the class gridworld,
deterministic and slippery.

Three figures:
  - the values and policies they learn, next to the true optimum;
  - their learning curves;
  - what happens in the slippery world when we follow the theory's convergence conditions.
"""
import matplotlib.pyplot as plt
import pandas as pd

from tabular_rl.envs import make_spec
from tabular_rl.metrics import moving_average
from tabular_rl.plotting import COLORS, ENV_NAMES, plot_band, plot_value_policy, save, smooth, snap_x
from tabular_rl.training import train

from .common import CONVERGENT_SLIPPERY, N_JOBS, SEEDS, save_table, summary_row, train_default

ENVS = ["gridworld_deterministic", "gridworld_slippery"]
ALGOS = ["SARSA", "Q-learning"]


def values_and_learning_curves(rows):
    fig_p, axes_p = plt.subplots(2, 3, figsize=(13, 7.5), layout="constrained")
    fig_c, axes_c = plt.subplots(2, 3, figsize=(16, 8), layout="constrained")
    for i, env_name in enumerate(ENVS):
        spec = make_spec(env_name)
        outs = {a: train_default(a, env_name) for a in ALGOS}
        plot_value_policy(axes_p[i, 0], spec, outs["SARSA"]["Q_star"], f"{ENV_NAMES[env_name]}\n" + r"Optimal $Q^*$ (value iteration)")
        for j, a in enumerate(ALGOS):
            o = outs[a]
            plot_value_policy(axes_p[i, j + 1], spec, o["Q"].mean(0), f"{a}\n(mean Q over {len(o['Q'])} seeds)")
            rows.append({"env": ENV_NAMES[env_name], "algorithm": a, **summary_row(o, env_name)})
            plot_band(axes_c[i, 0], moving_average(o["returns"], 50), a, COLORS[a], robust=True)
            plot_band(axes_c[i, 1], o["rmse_opt"], a, COLORS[a], x=snap_x(o))
            axes_c[i, 2].plot(snap_x(o), smooth(o["regret"].mean(0), 5), color=COLORS[a], label=a, lw=1.8)
        axes_c[i, 0].set(title=f"{ENV_NAMES[env_name]}\nTraining return (moving avg 50)", ylabel="return")
        axes_c[i, 1].set(title="Error of Q on optimal actions\n" + r"$\mathrm{RMSE}\,(Q(s,a^*),\ Q^*(s,a^*))$", yscale="log")
        axes_c[i, 2].set(title="Regret of the greedy policy (mean)\n" + r"$V^*(s_0) - V^{\pi}(s_0)$", ylabel="regret", xscale="log")
        for ax in axes_c[i]:
            ax.set_xlabel("episode")
            ax.legend()
    fig_p.suptitle("Learned values and greedy policies vs the optimum")
    fig_c.suptitle("SARSA vs Q-learning: learning curves (20 seeds, median/IQR or mean±std)")
    save(fig_p, "exp1_values_policies.png")
    save(fig_c, "exp1_learning_curves.png")


def convergent_schedule(rows):
    """With the default settings (eps never below 0.05, constant alpha) SARSA misses the optimal
    policy in some seeds of the slippery world: it learns the value of its own exploring policy and
    rarely visits the two states next to the pit. If exploration fades out and alpha shrinks with
    the visits, theory says both should converge. Do they?"""
    fig, axes = plt.subplots(1, 3, figsize=(17, 4.6), layout="constrained")
    for a in ALGOS:
        convergent = train(a, "gridworld_slippery", SEEDS, n_jobs=N_JOBS, **CONVERGENT_SLIPPERY)
        rows.append({"env": "Slippery (GLIE + decaying α)", "algorithm": a, **summary_row(convergent, "gridworld_slippery")})
        for tag, out, ls in [("default", train_default(a, "gridworld_slippery"), "--"), ("GLIE + decaying α", convergent, "-")]:
            x = snap_x(out)
            axes[0].plot(x, 100 * smooth((out["regret"] < 1e-9).mean(0), 10), ls, color=COLORS[a], lw=1.8, label=f"{a}, {tag}")
            axes[1].plot(x, smooth(out["regret"].mean(0), 10), ls, color=COLORS[a], lw=1.8, label=f"{a}, {tag}")
            axes[2].plot(x, out["rmse_opt"].mean(0), ls, color=COLORS[a], lw=1.8, label=f"{a}, {tag}")
    axes[0].set(title=r"Seeds whose greedy policy is exactly $\pi^*$ (%)", ylabel="% of 20 seeds", ylim=(0, 105))
    axes[1].set(title="Regret of the greedy policy (mean)", yscale="log")
    axes[2].set(title="RMSE of Q on optimal actions (mean)", yscale="log")
    for ax in axes:
        ax.set(xlabel="episode", xscale="log")
    axes[0].legend(loc="upper left")
    fig.suptitle(r"Slippery gridworld: default schedule (dashed, $\varepsilon_{\min}=0.05$, constant $\alpha$) vs "
                 r"convergence conditions (solid, GLIE $\varepsilon\to 0$ + Robbins–Monro $\alpha$)")
    save(fig, "exp1_convergent_schedule.png")


def main():
    print("exp1: SARSA & Q-learning on the class gridworld")
    rows = []
    values_and_learning_curves(rows)
    convergent_schedule(rows)
    save_table(pd.DataFrame(rows), "exp1_sarsa_qlearning")


if __name__ == "__main__":
    main()
