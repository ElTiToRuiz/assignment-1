"""Minimum requirement: SARSA and Q-learning on the class gridworld (deterministic and slippery)."""
import matplotlib.pyplot as plt
import pandas as pd

from tabular_rl.envs import make_spec
from tabular_rl.runner import moving_average, train
from tabular_rl.viz import COLORS, ENV_NAMES, plot_band, plot_value_policy, save, smooth, snap_x

from .common import CONVERGENT_SLIPPERY, N_JOBS, SEEDS, save_table, summary_row, train_default

ENVS = ["gridworld_deterministic", "gridworld_slippery"]
ALGOS = ["SARSA", "Q-learning"]


def main():
    print("exp1: SARSA & Q-learning on the class gridworld")
    fig_p, axes_p = plt.subplots(2, 3, figsize=(13, 7.5), layout="constrained")
    fig_c, axes_c = plt.subplots(2, 3, figsize=(16, 8), layout="constrained")
    rows = []
    for i, env_name in enumerate(ENVS):
        spec = make_spec(env_name)
        outs = {a: train_default(a, env_name) for a in ALGOS}
        plot_value_policy(axes_p[i, 0], spec, outs["SARSA"]["Q_star"], f"{ENV_NAMES[env_name]}\nOptimal Q* (value iteration)")
        for j, a in enumerate(ALGOS):
            o = outs[a]
            plot_value_policy(axes_p[i, j + 1], spec, o["Q"].mean(0), f"{a}\n(mean Q over {len(o['Q'])} seeds)")
            rows.append({"env": ENV_NAMES[env_name], "algorithm": a, **summary_row(o, env_name)})
            plot_band(axes_c[i, 0], moving_average(o["returns"], 50), a, COLORS[a], robust=True)
            plot_band(axes_c[i, 1], o["rmse_opt"], a, COLORS[a], x=snap_x(o))
            axes_c[i, 2].plot(snap_x(o), smooth(o["regret"].mean(0), 5), color=COLORS[a], label=a, lw=1.8)
        axes_c[i, 0].set(title=f"{ENV_NAMES[env_name]}\nTraining return (moving avg 50)", ylabel="return")
        axes_c[i, 1].set(title="Error of Q on optimal actions\nRMSE(Q(s,a*), Q*(s,a*))", yscale="log")
        axes_c[i, 2].set(title="Regret of the greedy policy (mean)\nV*(s0) − V^π(s0)", ylabel="regret", xscale="log")
        for ax in axes_c[i]:
            ax.set_xlabel("episode"); ax.legend()
    # Default schedule keeps eps_min = 0.05 and a constant alpha: SARSA then learns Q of the
    # eps-greedy policy and rarely visits the states next to the pit (6, 11). With GLIE exploration
    # and a decaying alpha, both algorithms reach the optimal policy in every seed.
    fig_g, axes_g = plt.subplots(1, 3, figsize=(17, 4.6), layout="constrained")
    for a in ALGOS:
        o = train(a, "gridworld_slippery", SEEDS, n_jobs=N_JOBS, **CONVERGENT_SLIPPERY)
        rows.append({"env": "Slippery (GLIE + decaying α)", "algorithm": a, **summary_row(o, "gridworld_slippery")})
        for tag, out, ls in [("default", train_default(a, "gridworld_slippery"), "--"), ("GLIE + decaying α", o, "-")]:
            x = snap_x(out)
            axes_g[0].plot(x, 100 * smooth((out["regret"] < 1e-9).mean(0), 10), ls, color=COLORS[a], lw=1.8, label=f"{a}, {tag}")
            axes_g[1].plot(x, smooth(out["regret"].mean(0), 10), ls, color=COLORS[a], lw=1.8, label=f"{a}, {tag}")
            axes_g[2].plot(x, out["rmse_opt"].mean(0), ls, color=COLORS[a], lw=1.8, label=f"{a}, {tag}")
    axes_g[0].set(title="Seeds whose greedy policy is exactly π* (%)", ylabel="% of 20 seeds", ylim=(0, 105))
    axes_g[1].set(title="Regret of the greedy policy (mean)", yscale="log")
    axes_g[2].set(title="RMSE of Q on optimal actions (mean)", yscale="log")
    for ax in axes_g:
        ax.set(xlabel="episode", xscale="log")
    axes_g[0].legend(loc="upper left")
    fig_g.suptitle("Slippery gridworld: default schedule (dashed, ε_min=0.05, constant α) vs "
                   "convergence conditions (solid, GLIE ε→0 + Robbins–Monro α)")
    save(fig_g, "exp1_convergent_schedule.png")
    fig_p.suptitle("Learned values and greedy policies vs the optimum")
    fig_c.suptitle("SARSA vs Q-learning: learning curves (20 seeds, median/IQR or mean±std)")
    save(fig_p, "exp1_values_policies.png")
    save(fig_c, "exp1_learning_curves.png")
    save_table(pd.DataFrame(rows), "exp1_sarsa_qlearning")


if __name__ == "__main__":
    main()
