"""Experiment 5: a closer look at training itself.

  (a) how the learning rate trades speed for noise
  (b) how much to explore, and for how long
  (c) Monte Carlo vs TD: the bias / variance trade-off from the slides
  (d) which states the agent actually visits
"""
import matplotlib.pyplot as plt
import numpy as np

from tabular_rl.envs import make_spec
from tabular_rl.metrics import moving_average
from tabular_rl.training import train
from tabular_rl.plotting import COLORS, plot_band, save, snap_x

from .common import N_JOBS, train_default

SEEDS10 = list(range(10))
N_EP = 3000
ENV = "gridworld_slippery"


def alpha_sensitivity():
    fig, ax = plt.subplots(1, 2, figsize=(13, 4.3), layout="constrained")
    alphas = [0.01, 0.03, 0.1, 0.3, 0.6, 1.0]
    for algo in ["SARSA", "Q-learning"]:
        outs = [train(algo, ENV, SEEDS10, n_jobs=N_JOBS, n_episodes=N_EP, alpha=al, eps=(1.0, 0.05, 0.998))
                for al in alphas]
        auc = np.array([o["regret"].mean(1) for o in outs])
        fin = np.array([o["rmse_opt"][:, -1] for o in outs])
        for a_, v in [(ax[0], auc), (ax[1], fin)]:
            a_.errorbar(alphas, v.mean(1), v.std(1), label=algo, color=COLORS[algo], marker="o", capsize=3, lw=1.8)
    ax[0].set(xscale="log", xlabel="alpha", ylabel="mean regret", title="Speed: mean regret during training")
    ax[1].set(xscale="log", yscale="log", xlabel="alpha", ylabel="RMSE",
              title="Accuracy: final RMSE of Q on optimal actions")
    ax[0].legend()
    fig.suptitle("Learning-rate sensitivity (slippery gridworld, 10 seeds): too small = slow, too large = noisy")
    save(fig, "exp5a_alpha_sensitivity.png")


def exploration():
    schedules = {"constant ε=0.01": (0.01, 0.01, 1.0), "constant ε=0.1": (0.1, 0.1, 1.0),
                 "constant ε=0.3": (0.3, 0.3, 1.0), "decay 1 → 0.05": (1.0, 0.05, 0.998)}
    cmap = plt.get_cmap("plasma")
    fig, ax = plt.subplots(1, 3, figsize=(17, 4.5), layout="constrained")
    for k, (name, sch) in enumerate(schedules.items()):
        o = train("Q-learning", ENV, SEEDS10, n_jobs=N_JOBS, n_episodes=N_EP, alpha=0.1, eps=sch)
        c = cmap(k / len(schedules))
        plot_band(ax[0], moving_average(o["returns"], 50), name, c, robust=True)
        plot_band(ax[1], o["regret"], name, c, x=snap_x(o), robust=True, floor=0)
        plot_band(ax[2], o["rmse_opt"], name, c, x=snap_x(o))
    ax[0].set(title="Training return (what the agent gets while exploring)", ylabel="return")
    ax[1].set(title="Regret of greedy policy (what it has learned)")
    ax[2].set(title="RMSE of Q on optimal actions", yscale="log")
    for a_ in ax:
        a_.set_xlabel("episode")
    ax[0].legend()
    fig.suptitle("Exploration–exploitation trade-off: Q-learning with different ε schedules (slippery gridworld)")
    save(fig, "exp5b_exploration.png")


def bias_variance():
    spec = make_spec(ENV)
    fig, ax = plt.subplots(1, 2, figsize=(13, 4.3), layout="constrained")
    for algo in ["Monte Carlo", "SARSA", "Q-learning"]:
        o = train(algo, ENV, list(range(30)), n_jobs=N_JOBS, n_episodes=N_EP, alpha=0.1, eps=(0.2, 0.2, 1.0))
        q = o["q_start_opt"]
        target = o["Q_star"][spec.start_state].max()
        ax[0].plot(snap_x(o), q.mean(0) - target, label=algo, color=COLORS[algo], lw=1.8)
        ax[1].plot(snap_x(o), q.std(0), label=algo, color=COLORS[algo], lw=1.8)
    ax[0].axhline(0, color="black", lw=.8)
    ax[0].set(xlabel="episode", title="Bias: mean over seeds of Q(s0,a*) − Q*(s0,a*)")
    ax[1].set(xlabel="episode", yscale="log", title="Variance: std over seeds of Q(s0,a*)")
    ax[0].legend()
    fig.suptitle("Monte Carlo vs TD (30 seeds, constant ε=0.2): MC targets the ε-greedy return, TD bootstraps")
    save(fig, "exp5c_bias_variance.png")


def coverage():
    spec = make_spec("gridworld_deterministic")
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.8), layout="constrained")
    for a_, algo in zip(axes, ["SARSA", "Q-learning"]):
        o = train_default(algo, "gridworld_deterministic")
        v = o["visits"].mean(0).sum(1).reshape(spec.shape).astype(float)
        a_.imshow(np.log10(v + 1), cmap="magma")
        for s in range(spec.n_states):
            r, c = divmod(s, spec.shape[1])
            label = "WALL" if s in spec.walls else ("GOAL" if spec.terminals.get(s, 0) > 0 else
                                                     "PIT" if s in spec.terminals else f"{v[r, c]:.0f}")
            a_.text(c, r, label, ha="center", va="center", color="white", fontsize=10, weight="bold")
        a_.set(title=f"{algo}: mean visits per state", xticks=[], yticks=[])
        a_.grid(False)
    fig.suptitle("State coverage (deterministic, 3000 episodes): rarely visited states keep inaccurate Q values")
    save(fig, "exp5d_state_visits.png")


def main():
    print("exp5: training-process analysis")
    alpha_sensitivity()
    exploration()
    bias_variance()
    coverage()


if __name__ == "__main__":
    main()
