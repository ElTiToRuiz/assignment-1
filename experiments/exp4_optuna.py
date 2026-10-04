"""Extra: hyper-parameter tuning with Optuna (TPE sampler) for SARSA and Q-learning (slippery gridworld).

Objective (minimise): mean regret of the greedy policy over the whole training run, i.e. the area
under the regret curve. It rewards learning *fast*, not only learning eventually. Averaged over 5
tuning seeds; budget fixed to 3000 episodes so the hyper-parameters matter.
The final comparison (default vs tuned) uses 20 *different* seeds to detect over-fitting.
The study is cached (results/cache/exp4_*.json + tables/exp4_trials_*.csv).
"""
import json

import matplotlib.pyplot as plt
import numpy as np
import optuna
import pandas as pd

from tabular_rl import runner
from tabular_rl.envs import make_spec
from tabular_rl.runner import CACHE_DIR, run_many, train
from tabular_rl.viz import COLORS, save, smooth, snap_x

from .common import N_JOBS, SEEDS, TABLES, save_table

ENV, N_EP, N_TRIALS = "gridworld_slippery", 3000, 40
TUNE_SEEDS = [100, 101, 102, 103, 104]  # disjoint from the evaluation seeds 0..19
DEFAULT = dict(alpha=0.1, eps=(1.0, 0.05, 0.995))
ALGOS = ["SARSA", "Q-learning"]


def objective_for(algo, spec):
    def objective(trial):
        alpha = trial.suggest_float("alpha", 0.01, 1.0, log=True)
        eps_start = trial.suggest_float("eps_start", 0.2, 1.0)
        eps_min = trial.suggest_float("eps_min", 0.001, 0.2, log=True)
        eps_decay = trial.suggest_float("eps_decay", 0.99, 0.9999, log=True)
        out = run_many(algo, spec, TUNE_SEEDS, n_jobs=len(TUNE_SEEDS), n_episodes=N_EP,
                       alpha=alpha, eps=(eps_start, eps_min, eps_decay))
        return float(out["regret"].mean())
    return objective


def tune(algo, spec):
    """Run (or load) the Optuna study. Returns (best_params, trials dataframe)."""
    best_path = CACHE_DIR / f"exp4_best_{runner.slug(algo)}.json"
    trials_path = TABLES / f"exp4_trials_{runner.slug(algo)}.csv"
    if best_path.exists() and trials_path.exists() and not runner.RETRAIN:
        return json.loads(best_path.read_text()), pd.read_csv(trials_path)
    print(f"  [optuna] tuning {algo}: {N_TRIALS} trials x {len(TUNE_SEEDS)} seeds ...", flush=True)
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.create_study(direction="minimize", sampler=optuna.samplers.TPESampler(seed=0))
    study.enqueue_trial(dict(alpha=0.1, eps_start=1.0, eps_min=0.05, eps_decay=0.995))  # default = trial 0
    study.optimize(objective_for(algo, spec), n_trials=N_TRIALS)
    best = {**study.best_params, "objective": study.best_value}
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    best_path.write_text(json.dumps(best, indent=2))
    df = study.trials_dataframe()
    save_table(df, f"exp4_trials_{runner.slug(algo)}")
    return best, df


def main():
    print("exp4: Optuna hyper-parameter tuning")
    spec = make_spec(ENV)
    fig, axes = plt.subplots(2, 3, figsize=(17, 9), layout="constrained")
    rows = []
    for j, algo in enumerate(ALGOS):
        best, df = tune(algo, spec)
        tuned = dict(alpha=best["alpha"], eps=(best["eps_start"], best["eps_min"], best["eps_decay"]))
        res_cfg = {"default": DEFAULT, "tuned": tuned}
        res = {tag: train(algo, ENV, SEEDS, n_jobs=N_JOBS, n_episodes=N_EP, **cfg)
               for tag, cfg in [("default", DEFAULT), ("tuned", tuned)]}
        for tag, cfg in [("default", DEFAULT), ("tuned", tuned)]:
            o = res[tag]
            rows.append({"algorithm": algo, "config": tag, "alpha": round(cfg["alpha"], 4),
                         "eps (start, min, decay)": tuple(round(v, 4) for v in cfg["eps"]),
                         "objective on tuning seeds": round(float(df["value"].iloc[0] if tag == "default" else best["objective"]), 4),
                         "mean regret held-out": round(float(o["regret"].mean()), 4),
                         "final regret held-out": round(float(o["regret"][:, -1].mean()), 4),
                         "seeds optimal": f"{int((o['regret'][:, -1] < 1e-9).sum())}/{len(SEEDS)}"})
        # (1) optimisation history
        ax = axes[j, 0]
        ax.scatter(df["number"], df["value"], s=18, color=COLORS[algo], alpha=.6, label="trial")
        ax.plot(df["number"], df["value"].cummin(), color="black", lw=1.5, label="best so far")
        ax.axhline(df["value"].iloc[0], ls=":", color="gray", label="default config")
        ax.set(yscale="log", xlabel="trial", ylabel="objective (mean regret)", title=f"{algo}: Optuna optimisation history")
        ax.legend()
        # (2) objective vs alpha, coloured by exploration decay
        ax = axes[j, 1]
        sc = ax.scatter(df["params_alpha"], df["value"], c=-np.log10(1 - df["params_eps_decay"]), cmap="viridis", s=28)
        ax.set(xscale="log", yscale="log", xlabel="alpha", ylabel="objective",
               title=f"{algo}: objective vs learning rate")
        fig.colorbar(sc, ax=ax, label="-log10(1 - eps_decay)  (higher = slower decay)")
        # (3) held-out default vs tuned
        ax = axes[j, 2]
        for tag, ls in [("default", "--"), ("tuned", "-")]:
            o = res[tag]
            ax.plot(snap_x(o), smooth(o["regret"].mean(0), 10), ls, color=COLORS[algo], lw=2,
                    label=f"{tag}: α={res_cfg[tag]['alpha']:.3f}, mean regret {o['regret'].mean():.4f}")
        ax.set(xlabel="episode", ylabel="regret (mean, smoothed)", xscale="log", title=f"{algo}: default vs tuned on 20 held-out seeds")
        ax.legend()
    fig.suptitle("Hyper-parameter optimisation with Optuna (slippery gridworld, 3000 episodes)")
    save(fig, "exp4_optuna.png")
    save_table(pd.DataFrame(rows), "exp4_default_vs_tuned")


if __name__ == "__main__":
    main()
