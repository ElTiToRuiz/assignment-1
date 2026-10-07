"""Hyper-parameter search with Optuna.

There is one study per (environment, algorithm), and each study tries to make two things small at once:

    speed      the regret of the greedy policy averaged over the whole training run (learn fast)
    exactness  the regret over the last 100 episodes (end up truly optimal)

These two pull against each other: a configuration can learn fast and still end up slightly wrong.
So instead of a single "best" we keep the whole Pareto front (the configurations that cannot get better
at one goal without getting worse at the other) and pick the one with the smallest sum of both.
Both are divided by |V*(s0)| so the two environments are on the same scale.

The studies live in an SQLite file. If a run stops halfway, running it again simply carries on.
Several processes share the file and test trials in parallel.

    uv run python -m experiments.tune                          # every study, 150 trials each
    uv run python -m experiments.tune --envs cliff_walking --algos SARSA Q-learning
    uv run python -m experiments.tune --quick --storage /tmp/smoke.db   # a few seconds, to check it works

Then `uv run python -m experiments.run_all` compares the tuned settings with the defaults (exp4).
"""
import argparse
import json
import math
import multiprocessing as mp
import os
import time
from pathlib import Path

import numpy as np
import optuna

from tabular_rl.envs import make_spec
from tabular_rl.paths import RESULTS, slug
from tabular_rl.training import run_many

from .common import CONFIG

STORAGE = RESULTS / "optuna" / "studies.db"
TUNE_SEEDS = [100, 101, 102, 103, 104]  # never used for the final comparison, which uses seeds 0..19
# Training budget per environment, the same for every trial and for the default-vs-tuned comparison.
BUDGET = {"gridworld_slippery": 3000, "cliff_walking": 500, "gridworld_deterministic": 300}
ENVS = ["gridworld_slippery", "cliff_walking"]
ALGOS = ["SARSA", "Expected SARSA", "Q-learning", "Double Q-learning", "n-step SARSA", "MC constant-α"]


def study_name(env, algo):
    return f"{env}__{slug(algo)}"


# ----------------------------------------------------------------------------- search space

def default_kwargs(env, algo):
    """The hand-picked settings used in the other experiments (with the tuning budget)."""
    kw = {k: v for k, v in CONFIG[env].items() if k != "n_episodes"}
    if algo == "n-step SARSA":
        kw["n"] = 3
    if algo == "MC constant-α":
        kw["alpha"] = 0.1
    return kw


def suggest(trial, env, algo):
    """What Optuna is allowed to change, turned into arguments for the agent."""
    is_mc = algo.startswith("MC")
    kw = {"alpha": trial.suggest_float("alpha", 0.005 if is_mc else 0.01, 0.5 if is_mc else 1.0, log=True)}
    # optionally shrink alpha with the visits (Robbins-Monro); Monte Carlo does not support it
    if not is_mc and trial.suggest_categorical("use_alpha_decay", [False, True]):
        kw["alpha_decay"] = trial.suggest_float("alpha_decay", 1e-4, 0.1, log=True)
    # The exploration schedule. The half-life (episodes until eps is halved) is easier to reason
    # about than the raw per-episode decay factor.
    eps_start = trial.suggest_float("eps_start", 0.05, 1.0)
    eps_min = trial.suggest_float("eps_min", 1e-3, 0.3, log=True)
    halflife = trial.suggest_float("eps_halflife", 10, BUDGET[env], log=True)
    kw["eps"] = (eps_start, min(eps_min, eps_start), 0.5 ** (1 / halflife))
    if algo == "n-step SARSA":
        kw["n"] = trial.suggest_int("n", 1, 8)
    return kw


def default_params(env, algo):
    """The default settings written in the search-space parameters, so they can be trial 0.
    That way every study shows where the defaults stand."""
    kw = default_kwargs(env, algo)
    eps_start, eps_min, decay = kw["eps"]
    halflife = BUDGET[env] if decay >= 1 else min(BUDGET[env], math.log(0.5) / math.log(decay))
    params = {"alpha": kw["alpha"], "eps_start": eps_start, "eps_min": eps_min, "eps_halflife": halflife}
    if not algo.startswith("MC"):
        params["use_alpha_decay"] = False
    if algo == "n-step SARSA":
        params["n"] = kw["n"]
    return params


# ----------------------------------------------------------------------------- objectives

def evaluate(env, algo, kw, seeds, n_episodes):
    """Train on the tuning seeds and return (speed, exactness, fraction of seeds exactly optimal)."""
    spec = make_spec(env)
    out = run_many(algo, spec, seeds, n_jobs=1, n_episodes=n_episodes, **kw)
    regret = np.clip(out["regret"], 0, None) / abs(out["V_star"][spec.start_state])
    return float(regret.mean()), float(regret[:, -10:].mean()), float((out["regret"][:, -1] < 1e-9).mean())


def make_objective(env, algo, seeds, n_episodes):
    def objective(trial):
        kw = suggest(trial, env, algo)
        speed, exactness, frac_optimal = evaluate(env, algo, kw, seeds, n_episodes)
        trial.set_user_attr("kwargs", json.dumps(kw))  # the exact agent arguments, so exp4 can retrain them
        trial.set_user_attr("seeds_optimal", frac_optimal)
        return speed, exactness
    return objective


def chosen_trial(study):
    """From the Pareto front, the trial with the smallest speed + exactness."""
    return min(study.best_trials, key=lambda t: sum(t.values))


# ----------------------------------------------------------------------------- running the studies

def storage_url(path):
    path = Path(path).resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    # a generous timeout, because several processes write to the same file
    return optuna.storages.RDBStorage(f"sqlite:///{path.as_posix()}", engine_kwargs={"connect_args": {"timeout": 120}})


def _worker(args):
    """One process: load the shared study and run its share of the trials."""
    storage, env, algo, n_trials, seeds, n_episodes, worker_id = args
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    sampler = optuna.samplers.TPESampler(seed=worker_id, multivariate=True, n_startup_trials=10)
    study = optuna.load_study(study_name=study_name(env, algo), storage=storage_url(storage), sampler=sampler)
    study.optimize(make_objective(env, algo, seeds, n_episodes), n_trials=n_trials)


def run(envs, algos, trials, workers, storage_path, seeds, quick=False):
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    storage = storage_url(storage_path)
    for env in envs:
        n_episodes = 150 if quick else BUDGET[env]
        for algo in algos:
            study = optuna.create_study(study_name=study_name(env, algo), storage=storage, load_if_exists=True,
                                        directions=["minimize", "minimize"])
            # Trial 0 is the default configuration. We run it here, before the workers start,
            # so two workers cannot both grab it.
            if not study.trials:
                study.enqueue_trial(default_params(env, algo))
                study.optimize(make_objective(env, algo, seeds, n_episodes), n_trials=1)

            done = sum(t.state == optuna.trial.TrialState.COMPLETE for t in study.trials)
            todo = max(0, trials - done)
            if todo == 0:
                print(f"  [optuna] {env:20s} {algo:18s} already has {done} trials")
                continue

            t0 = time.time()
            n_workers = max(1, min(workers, todo))
            shares = [todo // n_workers + (i < todo % n_workers) for i in range(n_workers)]
            jobs = [(storage_path, env, algo, k, seeds, n_episodes, 1000 * done + i) for i, k in enumerate(shares)]
            print(f"  [optuna] {env:20s} {algo:18s} {todo} trials x {len(seeds)} seeds, {n_workers} workers ...",
                  end="", flush=True)
            if n_workers == 1:
                _worker(jobs[0])
            else:
                with mp.get_context("spawn").Pool(n_workers) as pool:
                    pool.map(_worker, jobs)
            best = chosen_trial(optuna.load_study(study_name=study_name(env, algo), storage=storage))
            print(f" {time.time() - t0:6.1f}s  chosen: speed={best.values[0]:.4f} exact={best.values[1]:.4f}")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--envs", nargs="+", default=ENVS, choices=list(BUDGET))
    p.add_argument("--algos", nargs="+", default=ALGOS, choices=ALGOS)
    p.add_argument("--trials", type=int, default=150, help="finished trials per study (it resumes up to this number)")
    p.add_argument("--workers", type=int, default=os.cpu_count() or 1, help="processes running trials in parallel")
    p.add_argument("--storage", default=str(STORAGE), help="SQLite file holding the studies")
    p.add_argument("--quick", action="store_true", help="a few tiny trials, only to check that everything runs")
    a = p.parse_args()
    t0 = time.time()
    run(a.envs, a.algos, 4 if a.quick else a.trials, a.workers, a.storage,
        TUNE_SEEDS[:2] if a.quick else TUNE_SEEDS, quick=a.quick)
    print(f"done in {time.time() - t0:.1f}s. Now run: uv run python -m experiments.run_all")
