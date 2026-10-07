"""Hyper-parameter optimisation with Optuna, designed to run on a separate (bigger) machine.

Each (environment, algorithm) pair is one **multi-objective** Optuna study (TPE sampler):

    objective 1  speed      mean regret of the greedy policy over the whole training run
    objective 2  exactness  mean regret over the last 100 episodes (is the final policy optimal?)

Both are divided by |V*(s0)| so environments are comparable. The two objectives conflict (the
Optuna run on SARSA showed it: faster learning but fewer seeds exactly optimal), so we keep the
whole Pareto front and pick the point with the lowest sum of both objectives.

Studies are stored in SQLite (results/optuna/studies.db). They are **resumable**: rerunning adds
trials until each study has `--trials` finished trials. Several worker processes share the
database and evaluate trials in parallel.

    uv run python -m experiments.tune                       # all studies, default budget
    uv run python -m experiments.tune --workers 32 --trials 100
    uv run python -m experiments.tune --envs cliff_walking --algos SARSA Q-learning
    uv run python -m experiments.tune --quick --storage /tmp/smoke.db   # 1-minute smoke test

Afterwards `uv run python -m experiments.run_all` evaluates the tuned configurations on 20 held-out seeds.
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
from tabular_rl.runner import RESULTS, run_many, slug

from .common import CONFIG

STORAGE = RESULTS / "optuna" / "studies.db"
TUNE_SEEDS = [100, 101, 102, 103, 104]  # disjoint from the evaluation seeds 0..19
# Fixed training budget per environment during tuning (and for the default-vs-tuned comparison).
BUDGET = {"gridworld_slippery": 3000, "cliff_walking": 500, "gridworld_deterministic": 300}
ENVS = ["gridworld_slippery", "cliff_walking"]
ALGOS = ["SARSA", "Expected SARSA", "Q-learning", "Double Q-learning", "n-step SARSA", "MC constant-α"]


def study_name(env, algo):
    return f"{env}__{slug(algo)}"


def default_kwargs(env, algo):
    """The hand-picked configuration used in the other experiments, with the tuning budget."""
    kw = {k: v for k, v in CONFIG[env].items() if k != "n_episodes"}
    if algo == "n-step SARSA":
        kw["n"] = 3
    if algo == "MC constant-α":
        kw["alpha"] = 0.1
    return kw


def suggest(trial, env, algo):
    """Search space -> keyword arguments for the agent."""
    budget = BUDGET[env]
    mc = algo.startswith("MC")
    kw = {"alpha": trial.suggest_float("alpha", 0.005 if mc else 0.01, 0.5 if mc else 1.0, log=True)}
    if not mc and trial.suggest_categorical("use_alpha_decay", [False, True]):
        kw["alpha_decay"] = trial.suggest_float("alpha_decay", 1e-4, 0.1, log=True)  # Robbins-Monro
    eps_start = trial.suggest_float("eps_start", 0.05, 1.0)
    eps_min = trial.suggest_float("eps_min", 1e-3, 0.3, log=True)
    halflife = trial.suggest_float("eps_halflife", 10, budget, log=True)  # episodes to halve eps
    kw["eps"] = (eps_start, min(eps_min, eps_start), 0.5 ** (1 / halflife))
    if algo == "n-step SARSA":
        kw["n"] = trial.suggest_int("n", 1, 8)
    return kw


def default_params(env, algo):
    """The default configuration expressed in the search-space parameters (enqueued as trial 0)."""
    kw = default_kwargs(env, algo)
    eps_start, eps_min, decay = kw["eps"]
    p = {"alpha": kw["alpha"], "eps_start": eps_start, "eps_min": eps_min,
         "eps_halflife": BUDGET[env] if decay >= 1 else min(BUDGET[env], math.log(0.5) / math.log(decay))}
    if not algo.startswith("MC"):
        p["use_alpha_decay"] = False
    if algo == "n-step SARSA":
        p["n"] = kw["n"]
    return p


def evaluate(env, algo, kw, seeds, n_episodes):
    """The two objectives (normalised regret) for one configuration."""
    spec = make_spec(env)
    out = run_many(algo, spec, seeds, n_jobs=1, n_episodes=n_episodes, **kw)
    reg = np.clip(out["regret"], 0, None) / abs(out["V_star"][spec.start_state])
    return float(reg.mean()), float(reg[:, -10:].mean()), float((out["regret"][:, -1] < 1e-9).mean())


def _objective(env, algo, seeds, n_episodes):
    def objective(trial):
        kw = suggest(trial, env, algo)
        speed, exact, frac_opt = evaluate(env, algo, kw, seeds, n_episodes)
        trial.set_user_attr("kwargs", json.dumps(kw))
        trial.set_user_attr("seeds_optimal", frac_opt)
        return speed, exact
    return objective


def _worker(args):
    storage, env, algo, n_trials, seeds, n_episodes, worker_id = args
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.load_study(study_name=study_name(env, algo), storage=storage_url(storage),
                              sampler=optuna.samplers.TPESampler(seed=worker_id, multivariate=True,
                                                                 n_startup_trials=10))
    study.optimize(_objective(env, algo, seeds, n_episodes), n_trials=n_trials)


def storage_url(path):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    return optuna.storages.RDBStorage(f"sqlite:///{path}", engine_kwargs={"connect_args": {"timeout": 60}})


def chosen_trial(study):
    """Pareto-optimal trial with the lowest speed + exactness."""
    return min(study.best_trials, key=lambda t: sum(t.values))


def run(envs, algos, trials, workers, storage_path, seeds, quick=False):
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    storage = storage_url(storage_path)
    for env in envs:
        n_episodes = 150 if quick else BUDGET[env]
        for algo in algos:
            study = optuna.create_study(study_name=study_name(env, algo), storage=storage, load_if_exists=True,
                                        directions=["minimize", "minimize"])
            if not study.trials:  # trial 0 = default configuration, evaluated here before the workers start
                optuna.logging.set_verbosity(optuna.logging.WARNING)
                study.enqueue_trial(default_params(env, algo))
                study.optimize(_objective(env, algo, seeds, n_episodes), n_trials=1)
            done = len([t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE])
            todo = max(0, trials - done)
            if todo == 0:
                print(f"  [optuna] {env:20s} {algo:18s} already has {done} trials")
                continue
            t0 = time.time()
            n_w = min(workers, todo)
            shares = [todo // n_w + (i < todo % n_w) for i in range(n_w)]
            jobs = [(storage_path, env, algo, k, seeds, n_episodes, 1000 * done + i) for i, k in enumerate(shares)]
            print(f"  [optuna] {env:20s} {algo:18s} {todo} trials x {len(seeds)} seeds, {n_w} workers ...",
                  end="", flush=True)
            if n_w == 1:
                _worker(jobs[0])
            else:
                with mp.get_context("spawn").Pool(n_w) as pool:
                    pool.map(_worker, jobs)
            study = optuna.load_study(study_name=study_name(env, algo), storage=storage)
            best = chosen_trial(study)
            print(f" {time.time() - t0:6.1f}s  chosen: speed={best.values[0]:.4f} exact={best.values[1]:.4f}")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--envs", nargs="+", default=ENVS, choices=list(BUDGET))
    p.add_argument("--algos", nargs="+", default=ALGOS, choices=ALGOS)
    p.add_argument("--trials", type=int, default=150, help="finished trials per study (resumes up to this)")
    p.add_argument("--workers", type=int, default=os.cpu_count() or 1, help="parallel processes")
    p.add_argument("--storage", default=str(STORAGE), help="SQLite file with the studies")
    p.add_argument("--quick", action="store_true", help="smoke test: 4 trials, 2 seeds, 150 episodes")
    a = p.parse_args()
    t0 = time.time()
    run(a.envs, a.algos, 4 if a.quick else a.trials, a.workers, a.storage,
        TUNE_SEEDS[:2] if a.quick else TUNE_SEEDS, quick=a.quick)
    print(f"done in {time.time() - t0:.1f}s. Now run: uv run python -m experiments.run_all")
