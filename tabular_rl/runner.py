"""Run an algorithm over several seeds, compute metrics against the DP ground truth,
and cache everything on disk so plots never require retraining.

    out = train("Q-learning", "gridworld_slippery", seeds=range(20), alpha=0.1, ...)

The first call trains and writes `results/cache/<env>__<algo>__<hash>.npz`; every later call with
the same arguments just loads that file. Set `runner.RETRAIN = True` (or `--retrain` in
`experiments.run_all`) to force training again.
"""
import hashlib
import json
import re
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

from .agents import ALGORITHMS
from .dp import evaluate_policy, model_tensors, value_iteration
from .envs import make_spec

RESULTS = Path(__file__).resolve().parent.parent / "results"
CACHE_DIR = RESULTS / "cache"
MODELS_DIR = RESULTS / "models"
RETRAIN = False  # set to True to ignore the cache
_TRAINED_NOW = set()  # with RETRAIN, each configuration is still trained only once per process


# --------------------------------------------------------------------------- metrics
def snapshot_metrics(spec, snaps, Q_star, V_star, tensors):
    """Metrics for every Q snapshot taken during training.

    rmse      RMSE(Q, Q*) over every (s, a) of non-terminal states
    rmse_opt  RMSE only on the optimal action of each state (the values that matter for control)
    match     fraction of states whose greedy action is optimal
    regret    V*(s0) - V^pi(s0) for the greedy policy pi (exact, via policy evaluation with P)
    """
    valid = spec.valid_states
    opt_actions = Q_star[valid] >= Q_star[valid].max(axis=1, keepdims=True) - 1e-9
    a_star = Q_star.argmax(axis=1)[valid]
    rmse, rmse_opt, match, regret = [], [], [], []
    for Q in snaps:
        rmse.append(np.sqrt(np.mean((Q[valid] - Q_star[valid]) ** 2)))
        rmse_opt.append(np.sqrt(np.mean((Q[valid, a_star] - Q_star[valid, a_star]) ** 2)))
        pi = Q.argmax(axis=1)
        match.append(np.mean(opt_actions[np.arange(len(valid)), pi[valid]]))
        V_pi = evaluate_policy(spec, pi, tensors)
        regret.append(V_star[spec.start_state] - V_pi[spec.start_state])
    return {"rmse": rmse, "rmse_opt": rmse_opt, "match": match, "regret": regret}


# --------------------------------------------------------------------------- training
def _one(args):
    algo, spec, seed, kw = args
    return ALGORITHMS[algo](spec, seed=seed, **kw)


def run_many(algo, spec, seeds, n_jobs=1, **kw):
    """Train `algo` once per seed (optionally in parallel). Returns a dict of arrays, one row per seed."""
    Q_star, V_star, _ = value_iteration(spec)
    tensors = model_tensors(spec)
    jobs = [(algo, spec, s, kw) for s in seeds]
    if n_jobs > 1:
        with ProcessPoolExecutor(n_jobs) as ex:
            results = list(ex.map(_one, jobs))
    else:
        results = [_one(j) for j in jobs]

    a0 = Q_star[spec.start_state].argmax()
    keys = ["returns", "lengths", "rmse", "rmse_opt", "match", "regret", "Q", "visits", "q_start_opt"]
    out = {k: [] for k in keys}
    for res in results:
        for k, v in snapshot_metrics(spec, res.snaps, Q_star, V_star, tensors).items():
            out[k].append(v)
        out["returns"].append(res.returns)
        out["lengths"].append(res.lengths)
        out["Q"].append(res.Q)
        out["visits"].append(res.visits)
        out["q_start_opt"].append(res.snaps[:, spec.start_state, a0])
    out = {k: np.array(v) for k, v in out.items()}
    out["Q_star"], out["V_star"] = Q_star, V_star
    return out


def slug(text):
    return re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")


def _key(algo, env_name, seeds, kw):
    cfg = json.dumps({"algo": algo, "env": env_name, "seeds": list(seeds), **kw}, sort_keys=True, default=str)
    return cfg, hashlib.sha1(cfg.encode()).hexdigest()[:10]


def _compact(a):
    """Shrink big arrays before saving (float32 is plenty for plots)."""
    if a.dtype == np.float64 and a.size > 1000:
        return a.astype(np.float32)
    if a.dtype.kind in "iu":
        return a.astype(np.int32)
    return a


def train(algo, env_name, seeds, n_jobs=1, **kw):
    """Cached version of `run_many`. Same arguments -> loads from disk instead of training."""
    cfg, h = _key(algo, env_name, seeds, kw)
    path = CACHE_DIR / f"{env_name}__{slug(algo)}__{h}.npz"
    if path.exists() and (not RETRAIN or path in _TRAINED_NOW):
        with np.load(path) as z:
            return {k: z[k] for k in z.files}
    t0 = time.time()
    print(f"  [train] {algo:18s} on {env_name:24s} ({len(list(seeds))} seeds) ...", end="", flush=True)
    out = run_many(algo, make_spec(env_name), list(seeds), n_jobs=n_jobs, **kw)
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **{k: _compact(v) for k, v in out.items()})
    path.with_suffix(".json").write_text(json.dumps(
        {"config": json.loads(cfg), "train_seconds": round(time.time() - t0, 1)}, indent=2))
    _TRAINED_NOW.add(path)
    print(f" {time.time() - t0:5.1f}s")
    return out


# --------------------------------------------------------------------------- models
def save_model(env_name, algo, Q):
    """Save a learned Q-table (the 'model' of a tabular agent)."""
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    np.save(MODELS_DIR / f"{env_name}__{slug(algo)}.npy", np.asarray(Q, dtype=np.float64))


def load_model(env_name, algo):
    path = MODELS_DIR / f"{env_name}__{slug(algo)}.npy"
    if not path.exists():
        raise FileNotFoundError(f"{path} not found. Run `python -m experiments.run_all` once first.")
    return np.load(path)


def moving_average(x, w=50):
    c = np.cumsum(np.insert(np.asarray(x, dtype=float), 0, 0.0, axis=-1), axis=-1)
    return (c[..., w:] - c[..., :-w]) / w
