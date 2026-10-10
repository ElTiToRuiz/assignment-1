"""Train an algorithm over many seeds, and remember the result on disk.

    out = train("Q-learning", "gridworld_slippery", seeds=range(20), alpha=0.1, ...)

The first call trains and saves results/cache/<env>__<algo>__<hash>.npz. Any later call with exactly
the same arguments just loads that file, so redrawing a figure never means retraining. The hash
comes from the arguments, so a new configuration simply gets a new file.

Set training.RETRAIN = True (or pass --retrain to run_all) to ignore the cache.
"""
import hashlib
import json
import time
from concurrent.futures import ProcessPoolExecutor

import numpy as np

from .agents import ALGORITHMS
from .envs import make_spec
from .metrics import snapshot_metrics
from .paths import CACHE_DIR, slug
from .planning import model_tensors, value_iteration

RETRAIN = False
_trained_this_session = set()  # with RETRAIN, still train each configuration only once per run


def _train_one_seed(args):
    algo, spec, seed, kw = args
    return ALGORITHMS[algo](spec, seed=seed, **kw)


def run_many(algo, spec, seeds, n_jobs=1, **kw):
    """Train `algo` once per seed (in parallel if n_jobs > 1) and score every run.

    Returns a dict of arrays with one row per seed, plus the true Q* and V* for reference.
    """
    Q_star, V_star, _ = value_iteration(spec)
    tensors = model_tensors(spec)
    jobs = [(algo, spec, s, kw) for s in seeds]
    if n_jobs > 1:
        with ProcessPoolExecutor(n_jobs) as pool:
            runs = list(pool.map(_train_one_seed, jobs))
    else:
        runs = [_train_one_seed(j) for j in jobs]

    best_start_action = Q_star[spec.start_state].argmax()
    keys = ["returns", "lengths", "rmse", "rmse_opt", "match", "regret", "Q", "visits", "q_start_opt"]
    out = {k: [] for k in keys}
    for run in runs:
        for k, v in snapshot_metrics(spec, run.snaps, Q_star, V_star, tensors).items():
            out[k].append(v)
        out["returns"].append(run.returns)
        out["lengths"].append(run.lengths)
        out["Q"].append(run.Q)
        out["visits"].append(run.visits)
        out["q_start_opt"].append(run.snaps[:, spec.start_state, best_start_action])
    out = {k: np.array(v) for k, v in out.items()}
    out["Q_star"], out["V_star"] = Q_star, V_star
    return out


def _cache_key(algo, env_name, seeds, kw):
    config = json.dumps({"algo": algo, "env": env_name, "seeds": list(seeds), **kw}, sort_keys=True, default=str)
    return config, hashlib.sha1(config.encode()).hexdigest()[:10]


def _shrink(a):
    """Integers as int32 to save some space. Floats stay at full precision, so a table computed
    from the cache is identical to one computed right after training."""
    if a.dtype.kind in "iu":
        return a.astype(np.int32)
    return a


def train(algo, env_name, seeds, n_jobs=1, **kw):
    """Like run_many, but loads the result from the cache when this exact configuration was already trained."""
    config, h = _cache_key(algo, env_name, seeds, kw)
    path = CACHE_DIR / f"{env_name}__{slug(algo)}__{h}.npz"
    if path.exists() and (not RETRAIN or path in _trained_this_session):
        with np.load(path) as z:
            return {k: z[k] for k in z.files}

    t0 = time.time()
    print(f"  [train] {algo:18s} on {env_name:24s} ({len(list(seeds))} seeds) ...", end="", flush=True)
    out = run_many(algo, make_spec(env_name), list(seeds), n_jobs=n_jobs, **kw)
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **{k: _shrink(v) for k, v in out.items()})
    # A readable note next to the data saying what it is and how long it took.
    path.with_suffix(".json").write_text(json.dumps(
        {"config": json.loads(config), "train_seconds": round(time.time() - t0, 1)}, indent=2))
    _trained_this_session.add(path)
    print(f" {time.time() - t0:5.1f}s")
    return out
