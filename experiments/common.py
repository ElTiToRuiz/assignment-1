"""Shared settings for all experiments."""
import os
from pathlib import Path

import numpy as np

from tabular_rl.envs import make_spec
from tabular_rl.runner import train

SEEDS = list(range(20))
N_JOBS = min(8, os.cpu_count() or 1)
LOG_EVERY = 10  # Q snapshot every 10 episodes (agents' default)
TABLES = Path(__file__).resolve().parent.parent / "results" / "tables"

ALGOS = ["Monte Carlo", "SARSA", "n-step SARSA", "Expected SARSA", "Q-learning", "Double Q-learning"]
ENVS = ["gridworld_deterministic", "gridworld_slippery", "cliff_walking"]

# Default hyper-parameters per environment. eps = (eps_start, eps_min, eps_decay per episode).
CONFIG = {
    "gridworld_deterministic": dict(n_episodes=3000, alpha=0.1, eps=(1.0, 0.05, 0.995)),
    # noisy transitions -> more episodes and slower exploration decay
    "gridworld_slippery": dict(n_episodes=10000, alpha=0.1, eps=(1.0, 0.05, 0.9995)),
    # classic Sutton & Barto setting: constant eps = 0.1, alpha = 0.5
    "cliff_walking": dict(n_episodes=1000, alpha=0.5, eps=(0.1, 0.1, 1.0)),
}


def train_default(algo, env_name):
    """Train (or load from cache) `algo` on `env_name` with the default config and 20 seeds."""
    return train(algo, env_name, SEEDS, n_jobs=N_JOBS, **CONFIG[env_name])


def episodes_to_optimal(out, env_name, rel_tol=0.02, window=20):
    """Per seed: first episode from which the greedy policy stays near-optimal until the end of training.

    Near-optimal = regret, averaged over a sliding window of `window` snapshots (200 episodes), below
    2% of |V*(s0)|. The window ignores single-snapshot flips between almost-tied actions (slippery world).
    NaN if the seed never settles.
    """
    tol = rel_tol * abs(out["V_star"][make_spec(env_name).start_state])
    res = []
    for reg in out["regret"]:
        sm = np.convolve(np.abs(reg), np.ones(window) / window, mode="full")[:len(reg)]
        sm[:window] = np.cumsum(np.abs(reg[:window])) / np.arange(1, window + 1)
        bad = np.flatnonzero(sm > tol)
        if len(bad) == 0:
            res.append(LOG_EVERY)
        elif bad[-1] == len(reg) - 1:
            res.append(np.nan)
        else:
            res.append((bad[-1] + 2) * LOG_EVERY)
    return np.array(res, dtype=float)


def summary_row(out, env_name, tail=100):
    """Final metrics (mean ± std over seeds)."""
    ret = out["returns"][:, -tail:].mean(1)
    eto = episodes_to_optimal(out, env_name)
    return {
        "return (last 100 ep)": f"{ret.mean():.3f} ± {ret.std():.3f}",
        "return median": f"{np.median(ret):.3f}",
        "RMSE Q (all)": f"{out['rmse'][:, -1].mean():.3f}",
        "RMSE Q (optimal a)": f"{out['rmse_opt'][:, -1].mean():.3f} ± {out['rmse_opt'][:, -1].std():.3f}",
        "optimal actions %": f"{100 * out['match'][:, -1].mean():.1f}",
        "regret at s0": f"{abs(out['regret'][:, -1].mean()):.4f}",
        "seeds optimal": f"{int((out['regret'][:, -1] < 1e-9).sum())}/{len(out['regret'])}",
        "seeds near-optimal": f"{int((~np.isnan(eto)).sum())}/{len(eto)}",
        "episodes to near-optimal (median of those)": "—" if np.all(np.isnan(eto)) else f"{np.nanmedian(eto):.0f}",
    }


def save_table(df, name):
    TABLES.mkdir(parents=True, exist_ok=True)
    df.to_csv(TABLES / f"{name}.csv", index=False)
    print(f"  [table]  results/tables/{name}.csv")


def to_markdown(df):
    cols = list(df.columns)
    lines = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    lines += ["| " + " | ".join(str(v) for v in row) + " |" for row in df.itertuples(index=False)]
    return "\n".join(lines)
