"""Settings shared by all experiments: seeds, default hyper-parameters and the summary table."""
import os

import numpy as np

from tabular_rl.envs import make_spec
from tabular_rl.paths import TABLES
from tabular_rl.training import train

SEEDS = list(range(20))   # every configuration is trained 20 times, results are reported over the 20
N_JOBS = int(os.environ.get("RL_JOBS", min(20, os.cpu_count() or 1)))  # processes used to train the seeds
LOG_EVERY = 10            # the agents keep a copy of Q every 10 episodes

ALGOS = ["Monte Carlo", "SARSA", "n-step SARSA", "Expected SARSA", "Q-learning", "Double Q-learning"]
ENVS = ["gridworld_deterministic", "gridworld_slippery", "cliff_walking"]

# Hand-picked hyper-parameters, one set per environment. eps = (start, minimum, decay per episode).
CONFIG = {
    "gridworld_deterministic": dict(n_episodes=3000, alpha=0.1, eps=(1.0, 0.05, 0.995)),
    # random transitions make every update noisy, so it gets more episodes and explores for longer
    "gridworld_slippery": dict(n_episodes=10000, alpha=0.1, eps=(1.0, 0.05, 0.9995)),
    # the classic Sutton & Barto setting: constant eps = 0.1 and alpha = 0.5
    "cliff_walking": dict(n_episodes=1000, alpha=0.5, eps=(0.1, 0.1, 1.0)),
}

# A schedule that meets the theory's convergence conditions in the slippery gridworld (used in exp1):
# exploration fades to zero (GLIE) and the step size shrinks with the visits (Robbins-Monro).
CONVERGENT_SLIPPERY = dict(n_episodes=30000, alpha=0.5, eps=(1.0, 0.0, 0.9998), alpha_decay=0.005)


def train_default(algo, env_name):
    """Train `algo` on `env_name` with the default settings above (or load it from the cache)."""
    return train(algo, env_name, SEEDS, n_jobs=N_JOBS, **CONFIG[env_name])


def episodes_to_optimal(out, env_name, rel_tol=0.02, window=20):
    """For each seed, the episode from which the greedy policy stays (nearly) optimal until the end.

    "Nearly optimal" = regret below 2% of |V*(s0)|, averaged over 20 snapshots (200 episodes). The
    averaging matters in the slippery world, where some actions are almost tied and the greedy choice
    flips back and forth for a long time. NaN if the seed never settles.
    """
    tol = rel_tol * abs(out["V_star"][make_spec(env_name).start_state])
    result = []
    for reg in out["regret"]:
        smoothed = np.convolve(np.abs(reg), np.ones(window) / window, mode="full")[:len(reg)]
        smoothed[:window] = np.cumsum(np.abs(reg[:window])) / np.arange(1, window + 1)
        bad = np.flatnonzero(smoothed > tol)
        if len(bad) == 0:
            result.append(LOG_EVERY)
        elif bad[-1] == len(reg) - 1:
            result.append(np.nan)
        else:
            result.append((bad[-1] + 2) * LOG_EVERY)
    return np.array(result, dtype=float)


def summary_row(out, env_name, tail=100):
    """One table row with how training ended, averaged over the seeds."""
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
