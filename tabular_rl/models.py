"""Learned Q-tables. For a tabular agent the Q-table *is* the model: with it you can act greedily
without any training (see experiments/demo.py)."""
import numpy as np

from .paths import MODELS_DIR, slug


def save_model(env_name, algo, Q):
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    np.save(MODELS_DIR / f"{env_name}__{slug(algo)}.npy", np.asarray(Q, dtype=np.float64))


def load_model(env_name, algo):
    path = MODELS_DIR / f"{env_name}__{slug(algo)}.npy"
    if not path.exists():
        raise FileNotFoundError(f"{path} not found. Run `uv run python -m experiments.run_all` once first.")
    return np.load(path)
