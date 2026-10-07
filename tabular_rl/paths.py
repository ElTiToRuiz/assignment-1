"""Where everything is written. All outputs live under results/."""
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results"
CACHE_DIR = RESULTS / "cache"      # raw training results (.npz), so we never retrain by accident
MODELS_DIR = RESULTS / "models"    # final Q-tables, one per environment x algorithm
FIGURES = RESULTS / "figures"
TABLES = RESULTS / "tables"


def slug(text):
    """'Double Q-learning' -> 'double-q-learning', safe to use in file names."""
    return re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")
