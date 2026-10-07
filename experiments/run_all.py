"""Run every experiment. Uses the cache in results/cache, so nothing is retrained unless --retrain.

    uv run python -m experiments.run_all                 # plots + tables from cache (seconds)
    uv run python -m experiments.run_all --retrain       # train everything again (~20-30 min)
    uv run python -m experiments.run_all --only exp1 exp3
"""
import argparse
import time

from tabular_rl import runner

from . import (exp1_main, exp2_compare_all, exp3_cliff_walking, exp4_optuna, exp5_training_analysis,
               exp6_failure_analysis)

EXPERIMENTS = {"exp1": exp1_main, "exp2": exp2_compare_all, "exp3": exp3_cliff_walking,
               "exp4": exp4_optuna, "exp5": exp5_training_analysis,
               "exp6": exp6_failure_analysis}

if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--retrain", action="store_true", help="ignore the cache and train again")
    p.add_argument("--only", nargs="+", choices=EXPERIMENTS, help="run only these experiments")
    args = p.parse_args()
    runner.RETRAIN = args.retrain
    t0 = time.time()
    for name in args.only or EXPERIMENTS:
        EXPERIMENTS[name].main()
    print(f"done in {time.time() - t0:.1f}s")
