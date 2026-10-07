"""Extra: analysis of the Optuna studies produced by `uv run python -m experiments.tune`.

For every (environment, algorithm) study it:
  1. plots all trials in the (speed, exactness) plane with the Pareto front and the chosen trial;
  2. re-trains the default and the tuned configuration on 20 held-out seeds (cached) and compares
     them with bootstrap 95% confidence intervals and a permutation test;
  3. estimates which hyper-parameters matter (PED-ANOVA importance).
The studies themselves are never re-run here, so this is fast once the cache exists.
Set RL_OPTUNA_DB=<file> to read another database.
"""
import json
import os
from pathlib import Path

import warnings

import matplotlib.pyplot as plt
from matplotlib.ticker import NullFormatter
import numpy as np
import optuna
import pandas as pd

from tabular_rl.envs import make_spec
from tabular_rl.runner import train
from tabular_rl.stats import bootstrap_ci, permutation_test
from tabular_rl.viz import COLORS, ENV_NAMES, save

from .common import N_JOBS, SEEDS, save_table
from .tune import ALGOS, BUDGET, ENVS, STORAGE, chosen_trial, default_kwargs, storage_url, study_name

PARAMS = ["alpha", "use_alpha_decay", "alpha_decay", "eps_start", "eps_min", "eps_halflife", "n"]


def load(storage_path):
    if not Path(storage_path).exists():
        return {}
    storage = storage_url(storage_path)
    names = set(optuna.get_all_study_names(storage))
    studies = {}
    for env in ENVS:
        for algo in ALGOS:
            if study_name(env, algo) in names:
                st = optuna.load_study(study_name=study_name(env, algo), storage=storage)
                if any(t.state == optuna.trial.TrialState.COMPLETE for t in st.trials):
                    studies[(env, algo)] = st
    return studies


def tuned_kwargs(study):
    kw = json.loads(chosen_trial(study).user_attrs["kwargs"])
    kw["eps"] = tuple(kw["eps"])
    return kw


def held_out(env, algo, kw, n_episodes):
    """Normalised regret per held-out seed: (speed, exactness, exactly optimal at the end)."""
    out = train(algo, env, SEEDS, n_jobs=N_JOBS, n_episodes=n_episodes, **kw)
    norm = abs(out["V_star"][make_spec(env).start_state])
    reg = np.clip(out["regret"], 0, None) / norm
    return reg.mean(1), reg[:, -10:].mean(1), out["regret"][:, -1] < 1e-9


def pareto_figure(studies, envs, algos):
    fig, axes = plt.subplots(len(envs), len(algos), figsize=(3.3 * len(algos), 3.2 * len(envs)),
                             layout="constrained", squeeze=False)
    for i, env in enumerate(envs):
        for j, algo in enumerate(algos):
            ax = axes[i, j]
            st = studies.get((env, algo))
            if st is None:
                ax.axis("off"); continue
            done = [t for t in st.trials if t.state == optuna.trial.TrialState.COMPLETE]
            v = np.array([t.values for t in done]) + 1e-5  # log axes: shift exact zeros
            front = {t.number for t in st.best_trials}
            on = np.array([t.number in front for t in done])
            ax.scatter(v[~on, 0], v[~on, 1], s=12, color="lightgray", label="trial")
            ax.scatter(v[on, 0], v[on, 1], s=22, color=COLORS[algo] if algo in COLORS else "C0", label="Pareto front")
            c = chosen_trial(st)
            ax.scatter(*(np.array(c.values) + 1e-5), marker="*", s=180, color="gold", edgecolor="black", label="chosen")
            d = [t for t in done if t.number == 0]
            if d:
                ax.scatter(*(np.array(d[0].values) + 1e-5), marker="s", s=70, facecolor="none", edgecolor="black",
                           lw=1.5, label="default", zorder=5)
            ax.set(xscale="log", yscale="log", title=f"{algo}\n{ENV_NAMES[env]}")
            for axis in (ax.xaxis, ax.yaxis):
                axis.set_minor_formatter(NullFormatter())
            if i == len(envs) - 1:
                ax.set_xlabel("speed: mean regret")
            if j == 0:
                ax.set_ylabel("exactness: final regret")
    axes[0, 0].legend(fontsize=7, loc="lower right")
    fig.suptitle("Optuna multi-objective search (TPE): every trial, Pareto front, default and chosen configuration")
    save(fig, "exp4_pareto.png")


def comparison(studies, envs, algos):
    rows, res = [], {}
    for env in envs:
        for algo in algos:
            st = studies.get((env, algo))
            if st is None:
                continue
            tuned = tuned_kwargs(st)
            d = held_out(env, algo, default_kwargs(env, algo), BUDGET[env])
            t = held_out(env, algo, tuned, BUDGET[env])
            res[(env, algo)] = (d, t)
            for tag, (sp, ex, opt), cfg in [("default", d, default_kwargs(env, algo)), ("tuned", t, tuned)]:
                lo, hi = bootstrap_ci(sp)
                rows.append({"env": ENV_NAMES[env], "algorithm": algo, "config": tag,
                             "hyper-parameters": json.dumps({k: (round(v, 4) if isinstance(v, float) else
                                                                 [round(x, 4) for x in v] if isinstance(v, tuple) else v)
                                                             for k, v in cfg.items()}),
                             "speed (mean regret)": round(float(sp.mean()), 4),
                             "95% CI": f"[{lo:.4f}, {hi:.4f}]",
                             "exactness (final regret)": round(float(ex.mean()), 4),
                             "seeds exactly optimal": f"{int(opt.sum())}/{len(opt)}",
                             "p-value speed (vs default)": "" if tag == "default" else f"{permutation_test(d[0], t[0]):.4f}",
                             "p-value exactness (vs default)": "" if tag == "default" else f"{permutation_test(d[1], t[1]):.4f}"})
    save_table(pd.DataFrame(rows), "exp4_default_vs_tuned")

    fig, axes = plt.subplots(2, len(envs), figsize=(8 * len(envs), 8.5), layout="constrained", squeeze=False)
    w = 0.38
    for i, env in enumerate(envs):
        al = [a for a in algos if (env, a) in res]
        for k, metric in enumerate(["speed (mean regret)", "seeds exactly optimal (%)"]):
            ax = axes[k, i]
            for j, a in enumerate(al):
                d, t = res[(env, a)]
                for off, tag, vals, hatch in [(-w / 2, "default", d, "//"), (w / 2, "tuned", t, None)]:
                    if k == 0:
                        m = vals[0].mean(); lo, hi = bootstrap_ci(vals[0])
                        ax.bar(j + off, max(m, 1e-5), w, color=COLORS.get(a, "C0"), alpha=.45 if tag == "default" else 1,
                               hatch=hatch, edgecolor="black", lw=.5, yerr=[[max(m - lo, 0)], [hi - m]], capsize=2,
                               label=tag if j == 0 else None)
                    else:
                        ax.bar(j + off, 100 * vals[2].mean(), w, color=COLORS.get(a, "C0"), alpha=.45 if tag == "default" else 1,
                               hatch=hatch, edgecolor="black", lw=.5, label=tag if j == 0 else None)
                if k == 0:
                    p = permutation_test(d[0], t[0])
                    star = "***" if p < 1e-3 else "**" if p < 1e-2 else "*" if p < 0.05 else "n.s."
                    top = max(d[0].mean() + (bootstrap_ci(d[0])[1] - d[0].mean()), bootstrap_ci(t[0])[1], 1e-5)
                    ax.text(j, top * 1.25, star, ha="center", va="bottom", fontsize=9)
            ax.set_xticks(range(len(al)), [a.replace(" ", "\n", 1) for a in al], fontsize=9)
            ax.grid(axis="x", visible=False)
            if k == 0:
                ax.set_yscale("log")
                lo_, hi_ = ax.get_ylim()
                ax.set_ylim(lo_, hi_ * 4)  # room for the significance marks
                ax.set(title=f"{ENV_NAMES[env]}: mean regret on 20 held-out seeds (95% CI)",
                       ylabel="normalised regret (lower = faster)")
            else:
                ax.set(ylim=(0, 105), title="Seeds whose final greedy policy is exactly π*", ylabel="% of seeds")
            ax.legend(loc="upper left")
    fig.suptitle("Default vs Optuna-tuned hyper-parameters (permutation test: * p<0.05, ** p<0.01, *** p<0.001)")
    save(fig, "exp4_default_vs_tuned.png")


def importance_figure(studies, envs, algos):
    fig, axes = plt.subplots(1, len(envs), figsize=(7 * len(envs), 0.6 * len(algos) + 2.2), layout="constrained", squeeze=False)
    rows = []
    for i, env in enumerate(envs):
        al = [a for a in algos if (env, a) in studies]
        M = np.full((len(al), len(PARAMS)), np.nan)
        for j, a in enumerate(al):
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    imp = optuna.importance.get_param_importances(
                        studies[(env, a)], evaluator=optuna.importance.PedAnovaImportanceEvaluator(),
                        target=lambda t: sum(t.values))
            except Exception as e:  # too few trials, constant objective, ...
                print(f"  [warn] importance {env}/{a}: {e}")
                continue
            for k, p in enumerate(PARAMS):
                if p in imp:
                    M[j, k] = imp[p]
                    rows.append({"env": ENV_NAMES[env], "algorithm": a, "parameter": p, "importance": round(imp[p], 3)})
        ax = axes[0, i]
        im = ax.imshow(M, cmap="Blues", vmin=0, vmax=max(np.nanmax(M), 1e-9) if np.isfinite(M).any() else 1)
        for (r, c), val in np.ndenumerate(M):
            ax.text(c, r, "—" if np.isnan(val) else f"{val:.2f}", ha="center", va="center", fontsize=8)
        ax.set_xticks(range(len(PARAMS)), PARAMS, rotation=30, ha="right")
        ax.set_yticks(range(len(al)), al)
        ax.set_title(ENV_NAMES[env]); ax.grid(False)
        fig.colorbar(im, ax=ax, shrink=.8)
    fig.suptitle("Which hyper-parameters matter? (PED-ANOVA importance for speed + exactness)")
    save(fig, "exp4_param_importance.png")
    if rows:
        save_table(pd.DataFrame(rows), "exp4_param_importance")


def main():
    print("exp4: Optuna studies")
    path = os.environ.get("RL_OPTUNA_DB", str(STORAGE))
    studies = load(path)
    if not studies:
        print(f"  no Optuna studies in {path}. Run `uv run python -m experiments.tune` first (ideally on a big machine).")
        return
    envs = [e for e in ENVS if any(k[0] == e for k in studies)]
    algos = [a for a in ALGOS if any(k[1] == a for k in studies)]
    rows = [{"env": ENV_NAMES[e], "algorithm": a, "trials": len(st.trials), "pareto size": len(st.best_trials),
             "chosen speed": round(chosen_trial(st).values[0], 4), "chosen exactness": round(chosen_trial(st).values[1], 4),
             "tuned hyper-parameters": chosen_trial(st).user_attrs["kwargs"]} for (e, a), st in studies.items()]
    save_table(pd.DataFrame(rows), "exp4_tuned_params")
    pareto_figure(studies, envs, algos)
    importance_figure(studies, envs, algos)
    comparison(studies, envs, algos)


if __name__ == "__main__":
    main()
