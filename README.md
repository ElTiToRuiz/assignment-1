# Assignment 1: Tabular Reinforcement Learning

**Authors:** Igor Ruiz, Andoni Garrido

Reinforcement Learning, Universidad de Deusto. **SARSA and Q-learning** on the class 3x4 gridworld, deterministic and stochastic (the minimum requirement). It also covers:
- a comparison of **6 tabular algorithms**;
- a second environment (**Cliff Walking**);
- **Optuna** hyper-parameter tuning;
- a deeper **analysis of the training process**.

All results are **cached**: the plots and tables are regenerated in a few seconds without training.

---

## Setup

The project is managed with [uv](https://docs.astral.sh/uv/). Dependencies are in `pyproject.toml`, and the exact versions are pinned in `uv.lock`. Python ≥ 3.11.

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh   # install uv (once)
uv sync                                           # creates .venv with the locked versions
```

Every command below runs through `uv run`, so there is no need to activate the virtual environment.

Libraries beyond the ones used in class: `optuna` (tuning), `pandas` (tables), `pytest` (tests). The slides are built with Node.js (`pptxgenjs`), only needed to rebuild them.

## How to run

| Command | What it does |
|---|---|
| `uv run python -m experiments.run_all` | Rebuilds every figure and table **from the cache** (~10 s, no training) |
| `uv run python -m experiments.run_all --retrain` | Trains everything again from scratch (~15 min on 8 cores) |
| `uv run python -m experiments.run_all --only exp1 exp3` | Runs only some experiments |
| `uv run python -m experiments.demo` | Loads a **saved model** (Q-table) and plays it in the class pygame `GridworldEnv` |
| `uv run python -m experiments.demo --env gridworld_slippery --algo SARSA --render ansi --episodes 3` | Same, in the terminal |
| `cd docs/presentation && npm install && npm run build` | Builds `docs/presentation.pptx` (the 5-minute talk + backup slides) from the results. Needs Node.js |
| `uv run python -m experiments.tune` | Optuna hyper-parameter search (about 1 CPU-hour; resumable, so it can be stopped and rerun). `--quick` checks it works in seconds |
| `uv run pytest -q` | Runs the tests (DP ground truth, policy iteration = value iteration, env equivalence, convergence, Expected SARSA = Q-learning at ε=0, cache, statistics, search space) |

## Repository layout

```
class_code/              env.py + tools.py from class (the GridworldEnv)
tabular_rl/              the library
  envs.py                the environments (class gridworld, Cliff Walking) in the class P format
  simulator.py           plays an environment step by step: all the agents ever see
  planning.py            value / policy iteration and exact policy evaluation (the answer key)
  agents/
    base.py              epsilon-greedy, the epsilon schedule, the step size, run bookkeeping
    monte_carlo.py       Monte Carlo control (1/N or constant α, exploring starts)
    sarsa.py             SARSA, Expected SARSA, n-step SARSA (on-policy)
    q_learning.py        Q-learning, Double Q-learning (off-policy)
  metrics.py             how a Q-table is graded against Q*
  training.py            many seeds in parallel + the disk cache
  models.py              save / load Q-tables
  plotting.py            figure style and the shared plots
  stats.py               bootstrap confidence intervals and permutation test
experiments/             exp1 ... exp6, tune.py (Optuna), run_all.py, demo.py
results/
  cache/                 training results (.npz) with a readable .json next to each
  optuna/                the Optuna studies (SQLite)
  models/                final Q-table per environment x algorithm (.npy)
  figures/, tables/      everything shown below
docs/                    presentation.pptx (+ presentation/build.js that makes it), THEORY.md (notes for the oral part)
tests/
```

## Method

- **Environments**:
  - **Gridworld 3x4** from class (`class_code/env.py`): +1 goal, −1 pit, one wall. Deterministic or slippery (80% intended move, 10% each perpendicular).
  - **Cliff Walking 4x12** (Sutton & Barto): −1 per step; −100 and back to the start on the cliff.
  - Both are written in the same `P[s][a] = [(prob, s', r, done)]` format.
- **Model-free agents**: they only use `reset()` / `step()`. The model `P` is used **only** by value iteration, to get the ground truth Q\*, and by exact policy evaluation, for the metrics.
- **Exploration**: ε-greedy with random tie-breaking and exponential decay ε_k = max(ε_min, ε_0·decay^k). γ = 0.99.
- **Statistics**: every configuration is run with **20 seeds**. The plots show median + IQR for noisy curves, or mean ± std.
- **Metrics**:
  - **training return**, which includes exploration;
  - **regret of the greedy policy**, V\*(s0) − V^π(s0), computed exactly;
  - **RMSE of Q on the optimal actions**, compared with Q\*;
  - **% of states with an optimal greedy action**.

Default hyper-parameters:

| Environment | Episodes | α | ε (start, min, decay) |
|---|---|---|---|
| Gridworld deterministic | 3000 | 0.1 | (1.0, 0.05, 0.995) |
| Gridworld slippery | 10000 | 0.1 | (1.0, 0.05, 0.9995) |
| Cliff Walking | 1000 | 0.5 | (0.1, 0.1, 1.0), i.e. constant 0.1 |
| Gridworld slippery, convergent schedule (exp1 only) | 30000 | 0.5 / (1 + 0.005·N(s,a)) | (1.0, 0.0, 0.9998) |

---

## Results

### 1. Minimum requirement: SARSA and Q-learning (`exp1`)

![values](results/figures/exp1_values_policies.png)
![curves](results/figures/exp1_learning_curves.png)

| Environment | Algorithm | Return (last 100 ep) | RMSE Q (optimal a) | Optimal actions % | Regret at s0 | Seeds with π\* |
|---|---|---|---|---|---|---|
| Deterministic | SARSA | 0.999 | 0.549 | 81.1 | 0.0000 | 20/20 |
| Deterministic | Q-learning | 0.993 | 0.204 | 94.4 | 0.0000 | 20/20 |
| Slippery | SARSA | 0.979 | 0.374 | 85.0 | 0.0512 | 15/20 |
| Slippery | Q-learning | 0.987 | **0.011** | **97.8** | **0.0039** | **19/20** |
| Slippery, convergent schedule | SARSA | 0.999 | 0.244 | 88.9 | **0.0000** | **20/20** |
| Slippery, convergent schedule | Q-learning | 1.000 | **0.008** | **99.4** | **0.0000** | **20/20** |

**Deterministic.** Both algorithms find the optimal path (up, up, right, right, right) in every seed.

**Slippery.** The optimal policy changes:
- in state 6, go LEFT, away from the pit;
- in state 11, push DOWN into the wall, so a slip can never take the agent into the pit.

Q-learning recovers Q\* almost exactly. SARSA learns the value of its own ε-greedy policy, so its values next to the pit are lower. That is the expected on-policy behaviour (Bellman equation vs BOE), not an error.

**Why SARSA misses π\* in 5 seeds with the default schedule, and the fix.** With ε_min = 0.05 and a constant α:
- SARSA learns Q of the ε-greedy policy, not Q\*;
- the states next to the pit (6 and 11) are rarely visited, so their Q values stay noisy and the greedy action there can be wrong.

The theory gives two conditions for SARSA to converge to the optimum:
- **GLIE** exploration: every pair (s,a) visited infinitely often and ε → 0;
- a **Robbins-Monro** step size: Σα = ∞, Σα² < ∞.

The "convergent schedule" meets both:
- ε decays to 0 (ε_0 = 1, decay = 0.9998);
- α(s,a) = 0.5 / (1 + 0.005·N(s,a)), where N(s,a) is the visit count (`agents.step_size`);
- 30 000 episodes.

With it, **both algorithms reach the optimal policy in 20/20 seeds**. SARSA still needs about 15× more episodes than Q-learning to settle (median 13 510 vs 900), which is the cost of learning on-policy.

![convergent](results/figures/exp1_convergent_schedule.png)

The figure also shows the price of these guarantees. Q-learning with the default schedule reaches π\* in most seeds within about 500 episodes, sooner than with the convergent schedule. The default schedule just never gets *all* seeds there, because ε_min = 0.05 and a constant α keep it noisy. Theory guarantees the limit, not the speed.

RMSE over *all* actions stays high in the deterministic world. Once the agent has found the path, it rarely visits bad actions or far states again (`exp5d`). That is why we report the error on the optimal actions.

### 2. All tabular algorithms (`exp2`)

![bars](results/figures/exp2_summary_bars.png)

Per-environment curves and policies are in `results/figures/exp2_curves_*.png` and `exp2_policies_*.png`. The full table is in `results/tables/exp2_all_algorithms.csv`.

- **Q-learning** gives the most accurate Q\* and the lowest regret in all 3 environments.
- **Expected SARSA** is the best on-policy method: it has lower variance than SARSA and the best online return in Cliff Walking (−19.9).
- **Double Q-learning** removes maximisation bias, but each table gets half the updates, so it is slower. In Cliff Walking it is unstable in a few seeds: the median return is −25 but the mean is −221.
- **n-step SARSA (n=3)** propagates the sparse reward faster, but the multi-step returns are noisier under slippery transitions.
- **Monte Carlo** works in the gridworld but **fails in Cliff Walking** (return ≈ −374). With a bad initial policy, episodes hit the 500-step limit, so there is no informative return. "MC needs finished episodes" is exactly the limitation from the slides.

### 3. On-policy vs off-policy: Cliff Walking (`exp3`)

![cliff](results/figures/exp3_cliff_walking.png)

| Algorithm | Greedy path | Online return (ε=0.1) | Greedy policy optimal |
|---|---|---|---|
| SARSA | 17 steps (safe, away from the edge) | **−25.8** | 0/20 seeds |
| Q-learning | 13 steps (along the cliff edge) | −51.9 | **20/20 seeds** |

This is the classic result. Q-learning learns the optimal path but keeps falling off while exploring. SARSA includes its own exploration in its target, so it prefers the safe path.

### 4. Hyper-parameter tuning with Optuna (`exp4`)

The search is `experiments/tune.py` and the analysis is `exp4`.
- **Studies:** one Optuna study per environment × algorithm. Environments: slippery gridworld, Cliff Walking. Algorithms: SARSA, Expected SARSA, Q-learning, Double Q, n-step SARSA, constant-α MC.
- **Search space:**
  - α, plus optionally the Robbins–Monro decay α/(1+k·N(s,a));
  - the full ε schedule: start, minimum, and half-life in episodes;
  - n, for n-step SARSA.
- **Two objectives**, both regret normalised by |V\*(s0)|, so environments are comparable:
  - **speed**: mean regret during training;
  - **exactness**: regret over the last 100 episodes.

  They pull against each other: a configuration can learn fast and still end up slightly wrong. So we keep the whole **Pareto front** and pick the point with the lowest speed + exactness.
- **Sampler:** multivariate TPE, 150 trials, 5 tuning seeds. Trial 0 is always the default configuration.
- **Storage:** SQLite (`results/optuna/studies.db`), so it is resumable and runs in parallel across processes.
- **Evaluation** (`exp4`) on **20 held-out seeds**:
  - bootstrap 95% confidence intervals;
  - a permutation test (default vs tuned);
  - PED-ANOVA parameter importance.

**Results.** All 12 studies, 150 trials each.

![tuned](results/figures/exp4_default_vs_tuned.png)

Default and tuned use the same training budget: 3000 episodes in the slippery gridworld, 500 in Cliff Walking. Both are evaluated on the 20 held-out seeds. In the tables, "speed" is the mean regret during training, "exact π\*" counts the seeds whose final greedy policy is exactly optimal, and p is the permutation test on speed.

*Gridworld (slippery 80/10/10)*

| Algorithm | Speed: default → tuned | p | Exact π\*: default → tuned | Tuned hyper-parameters |
|---|---|---|---|---|
| SARSA | 0.069 → 0.040 | 0.03 | 4 → 6 /20 | α = 0.011; ε 0.68 → 0.018 |
| Expected SARSA | 0.034 → 0.020 | 0.02 | 4 → **11** /20 | α = 0.45 with decay k = 0.005; ε 0.12 → 0.002 |
| Q-learning | 0.010 → **0.003** | <0.001 | 19 → **20** /20 | α = 0.49 with decay k = 0.05; ε 0.82 → 0.001 |
| Double Q-learning | 0.073 → 0.016 | <0.001 | 10 → 13 /20 | α = 0.13 with decay k = 0.009; ε 0.88 → 0.13 |
| n-step SARSA | 0.183 → 0.032 | <0.001 | 0 → **12** /20 | **n = 1**; α = 0.34 with decay k = 0.002 |
| MC constant-α | 0.400 → 0.046 | <0.001 | 0 → 0 /20 | α = 0.016; ε 0.59 → 0.001 |

*Cliff Walking*

| Algorithm | Speed: default → tuned | p | Exact π\*: default → tuned | Tuned hyper-parameters |
|---|---|---|---|---|
| SARSA | 2.39 → 0.90 | <0.001 | 0 → 0 /20 | α = 0.25 with decay k = 0.005; ε 0.90 → 0.03 |
| Expected SARSA | 0.86 → 0.54 | <0.001 | 0 → **18** /20 | α = 0.75; ε 0.07 → 0.002 (fast decay) |
| Q-learning | 0.59 → **0.09** | <0.001 | 20 → 20 /20 | α = 0.96 with decay k = 0.03; ε 0.78 → **0.27** |
| Double Q-learning | 3.53 → 0.67 | <0.001 | 0 → **12** /20 | α = 0.30; ε 0.81 → 0.07 |
| n-step SARSA | 2.68 → 0.53 | <0.001 | 0 → 0 /20 | **n = 4**; α = 0.20 with decay k = 0.003 |
| MC constant-α | 2.80 → 1.82 | <0.001 | 0 → 0 /20 | α = 0.076; ε 0.53 → 0.001 |

The full numbers (95% CIs, exactness p-values) are in `results/tables/exp4_default_vs_tuned.csv`.

**Findings:**
1. **Tuning helps every algorithm.** Speed improves in all 12 cases: p < 0.05 in all of them and p < 0.001 in 10. The number of seeds that end exactly at π\* grows too:
   - Expected SARSA in Cliff Walking: 0 → 18;
   - n-step SARSA in the slippery world: 0 → 12;
   - Double Q in Cliff Walking: 0 → 12.

   Q-learning reaches π\* in 20/20 seeds in only 3000 episodes. The hand-made convergent schedule in `exp1` needed 30 000.
2. **Optuna rediscovered the theory.** The Robbins–Monro step size α/(1+k·N(s,a)) was chosen in **8 of the 10** TD configurations, and almost every tuned schedule lets ε fall close to 0 (GLIE).
3. **On-policy vs off-policy in Cliff Walking.**
   - Expected SARSA (on-policy) only finds the optimal edge path (18/20 seeds) when ε decays quickly to ≈ 0. With exploration left on, its target keeps preferring the safe path (`exp3`).
   - Q-learning (off-policy) is the opposite: Optuna keeps **ε_min = 0.27**, because exploration does not change what Q-learning learns (the greedy target). More exploration simply means more data.
4. **n-step SARSA: the best n depends on the environment.**
   - In the slippery world Optuna picks **n = 1**, i.e. plain SARSA: with noisy transitions, longer returns add variance.
   - In the deterministic Cliff Walking it picks **n = 4**: the reward propagates faster along a long path, with no extra noise.
5. **What matters** (`exp4_param_importance.png`): α or its decay for Q-learning and Double Q in the slippery world, and the ε schedule (half-life) for Expected SARSA and Double Q in Cliff Walking.
6. **Honest caveats:**
   - Each algorithm only gets 3000 (slippery) or 500 (Cliff) episodes here, fewer than in `exp2`. The default numbers in this table are therefore worse than in `exp2`, and this is a comparison at a **fixed budget**.
   - Over-fitting to the 5 tuning seeds still shows. Tuned SARSA in the slippery world had a final regret of 0 on the tuning seeds but 0.03 on the held-out ones.
   - On-policy methods that keep exploring (SARSA, n-step SARSA, MC in Cliff Walking) cannot reach the exact optimal path. That is a property of the algorithm, not a tuning failure.

Other figures:
- [`exp4_pareto.png`](results/figures/exp4_pareto.png): every trial with its Pareto front;
- [`exp4_param_importance.png`](results/figures/exp4_param_importance.png): parameter importance.

### 5. Analysis of the training process (`exp5`)

| | |
|---|---|
| ![alpha](results/figures/exp5a_alpha_sensitivity.png) | ![eps](results/figures/exp5b_exploration.png) |
| ![biasvar](results/figures/exp5c_bias_variance.png) | ![visits](results/figures/exp5d_state_visits.png) |

- **Learning rate.** In the stochastic world, a large α (≥ 0.6) makes Q chase single noisy samples: high regret and high error. A very small α (0.01) has not converged after 3000 episodes. Q-learning is most accurate around α = 0.1–0.3; SARSA prefers smaller α, because its target is noisier.
- **Exploration.** A small constant ε gives a high training return but learns Q poorly, while a large ε explores but behaves worse. Decaying ε gets both.
- **MC vs TD.**
  - MC converges to Q of the ε-greedy policy (unbiased for that policy) but its spread across seeds decreases slowly (high variance).
  - Q-learning bootstraps towards Q\* (bias → 0).
- **Coverage.** States off the optimal path are visited orders of magnitude less often, so their Q values stay inaccurate.

---

### 6. Failure analysis: why some methods fail in Cliff Walking (`exp6`)

![failures](results/figures/exp6_failure_analysis.png)

Cliff Walking pays −1 per step, so with γ = 0.99 a policy that **never reaches the goal is worth −1/(1−γ) = −100**. The first episodes are very long: hundreds of steps and many falls. If the estimates collapse to that −100 plateau before the goal is found, the following happens:
- every action looks equally bad;
- the greedy policy loops;
- the goal signal never propagates back.

| Variant | Stuck seeds | Median return (last 100 ep) | Greedy regret (median) |
|---|---|---|---|
| Monte Carlo, step 1/N (class version) | **14/20** | −507.9 | 87.75 |
| MC, constant α = 0.1 | 0/20 | −39.0 | 5.14 |
| MC, constant α = 0.1 + exploring starts | 0/20 | −27.5\* | 3.46 |
| Double Q-learning, α = 0.5 | **3/20** | −25.1 (mean −221) | 3.46 |
| Double Q-learning, α = 0.1 | 0/20 | −26.1 | 3.46 |
| SARSA, α = 0.5 (reference) | 0/20 | −24.4 | 3.46 |
| Q-learning, α = 0.5 (reference) | 0/20 | −50.7 | **0.00** |

\* With exploring starts the episodes begin at random states, so the return is not directly comparable.

**Monte Carlo.** With the class step size 1/N(s,a), Q is the mean of *all* returns ever seen. The catastrophic returns of the first, random episodes (about −1500) stay in that mean forever. In control the policy keeps improving, so the target is **non-stationary**. A constant α forgets old returns and fixes the collapse. Exploring starts (class *Monte-Carlo ES*) helps further: every (s,a) keeps being tried from short episodes near the goal.

**Double Q-learning.** Each table gets only half of the updates. With a large α (0.5), in 3 seeds the Q values at the start state fall to −100 (red curves) and never recover. With α = 0.1, 0/20 seeds get stuck.

A greedy regret of 3.46 is the safe path (17 steps instead of 13). Every on-policy method converges to it, because its target includes the ε = 0.1 exploration (see `exp3`).

## Conclusions

1. SARSA and Q-learning both solve the deterministic and the stochastic gridworld.
2. Q-learning (BOE) converges to Q\*. SARSA (Bellman equation) converges to the value of the ε-greedy policy, which makes it **safer while exploring** (Cliff Walking).
3. Stochastic transitions need a decaying α and ε → 0 (Robbins–Monro + GLIE) for SARSA and Q-learning to reach π\* in every seed.
4. TD bootstraps: low variance, biased. MC uses real returns: unbiased, high variance, and episodes must terminate.
5. Expected SARSA reduces variance. Double Q-learning removes maximisation bias but learns more slowly.
6. Hyper-parameters matter.
   - Multi-objective Optuna improved every algorithm significantly, and Q-learning reached π\* in 20/20 seeds with 10× fewer episodes.
   - It independently chose the theory's convergence conditions: Robbins–Monro α and ε → 0.
   - Tuning must still be validated on held-out seeds.
7. The failures in Cliff Walking are explained by theory. Monte Carlo's 1/N mean cannot forget the first catastrophic returns (the target is non-stationary in control), and too large an α makes Double Q collapse to the −100 "never arrive" plateau. Constant-α MC, exploring starts and a smaller α fix both.

Theory Q&A for the oral part: [`docs/THEORY.md`](docs/THEORY.md). Slides: [`docs/presentation.pptx`](docs/presentation.pptx).
