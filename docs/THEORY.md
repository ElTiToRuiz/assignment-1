# Tabular RL: theory notes for the oral questions

Short answers to the questions most likely to come up. Each one is linked to this repo's code and results.

---

## 1. MDP, return, value functions
- **MDP** = (S, A, P, R, γ). `env.P[s][a] = [(prob, s', r, done), ...]` *is* P(s', r | s, a).
- **Return**: G_t = r_{t+1} + γ r_{t+2} + γ² r_{t+3} + … We use γ = 0.99. γ < 1 keeps sums finite and makes shorter paths preferable.
- **State value** V^π(s) = E_π[G_t | s_t = s]. **Action value** Q^π(s,a) = E_π[G_t | s_t = s, a_t = a].
- V^π(s) = Σ_a π(a|s) Q^π(s,a). For a greedy policy, V*(s) = max_a Q*(s,a).

## 2. Bellman equation (BE) vs Bellman optimality equation (BOE)
- **BE** (evaluates a fixed policy π):
  Q^π(s,a) = Σ_{s',r} p(s',r|s,a) [ r + γ Σ_{a'} π(a'|s') Q^π(s',a') ]
- **BOE** (characterises the optimum):
  Q*(s,a) = Σ_{s',r} p(s',r|s,a) [ r + γ max_{a'} Q*(s',a') ]
- **Policy iteration** applies the BE (evaluation), then improves greedily. **Value iteration** applies the BOE directly.
- Both need the **model** P. We implement both in `tabular_rl/dp.py`, and a test checks that they give the same V*. Number of sweeps to converge:

  | | Value iteration (tol 1e-8) | Policy iteration |
  |---|---|---|
  | Deterministic grid | 6 | 6 |
  | Slippery grid | **145** | **7** |
  | Cliff Walking | 15 | 15 |

  This matches the comparison table from class:
  - **value iteration** converges only asymptotically; each sweep is cheap, but it needs many of them;
  - **policy iteration** converges exactly in a few steps; each step is expensive, because it solves a linear system.
- Here, `tabular_rl/dp.py` uses value iteration only to compute the *ground truth* Q* for the metrics; the agents never see P.

## 3. Model-free: Monte Carlo vs TD
| | Monte Carlo | TD (SARSA / Q-learning) |
|---|---|---|
| Target | real return G_t | r + γ·(estimate of next value) → **bootstrapping** |
| Update | at episode end | every step (online, incremental) |
| Tasks | episodic only | episodic and continuing |
| Bias | unbiased target | biased: depends on the current estimate |
| Variance | high: sum of many random rewards | low: one random step |

- Evidence: `exp5c_bias_variance.png`. MC's Q(s0,a*) has the largest spread across seeds.
- In Cliff Walking, MC fails completely (`exp2`). Episodes under a bad policy hit the 500-step limit, so there's no informative return. "MC needs finished episodes" is exactly the con from the slides.

## 4. TD error and the update rule
Q(s,a) ← Q(s,a) + α [ target − Q(s,a) ], where δ = target − Q(s,a) is the **TD error**.
This works like a gradient step: it moves the estimate towards the Bellman target with learning rate α.

## 5. SARSA (on-policy TD control)
- Target: r + γ Q(s', a'), where a' is the action the ε-greedy policy **really takes next**. The name comes from the tuple (S, A, R, S', A').
- It is a sample of the **Bellman equation** for the ε-greedy policy, so it learns Q^{π_ε}: the value of the policy *including its exploration*.
- Code: `tabular_rl/agents.py::sarsa`. a' is sampled **before** the update and then executed.
- It converges to Q* only if exploration fades out (GLIE) and the step sizes satisfy Robbins–Monro (Σα = ∞, Σα² < ∞).
- **We verified this (`exp1`).** In the slippery world, with the default schedule (ε_min = 0.05, constant α), SARSA reaches π* in only 15/20 seeds:
  - its greedy action is wrong in states 6 and 11, next to the pit;
  - it rarely visits those states, and it learns Q of the ε-greedy policy.

  With ε → 0 and α(s,a) = 0.5 / (1 + 0.005·N(s,a)) it reaches π* in **20/20** seeds. The cost: it needs about 15× more episodes than Q-learning.

## 6. Q-learning (off-policy TD control)
- Target: r + γ max_{a'} Q(s', a'), a sample of the **BOE**. It learns Q* directly, whatever policy collected the data.
- **Behaviour policy** = ε-greedy (explores). **Target policy** = greedy (what is learned). Because these differ, it is **off-policy**.
- It converges to Q* if every (s,a) is visited infinitely often and α satisfies Robbins–Monro.

## 7. Why SARSA and Q-learning differ (Cliff Walking, `exp3`)
- Q-learning learns the **optimal** path along the cliff edge (13 steps): its target assumes greedy behaviour from then on.
- SARSA learns the **safe** path one row up or more. Its target includes the ε random actions, which near the edge sometimes mean falling (−100).
- Result: **online** return with ε = 0.1 is better for SARSA (about −26 vs −52). The **greedy** policy is optimal only for Q-learning (regret 0).
- In the slippery gridworld you see the same pattern: SARSA's values near the pit are lower than Q*. That is the cost of its own exploration.

## 8. Expected SARSA
- Target: r + γ Σ_{a'} π(a'|s') Q(s',a'), the expectation over the next action instead of a sample.
- This gives lower variance than SARSA at the same bias. With π greedy (ε = 0) it equals Q-learning; `tests/test_basic.py` checks this exactly.
- It was the best on-policy method in our runs. In Cliff Walking it had the best online return.

## 9. Double Q-learning (maximisation bias)
- max_a Q(s',a) over *noisy* estimates is biased upward: E[max] ≥ max E.
- Double Q keeps two tables. One picks the argmax and the other evaluates it, which removes that bias.
- Cost: each table gets half the updates, so it learns more slowly. That shows in the deterministic grid and in Cliff Walking.

## 10. n-step SARSA
- Target: r_{t+1} + γ r_{t+2} + … + γ^{n−1} r_{t+n} + γ^n Q(s_{t+n}, a_{t+n}).
- n = 1 is SARSA; n → ∞ is Monte Carlo. Intermediate n trades bias against variance and propagates the sparse goal reward faster.
- We use n = 3.

## 11. Exploration: ε-greedy and schedules
- With probability ε take a random action, otherwise the greedy one. Ties are broken randomly; important because Q starts at 0, so all actions tie at first.
- We use an exponential decay: ε_k = max(ε_min, ε_0 · decay^k).
- **GLIE** (Greedy in the Limit with Infinite Exploration) is what SARSA needs to reach Q*.
- `exp5b` shows the trade-off:
  - small constant ε gives good training return but slow, unreliable learning;
  - large ε learns Q well but behaves badly;
  - decay gives the best of both.

## 12. Learning rate α
- Constant α weights recent samples more. This is needed in non-stationary settings and gives faster learning, but Q keeps oscillating with stochastic transitions.
- α = 1/N(s,a) is the exact sample mean, which is what our Monte Carlo uses.
- `exp5a` shows the trade-off in the slippery world:
  - too large an α (≥ 0.6): Q follows single noisy transitions, so regret and error are high;
  - too small an α (0.01): still far from Q* after 3000 episodes;
  - Q-learning works best around α = 0.1–0.3. SARSA prefers smaller α, because its target also samples a'.

## 13. Deterministic vs stochastic environment
- Deterministic: one sample is enough, so α can be large. Every algorithm finds π* within about 20–500 episodes.
- Slippery (80/10/10): targets are noisy, so a smaller or decaying α and more episodes are needed.
- The optimal policy changes:
  - from state 6 it is LEFT, not UP (moving away from the pit);
  - from state 11 it is DOWN (bump into the wall, so a slip can never take the agent into the pit).

## 14. Metrics we use (and why)
- **Training return**: what the agent gets *while learning*. This includes exploration.
- **Regret of the greedy policy**: V*(s0) − V^π(s0), computed exactly with the model (policy evaluation). It measures *what the agent has learned*.
- **RMSE on optimal actions**: accuracy of Q where it matters for control. RMSE over all (s,a) is dominated by rarely visited bad actions (see `exp5d_state_visits.png`).
- 20 seeds per configuration. We plot median + IQR for noisy curves and mean ± std for errors.

## 15. Hyper-parameter tuning (Optuna)
- The TPE sampler proposes (α, ε_0, ε_min, decay). The objective is the area under the regret curve (speed of learning) on 5 tuning seeds.
- We re-evaluated on 20 **held-out** seeds to check for over-fitting to the tuning seeds. See the README table for the honest result.

## 16. What goes beyond the class slides, and how it relates to them
The assignment asks for "improvements/optimizations of seen algorithms". Each extra is a small change to something from class:

| Extra | Explanation in class notation |
|---|---|
| **Expected SARSA** | SARSA, with the sampled q(s',a') replaced by its expectation Σ_{a'} π(a'\|s') q(s',a'). That is Bellman equation (3) from slides 1.3 inside the TD target. Same fixed point as SARSA, lower variance. |
| **Double Q-learning** | Q-learning, where the max in the BOE target over *noisy* estimates overestimates (E[max] ≥ max E). Two tables: one chooses argmax, the other evaluates it. |
| **Decaying α(s,a)** | α / (1 + k·N(s,a)) satisfies Robbins–Monro. The MC "online mean" from slides 1.5 is the case α = 1/N. |
| **GLIE** | The ε-greedy policy from slides 1.5/1.6 with ε → 0, so that it is greedy in the limit while every (s,a) is still visited infinitely often. |
| **Regret** | V*(s0) − V^π(s0), where V* = max_π V_π from slides 1.2. V^π is computed exactly by solving the Bellman equation (I − γP_π)V = r_π. |
| **Median / IQR** | Plot statistics only: the band between the 25th and 75th percentile over the 20 seeds. It is robust to the few seeds with extreme values. |
| **TPE (Optuna)** | Optuna's default sampler. It models which hyper-parameters gave good vs bad trials and proposes new ones where the good ones concentrate. |
| **Held-out seeds** | Same idea as a train/test split: tune on 5 seeds, report on 20 different ones. |
