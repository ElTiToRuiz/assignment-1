"""How we grade a Q-table. We know the true Q* (from planning.py), so we can measure exactly how far
the agent is from it, instead of only looking at noisy training returns."""
import numpy as np

from .planning import evaluate_policy


def snapshot_metrics(spec, snaps, Q_star, V_star, tensors):
    """Scores for every Q snapshot of a training run.

    rmse      error of Q against Q* over every (state, action)
    rmse_opt  error only on the best action of each state, the values that decide what the agent does.
              Bad actions are rarely tried once the agent knows better, so their values stay rough,
              and that is fine.
    match     fraction of states where the agent's greedy action is an optimal one
    regret    how much value the greedy policy loses from the start state: V*(s0) - V_pi(s0).
              Computed exactly with the model, so 0 really means "this policy is optimal".
    """
    valid = spec.valid_states
    optimal_actions = Q_star[valid] >= Q_star[valid].max(axis=1, keepdims=True) - 1e-9
    a_star = Q_star.argmax(axis=1)[valid]
    rmse, rmse_opt, match, regret = [], [], [], []
    for Q in snaps:
        rmse.append(np.sqrt(np.mean((Q[valid] - Q_star[valid]) ** 2)))
        rmse_opt.append(np.sqrt(np.mean((Q[valid, a_star] - Q_star[valid, a_star]) ** 2)))
        pi = Q.argmax(axis=1)
        match.append(np.mean(optimal_actions[np.arange(len(valid)), pi[valid]]))
        V_pi = evaluate_policy(spec, pi, tensors)
        regret.append(V_star[spec.start_state] - V_pi[spec.start_state])
    return {"rmse": rmse, "rmse_opt": rmse_opt, "match": match, "regret": regret}


def moving_average(x, w=50):
    """Mean of the last `w` values along the last axis (the result is w-1 points shorter)."""
    c = np.cumsum(np.insert(np.asarray(x, dtype=float), 0, 0.0, axis=-1), axis=-1)
    return (c[..., w:] - c[..., :-w]) / w
