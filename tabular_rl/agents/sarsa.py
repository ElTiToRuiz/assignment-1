"""On-policy TD control (class 1.6): SARSA and two of its relatives.

They all learn the value of the policy they are actually following, exploration included. That is
why they play it safe near cliffs and pits.
"""
import numpy as np

from ..envs import EnvSpec
from .base import RunResult, epsilon_at, epsilon_greedy, new_run, step_size


def sarsa(spec: EnvSpec, n_episodes=2000, alpha=0.1, eps=(1.0, 0.05, 0.995), seed=0, log_every=10,
          alpha_decay=0.0) -> RunResult:
    """SARSA: Q(s,a) moves towards r + gamma * Q(s', a'), where a' is the action we will really take next.

    That is a sample of the Bellman equation for the epsilon-greedy policy, so SARSA learns how good
    that policy is, random moves and all.
    """
    env, rng, rec = new_run(spec, seed, n_episodes, log_every)
    Q = np.zeros((spec.n_states, spec.n_actions))
    for ep in range(n_episodes):
        e = epsilon_at(ep, eps)
        s = env.reset()
        a = epsilon_greedy(Q[s], e, rng)
        G, steps = 0.0, 0
        for _ in range(spec.max_steps):
            s2, r, done = env.step(a)
            # Pick the next action *before* updating, and then really take it: that is the "SA" at the end.
            a2 = epsilon_greedy(Q[s2], e, rng)
            target = r + (0.0 if done else spec.gamma * Q[s2, a2])
            Q[s, a] += step_size(alpha, alpha_decay, rec.visits[s, a]) * (target - Q[s, a])
            rec.visits[s, a] += 1
            G, steps = G + r, steps + 1
            s, a = s2, a2
            if done:
                break
        rec.end_episode(ep, G, steps, Q)
    return rec.result(Q)


def expected_sarsa(spec: EnvSpec, n_episodes=2000, alpha=0.1, eps=(1.0, 0.05, 0.995), seed=0, log_every=10,
                   alpha_decay=0.0) -> RunResult:
    """Expected SARSA: like SARSA, but instead of the one next action we happened to sample, use the
    average over all of them, weighted by how likely the epsilon-greedy policy is to pick each.

    Same answer as SARSA on average, with less noise. With eps = 0 it becomes exactly Q-learning.
    """
    env, rng, rec = new_run(spec, seed, n_episodes, log_every)
    Q = np.zeros((spec.n_states, spec.n_actions))
    n_actions = spec.n_actions
    for ep in range(n_episodes):
        e = epsilon_at(ep, eps)
        s = env.reset()
        G, steps = 0.0, 0
        for _ in range(spec.max_steps):
            a = epsilon_greedy(Q[s], e, rng)
            s2, r, done = env.step(a)
            if done:
                target = r
            else:
                # epsilon-greedy probabilities in s': eps spread evenly, the rest shared by the best actions
                best = Q[s2] == Q[s2].max()
                pi = e / n_actions + (1 - e) * best / best.sum()
                target = r + spec.gamma * pi @ Q[s2]
            Q[s, a] += step_size(alpha, alpha_decay, rec.visits[s, a]) * (target - Q[s, a])
            rec.visits[s, a] += 1
            G, steps = G + r, steps + 1
            s = s2
            if done:
                break
        rec.end_episode(ep, G, steps, Q)
    return rec.result(Q)


def n_step_sarsa(spec: EnvSpec, n_episodes=2000, alpha=0.1, eps=(1.0, 0.05, 0.995), seed=0,
                 log_every=10, n=3, alpha_decay=0.0) -> RunResult:
    """n-step SARSA: wait n steps, then update with the n real rewards plus an estimate for the rest:

        r1 + gamma r2 + ... + gamma^(n-1) rn + gamma^n Q(s_n, a_n)

    n = 1 is plain SARSA and a very large n is Monte Carlo, so n sets how much we trust real rewards
    vs our own estimates.
    """
    env, rng, rec = new_run(spec, seed, n_episodes, log_every)
    Q = np.zeros((spec.n_states, spec.n_actions))
    g = spec.gamma
    for ep in range(n_episodes):
        e = epsilon_at(ep, eps)
        # States, actions and rewards of this episode. R[t+1] is the reward after action A[t].
        S, A, R = [env.reset()], [0], [0.0]
        A[0] = epsilon_greedy(Q[S[0]], e, rng)
        T, t, G = spec.max_steps, 0, 0.0   # T = when the episode ended (max_steps until we know)
        while True:
            if t < T:
                s2, r, done = env.step(A[t])
                S.append(s2); R.append(r); G += r
                if done:
                    T = t + 1
                else:
                    A.append(epsilon_greedy(Q[s2], e, rng))
            tau = t - n + 1  # the step whose estimate we can update now
            if tau >= 0:
                last = min(tau + n, T)
                target = sum(g ** (i - tau - 1) * R[i] for i in range(tau + 1, last + 1))
                if tau + n < T:  # the episode is still going after n steps, so add our estimate
                    target += g ** n * Q[S[tau + n], A[tau + n]]
                s_, a_ = S[tau], A[tau]
                Q[s_, a_] += step_size(alpha, alpha_decay, rec.visits[s_, a_]) * (target - Q[s_, a_])
                rec.visits[s_, a_] += 1
            if tau >= T - 1:
                break
            t += 1
        rec.end_episode(ep, G, T, Q)
    return rec.result(Q)
