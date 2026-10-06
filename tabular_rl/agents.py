"""Tabular model-free algorithms: Monte Carlo (e-greedy), SARSA, n-step SARSA,
Expected SARSA, Q-learning and Double Q-learning.

All of them share the same signature and return a `RunResult`. They interact with
the environment only through `Simulator.reset()/step()` (model-free).

Exploration: e-greedy with exponentially decaying epsilon,
    eps(ep) = max(eps_min, eps_start * eps_decay**ep)
(eps_decay=1 -> constant epsilon).
"""
import random
from dataclasses import dataclass

import numpy as np

from .envs import EnvSpec, Simulator


@dataclass
class RunResult:
    Q: np.ndarray        # final Q (S, A)
    returns: np.ndarray  # undiscounted return of every training episode
    lengths: np.ndarray  # number of steps of every training episode
    snaps: np.ndarray    # Q snapshots every `log_every` episodes (n_snaps, S, A)
    visits: np.ndarray   # (s, a) visit counts during training


class _Recorder:
    def __init__(self, spec, n_episodes, log_every):
        self.returns = np.zeros(n_episodes)
        self.lengths = np.zeros(n_episodes, dtype=int)
        self.visits = np.zeros((spec.n_states, spec.n_actions), dtype=int)
        self.snaps, self.log_every = [], log_every

    def end_episode(self, ep, G, steps, Q):
        self.returns[ep], self.lengths[ep] = G, steps
        if (ep + 1) % self.log_every == 0:
            self.snaps.append(Q.copy())

    def result(self, Q):
        return RunResult(Q, self.returns, self.lengths, np.array(self.snaps), self.visits)


def greedy_action(q: np.ndarray, rng: random.Random) -> int:
    """argmax with random tie-breaking (Q starts at 0, so ties are the norm early on)."""
    best = np.flatnonzero(q == q.max())
    return int(best[0]) if len(best) == 1 else int(best[rng.randrange(len(best))])


def epsilon_greedy(q: np.ndarray, eps: float, rng: random.Random) -> int:
    """With prob. eps a uniformly random action, otherwise the greedy one."""
    if rng.random() < eps:
        return rng.randrange(len(q))
    return greedy_action(q, rng)


def epsilon_at(ep: int, eps: tuple) -> float:
    """Exploration rate of episode `ep` for the schedule eps = (eps_start, eps_min, eps_decay)."""
    eps_start, eps_min, eps_decay = eps
    return max(eps_min, eps_start * eps_decay ** ep)


def step_size(alpha: float, alpha_decay: float, n_visits: int) -> float:
    """alpha_t(s,a) = alpha / (1 + alpha_decay * N(s,a)). alpha_decay=0 -> constant alpha.

    With alpha_decay > 0 the step size satisfies the Robbins-Monro conditions
    (sum alpha = inf, sum alpha^2 < inf), needed for exact convergence under stochastic transitions.
    """
    return alpha / (1.0 + alpha_decay * n_visits)


def _setup(spec, seed, n_episodes, log_every):
    return Simulator(spec, seed), random.Random(seed + 10_000), _Recorder(spec, n_episodes, log_every)


def sarsa(spec: EnvSpec, n_episodes=2000, alpha=0.1, eps=(1.0, 0.05, 0.995), seed=0, log_every=10,
          alpha_decay=0.0) -> RunResult:
    """On-policy TD control. Target: r + gamma * Q(s', a') with a' ~ e-greedy (same policy that acts)."""
    env, rng, rec = _setup(spec, seed, n_episodes, log_every)
    Q = np.zeros((spec.n_states, spec.n_actions))
    for ep in range(n_episodes):
        e = epsilon_at(ep, eps)
        s = env.reset()
        a = epsilon_greedy(Q[s], e, rng)
        G, steps = 0.0, 0
        for _ in range(spec.max_steps):
            s2, r, done = env.step(a)
            a2 = epsilon_greedy(Q[s2], e, rng)  # a' chosen BEFORE the update and actually executed next
            target = r + (0.0 if done else spec.gamma * Q[s2, a2])
            Q[s, a] += step_size(alpha, alpha_decay, rec.visits[s, a]) * (target - Q[s, a])
            rec.visits[s, a] += 1
            G, steps = G + r, steps + 1
            s, a = s2, a2
            if done:
                break
        rec.end_episode(ep, G, steps, Q)
    return rec.result(Q)


def q_learning(spec: EnvSpec, n_episodes=2000, alpha=0.1, eps=(1.0, 0.05, 0.995), seed=0, log_every=10,
               alpha_decay=0.0) -> RunResult:
    """Off-policy TD control (Bellman optimality). Target: r + gamma * max_a' Q(s', a')."""
    env, rng, rec = _setup(spec, seed, n_episodes, log_every)
    Q = np.zeros((spec.n_states, spec.n_actions))
    for ep in range(n_episodes):
        e = epsilon_at(ep, eps)
        s = env.reset()
        G, steps = 0.0, 0
        for _ in range(spec.max_steps):
            a = epsilon_greedy(Q[s], e, rng)  # behaviour policy: e-greedy
            s2, r, done = env.step(a)
            target = r + (0.0 if done else spec.gamma * Q[s2].max())  # target policy: greedy
            Q[s, a] += step_size(alpha, alpha_decay, rec.visits[s, a]) * (target - Q[s, a])
            rec.visits[s, a] += 1
            G, steps = G + r, steps + 1
            s = s2
            if done:
                break
        rec.end_episode(ep, G, steps, Q)
    return rec.result(Q)


def expected_sarsa(spec: EnvSpec, n_episodes=2000, alpha=0.1, eps=(1.0, 0.05, 0.995), seed=0, log_every=10,
                   alpha_decay=0.0) -> RunResult:
    """Like SARSA but replaces Q(s',a') by its expectation under the e-greedy policy (lower variance)."""
    env, rng, rec = _setup(spec, seed, n_episodes, log_every)
    Q = np.zeros((spec.n_states, spec.n_actions))
    nA = spec.n_actions
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
                best = Q[s2] == Q[s2].max()
                pi = e / nA + (1 - e) * best / best.sum()  # e-greedy action probabilities in s'
                target = r + spec.gamma * pi @ Q[s2]
            Q[s, a] += step_size(alpha, alpha_decay, rec.visits[s, a]) * (target - Q[s, a])
            rec.visits[s, a] += 1
            G, steps = G + r, steps + 1
            s = s2
            if done:
                break
        rec.end_episode(ep, G, steps, Q)
    return rec.result(Q)


def double_q_learning(spec: EnvSpec, n_episodes=2000, alpha=0.1, eps=(1.0, 0.05, 0.995), seed=0, log_every=10) -> RunResult:
    """Two tables: one picks argmax, the other evaluates it (removes maximisation bias)."""
    env, rng, rec = _setup(spec, seed, n_episodes, log_every)
    Q1 = np.zeros((spec.n_states, spec.n_actions))
    Q2 = np.zeros_like(Q1)
    for ep in range(n_episodes):
        e = epsilon_at(ep, eps)
        s = env.reset()
        G, steps = 0.0, 0
        for _ in range(spec.max_steps):
            a = epsilon_greedy(Q1[s] + Q2[s], e, rng)
            s2, r, done = env.step(a)
            A, B = (Q1, Q2) if rng.random() < 0.5 else (Q2, Q1)  # update A, evaluate with B
            target = r if done else r + spec.gamma * B[s2, greedy_action(A[s2], rng)]
            A[s, a] += alpha * (target - A[s, a])
            rec.visits[s, a] += 1
            G, steps = G + r, steps + 1
            s = s2
            if done:
                break
        rec.end_episode(ep, G, steps, (Q1 + Q2) / 2)
    return rec.result((Q1 + Q2) / 2)


def n_step_sarsa(spec: EnvSpec, n_episodes=2000, alpha=0.1, eps=(1.0, 0.05, 0.995), seed=0,
                 log_every=10, n=3) -> RunResult:
    """n-step SARSA: target = r_t+1 + ... + gamma^(n-1) r_t+n + gamma^n Q(s_t+n, a_t+n)."""
    env, rng, rec = _setup(spec, seed, n_episodes, log_every)
    Q = np.zeros((spec.n_states, spec.n_actions))
    g = spec.gamma
    for ep in range(n_episodes):
        e = epsilon_at(ep, eps)
        S, A, R = [env.reset()], [0], [0.0]  # R[t+1] = reward received after action A[t]
        A[0] = epsilon_greedy(Q[S[0]], e, rng)
        T, t, G = spec.max_steps, 0, 0.0
        while True:
            if t < T:
                s2, r, done = env.step(A[t])
                S.append(s2); R.append(r); G += r
                if done:
                    T = t + 1
                else:
                    A.append(epsilon_greedy(Q[s2], e, rng))
            tau = t - n + 1  # time step whose estimate is updated
            if tau >= 0:
                last = min(tau + n, T)
                Gn = sum(g ** (i - tau - 1) * R[i] for i in range(tau + 1, last + 1))
                if tau + n < T:  # bootstrap if the episode has not ended within n steps
                    Gn += g ** n * Q[S[tau + n], A[tau + n]]
                Q[S[tau], A[tau]] += alpha * (Gn - Q[S[tau], A[tau]])
                rec.visits[S[tau], A[tau]] += 1  # counts updates of (s,a)
            if tau >= T - 1:
                break
            t += 1
        rec.end_episode(ep, G, T, Q)
    return rec.result(Q)


def monte_carlo(spec: EnvSpec, n_episodes=2000, alpha=None, eps=(1.0, 0.05, 0.995), seed=0,
                log_every=10, first_visit=False) -> RunResult:
    """Monte Carlo e-greedy control (class "Montecarlo improved", without exploring starts).

    Backward return computation + online sample-mean update Q <- Q + (G - Q)/N(s,a).
    `alpha` is ignored (the step size is 1/N); kept for a common signature.
    """
    env, rng, rec = _setup(spec, seed, n_episodes, log_every)
    Q = np.zeros((spec.n_states, spec.n_actions))
    N = np.zeros_like(Q)
    for ep in range(n_episodes):
        e = epsilon_at(ep, eps)
        s = env.reset()
        traj = []
        for _ in range(spec.max_steps):
            a = epsilon_greedy(Q[s], e, rng)
            s2, r, done = env.step(a)
            traj.append((s, a, r))
            s = s2
            if done:
                break
        first = {}
        if first_visit:
            for i, (s, a, _) in enumerate(traj):
                first.setdefault((s, a), i)
        G_ret = 0.0
        for i in range(len(traj) - 1, -1, -1):
            s, a, r = traj[i]
            G_ret = r + spec.gamma * G_ret
            if first_visit and first[(s, a)] != i:
                continue
            N[s, a] += 1
            Q[s, a] += (G_ret - Q[s, a]) / N[s, a]
        for s, a, _ in traj:
            rec.visits[s, a] += 1
        rec.end_episode(ep, sum(t[2] for t in traj), len(traj), Q)
    return rec.result(Q)


ALGORITHMS = {
    "Monte Carlo": monte_carlo,
    "SARSA": sarsa,
    "n-step SARSA": n_step_sarsa,
    "Expected SARSA": expected_sarsa,
    "Q-learning": q_learning,
    "Double Q-learning": double_q_learning,
}
