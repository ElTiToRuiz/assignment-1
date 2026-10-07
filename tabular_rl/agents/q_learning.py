"""Off-policy TD control (class 1.6): Q-learning and Double Q-learning.

They explore with epsilon-greedy but learn about the greedy policy, so exploring never changes what
they end up learning. That is why they find the optimal path along the cliff edge.
"""
import numpy as np

from ..envs import EnvSpec
from .base import RunResult, epsilon_at, epsilon_greedy, greedy_action, new_run, step_size


def q_learning(spec: EnvSpec, n_episodes=2000, alpha=0.1, eps=(1.0, 0.05, 0.995), seed=0, log_every=10,
               alpha_decay=0.0) -> RunResult:
    """Q-learning: Q(s,a) moves towards r + gamma * max Q(s', .), a sample of the Bellman optimality equation."""
    env, rng, rec = new_run(spec, seed, n_episodes, log_every)
    Q = np.zeros((spec.n_states, spec.n_actions))
    for ep in range(n_episodes):
        e = epsilon_at(ep, eps)
        s = env.reset()
        G, steps = 0.0, 0
        for _ in range(spec.max_steps):
            a = epsilon_greedy(Q[s], e, rng)                          # how we act: explore
            s2, r, done = env.step(a)
            target = r + (0.0 if done else spec.gamma * Q[s2].max())  # what we learn about: the greedy policy
            Q[s, a] += step_size(alpha, alpha_decay, rec.visits[s, a]) * (target - Q[s, a])
            rec.visits[s, a] += 1
            G, steps = G + r, steps + 1
            s = s2
            if done:
                break
        rec.end_episode(ep, G, steps, Q)
    return rec.result(Q)


def double_q_learning(spec: EnvSpec, n_episodes=2000, alpha=0.1, eps=(1.0, 0.05, 0.995), seed=0, log_every=10,
                      alpha_decay=0.0) -> RunResult:
    """Double Q-learning: two tables, one chooses the best next action and the other says how good it is.

    Taking the max of noisy estimates is biased upwards (we pick whichever got lucky). Splitting
    "choose" and "evaluate" between two independent tables removes that bias. The price: each table
    only gets half of the updates, so it learns more slowly.
    """
    env, rng, rec = new_run(spec, seed, n_episodes, log_every)
    Q1 = np.zeros((spec.n_states, spec.n_actions))
    Q2 = np.zeros_like(Q1)
    for ep in range(n_episodes):
        e = epsilon_at(ep, eps)
        s = env.reset()
        G, steps = 0.0, 0
        for _ in range(spec.max_steps):
            a = epsilon_greedy(Q1[s] + Q2[s], e, rng)  # act on what both tables know
            s2, r, done = env.step(a)
            learner, judge = (Q1, Q2) if rng.random() < 0.5 else (Q2, Q1)  # flip a coin for who learns
            target = r if done else r + spec.gamma * judge[s2, greedy_action(learner[s2], rng)]
            learner[s, a] += step_size(alpha, alpha_decay, rec.visits[s, a]) * (target - learner[s, a])
            rec.visits[s, a] += 1
            G, steps = G + r, steps + 1
            s = s2
            if done:
                break
        rec.end_episode(ep, G, steps, (Q1 + Q2) / 2)
    return rec.result((Q1 + Q2) / 2)
