"""What every algorithm shares: exploration, step size and the bookkeeping of a training run."""
import random
from dataclasses import dataclass

import numpy as np

from ..simulator import Simulator


@dataclass
class RunResult:
    Q: np.ndarray        # the learned Q-table, shape (states, actions)
    returns: np.ndarray  # total reward of each training episode
    lengths: np.ndarray  # number of steps of each training episode
    snaps: np.ndarray    # a copy of Q every `log_every` episodes, to draw learning curves later
    visits: np.ndarray   # how many times each (s, a) was updated


class Recorder:
    """Keeps track of a training run so the algorithms can stay focused on the learning rule."""

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


def new_run(spec, seed, n_episodes, log_every):
    """Everything a training run needs. The environment and the agent get different random streams,
    so changing how the agent explores does not change the dice the environment rolls."""
    return Simulator(spec, seed), random.Random(seed + 10_000), Recorder(spec, n_episodes, log_every)


def greedy_action(q: np.ndarray, rng: random.Random) -> int:
    """The best action, breaking ties at random.

    Ties matter: Q starts at zero, so at the beginning every action is tied. Always picking the
    first one would make the agent walk LEFT forever.
    """
    best = np.flatnonzero(q == q.max())
    return int(best[0]) if len(best) == 1 else int(best[rng.randrange(len(best))])


def epsilon_greedy(q: np.ndarray, eps: float, rng: random.Random) -> int:
    """With probability eps try a random action, otherwise take the best one."""
    if rng.random() < eps:
        return rng.randrange(len(q))
    return greedy_action(q, rng)


def epsilon_at(ep: int, eps: tuple) -> float:
    """Exploration rate in episode `ep`.

    eps = (start, minimum, decay): it starts at `start`, is multiplied by `decay` every episode and
    never goes below `minimum`. decay = 1 means a constant epsilon.
    """
    eps_start, eps_min, eps_decay = eps
    return max(eps_min, eps_start * eps_decay ** ep)


def step_size(alpha: float, alpha_decay: float, n_visits: int) -> float:
    """Learning rate for a pair that has been updated `n_visits` times: alpha / (1 + alpha_decay * N).

    alpha_decay = 0 gives the usual constant alpha. Any alpha_decay > 0 shrinks the steps just slowly
    enough to satisfy the Robbins-Monro conditions (sum of steps infinite, sum of squares finite).
    Theory needs this for the estimates to settle exactly when transitions are random.
    """
    return alpha / (1.0 + alpha_decay * n_visits)
