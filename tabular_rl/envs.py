"""Environment specifications and a fast sampler.

An environment is described by an `EnvSpec`, which holds the MDP dynamics in the
same format used in class (`env.P[s][a] = [(prob, s', reward, done), ...]`).
The learning agents never look at `P`: they only call `Simulator.reset/step`,
exactly like with the class `GridworldEnv`. `P` is only used by the dynamic
programming code (ground truth `Q*`) and by the exact policy evaluation used for
metrics.
"""
import os
import random
from dataclasses import dataclass, field

import numpy as np

os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")
from class_code.env import GridworldEnv

ACTIONS = {0: (0, -1), 1: (1, 0), 2: (0, 1), 3: (-1, 0)}  # LEFT, DOWN, RIGHT, UP
ARROWS = {0: "←", 1: "↓", 2: "→", 3: "↑"}


@dataclass
class EnvSpec:
    name: str
    P: dict
    n_states: int
    n_actions: int
    start_state: int
    shape: tuple
    gamma: float
    max_steps: int
    terminals: dict = field(default_factory=dict)  # state -> terminal reward
    walls: set = field(default_factory=set)
    cliffs: set = field(default_factory=set)

    @property
    def valid_states(self):
        """Non-terminal states that the agent can actually be in."""
        return [
            s for s in range(self.n_states)
            if s not in self.terminals and s not in self.walls and s not in self.cliffs
        ]


def gridworld_spec(slippery: bool) -> EnvSpec:
    """3x4 Russell & Norvig gridworld from class (deterministic or 80/10/10 slippery)."""
    env = GridworldEnv(is_slippery=slippery)
    return EnvSpec(
        name="gridworld_slippery" if slippery else "gridworld_deterministic",
        P=env.P,
        n_states=env.n_states,
        n_actions=env.n_actions,
        start_state=env.start_state,
        shape=(env.n_rows, env.n_cols),
        gamma=0.99,
        max_steps=200,
        terminals=dict(env.terminal_states),
        walls={env.wall_state},
    )


def cliff_spec() -> EnvSpec:
    """Sutton & Barto's Cliff Walking (4x12). -1 per step, -100 and back to start on the cliff."""
    n_rows, n_cols = 4, 12
    start, goal = (n_rows - 1) * n_cols, n_rows * n_cols - 1
    cliffs = set(range(start + 1, goal))
    P = {s: {a: [] for a in range(4)} for s in range(n_rows * n_cols)}
    for s in range(n_rows * n_cols):
        if s in cliffs:
            continue
        r, c = divmod(s, n_cols)
        for a, (dr, dc) in ACTIONS.items():
            if s == goal:
                P[s][a] = [(1.0, s, 0.0, True)]
                continue
            nr, nc = min(max(r + dr, 0), n_rows - 1), min(max(c + dc, 0), n_cols - 1)
            s2 = nr * n_cols + nc
            if s2 in cliffs:
                P[s][a] = [(1.0, start, -100.0, False)]
            else:
                P[s][a] = [(1.0, s2, -1.0, s2 == goal)]
    return EnvSpec(
        name="cliff_walking", P=P, n_states=n_rows * n_cols, n_actions=4,
        start_state=start, shape=(n_rows, n_cols), gamma=0.99, max_steps=500,
        terminals={goal: -1.0}, cliffs=cliffs,
    )


def make_spec(name: str) -> EnvSpec:
    return {
        "gridworld_deterministic": lambda: gridworld_spec(False),
        "gridworld_slippery": lambda: gridworld_spec(True),
        "cliff_walking": cliff_spec,
    }[name]()


class Simulator:
    """Samples transitions from `spec.P` (same behaviour as `GridworldEnv.step`, but fast).

    `step` returns (next_state, reward, done). Own RNG -> reproducible runs.
    """

    def __init__(self, spec: EnvSpec, seed: int = 0):
        self.spec = spec
        self.rng = random.Random(seed)
        self.s = spec.start_state
        self._table = {
            (s, a): ([t[0] for t in trs], trs)
            for s, acts in spec.P.items() for a, trs in acts.items() if trs
        }

    def reset(self, state=None):
        """Back to the start state (or to `state`, used by Monte Carlo exploring starts)."""
        self.s = self.spec.start_state if state is None else state
        return self.s

    def step(self, a):
        probs, trs = self._table[(self.s, a)]
        if len(trs) == 1:
            _, s2, r, done = trs[0]
        else:
            _, s2, r, done = self.rng.choices(trs, weights=probs)[0]
        self.s = s2
        return s2, r, done


def greedy_path(spec: EnvSpec, Q, limit=60):
    """States visited by the greedy policy from the start state (follows the most likely transition)."""
    s, path = spec.start_state, [spec.start_state]
    for _ in range(limit):
        prob, s, _, done = max(spec.P[s][int(np.argmax(Q[s]))], key=lambda t: t[0])
        path.append(s)
        if done:
            break
    return path
