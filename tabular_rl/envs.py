"""The environments.

Each one is an `EnvSpec`: the grid layout plus the full dynamics in the same format as in class,
P[s][a] = [(prob, next_state, reward, done), ...].

The agents never read P directly. They only play the environment through `Simulator`, exactly as
they would with the class GridworldEnv. We keep P around for two things only: computing the true
optimum with dynamic programming, and grading the learned policies exactly.
"""
import os
from dataclasses import dataclass, field

import numpy as np

os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")  # the class env imports pygame, which is chatty
from class_code.env import GridworldEnv

# Same action numbering as the class env.
ACTIONS = {0: (0, -1), 1: (1, 0), 2: (0, 1), 3: (-1, 0)}  # LEFT, DOWN, RIGHT, UP
ARROWS = {0: "←", 1: "↓", 2: "→", 3: "↑"}


@dataclass
class EnvSpec:
    name: str
    P: dict
    n_states: int
    n_actions: int
    start_state: int
    shape: tuple          # (rows, cols), states are numbered row by row
    gamma: float
    max_steps: int        # episodes are cut here so a bad policy cannot loop forever
    terminals: dict = field(default_factory=dict)  # terminal state -> its reward
    walls: set = field(default_factory=set)
    cliffs: set = field(default_factory=set)

    @property
    def valid_states(self):
        """States the agent can actually stand on and act from."""
        blocked = set(self.terminals) | self.walls | self.cliffs
        return [s for s in range(self.n_states) if s not in blocked]


def gridworld_spec(slippery: bool) -> EnvSpec:
    """The 3x4 Russell & Norvig gridworld from class. Slippery: 80% intended move, 10% to each side."""
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
    """Cliff Walking from Sutton & Barto (4x12).

    Every step costs -1. Stepping into the cliff costs -100 and sends you back to the start,
    but the episode goes on. Only the goal ends it.
    """
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
            # walking into the border just leaves you where you are
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
    builders = {
        "gridworld_deterministic": lambda: gridworld_spec(slippery=False),
        "gridworld_slippery": lambda: gridworld_spec(slippery=True),
        "cliff_walking": cliff_spec,
    }
    return builders[name]()


def greedy_path(spec: EnvSpec, Q, limit=60):
    """The states the greedy policy walks through from the start (taking the most likely outcome)."""
    s, path = spec.start_state, [spec.start_state]
    for _ in range(limit):
        _, s, _, done = max(spec.P[s][int(np.argmax(Q[s]))], key=lambda t: t[0])
        path.append(s)
        if done:
            break
    return path
