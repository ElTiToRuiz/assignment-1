"""Plays an environment one step at a time. This is all the agents ever see."""
import random

from .envs import EnvSpec


class Simulator:
    """Same behaviour as GridworldEnv.step from class, just faster and with its own random generator,
    so every run is reproducible from its seed.

    step(a) returns (next_state, reward, done).
    """

    def __init__(self, spec: EnvSpec, seed: int = 0):
        self.spec = spec
        self.rng = random.Random(seed)
        self.s = spec.start_state
        # Pre-split each transition list into (probabilities, outcomes) once, instead of every step.
        self._table = {
            (s, a): ([t[0] for t in outcomes], outcomes)
            for s, actions in spec.P.items() for a, outcomes in actions.items() if outcomes
        }

    def reset(self, state=None):
        """Back to the start (or to a chosen state, which Monte Carlo exploring starts needs)."""
        self.s = self.spec.start_state if state is None else state
        return self.s

    def step(self, a):
        probs, outcomes = self._table[(self.s, a)]
        if len(outcomes) == 1:
            _, s2, r, done = outcomes[0]
        else:
            _, s2, r, done = self.rng.choices(outcomes, weights=probs)[0]
        self.s = s2
        return s2, r, done
