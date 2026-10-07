"""The learning algorithms. They all take an environment plus hyper-parameters and return a RunResult.

ALGORITHMS maps the names used in the experiments (and in the cache files) to the functions.
"""
from functools import partial

from .base import RunResult, epsilon_at, epsilon_greedy, greedy_action, step_size
from .monte_carlo import monte_carlo
from .q_learning import double_q_learning, q_learning
from .sarsa import expected_sarsa, n_step_sarsa, sarsa

ALGORITHMS = {
    "Monte Carlo": monte_carlo,
    "MC constant-α": partial(monte_carlo, constant_alpha=True),
    "MC Exploring Starts": partial(monte_carlo, constant_alpha=True, exploring_starts=True),
    "SARSA": sarsa,
    "n-step SARSA": n_step_sarsa,
    "Expected SARSA": expected_sarsa,
    "Q-learning": q_learning,
    "Double Q-learning": double_q_learning,
}
