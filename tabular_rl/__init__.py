"""Small tabular RL library for the assignment.

    class_gridworld.py  the GridworldEnv from class, unchanged
    envs.py          the environments (class gridworld + Cliff Walking) in the class P format
    simulator.py     plays an environment step by step (what the agents see)
    planning.py      value / policy iteration: the exact answer, used only to grade the agents
    agents/          the learning algorithms
    metrics.py       how we score a Q-table against the exact answer
    training.py      run an algorithm over many seeds, with a disk cache
    models.py        save / load learned Q-tables
    plotting.py      figure style and helpers
    stats.py         bootstrap confidence intervals and permutation test
"""
