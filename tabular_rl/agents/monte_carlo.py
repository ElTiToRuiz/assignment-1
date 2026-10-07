"""Monte Carlo control (class 1.5): play a whole episode, then learn from the returns it actually got."""
import numpy as np

from ..envs import EnvSpec
from .base import RunResult, epsilon_at, epsilon_greedy, new_run


def monte_carlo(spec: EnvSpec, n_episodes=2000, alpha=None, eps=(1.0, 0.05, 0.995), seed=0,
                log_every=10, first_visit=False, exploring_starts=False,
                constant_alpha=False) -> RunResult:
    """Monte Carlo with an epsilon-greedy policy (the class "Montecarlo improved").

    After each episode we walk the trajectory backwards, building the return G step by step, and
    move Q(s, a) towards it.

    By default the step is 1/N(s, a), so Q is exactly the average of every return seen, as in class
    (`alpha` is ignored). With constant_alpha=True we use `alpha` instead, which forgets old returns.
    That helps a lot in control: the policy keeps getting better, so the very bad returns of the
    first random episodes should not stay in the average forever (see exp6).

    exploring_starts=True starts every episode from a random state and action (the class
    "Monte-Carlo Exploring Starts"), so every pair keeps being tried even while the policy is bad.
    """
    env, rng, rec = new_run(spec, seed, n_episodes, log_every)
    Q = np.zeros((spec.n_states, spec.n_actions))
    N = np.zeros_like(Q)
    for ep in range(n_episodes):
        e = epsilon_at(ep, eps)

        # 1) play one episode
        if exploring_starts:
            s, first_action = env.reset(rng.choice(spec.valid_states)), rng.randrange(spec.n_actions)
        else:
            s, first_action = env.reset(), None
        trajectory = []
        for t in range(spec.max_steps):
            a = first_action if t == 0 and first_action is not None else epsilon_greedy(Q[s], e, rng)
            s2, r, done = env.step(a)
            trajectory.append((s, a, r))
            s = s2
            if done:
                break

        # 2) learn from it, going backwards so the return is just G = r + gamma * G
        first_seen = {}
        if first_visit:
            for i, (s, a, _) in enumerate(trajectory):
                first_seen.setdefault((s, a), i)
        G = 0.0
        for i in range(len(trajectory) - 1, -1, -1):
            s, a, r = trajectory[i]
            G = r + spec.gamma * G
            if first_visit and first_seen[(s, a)] != i:
                continue
            N[s, a] += 1
            Q[s, a] += (G - Q[s, a]) * (alpha if constant_alpha else 1.0 / N[s, a])

        for s, a, _ in trajectory:
            rec.visits[s, a] += 1
        rec.end_episode(ep, sum(step[2] for step in trajectory), len(trajectory), Q)
    return rec.result(Q)
