"""Dynamic programming (class 1.3-1.4). It needs the model P, so the agents never use it.

We use it as the answer key: value iteration gives the true Q*, and exact policy evaluation tells us
how good a learned policy really is.
"""
import numpy as np

from .envs import EnvSpec


def model_tensors(spec: EnvSpec):
    """P as two arrays, so the Bellman backups become matrix products.

    T[s, a, s'] = probability of landing in s' and the episode going on
    R[s, a]     = expected immediate reward

    A transition that ends the episode adds its reward but no future value, since V(terminal) = 0.
    """
    S, A = spec.n_states, spec.n_actions
    T, R = np.zeros((S, A, S)), np.zeros((S, A))
    for s in range(S):
        for a in range(A):
            for prob, s2, r, done in spec.P[s][a]:
                R[s, a] += prob * r
                if not done:
                    T[s, a, s2] += prob
    return T, R


def value_iteration(spec: EnvSpec, tol: float = 1e-12):
    """Apply the Bellman optimality equation until V stops changing. Returns Q*, V* and the greedy policy."""
    T, R = model_tensors(spec)
    V = np.zeros(spec.n_states)
    while True:
        Q = R + spec.gamma * T @ V
        V_new = Q.max(axis=1)
        if np.max(np.abs(V_new - V)) < tol:
            return Q, V_new, Q.argmax(axis=1)
        V = V_new


def evaluate_policy(spec: EnvSpec, policy, tensors=None):
    """Exact V of a deterministic policy, by solving the Bellman equation (I - gamma P_pi) V = r_pi."""
    T, R = tensors if tensors is not None else model_tensors(spec)
    idx = np.arange(spec.n_states)
    P_pi, r_pi = T[idx, policy], R[idx, policy]
    return np.linalg.solve(np.eye(spec.n_states) - spec.gamma * P_pi, r_pi)


def policy_iteration(spec: EnvSpec):
    """Evaluate the policy exactly, improve it greedily, and repeat until it stops changing.

    Returns Q, V, the policy and how many improvement rounds it took.
    """
    tensors = model_tensors(spec)
    T, R = tensors
    policy = np.zeros(spec.n_states, dtype=int)
    for k in range(1, 1000):
        V = evaluate_policy(spec, policy, tensors)
        Q = R + spec.gamma * T @ V
        new = Q.argmax(axis=1)
        # If the current action is already as good as the best one, keep it. Otherwise the loop
        # could keep swapping between equally good policies and never stop.
        already_best = Q[np.arange(spec.n_states), policy] >= Q.max(axis=1) - 1e-12
        new[already_best] = policy[already_best]
        if np.array_equal(new, policy):
            return Q, V, policy, k
        policy = new
    raise RuntimeError("policy iteration did not converge")
