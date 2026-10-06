"""Dynamic programming ground truth (needs the model P) and exact policy evaluation."""
import numpy as np

from .envs import EnvSpec


def model_tensors(spec: EnvSpec):
    """T[s,a,s'] = prob of moving to s' *and continuing*; R[s,a] = expected reward.

    Terminal transitions (done=True) contribute reward but no continuation (V(s')=0).
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
    """Optimal Q*, V*, greedy policy via value iteration (Bellman optimality)."""
    T, R = model_tensors(spec)
    V = np.zeros(spec.n_states)
    while True:
        Q = R + spec.gamma * T @ V
        V_new = Q.max(axis=1)
        if np.max(np.abs(V_new - V)) < tol:
            return Q, V_new, Q.argmax(axis=1)
        V = V_new


def evaluate_policy(spec: EnvSpec, policy, tensors=None):
    """Exact V^pi for a deterministic policy: solve (I - gamma P_pi) V = r_pi."""
    T, R = tensors if tensors is not None else model_tensors(spec)
    idx = np.arange(spec.n_states)
    P_pi, r_pi = T[idx, policy], R[idx, policy]
    return np.linalg.solve(np.eye(spec.n_states) - spec.gamma * P_pi, r_pi)


def policy_iteration(spec: EnvSpec):
    """Policy iteration (class 1.4): exact policy evaluation (Bellman eq.) + greedy improvement,
    until the policy is stable. Converges in a finite number of iterations.

    Returns Q, V, policy and the number of improvement iterations.
    """
    tensors = model_tensors(spec)
    T, R = tensors
    policy = np.zeros(spec.n_states, dtype=int)
    for k in range(1, 1000):
        V = evaluate_policy(spec, policy, tensors)    # policy evaluation
        Q = R + spec.gamma * T @ V                    # policy improvement
        new = Q.argmax(axis=1)
        # keep the current action on ties, so the loop cannot cycle between equivalent policies
        stable = Q[np.arange(spec.n_states), policy] >= Q.max(axis=1) - 1e-12
        new[stable] = policy[stable]
        if np.array_equal(new, policy):
            return Q, V, policy, k
        policy = new
    raise RuntimeError("policy iteration did not converge")
