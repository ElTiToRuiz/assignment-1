"""Quick checks (about 2 s): the ground truth is right, the agents learn, and the tooling works."""
import numpy as np

from tabular_rl import models, training
from tabular_rl.agents import ALGORITHMS, expected_sarsa, q_learning
from tabular_rl.class_gridworld import GridworldEnv
from tabular_rl.envs import greedy_path, make_spec
from tabular_rl.planning import evaluate_policy, policy_iteration, value_iteration
from tabular_rl.simulator import Simulator
from tabular_rl.training import run_many


def test_value_iteration_matches_closed_form_deterministic():
    spec = make_spec("gridworld_deterministic")
    _, V, pi = value_iteration(spec)
    g = spec.gamma
    assert np.isclose(V[2], 1.0)                    # one step from the goal
    assert np.isclose(V[spec.start_state], g ** 4)  # 8->4->0->1->2->3: the +1 arrives on the 5th step
    assert np.allclose(evaluate_policy(spec, pi), V)


def test_policy_iteration_matches_value_iteration():
    for env_name in ["gridworld_deterministic", "gridworld_slippery", "cliff_walking"]:
        spec = make_spec(env_name)
        _, V_vi, _ = value_iteration(spec)
        _, V_pi, pi, _ = policy_iteration(spec)
        assert np.allclose(V_pi, V_vi, atol=1e-8)
        assert np.allclose(evaluate_policy(spec, pi), V_vi, atol=1e-8)


def test_simulator_matches_class_env_transitions():
    spec = make_spec("gridworld_slippery")
    assert GridworldEnv(is_slippery=True).P == spec.P
    sim, n, hits = Simulator(spec, seed=1), 20000, np.zeros(spec.n_states)
    for _ in range(n):
        sim.s = 10
        hits[sim.step(2)[0]] += 1  # RIGHT from state 10
    for p, s2, _, _ in spec.P[10][2]:
        assert abs(hits[s2] / n - p) < 0.02


def test_all_algorithms_learn_optimal_policy_deterministic():
    spec = make_spec("gridworld_deterministic")
    for name in ALGORITHMS:
        out = run_many(name, spec, seeds=[0], n_episodes=1500, alpha=0.2)
        assert out["regret"][0, -1] < 1e-9, name


def test_expected_sarsa_with_greedy_policy_equals_q_learning():
    """With eps = 0 the average over the next action is just the max, so the updates are identical."""
    spec = make_spec("gridworld_slippery")
    kw = dict(n_episodes=200, alpha=0.3, eps=(0.0, 0.0, 1.0), seed=3)
    assert np.allclose(expected_sarsa(spec, **kw).Q, q_learning(spec, **kw).Q)


def test_cliff_greedy_path_of_optimal_q_is_shortest():
    spec = make_spec("cliff_walking")
    Q_star, _, _ = value_iteration(spec)
    assert len(greedy_path(spec, Q_star)) - 1 == 13  # up, 11 x right, down


def test_cache_and_model_roundtrip(tmp_path, monkeypatch):
    monkeypatch.setattr(training, "CACHE_DIR", tmp_path / "cache")
    monkeypatch.setattr(models, "MODELS_DIR", tmp_path / "models")
    kw = dict(n_episodes=50, alpha=0.2)
    a = training.train("SARSA", "gridworld_deterministic", [0, 1], **kw)
    assert len(list((tmp_path / "cache").glob("*.npz"))) == 1
    b = training.train("SARSA", "gridworld_deterministic", [0, 1], **kw)  # loaded, not retrained
    assert np.allclose(a["Q"], b["Q"], atol=1e-6)
    models.save_model("gridworld_deterministic", "SARSA", a["Q"].mean(0))
    assert np.allclose(models.load_model("gridworld_deterministic", "SARSA"), a["Q"].mean(0))


def test_stats_helpers():
    from tabular_rl.stats import bootstrap_ci, permutation_test
    rng = np.random.default_rng(0)
    a, b = rng.normal(0, 1, 40), rng.normal(2, 1, 40)
    lo, hi = bootstrap_ci(a)
    assert lo < a.mean() < hi
    assert permutation_test(a, b, n_perm=2000) < 0.01      # clearly different
    assert permutation_test(a, a.copy(), n_perm=2000) > 0.5  # identical groups


def test_tune_default_trial_reproduces_default_config():
    import optuna
    from experiments.tune import ALGOS, ENVS, default_kwargs, default_params, suggest
    for env in ENVS:
        for algo in ALGOS:
            kw = suggest(optuna.trial.FixedTrial(default_params(env, algo)), env, algo)
            ref = default_kwargs(env, algo)
            assert np.isclose(kw["alpha"], ref["alpha"]) and "alpha_decay" not in kw
            assert np.allclose(kw["eps"][:2], ref["eps"][:2])
            assert abs(kw["eps"][2] - ref["eps"][2]) < 1e-6 or ref["eps"][2] == 1.0
            assert kw.get("n") == ref.get("n")
