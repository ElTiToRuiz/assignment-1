import numpy as np

from class_code.env import GridworldEnv
from tabular_rl import runner
from tabular_rl.agents import ALGORITHMS, expected_sarsa, q_learning
from tabular_rl.dp import evaluate_policy, value_iteration
from tabular_rl.envs import Simulator, greedy_path, make_spec
from tabular_rl.runner import run_many


def test_value_iteration_matches_closed_form_deterministic():
    spec = make_spec("gridworld_deterministic")
    _, V, pi = value_iteration(spec)
    g = spec.gamma
    assert np.isclose(V[2], 1.0)                    # one step from the goal
    assert np.isclose(V[spec.start_state], g ** 4)  # 8->4->0->1->2->3: reward +1 on the 5th step
    assert np.allclose(evaluate_policy(spec, pi), V)


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
    """With eps=0 the expectation over the greedy policy is the max -> identical updates."""
    spec = make_spec("gridworld_slippery")
    kw = dict(n_episodes=200, alpha=0.3, eps=(0.0, 0.0, 1.0), seed=3)
    assert np.allclose(expected_sarsa(spec, **kw).Q, q_learning(spec, **kw).Q)


def test_cliff_greedy_path_of_optimal_q_is_shortest():
    spec = make_spec("cliff_walking")
    Q_star, _, _ = value_iteration(spec)
    assert len(greedy_path(spec, Q_star)) - 1 == 13  # up, 11 x right, down


def test_cache_and_model_roundtrip(tmp_path, monkeypatch):
    monkeypatch.setattr(runner, "CACHE_DIR", tmp_path / "cache")
    monkeypatch.setattr(runner, "MODELS_DIR", tmp_path / "models")
    kw = dict(n_episodes=50, alpha=0.2)
    a = runner.train("SARSA", "gridworld_deterministic", [0, 1], **kw)
    assert len(list((tmp_path / "cache").glob("*.npz"))) == 1
    b = runner.train("SARSA", "gridworld_deterministic", [0, 1], **kw)  # loaded, not retrained
    assert np.allclose(a["Q"], b["Q"], atol=1e-6)
    runner.save_model("gridworld_deterministic", "SARSA", a["Q"].mean(0))
    assert np.allclose(runner.load_model("gridworld_deterministic", "SARSA"), a["Q"].mean(0))
