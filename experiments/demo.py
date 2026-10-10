"""Watch a trained agent play the class gridworld. No training: it loads a saved Q-table from
results/models/ and always takes the greedy action.

    uv run python -m experiments.demo                     # Q-learning, deterministic grid, pygame window
    uv run python -m experiments.demo --env gridworld_slippery --algo SARSA --render ansi --episodes 3
"""
import argparse
import time

from tabular_rl.agents import ALGORITHMS
from tabular_rl.class_gridworld import GridworldEnv
from tabular_rl.envs import ARROWS
from tabular_rl.models import load_model

WALL, GOAL, PIT = 5, 3, 7  # special cells of the 3x4 class gridworld


def print_policy(policy):
    for row in range(3):
        cells = []
        for col in range(4):
            s = 4 * row + col
            cells.append({WALL: "#", GOAL: "G", PIT: "P"}.get(s, ARROWS[policy[s]]))
        print("  " + " ".join(cells))


def play(env, policy, render):
    s, _ = env.reset()
    total, done, steps = 0.0, False, 0
    while not done and steps < 100:
        if render == "ansi":
            print(env.render())
            time.sleep(0.3)
        s, r, done, _, _ = env.step(int(policy[s]))
        total, steps = total + r, steps + 1
    return total, steps


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--env", default="gridworld_deterministic", choices=["gridworld_deterministic", "gridworld_slippery"])
    p.add_argument("--algo", default="Q-learning", choices=list(ALGORITHMS))
    p.add_argument("--render", default="human", choices=["human", "ansi"])
    p.add_argument("--episodes", type=int, default=1)
    args = p.parse_args()

    policy = load_model(args.env, args.algo).argmax(axis=1)
    print(f"{args.algo} on {args.env}. Greedy policy:")
    print_policy(policy)

    env = GridworldEnv(render_mode=args.render, is_slippery=args.env == "gridworld_slippery")
    for ep in range(args.episodes):
        total, steps = play(env, policy, args.render)
        print(f"episode {ep + 1}: return {total:+.1f} in {steps} steps")
    time.sleep(0.5)
    env.close()
