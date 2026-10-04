"""Watch a saved agent (no training): loads results/models/<env>__<algo>.npy and runs the greedy policy
in the class GridworldEnv.

    python -m experiments.demo                                  # Q-learning, deterministic, pygame window
    python -m experiments.demo --env gridworld_slippery --algo SARSA --render ansi --episodes 3
"""
import argparse
import time

import numpy as np

from class_code.env import GridworldEnv
from tabular_rl.agents import ALGORITHMS
from tabular_rl.envs import ARROWS
from tabular_rl.runner import load_model

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--env", default="gridworld_deterministic", choices=["gridworld_deterministic", "gridworld_slippery"])
    p.add_argument("--algo", default="Q-learning", choices=list(ALGORITHMS))
    p.add_argument("--render", default="human", choices=["human", "ansi"])
    p.add_argument("--episodes", type=int, default=1)
    args = p.parse_args()

    Q = load_model(args.env, args.algo)
    policy = Q.argmax(axis=1)
    print(f"{args.algo} on {args.env}. Greedy policy:")
    for r in range(3):
        print("  " + " ".join("#" if 4 * r + c == 5 else "G" if 4 * r + c == 3 else "P" if 4 * r + c == 7
                               else ARROWS[policy[4 * r + c]] for c in range(4)))
    env = GridworldEnv(render_mode=args.render, is_slippery=args.env == "gridworld_slippery")
    for ep in range(args.episodes):
        s, _ = env.reset()
        G, done, steps = 0.0, False, 0
        while not done and steps < 100:
            if args.render == "ansi":
                print(env.render()); time.sleep(0.3)
            s, r, done, _, _ = env.step(int(policy[s]))
            G, steps = G + r, steps + 1
        print(f"episode {ep + 1}: return {G:+.1f} in {steps} steps")
    time.sleep(0.5)
    env.close()
