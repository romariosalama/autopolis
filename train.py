"""Train the PPO traffic-light policy.

    python train.py --steps 300000

One shared policy runs every intersection (see engine/rl.py). Training
uses several 3x3 cities at once with random traffic levels.
"""

import argparse
import os
import time

import torch
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import VecMonitor

from engine.policy import export_policy
from engine.rl import SharedPolicyVecEnv


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=300_000)
    parser.add_argument("--cities", type=int, default=4)
    parser.add_argument("--size", type=int, default=3)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", default="models/ppo_lights.zip")
    args = parser.parse_args()

    torch.set_num_threads(2)
    env = VecMonitor(SharedPolicyVecEnv(num_cities=args.cities, rows=args.size, cols=args.size, seed=args.seed))

    model = PPO(
        "MlpPolicy",
        env,
        n_steps=128,            # per intersection, so 128 * 36 = 4608 samples per update
        batch_size=512,
        n_epochs=5,
        gamma=0.97,             # decisions are 2.5 s apart, so this looks ~80 s ahead
        learning_rate=3e-4,
        ent_coef=0.01,
        policy_kwargs={"net_arch": [64, 64]},
        seed=args.seed,
        device="cpu",
        verbose=1,
    )

    start = time.perf_counter()
    model.learn(total_timesteps=args.steps)
    print(f"trained {args.steps} steps in {time.perf_counter() - start:.0f}s")

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    model.save(args.out)
    npz = os.path.join(os.path.dirname(args.out), "policy.npz")
    export_policy(model, npz)
    print(f"saved {args.out} and {npz}")


if __name__ == "__main__":
    main()
