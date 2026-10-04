"""Reinforcement learning environment for the traffic lights.

Design: every intersection is controlled by the SAME small policy network
(parameter sharing). Each intersection only sees its own neighborhood:

    observation (19 numbers per intersection)
        - waiting cars on each of the 4 incoming roads (N, S, E, W)
        - moving cars on each incoming road (traffic about to arrive)
        - cars on each of the 4 outgoing roads (is there room downstream?)
        - current phase, one-hot (6 phases)
        - seconds in the current phase, scaled

    action: 0 = keep the current light, 1 = switch
    reward: minus the waiting time that piled up on its incoming roads

Because the policy only ever sees one intersection, a model trained on a
3x3 grid can run every light in a 6x6 (or any size) city.

CityEnv is a normal Gymnasium env for the whole grid. SharedPolicyVecEnv
turns N copies of it into a stable-baselines3 VecEnv where each
intersection is its own "env", which is how SB3's PPO trains one shared
policy on all of them at once.
"""

import gymnasium as gym
import numpy as np
from gymnasium import spaces
from stable_baselines3.common.vec_env.base_vec_env import VecEnv

from .control import DECISION_INTERVAL, OBS_SIZE, REWARD_SCALE, LocalView, apply_actions
from .sim import Simulation


class CityEnv(gym.Env):
    """Whole-grid environment: obs is (n_intersections, 19), action is one
    keep/switch bit per intersection. info["local_rewards"] has each
    intersection's own reward."""

    metadata = {"render_modes": []}

    def __init__(self, rows=3, cols=3, episode_seconds=600, spawn_rate=None,
                 rate_range=(20, 70), seed=None):
        super().__init__()
        self.rows, self.cols = rows, cols
        self.episode_seconds = episode_seconds
        self.fixed_rate = spawn_rate
        self.rate_range = rate_range
        self.n = rows * cols
        self.observation_space = spaces.Box(0.0, np.inf, shape=(self.n, OBS_SIZE), dtype=np.float32)
        self.action_space = spaces.MultiBinary(self.n)
        self._np_rng = np.random.default_rng(seed)
        self.sim = None

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            self._np_rng = np.random.default_rng(seed)
        sim_seed = int(self._np_rng.integers(1 << 31))
        # train on a range of traffic levels so the policy doesn't overfit one
        rate = self.fixed_rate if self.fixed_rate is not None else float(self._np_rng.uniform(*self.rate_range))
        self.sim = Simulation(self.rows, self.cols, seed=sim_seed, spawn_rate=rate, light_mode="ai")
        self.view = LocalView(self.sim)
        # warm up so the episode doesn't start with an empty city
        self.sim.run(60)
        return self.view.observe_all(), {}

    def step(self, action):
        apply_actions(self.sim, self.view, action)
        ticks = int(round(DECISION_INTERVAL / self.sim.dt))
        local_wait = np.zeros(self.n, dtype=np.float32)
        for _ in range(ticks):
            self.sim.step()
            for i, n in enumerate(self.view.nodes):
                local_wait[i] += self.view.waiting_on(n) * self.sim.dt
        local_rewards = -local_wait / REWARD_SCALE
        done = self.sim.time >= self.episode_seconds + 60
        info = {"local_rewards": local_rewards}
        return self.view.observe_all(), float(local_rewards.sum()), False, done, info


class SharedPolicyVecEnv(VecEnv):
    """Exposes every intersection of `num_cities` CityEnvs as one VecEnv slot."""

    def __init__(self, num_cities=4, rows=3, cols=3, seed=0, **env_kwargs):
        self.envs = [CityEnv(rows, cols, seed=seed + i, **env_kwargs) for i in range(num_cities)]
        self.per_city = rows * cols
        obs_space = spaces.Box(0.0, np.inf, shape=(OBS_SIZE,), dtype=np.float32)
        super().__init__(num_cities * self.per_city, obs_space, spaces.Discrete(2))
        self._actions = None

    def reset(self):
        obs = [env.reset()[0] for env in self.envs]
        return np.concatenate(obs)

    def step_async(self, actions):
        self._actions = np.asarray(actions).reshape(len(self.envs), self.per_city)

    def step_wait(self):
        all_obs, all_rew, all_done, infos = [], [], [], []
        for env, act in zip(self.envs, self._actions):
            obs, _, _, done, info = env.step(act)
            rew = info["local_rewards"]
            if done:
                final = obs
                obs, _ = env.reset()
                for i in range(self.per_city):
                    infos.append({"terminal_observation": final[i], "TimeLimit.truncated": True})
            else:
                infos.extend({} for _ in range(self.per_city))
            all_obs.append(obs)
            all_rew.append(rew)
            all_done.append(np.full(self.per_city, done))
        return (np.concatenate(all_obs), np.concatenate(all_rew).astype(np.float32),
                np.concatenate(all_done), infos)

    def close(self):
        pass

    def get_attr(self, attr_name, indices=None):
        return [getattr(self, attr_name, None)] * len(self._get_indices(indices))

    def set_attr(self, attr_name, value, indices=None):
        pass

    def env_method(self, method_name, *args, indices=None, **kwargs):
        return [None] * len(self._get_indices(indices))

    def env_is_wrapped(self, wrapper_class, indices=None):
        return [False] * len(self._get_indices(indices))

    def seed(self, seed=None):
        return [None] * self.num_envs
