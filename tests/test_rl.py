import os

import numpy as np
import pytest

from engine.control import DECISION_INTERVAL, OBS_SIZE, AIController, LocalView
from engine.lights import EW_GREEN
from engine.sim import Simulation

MODEL_NPZ = "models/policy.npz"
MODEL_ZIP = "models/ppo_lights.zip"


def test_observation_shape_and_phase_one_hot():
    sim = Simulation(3, 3, seed=0, spawn_rate=40)
    sim.run(60)
    obs = LocalView(sim).observe_all()
    assert obs.shape == (9, OBS_SIZE)
    assert np.allclose(obs[:, 12:18].sum(axis=1), 1.0)
    assert (obs >= 0).all()


def test_corner_intersection_has_no_missing_roads():
    # gates mean every intersection, even corners, has 4 incoming roads
    sim = Simulation(3, 3)
    view = LocalView(sim)
    for n in view.nodes:
        assert all(e is not None for e in view.incoming[n].values())


def test_waiting_cars_show_up_in_observation():
    sim = Simulation(1, 1, seed=0, spawn_rate=60, light_mode="ai")
    sim.lights[sim.city.intersections[0]].phase = EW_GREEN
    sim.run(120)
    obs = LocalView(sim).observe_all()[0]
    assert obs[0] + obs[1] > 0   # cars queued on the red north/south roads


def test_gate_queue_time_counts_as_waiting():
    sim = Simulation(1, 1, seed=0, spawn_rate=0)
    gate = sim.city.gates[0]
    car = sim.spawn_car(gate)
    car.spawn_time = -30.0   # pretend it waited 30 s to get in
    sim.step()
    assert car.wait_time >= 30.0


class AlwaysSwitch:
    def __init__(self):
        self.calls = 0

    def predict(self, obs, deterministic=True):
        self.calls += 1
        return np.ones(len(obs), dtype=int), None


def test_controller_decides_on_an_interval():
    sim = Simulation(2, 2, seed=0, light_mode="ai")
    model = AlwaysSwitch()
    ctrl = AIController(model)
    seconds = 30
    for _ in range(int(seconds / sim.dt)):
        ctrl(sim)
        sim.step()
    assert model.calls == pytest.approx(seconds / DECISION_INTERVAL, abs=1)


def test_controller_lights_still_cycle_safely():
    sim = Simulation(2, 2, seed=0, light_mode="ai")
    ctrl = AIController(AlwaysSwitch())
    phases = set()
    for _ in range(int(60 / sim.dt)):
        ctrl(sim)
        sim.step()
        phases.add(sim.lights[sim.city.intersections[0]].phase)
    assert len(phases) == 6   # went through yellow and all-red, never skipped them


# ---------- tests that need the training stack ----------

def test_city_env_step():
    pytest.importorskip("gymnasium")
    pytest.importorskip("stable_baselines3")
    from engine.rl import CityEnv
    env = CityEnv(2, 2, episode_seconds=20, seed=0)
    obs, _ = env.reset()
    assert env.observation_space.contains(obs)
    done = False
    steps = 0
    while not done:
        obs, reward, _, done, info = env.step(env.action_space.sample())
        assert reward <= 0
        assert info["local_rewards"].shape == (4,)
        steps += 1
    assert steps == pytest.approx(20 / DECISION_INTERVAL, abs=1)


def test_shared_policy_vec_env_shapes():
    pytest.importorskip("stable_baselines3")
    from engine.rl import SharedPolicyVecEnv
    venv = SharedPolicyVecEnv(num_cities=2, rows=2, cols=2, episode_seconds=10)
    obs = venv.reset()
    assert obs.shape == (8, OBS_SIZE)
    obs, rew, done, infos = venv.step(np.zeros(8, dtype=int))
    assert obs.shape == (8, OBS_SIZE) and rew.shape == (8,) and len(infos) == 8


@pytest.mark.skipif(not (os.path.exists(MODEL_NPZ) and os.path.exists(MODEL_ZIP)), reason="no trained model")
def test_numpy_policy_matches_sb3():
    sb3 = pytest.importorskip("stable_baselines3")
    from engine.policy import NumpyPolicy
    model = sb3.PPO.load(MODEL_ZIP, device="cpu")
    fast = NumpyPolicy(MODEL_NPZ)
    sim = Simulation(4, 4, seed=7, spawn_rate=80, light_mode="ai")
    view = LocalView(sim)
    for _ in range(20):
        sim.run(10)
        obs = view.observe_all()
        expected, _ = model.predict(obs, deterministic=True)
        got, _ = fast.predict(obs)
        assert (expected == got).all()
