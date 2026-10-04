"""What one traffic light "sees" and how the trained policy drives the lights.

Kept separate from rl.py so the API server can run the AI lights without
installing Gymnasium, Stable-Baselines3, or PyTorch.

Observation (19 numbers per intersection):
    - waiting cars on each of the 4 incoming roads (N, S, E, W)
    - moving cars on each incoming road (traffic about to arrive)
    - cars on each of the 4 outgoing roads (is there room downstream?)
    - current phase, one-hot (6 phases)
    - seconds in the current phase, scaled
"""

import numpy as np

from .car import WAITING_SPEED
from .lights import NS_GREEN, NS_YELLOW, EW_GREEN, EW_YELLOW, ALL_RED_1, ALL_RED_2

PHASES = [NS_GREEN, NS_YELLOW, ALL_RED_1, EW_GREEN, EW_YELLOW, ALL_RED_2]
SIDES = ["N", "S", "E", "W"]
OBS_SIZE = 19
DECISION_INTERVAL = 2.5     # seconds of sim time between decisions
QUEUE_SCALE = 10.0          # divide car counts by this so inputs stay around 0..1
REWARD_SCALE = 100.0


def _side(city, node_id, other_id):
    node, other = city.nodes[node_id], city.nodes[other_id]
    if other.y < node.y:
        return "N"
    if other.y > node.y:
        return "S"
    if other.x > node.x:
        return "E"
    return "W"


class LocalView:
    """Precomputes which edges belong to each intersection so building the
    observation every step is fast."""

    def __init__(self, sim):
        city = sim.city
        self.sim = sim
        self.nodes = list(city.intersections)
        self.incoming = {}
        self.outgoing = {}
        for n in self.nodes:
            self.incoming[n] = {s: None for s in SIDES}
            self.outgoing[n] = {s: None for s in SIDES}
            for e in city.incoming_edges(n):
                self.incoming[n][_side(city, n, e.start)] = e
            for eid in city.out_edges[n]:
                e = city.edges[eid]
                self.outgoing[n][_side(city, n, e.end)] = e

    def waiting_on(self, n):
        total = 0
        for e in self.incoming[n].values():
            if e is not None:
                total += sum(1 for c in e.cars if c.speed < WAITING_SPEED)
        return total

    def observe(self, n):
        light = self.sim.lights[n]
        obs = np.zeros(OBS_SIZE, dtype=np.float32)
        for i, s in enumerate(SIDES):
            e = self.incoming[n][s]
            if e is not None:
                waiting = sum(1 for c in e.cars if c.speed < WAITING_SPEED)
                obs[i] = waiting / QUEUE_SCALE
                obs[4 + i] = (len(e.cars) - waiting) / QUEUE_SCALE
            e = self.outgoing[n][s]
            if e is not None:
                obs[8 + i] = len(e.cars) / QUEUE_SCALE
        obs[12 + PHASES.index(light.phase)] = 1.0
        obs[18] = min(light.time_in_phase / 60.0, 1.0)
        return obs

    def observe_all(self):
        return np.stack([self.observe(n) for n in self.nodes])


def apply_actions(sim, view, actions):
    for n, a in zip(view.nodes, actions):
        if a == 1:
            sim.lights[n].request_switch()


class AIController:
    """Runs a trained model on every light in a live Simulation.
    Call it once per tick; it only decides every DECISION_INTERVAL seconds."""

    def __init__(self, model):
        self.model = model
        self.view = None
        self.next_decision = 0.0

    def __call__(self, sim):
        if self.view is None or self.view.sim is not sim:
            self.view = LocalView(sim)
            self.next_decision = sim.time
        if sim.time + 1e-9 < self.next_decision:
            return
        obs = self.view.observe_all()
        actions, _ = self.model.predict(obs, deterministic=True)
        apply_actions(sim, self.view, actions)
        self.next_decision = sim.time + DECISION_INTERVAL
