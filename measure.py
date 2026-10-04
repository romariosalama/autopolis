"""Measures headless speed and how many cars the demo city holds at once."""
import json
import time

from engine.control import AIController
from engine.policy import NumpyPolicy
from engine.sim import Simulation

results = {}

# headless speed: how many seconds of city time per second of real time
for size, rate in [(3, 50), (6, 120), (6, 300)]:
    sim = Simulation(size, size, seed=1, spawn_rate=rate)
    sim.run(120)  # warm up
    start = time.perf_counter()
    sim.run(600)
    elapsed = time.perf_counter() - start
    key = f"{size}x{size}@{rate}"
    results[f"speed_{key}"] = round(600 / elapsed)
    print(f"{key:>10} cars/min: {600 / elapsed:6.0f}x real time with ~{len(sim.cars)} cars")

# the live demo city (6x6) during rush hour, AI lights on
sim = Simulation(6, 6, seed=1, spawn_rate=120, light_mode="ai")
sim.rush_hour = True
ctrl = AIController(NumpyPolicy("models/policy.npz"))
peak = 0
for _ in range(int(900 / sim.dt)):
    ctrl(sim)
    sim.step()
    peak = max(peak, len(sim.cars))
results["peak_cars_6x6_rush_hour"] = peak
print(f"6x6 rush hour: peak {peak} cars on the road at once")

with open("docs/measurements.json", "w") as f:
    json.dump(results, f, indent=2)
