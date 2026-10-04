"""Measures fixed-timer lights vs the trained AI lights on the same traffic.

Every controller is run on the exact same seeds and traffic levels, so the
only difference between runs is who controls the lights.

    python benchmark.py                       # 3x3 grid, default settings
    python benchmark.py --size 6 --rates 100 150
"""

import argparse
import json
import statistics
import time

from engine.sim import Simulation

SEEDS = [101, 202, 303, 404, 505]
DEFAULT_RATES = [30, 50, 70]
GREEN_TIMES = [10, 15, 20, 30, 40]   # fixed-timer settings to compare against


def run(size, rate, seed, minutes, green_time=30, controller=None):
    mode = "ai" if controller else "fixed"
    sim = Simulation(size, size, seed=seed, spawn_rate=rate, light_mode=mode, green_time=green_time)
    ticks = int(minutes * 60 / sim.dt)
    for _ in range(ticks):
        if controller:
            controller(sim)
        sim.step()
    return {
        "avg_wait": sim.avg_wait(),
        "trips": sim.trips_completed,
        "backlog": sum(len(q) for q in sim.spawn_queues.values()),
    }


def evaluate(size, rates, minutes, green_time=30, make_controller=None):
    """Average wait over all seeds and rates (each run weighted equally)."""
    waits, trips = [], []
    for rate in rates:
        for seed in SEEDS:
            controller = make_controller() if make_controller else None
            r = run(size, rate, seed, minutes, green_time, controller)
            waits.append(r["avg_wait"])
            trips.append(r["trips"])
    return statistics.mean(waits), statistics.mean(trips), waits


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--size", type=int, default=3)
    parser.add_argument("--rates", type=float, nargs="+", default=DEFAULT_RATES)
    parser.add_argument("--minutes", type=float, default=20)
    parser.add_argument("--model", default="models/policy.npz")
    parser.add_argument("--no-ai", action="store_true")
    parser.add_argument("--out", default=None, help="save results as JSON")
    args = parser.parse_args()

    print(f"{args.size}x{args.size} grid, rates {args.rates} cars/min, "
          f"{len(SEEDS)} seeds, {args.minutes:g} sim-minutes each\n")

    results = {"size": args.size, "rates": args.rates, "seeds": SEEDS, "minutes": args.minutes, "fixed": {}}
    for g in GREEN_TIMES:
        wait, trips, _ = evaluate(args.size, args.rates, args.minutes, green_time=g)
        results["fixed"][g] = {"avg_wait": wait, "trips": trips}
        print(f"fixed timer {g:>2}s green   avg wait {wait:6.1f}s   trips/run {trips:7.0f}")

    default = results["fixed"][30]["avg_wait"]
    best_g = min(results["fixed"], key=lambda g: results["fixed"][g]["avg_wait"])
    best = results["fixed"][best_g]["avg_wait"]

    if not args.no_ai:
        from engine.policy import NumpyPolicy
        from engine.control import AIController
        model = NumpyPolicy(args.model)
        wait, trips, _ = evaluate(args.size, args.rates, args.minutes,
                                  make_controller=lambda: AIController(model))
        results["ai"] = {"avg_wait": wait, "trips": trips}
        print(f"AI (PPO)                avg wait {wait:6.1f}s   trips/run {trips:7.0f}")
        print(f"\nvs default 30s timer:     {100 * (default - wait) / default:5.1f}% less waiting")
        print(f"vs best fixed ({best_g}s):     {100 * (best - wait) / best:5.1f}% less waiting")
        results["reduction_vs_default"] = (default - wait) / default
        results["reduction_vs_best_fixed"] = (best - wait) / best
        results["best_fixed_green"] = best_g

    if args.out:
        with open(args.out, "w") as f:
            json.dump(results, f, indent=2)


if __name__ == "__main__":
    start = time.perf_counter()
    main()
    print(f"\n(took {time.perf_counter() - start:.0f}s)")
