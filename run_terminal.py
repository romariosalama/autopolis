"""Quick headless run that prints city stats to the terminal."""
import argparse
import time

from engine.sim import Simulation

parser = argparse.ArgumentParser()
parser.add_argument("--size", type=int, default=3)
parser.add_argument("--rate", type=float, default=30.0, help="cars per minute")
parser.add_argument("--minutes", type=float, default=10.0)
parser.add_argument("--seed", type=int, default=0)
args = parser.parse_args()

sim = Simulation(rows=args.size, cols=args.size, seed=args.seed, spawn_rate=args.rate)
start = time.perf_counter()
for minute in range(int(args.minutes)):
    sim.run(60)
    s = sim.stats()
    print(f"t={s['time']:6.0f}s  cars={s['cars']:4d}  waiting={s['waiting']:4d}  "
          f"trips={s['trips_completed']:5d}  avg_wait={s['avg_wait']:5.1f}s  "
          f"cars/min={s['cars_per_minute']:5.1f}  backlog={s['spawn_backlog']}")
elapsed = time.perf_counter() - start
print(f"\nsimulated {sim.time:.0f}s in {elapsed:.2f}s real time ({sim.time / elapsed:.0f}x)")
