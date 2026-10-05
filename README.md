# Autopolis

A 3D traffic simulation where a reinforcement learning agent controls the traffic lights.

**Live demo: [autopolis.app](https://autopolis.app)**

The simulation runs in Python and streams to the browser over a WebSocket, where React and Three.js render it. You can watch hundreds of cars drive around the city, add more traffic, start rush hour, and switch the lights between fixed timers and the trained AI. The chart on the left shows average wait time, with the AI periods shaded green, so you can see the wait drop after switching.

![Autopolis running a 6x6 city](docs/screenshot.png)

## Results

I trained the AI with PPO on a 3x3 grid and compared it to fixed-timer lights on the same traffic (5 random seeds, 3 traffic levels, 15 simulated minutes each). I tested several timer lengths instead of just one, since comparing against a badly tuned timer would make the AI look better than it is.

![Average wait per trip: fixed timers vs AI](docs/results.png)

| Grid | Best fixed timer | Default timer (30s) | AI (PPO) | vs best fixed | vs default |
|---|---|---|---|---|---|
| 3x3 (trained on) | 16.1s | 27.2s | 12.3s | -23.7% | -55.0% |
| 6x6 (not trained on) | 39.7s | 51.6s | 28.2s | -28.9% | -45.3% |

The AI completes at least as many trips as the timers, so it isn't lowering wait times by holding cars back. Time spent waiting to enter the city also counts as waiting, so blocking the entrances doesn't help either.

Other numbers (from `measure.py`):

- Runs headless at about 2,260x real time on a 3x3 city and 590x on a 6x6 city with ~250 cars
- The 6x6 city peaks at 722 cars at once during rush hour

To reproduce: `python benchmark.py` and `python benchmark.py --size 6 --rates 100 150`. Raw results are in `docs/`.

## How it works

There are three parts:

- `engine/`: the simulation, in plain Python with no rendering
- `api/`: a FastAPI server that runs one shared city and streams it to browsers over a WebSocket 10 times a second
- `web/`: the React + Three.js frontend that draws the city and sends controls back

Keeping the engine separate from the frontend means the same code can run without a browser for training, which is much faster than real time.

### Simulation

- The city is a grid of intersections connected by two-way roads, stored as a directed graph with one edge per direction. Cars enter and leave through gates on the edges of the map.
- Each car gets a route from A* search, using Manhattan distance as the heuristic. A bit of random cost is added so cars spread out across routes that are the same length.
- Cars follow each other using the [Intelligent Driver Model](https://en.wikipedia.org/wiki/Intelligent_driver_model), which sets each car's acceleration based on the gap to the car (or red light) in front. There's also a hard limit so cars can never overlap, and a test checks this every tick.
- Each light goes green, yellow, all-red, then the other direction gets green. The all-red part matters because it means switching costs a few seconds, so switching constantly isn't free.

### AI lights

Every light uses the same small neural network. Each light only looks at its own intersection:

- Input (19 numbers): stopped and moving cars on each road coming in, cars on each road going out, the current phase, and how long it's been in that phase
- Output: keep the light the same or switch it. It still has to go through yellow and all-red, and has to stay green for at least 5 seconds.
- Reward: negative waiting time on the roads coming into that intersection

For training I wrote a custom Stable-Baselines3 `VecEnv` where every intersection in 4 parallel cities counts as its own environment, so PPO trains one shared policy on 36 intersections at the same time. Since the policy only sees one intersection, a model trained on 3x3 also works on a 6x6 city, which is what the second row of the results table tests.

The trained network is exported to NumPy (`engine/policy.py`) so the server doesn't need PyTorch installed. There's a test that checks the NumPy version picks the same actions as the original model.


## Project structure

```
engine/   city.py, routing.py, car.py, lights.py, sim.py (simulation)
          control.py, rl.py, policy.py (AI lights)
api/      FastAPI server
web/      React + react-three-fiber frontend
tests/    pytest tests
models/   trained policy
docs/     screenshots and benchmark results
```

## Limitations

- Intersections are treated as points, so turning conflicts inside an intersection aren't simulated
- One lane per direction and no lane changes
- Everyone on the live site shares the same city and controls
- I've only tested the AI on grid cities
