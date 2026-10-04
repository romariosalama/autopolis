"""FastAPI server: runs one shared city and streams it over a WebSocket.

The simulation runs in a background task. Every 100 ms it advances the
city by (0.1 s * speed) of sim time and sends a snapshot to every
connected browser, so the stream is 10 updates a second.
"""

import asyncio
import os
import time
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from engine.policy import NumpyPolicy
from engine.control import AIController
from engine.sim import Simulation

CITY_SIZE = int(os.getenv("CITY_SIZE", "6"))
SPAWN_RATE = float(os.getenv("SPAWN_RATE", "120"))
FRAME_TIME = 0.1          # seconds between frames (10 Hz)
IDLE_PAUSE = os.getenv("IDLE_PAUSE", "1") == "1"
MAX_CARS = 1500           # keeps a public demo from being spammed into the ground
MODEL_PATH = os.getenv("MODEL_PATH", "models/policy.npz")
HISTORY_EVERY = 5.0       # sim seconds between points on the dashboard chart
HISTORY_LEN = 120         # 120 points * 5 s = the last 10 minutes of city time


class SimRunner:
    def __init__(self):
        self.ai_policy = None
        if os.path.exists(MODEL_PATH):
            self.ai_policy = AIController(NumpyPolicy(MODEL_PATH))
        self.reset()

    def reset(self, seed=0):
        self.sim = Simulation(CITY_SIZE, CITY_SIZE, seed=seed, spawn_rate=SPAWN_RATE)
        self.paused = False
        self.speed = 1.0
        self._sim_time_owed = 0.0
        # points for the wait-time chart: [sim time, avg wait last minute, light mode]
        self.history = []
        self.samples_taken = 0
        self._next_sample = HISTORY_EVERY

    def advance(self, real_seconds):
        if self.paused:
            return
        self._sim_time_owed += real_seconds * self.speed
        # cap catch-up so a slow frame can't snowball
        self._sim_time_owed = min(self._sim_time_owed, 2.0)
        while self._sim_time_owed >= self.sim.dt:
            if self.sim.light_mode == "ai" and self.ai_policy is not None:
                self.ai_policy(self.sim)
            self.sim.step()
            self._sim_time_owed -= self.sim.dt
            if self.sim.time >= self._next_sample:
                self._take_sample()

    def _take_sample(self):
        self._next_sample += HISTORY_EVERY
        self.history.append([round(self.sim.time, 1), round(self.sim.recent_avg_wait(), 2), self.sim.light_mode])
        self.samples_taken += 1
        if len(self.history) > HISTORY_LEN:
            self.history.pop(0)

    def samples_since(self, count):
        """History points taken after the first `count` samples."""
        new = self.samples_taken - count
        if new <= 0:
            return []
        return self.history[-min(new, len(self.history)):]


runner = SimRunner()
clients = set()


async def sim_loop():
    last = time.perf_counter()
    sent_samples = runner.samples_taken
    while True:
        now = time.perf_counter()
        # nobody watching: freeze the city so an idle server isn't burning CPU
        if clients or not IDLE_PAUSE:
            runner.advance(now - last)
        last = now
        if runner.samples_taken < sent_samples:   # the city was reset
            sent_samples = 0
        if clients:
            frame = {"type": "state", **runner.sim.snapshot(),
                     "paused": runner.paused, "speed": runner.speed,
                     "samples": runner.samples_since(sent_samples)}
            sent_samples = runner.samples_taken
            dead = []
            for ws in list(clients):
                try:
                    await ws.send_json(frame)
                except Exception:
                    dead.append(ws)
            for ws in dead:
                clients.discard(ws)
        await asyncio.sleep(max(0.0, FRAME_TIME - (time.perf_counter() - now)))


@asynccontextmanager
async def lifespan(app):
    task = asyncio.create_task(sim_loop())
    yield
    task.cancel()


app = FastAPI(title="Autopolis", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=os.getenv("CORS_ORIGINS", "http://localhost:5173").split(","),
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.websocket("/ws")
async def websocket_endpoint(ws: WebSocket):
    await ws.accept()
    await ws.send_json({"type": "layout", **runner.sim.city_layout(),
                        "history": runner.history, "ai_available": runner.ai_policy is not None})
    clients.add(ws)
    try:
        while True:
            await ws.receive_text()   # we don't expect messages, this just keeps it open
    except WebSocketDisconnect:
        clients.discard(ws)


# ---------- REST controls ----------

class SpeedBody(BaseModel):
    speed: float = Field(gt=0, le=20)


class SpawnBody(BaseModel):
    count: int = Field(default=20, ge=1, le=200)


class ToggleBody(BaseModel):
    on: bool


class LightsBody(BaseModel):
    mode: str


@app.get("/api/health")
def health():
    return {"ok": True}


@app.get("/api/stats")
def stats():
    return runner.sim.stats()


@app.post("/api/pause")
def pause():
    runner.paused = True
    return {"paused": True}


@app.post("/api/resume")
def resume():
    runner.paused = False
    return {"paused": False}


@app.post("/api/speed")
def set_speed(body: SpeedBody):
    runner.speed = body.speed
    return {"speed": runner.speed}


@app.post("/api/spawn")
def spawn(body: SpawnBody):
    room = MAX_CARS - len(runner.sim.cars)
    added = 0
    for _ in range(min(body.count, max(room, 0))):
        if runner.sim.spawn_car():
            added += 1
    return {"added": added}


@app.post("/api/rush-hour")
def rush_hour(body: ToggleBody):
    runner.sim.rush_hour = body.on
    return {"rush_hour": body.on}


@app.post("/api/lights")
def set_lights(body: LightsBody):
    if body.mode not in ("fixed", "ai"):
        raise HTTPException(400, "mode must be 'fixed' or 'ai'")
    if body.mode == "ai" and runner.ai_policy is None:
        raise HTTPException(409, "no trained model loaded yet")
    runner.sim.set_light_mode(body.mode)
    return {"mode": body.mode}


@app.post("/api/reset")
def reset():
    runner.reset()
    return {"ok": True}
