"""The simulation loop. This is the whole city: no rendering happens here.

The browser and the RL environment both just call step() and read
snapshot(), which is what lets training run headless and much faster
than real time.
"""

import math
import random
from collections import deque

from .car import Car, idm_accel, stopping_distance, WAITING_SPEED, MIN_GAP
from .city import City
from .lights import TrafficLight
from .routing import astar

DEFAULT_DT = 0.25          # seconds of sim time per tick
LANE_OFFSET = 2.5          # meters to the right of the road center line
HARD_GAP = 0.5             # cars are never closer than this, no matter what
STOP_LINE = 1.0            # meters before the intersection where cars stop
STATS_WINDOW = 60.0        # seconds, for "cars per minute"


class Simulation:
    def __init__(self, rows=3, cols=3, seed=0, dt=DEFAULT_DT, spawn_rate=20.0,
                 light_mode="fixed", green_time=30.0):
        """spawn_rate is cars per minute for the whole city."""
        self.city = City(rows, cols)
        self.rng = random.Random(seed)
        self.dt = dt
        self.time = 0.0
        self.tick_count = 0
        self.spawn_rate = spawn_rate
        self.rush_hour = False
        self.light_mode = light_mode

        self.lights = {}
        for i, node_id in enumerate(self.city.intersections):
            n = self.city.nodes[node_id]
            # stagger the lights a bit so they don't all switch together
            offset = ((n.x + n.y) / self.city.block_length) * 7.0
            self.lights[node_id] = TrafficLight(node_id, offset, green_time)

        self._incoming = {n: self.city.incoming_edges(n) for n in self.city.intersections}

        self.cars = {}
        self.next_car_id = 0
        self.spawn_queues = {g: deque() for g in self.city.gates}

        # stats
        self.trips_completed = 0
        self.total_wait_completed = 0.0
        self.total_travel_completed = 0.0
        self.recent_finishes = deque()   # (time, wait_time)

    # ---------- spawning ----------

    def current_spawn_rate(self):
        return self.spawn_rate * (2.5 if self.rush_hour else 1.0)

    def _random_route(self, start_gate):
        goal = self.rng.choice([g for g in self.city.gates if g != start_gate])
        # tiny random cost jitter spreads traffic across equally short routes.
        # multiplying by >= 1 keeps the Manhattan heuristic admissible.
        jitter = {e: 1.0 + self.rng.random() * 0.3 for e in self.city.edges}
        return astar(self.city, start_gate, goal, lambda edge: edge.length * jitter[edge.id])

    def spawn_car(self, gate=None):
        """Adds a car to a gate's waiting queue. It enters when there's room."""
        if gate is None:
            gate = self.rng.choice(self.city.gates)
        route = self._random_route(gate)
        if route is None or len(route) < 2:
            return None
        car = Car(id=self.next_car_id, route=route, spawn_time=self.time)
        self.next_car_id += 1
        self.spawn_queues[gate].append(car)
        return car

    def _spawn_step(self):
        per_gate = self.current_spawn_rate() / 60.0 / len(self.city.gates)
        for gate in self.city.gates:
            if self.rng.random() < per_gate * self.dt:
                self.spawn_car(gate)

        # let waiting cars enter if the first edge has room at the start
        for gate, queue in self.spawn_queues.items():
            if not queue:
                continue
            car = queue[0]
            edge = self.city.edge_between(car.route[0], car.route[1])
            if self._has_room_at_start(edge):
                queue.popleft()
                # time stuck waiting to get into the city counts as waiting too,
                # otherwise a controller could "win" by blocking the entrances
                car.wait_time += self.time - car.spawn_time
                car.edge_id = edge.id
                car.pos = 0.0
                car.speed = min(5.0, car.desired_speed)
                edge.cars.append(car)
                self.cars[car.id] = car

    def _has_room_at_start(self, edge, overflow=0.0):
        """True if a car can enter the edge at position `overflow` and still
        leave MIN_GAP behind the last car already on it."""
        if not edge.cars:
            return True
        last = edge.cars[-1]
        return last.pos - last.length - overflow >= MIN_GAP

    # ---------- the tick ----------

    def step(self):
        dt = self.dt
        for light in self.lights.values():
            light.step(dt, self.light_mode)

        self._spawn_step()
        self._compute_accelerations()
        self._move_cars()
        self._cross_intersections()

        self.time += dt
        self.tick_count += 1
        while self.recent_finishes and self.recent_finishes[0][0] < self.time - STATS_WINDOW:
            self.recent_finishes.popleft()

    def _light_blocks(self, car, edge):
        """True if the car has to treat the stop line as a wall."""
        end = edge.end
        if end not in self.lights:   # road ends at a gate
            return False
        color = self.lights[end].color_for(edge.axis)
        if color == "green":
            car.committed = False
            return False
        if car.committed:
            return False
        if color == "yellow":
            dist = edge.length - STOP_LINE - car.pos
            if dist < stopping_distance(car.speed):
                car.committed = True   # too close to stop safely, so go
                return False
        return True

    def _compute_accelerations(self):
        for edge in self.city.edges.values():
            for i, car in enumerate(edge.cars):
                if i > 0:
                    leader = edge.cars[i - 1]
                    gap = leader.pos - leader.length - car.pos
                    car.accel = idm_accel(car.speed, car.desired_speed, gap, leader.speed)
                    continue

                # first car on the edge: look at the light and the next road
                if self._light_blocks(car, edge):
                    gap = edge.length - STOP_LINE - car.pos
                    car.accel = idm_accel(car.speed, car.desired_speed, gap, 0.0)
                    continue

                pair = car.next_node_pair()
                if pair is None:
                    car.accel = idm_accel(car.speed, car.desired_speed)
                    continue
                nxt = self.city.edge_between(*pair)
                if nxt.cars:
                    last = nxt.cars[-1]
                    gap = (edge.length - car.pos) + last.pos - last.length
                    car.accel = idm_accel(car.speed, car.desired_speed, gap, last.speed)
                else:
                    car.accel = idm_accel(car.speed, car.desired_speed)

    def _move_cars(self):
        dt = self.dt
        for edge in self.city.edges.values():
            blocked = None
            for i, car in enumerate(edge.cars):
                v, a = car.speed, car.accel
                new_v = v + a * dt
                if new_v < 0:
                    dist = -v * v / (2 * a) if a < 0 else 0.0
                    new_v = 0.0
                else:
                    dist = v * dt + 0.5 * a * dt * dt
                new_pos = car.pos + max(0.0, dist)

                # hard safety limits so cars can never overlap
                if i > 0:
                    leader = edge.cars[i - 1]
                    limit = leader.pos - leader.length - HARD_GAP
                    if new_pos > limit:
                        new_pos = max(car.pos, limit)
                        new_v = min(new_v, leader.speed)
                else:
                    if blocked is None:
                        blocked = self._light_blocks(car, edge)
                    if blocked and new_pos > edge.length - STOP_LINE:
                        new_pos = max(car.pos, edge.length - STOP_LINE)
                        new_v = 0.0

                car.pos = new_pos
                car.speed = new_v
                if new_v < WAITING_SPEED:
                    car.wait_time += dt

    def _cross_intersections(self):
        for edge in self.city.edges.values():
            while edge.cars and edge.cars[0].pos >= edge.length:
                car = edge.cars[0]
                overflow = car.pos - edge.length
                pair = car.next_node_pair()

                if pair is None:   # reached the destination gate
                    edge.cars.pop(0)
                    self._finish_trip(car)
                    continue

                nxt = self.city.edge_between(*pair)
                if not self._has_room_at_start(nxt, overflow):
                    # the next road is full: wait at the end of this one
                    car.pos = edge.length
                    car.speed = 0.0
                    break

                edge.cars.pop(0)
                car.leg += 1
                car.edge_id = nxt.id
                car.pos = overflow
                car.committed = False
                nxt.cars.append(car)

    def _finish_trip(self, car):
        del self.cars[car.id]
        self.trips_completed += 1
        self.total_wait_completed += car.wait_time
        self.total_travel_completed += self.time - car.spawn_time
        self.recent_finishes.append((self.time, car.wait_time))

    # ---------- controls ----------

    def set_light_mode(self, mode):
        assert mode in ("fixed", "ai")
        self.light_mode = mode

    def run(self, seconds):
        for _ in range(int(round(seconds / self.dt))):
            self.step()

    # ---------- reading the state ----------

    def avg_wait(self):
        if self.trips_completed == 0:
            return 0.0
        return self.total_wait_completed / self.trips_completed

    def recent_avg_wait(self):
        """Average wait of trips that finished in the last minute. This is what
        the dashboard shows, so switching the lights shows up within a minute."""
        if not self.recent_finishes:
            return 0.0
        return sum(w for _, w in self.recent_finishes) / len(self.recent_finishes)

    def cars_per_minute(self):
        window = min(STATS_WINDOW, self.time)
        if window <= 0:
            return 0.0
        return len(self.recent_finishes) * 60.0 / window

    def waiting_cars(self):
        return sum(1 for c in self.cars.values() if c.speed < WAITING_SPEED)

    def queue_lengths(self, node_id):
        """Waiting cars on each road coming into an intersection, keyed N/S/E/W
        by the side they come from."""
        node = self.city.nodes[node_id]
        result = {"N": 0, "S": 0, "E": 0, "W": 0}
        for edge in self._incoming[node_id]:
            start = self.city.nodes[edge.start]
            if start.y < node.y:
                side = "N"
            elif start.y > node.y:
                side = "S"
            elif start.x > node.x:
                side = "E"
            else:
                side = "W"
            result[side] = sum(1 for c in edge.cars if c.speed < WAITING_SPEED)
        return result

    def stats(self):
        return {
            "time": round(self.time, 2),
            "cars": len(self.cars),
            "waiting": self.waiting_cars(),
            "trips_completed": self.trips_completed,
            "avg_wait": round(self.avg_wait(), 2),
            "avg_wait_recent": round(self.recent_avg_wait(), 2),
            "cars_per_minute": round(self.cars_per_minute(), 1),
            "spawn_backlog": sum(len(q) for q in self.spawn_queues.values()),
            "light_mode": self.light_mode,
            "rush_hour": self.rush_hour,
        }

    def car_position(self, car):
        edge = self.city.edges[car.edge_id]
        a, b = self.city.nodes[edge.start], self.city.nodes[edge.end]
        t = min(car.pos / edge.length, 1.0)
        dx, dy = (b.x - a.x) / edge.length, (b.y - a.y) / edge.length
        # drive on the right: shift along the right-hand normal
        x = a.x + (b.x - a.x) * t - dy * LANE_OFFSET
        y = a.y + (b.y - a.y) * t + dx * LANE_OFFSET
        heading = math.atan2(dy, dx)
        return x, y, heading

    def snapshot(self):
        cars = []
        for car in self.cars.values():
            x, y, h = self.car_position(car)
            cars.append([car.id, round(x, 2), round(y, 2), round(h, 3), round(car.speed, 2)])
        lights = []
        for node_id, light in self.lights.items():
            n = self.city.nodes[node_id]
            lights.append({"id": node_id, "x": n.x, "y": n.y, "phase": light.phase})
        return {"cars": cars, "lights": lights, "stats": self.stats()}

    def city_layout(self):
        """Static map data the frontend needs once, at connect time."""
        nodes = [{"id": n.id, "x": n.x, "y": n.y, "gate": n.is_gate} for n in self.city.nodes.values()]
        roads = []
        seen = set()
        for e in self.city.edges.values():
            key = tuple(sorted((e.start, e.end)))
            if key not in seen:
                seen.add(key)
                roads.append({"a": e.start, "b": e.end})
        return {"rows": self.city.rows, "cols": self.city.cols,
                "block": self.city.block_length, "nodes": nodes, "roads": roads}
