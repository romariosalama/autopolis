import pytest

from engine.car import Car, idm_accel, MAX_ACCEL
from engine.city import City
from engine.lights import (TrafficLight, NS_GREEN, NS_YELLOW, EW_GREEN, EW_YELLOW,
                           ALL_RED_1, ALL_RED_2, YELLOW_TIME, ALL_RED_TIME, MIN_GREEN_TIME)
from engine.routing import astar
from engine.sim import Simulation, HARD_GAP


# ---------- city ----------

def test_grid_has_right_number_of_intersections_and_gates():
    city = City(3, 4)
    assert len(city.intersections) == 12
    assert len(city.gates) == 2 * 3 + 2 * 4


def test_every_road_goes_both_ways():
    city = City(3, 3)
    for (a, b) in city.edge_lookup:
        assert (b, a) in city.edge_lookup


def test_edge_axis():
    city = City(2, 2)
    for e in city.edges.values():
        a, b = city.nodes[e.start], city.nodes[e.end]
        assert e.axis == ("NS" if a.x == b.x else "EW")


# ---------- routing ----------

def test_astar_finds_shortest_path_across_grid():
    city = City(3, 3)
    start = city.intersections[0]       # top left
    goal = city.intersections[-1]       # bottom right
    path = astar(city, start, goal)
    assert path[0] == start and path[-1] == goal
    length = sum(city.edge_between(a, b).length for a, b in zip(path, path[1:]))
    assert length == pytest.approx(4 * city.block_length)


def test_astar_path_only_uses_real_roads():
    city = City(4, 4)
    path = astar(city, city.gates[0], city.gates[-1])
    for a, b in zip(path, path[1:]):
        assert (a, b) in city.edge_lookup


def test_astar_never_cuts_through_a_gate():
    city = City(4, 4)
    for g1 in city.gates[:4]:
        for g2 in city.gates[-4:]:
            path = astar(city, g1, g2)
            assert all(not city.nodes[n].is_gate for n in path[1:-1])


def test_astar_same_start_and_goal():
    city = City(2, 2)
    n = city.intersections[0]
    assert astar(city, n, n) == [n]


def test_astar_respects_custom_cost():
    city = City(2, 2)
    a, b = city.intersections[0], city.intersections[1]
    direct = city.edge_lookup[(a, b)]
    # make the direct road really expensive so A* goes around
    path = astar(city, a, b, lambda e: 10_000 if e.id == direct else e.length)
    assert len(path) > 2


# ---------- lights ----------

def test_fixed_light_cycles_through_all_phases():
    light = TrafficLight(0, green_time=10)
    seen = [light.phase]
    for _ in range(int(40 / 0.5)):
        light.step(0.5, "fixed")
        if light.phase != seen[-1]:
            seen.append(light.phase)
    assert seen[:7] == [NS_GREEN, NS_YELLOW, ALL_RED_1, EW_GREEN, EW_YELLOW, ALL_RED_2, NS_GREEN]


def test_yellow_lasts_yellow_time():
    light = TrafficLight(0, green_time=10)
    light.phase = NS_YELLOW
    light.time_in_phase = 0
    light.step(YELLOW_TIME - 0.1, "fixed")
    assert light.phase == NS_YELLOW
    light.step(0.2, "fixed")
    assert light.phase == ALL_RED_1


def test_all_red_is_red_both_ways_then_goes_green():
    light = TrafficLight(0)
    light.phase = ALL_RED_2
    light.time_in_phase = 0
    assert light.color_for("NS") == "red" and light.color_for("EW") == "red"
    light.step(ALL_RED_TIME + 0.01, "fixed")
    assert light.phase == NS_GREEN


def test_light_colors_per_axis():
    light = TrafficLight(0)
    light.phase = NS_GREEN
    assert light.color_for("NS") == "green" and light.color_for("EW") == "red"
    light.phase = EW_YELLOW
    assert light.color_for("EW") == "yellow" and light.color_for("NS") == "red"


def test_ai_light_holds_green_without_request():
    light = TrafficLight(0)
    for _ in range(1000):
        light.step(0.5, "ai")
    assert light.phase == NS_GREEN


def test_ai_light_ignores_switch_before_min_green():
    light = TrafficLight(0)
    light.step(MIN_GREEN_TIME / 2, "ai")
    light.request_switch()
    light.step(0.1, "ai")
    assert light.phase == NS_GREEN


def test_ai_light_switches_after_min_green():
    light = TrafficLight(0)
    light.step(MIN_GREEN_TIME + 0.1, "ai")
    light.request_switch()
    light.step(0.1, "ai")
    assert light.phase == NS_YELLOW


# ---------- IDM ----------

def test_idm_accelerates_on_empty_road():
    assert idm_accel(0.0, 13.9) == pytest.approx(MAX_ACCEL)


def test_idm_free_road_accel_drops_near_desired_speed():
    assert idm_accel(13.9, 13.9) == pytest.approx(0.0)


def test_idm_brakes_when_close_to_stopped_leader():
    assert idm_accel(10.0, 13.9, gap=5.0, leader_speed=0.0) < -2.0


# ---------- simulation ----------

def all_gaps(sim):
    for edge in sim.city.edges.values():
        for front, back in zip(edge.cars, edge.cars[1:]):
            yield front.pos - front.length - back.pos


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_no_two_cars_overlap(seed):
    sim = Simulation(4, 4, seed=seed, spawn_rate=120)
    for _ in range(int(300 / sim.dt)):
        sim.step()
        for gap in all_gaps(sim):
            assert gap >= HARD_GAP - 1e-9


def test_cars_on_an_edge_stay_in_order():
    sim = Simulation(3, 3, seed=5, spawn_rate=80)
    for _ in range(int(200 / sim.dt)):
        sim.step()
        for edge in sim.city.edges.values():
            positions = [c.pos for c in edge.cars]
            assert positions == sorted(positions, reverse=True)


def test_no_car_runs_a_red_light_from_far_away():
    """A car that wasn't already committed on yellow must never cross on red."""
    sim = Simulation(3, 3, seed=3, spawn_rate=60)
    for _ in range(int(300 / sim.dt)):
        before = {c.id: (c.edge_id, c.committed) for c in sim.cars.values()}
        colors = {}
        for e in sim.city.edges.values():
            if e.end in sim.lights:
                colors[e.id] = sim.lights[e.end].color_for(e.axis)
        sim.step()
        for car in sim.cars.values():
            old = before.get(car.id)
            if old and old[0] != car.edge_id:
                # it crossed an intersection this tick
                assert colors.get(old[0]) != "red" or old[1]


def test_trips_complete_and_stats_update():
    sim = Simulation(3, 3, seed=0, spawn_rate=30)
    sim.run(300)
    s = sim.stats()
    assert s["trips_completed"] > 50
    assert s["avg_wait"] >= 0
    assert s["cars_per_minute"] > 0
    assert s["avg_wait_recent"] >= 0


def test_same_seed_is_deterministic():
    a = Simulation(3, 3, seed=42)
    b = Simulation(3, 3, seed=42)
    a.run(120)
    b.run(120)
    assert a.snapshot() == b.snapshot()


def test_cars_reach_their_destination_gate():
    sim = Simulation(3, 3, seed=1, spawn_rate=40)
    car = sim.spawn_car(sim.city.gates[0])
    goal = car.route[-1]
    for _ in range(int(400 / sim.dt)):
        sim.step()
        if car.id not in sim.cars and car not in sim.spawn_queues[sim.city.gates[0]]:
            break
    assert car.id not in sim.cars
    assert sim.city.nodes[goal].is_gate


def test_red_light_queue_forms():
    sim = Simulation(1, 1, seed=0, spawn_rate=0)
    node = sim.city.intersections[0]
    light = sim.lights[node]
    sim.set_light_mode("ai")   # hold the light so it never changes
    light.phase = EW_GREEN
    # send a few cars north->south (NS axis, which is red)
    top_gate = min(sim.city.gates, key=lambda g: sim.city.nodes[g].y)
    bottom_gate = max(sim.city.gates, key=lambda g: sim.city.nodes[g].y)
    for i in range(4):
        car = Car(id=1000 + i, route=[top_gate, node, bottom_gate], spawn_time=0)
        sim.spawn_queues[top_gate].append(car)
    sim.run(60)
    assert sim.queue_lengths(node)["N"] == 4


def test_rush_hour_increases_spawn_rate():
    sim = Simulation(3, 3)
    normal = sim.current_spawn_rate()
    sim.rush_hour = True
    assert sim.current_spawn_rate() > normal


def test_snapshot_shape():
    sim = Simulation(3, 3, seed=0)
    sim.run(60)
    snap = sim.snapshot()
    assert set(snap) == {"cars", "lights", "stats"}
    assert len(snap["lights"]) == 9
    for car in snap["cars"]:
        assert len(car) == 5


def test_city_layout_lists_each_road_once():
    sim = Simulation(3, 3)
    layout = sim.city_layout()
    assert len(layout["roads"]) == len(sim.city.edges) // 2
