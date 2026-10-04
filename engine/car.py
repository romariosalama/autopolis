"""Cars and the Intelligent Driver Model (IDM).

IDM is a standard car-following model from traffic research (Treiber et al.,
2000). Each car looks at the gap to whatever is in front of it (another car
or a red light's stop line) and picks an acceleration that keeps a safe
time gap. It produces smooth braking and realistic stop-and-go queues
without any physics engine.
"""

import math
from dataclasses import dataclass, field

CAR_LENGTH = 4.5   # meters

# IDM parameters (typical city driving values)
DESIRED_SPEED = 13.9   # m/s, about 50 km/h
TIME_HEADWAY = 1.5     # seconds of gap a driver wants to keep
MAX_ACCEL = 1.5        # m/s^2
COMFORT_DECEL = 2.0    # m/s^2
MIN_GAP = 2.0          # meters, gap kept when stopped
DELTA = 4

WAITING_SPEED = 0.5    # below this a car counts as "waiting"


@dataclass
class Car:
    id: int
    route: list            # list of node ids, gate to gate
    spawn_time: float
    leg: int = 0           # index into route; current edge is route[leg] -> route[leg+1]
    edge_id: int = -1
    pos: float = 0.0       # meters from the start of the current edge
    speed: float = 0.0
    accel: float = 0.0
    wait_time: float = 0.0
    committed: bool = False  # decided to go through a yellow light
    length: float = CAR_LENGTH
    desired_speed: float = DESIRED_SPEED
    extra: dict = field(default_factory=dict)

    def next_node_pair(self):
        """(start, end) of the edge after the current one, or None at the last leg."""
        if self.leg + 2 >= len(self.route):
            return None
        return self.route[self.leg + 1], self.route[self.leg + 2]


def idm_accel(speed, desired_speed, gap=None, leader_speed=None):
    """IDM acceleration. gap=None means the road ahead is empty."""
    free_road = MAX_ACCEL * (1 - (speed / desired_speed) ** DELTA)
    if gap is None:
        return free_road

    gap = max(gap, 0.1)
    dv = speed - leader_speed
    desired_gap = MIN_GAP + max(0.0, speed * TIME_HEADWAY + speed * dv / (2 * math.sqrt(MAX_ACCEL * COMFORT_DECEL)))
    return free_road - MAX_ACCEL * (desired_gap / gap) ** 2


def stopping_distance(speed, decel=COMFORT_DECEL):
    return speed * speed / (2 * decel)
