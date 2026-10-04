"""Traffic lights as a small state machine.

Each intersection cycles through six phases:

    NS_GREEN -> NS_YELLOW -> ALL_RED_1 -> EW_GREEN -> EW_YELLOW -> ALL_RED_2 -> ...

The short all-red interval gives the intersection time to clear, like real
signals do. It also means every switch costs a few seconds where nobody
moves, which is why switching too often is bad.

In "fixed" mode the light switches on a timer. In "ai" mode the light
stays green until something calls request_switch() (the RL agent), but it
still has to respect the minimum green time and the yellow time so the AI
can't do anything a real light couldn't.
"""

NS_GREEN = "NS_GREEN"
NS_YELLOW = "NS_YELLOW"
EW_GREEN = "EW_GREEN"
EW_YELLOW = "EW_YELLOW"
ALL_RED_1 = "ALL_RED_1"   # after NS yellow
ALL_RED_2 = "ALL_RED_2"   # after EW yellow

NEXT_PHASE = {
    NS_GREEN: NS_YELLOW,
    NS_YELLOW: ALL_RED_1,
    ALL_RED_1: EW_GREEN,
    EW_GREEN: EW_YELLOW,
    EW_YELLOW: ALL_RED_2,
    ALL_RED_2: NS_GREEN,
}

GREEN_TIME = 30.0      # seconds, fixed-timer mode
YELLOW_TIME = 3.0
ALL_RED_TIME = 2.0
MIN_GREEN_TIME = 5.0   # AI can't switch faster than this


class TrafficLight:
    def __init__(self, node_id, offset=0.0, green_time=GREEN_TIME):
        self.node_id = node_id
        self.phase = NS_GREEN
        self.green_time = green_time
        # offset lets neighboring lights start at different points in the cycle
        self.time_in_phase = offset % green_time
        self.switch_requested = False

    def color_for(self, axis):
        """What color a car on a road with this axis ("NS" or "EW") sees."""
        if self.phase == axis + "_GREEN":
            return "green"
        if self.phase == axis + "_YELLOW":
            return "yellow"
        return "red"

    def is_green_phase(self):
        return self.phase in (NS_GREEN, EW_GREEN)

    def can_switch(self):
        return self.is_green_phase() and self.time_in_phase >= MIN_GREEN_TIME

    def request_switch(self):
        """Used in AI mode. Ignored if the light isn't allowed to switch yet."""
        if self.can_switch():
            self.switch_requested = True

    def _advance(self):
        self.phase = NEXT_PHASE[self.phase]
        self.time_in_phase = 0.0
        self.switch_requested = False

    def step(self, dt, mode="fixed"):
        self.time_in_phase += dt

        if not self.is_green_phase():
            limit = ALL_RED_TIME if self.phase in (ALL_RED_1, ALL_RED_2) else YELLOW_TIME
            if self.time_in_phase >= limit:
                self._advance()
            return

        if mode == "fixed":
            if self.time_in_phase >= self.green_time:
                self._advance()
        elif self.switch_requested:
            self._advance()
