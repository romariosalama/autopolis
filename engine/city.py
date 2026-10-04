"""The city map: a grid of intersections connected by two-way roads.

Every road is stored as two directed edges (one per direction) so each
direction gets its own lane and its own queue of cars.

Around the outside of the grid there are "gates". Cars enter the city at
a gate and their trip is complete when they reach another gate.
"""

from dataclasses import dataclass, field

BLOCK_LENGTH = 100.0  # meters between neighboring intersections
GATE_LENGTH = 80.0    # meters from a gate to the first intersection


@dataclass
class Node:
    id: int
    x: float
    y: float
    is_gate: bool = False


@dataclass
class Edge:
    id: int
    start: int          # node id
    end: int            # node id
    length: float
    axis: str           # "NS" or "EW", used by the traffic lights
    cars: list = field(default_factory=list)  # ordered: index 0 is closest to the end


class City:
    def __init__(self, rows=3, cols=3, block_length=BLOCK_LENGTH):
        self.rows = rows
        self.cols = cols
        self.block_length = block_length
        self.nodes = {}
        self.edges = {}
        self.out_edges = {}       # node id -> list of edge ids leaving it
        self.edge_lookup = {}     # (start, end) -> edge id
        self.intersections = []   # node ids that have a traffic light
        self.gates = []

        self._build_grid()

    def _add_node(self, x, y, is_gate=False):
        node = Node(len(self.nodes), x, y, is_gate)
        self.nodes[node.id] = node
        self.out_edges[node.id] = []
        if is_gate:
            self.gates.append(node.id)
        else:
            self.intersections.append(node.id)
        return node.id

    def _add_road(self, a, b):
        """Adds a two-way road as two directed edges."""
        for start, end in [(a, b), (b, a)]:
            n1, n2 = self.nodes[start], self.nodes[end]
            length = abs(n1.x - n2.x) + abs(n1.y - n2.y)
            axis = "NS" if n1.x == n2.x else "EW"
            edge = Edge(len(self.edges), start, end, length, axis)
            self.edges[edge.id] = edge
            self.out_edges[start].append(edge.id)
            self.edge_lookup[(start, end)] = edge.id

    def _build_grid(self):
        L = self.block_length
        grid = {}
        for r in range(self.rows):
            for c in range(self.cols):
                grid[(r, c)] = self._add_node(c * L, r * L)

        # roads between neighbors
        for r in range(self.rows):
            for c in range(self.cols):
                if c + 1 < self.cols:
                    self._add_road(grid[(r, c)], grid[(r, c + 1)])
                if r + 1 < self.rows:
                    self._add_road(grid[(r, c)], grid[(r + 1, c)])

        # gates on all four sides
        g = GATE_LENGTH
        for c in range(self.cols):
            top = self._add_node(c * L, -g, is_gate=True)
            self._add_road(top, grid[(0, c)])
            bottom = self._add_node(c * L, (self.rows - 1) * L + g, is_gate=True)
            self._add_road(bottom, grid[(self.rows - 1, c)])
        for r in range(self.rows):
            left = self._add_node(-g, r * L, is_gate=True)
            self._add_road(left, grid[(r, 0)])
            right = self._add_node((self.cols - 1) * L + g, r * L, is_gate=True)
            self._add_road(right, grid[(r, self.cols - 1)])

    def neighbors(self, node_id):
        return [self.edges[e].end for e in self.out_edges[node_id]]

    def edge_between(self, a, b):
        return self.edges[self.edge_lookup[(a, b)]]

    def incoming_edges(self, node_id):
        return [e for e in self.edges.values() if e.end == node_id]
