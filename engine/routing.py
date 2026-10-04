"""A* pathfinding over the city graph."""

import heapq


def heuristic(city, a, b):
    # roads only go north/south/east/west, so Manhattan distance never
    # overestimates the real distance (which A* needs)
    n1, n2 = city.nodes[a], city.nodes[b]
    return abs(n1.x - n2.x) + abs(n1.y - n2.y)


def astar(city, start, goal, cost_fn=None):
    """Returns the list of node ids from start to goal, or None if unreachable.

    cost_fn(edge) lets callers add things like congestion or random jitter.
    By default the cost of an edge is its length.
    """
    if cost_fn is None:
        cost_fn = lambda edge: edge.length

    open_heap = [(heuristic(city, start, goal), 0, start)]
    came_from = {}
    best_cost = {start: 0.0}
    counter = 0  # tie breaker so heapq never compares node ids weirdly

    while open_heap:
        _, _, current = heapq.heappop(open_heap)
        if current == goal:
            path = [current]
            while current in came_from:
                current = came_from[current]
                path.append(current)
            return list(reversed(path))

        for edge_id in city.out_edges[current]:
            edge = city.edges[edge_id]
            nxt = edge.end
            # gates are only allowed as the start or the goal, never a shortcut
            if city.nodes[nxt].is_gate and nxt != goal:
                continue
            new_cost = best_cost[current] + cost_fn(edge)
            if new_cost < best_cost.get(nxt, float("inf")):
                best_cost[nxt] = new_cost
                came_from[nxt] = current
                counter += 1
                heapq.heappush(open_heap, (new_cost + heuristic(city, nxt, goal), counter, nxt))

    return None
