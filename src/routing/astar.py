"""A* on a NetworkX directed graph whose edges have a non-negative 'travel_time' weight."""
import heapq
import itertools

def astar_time(G, source, target, heuristic=None, weight="travel_time"):
    """Return (node_path, total_cost_seconds).

    `heuristic(node, target)` must be an admissible lower bound in seconds
    if optimality is required. Use heuristic=None for Dijkstra-equivalent A*.
    Parallel edges are handled by taking the minimum edge weight.
    """
    if source not in G or target not in G:
        raise nx_node_error(source, target)
    if source == target:
        return [source], 0.0
    heuristic = heuristic or (lambda u, v: 0.0)
    counter = itertools.count()
    open_heap = [(heuristic(source, target), next(counter), 0.0, source)]
    came_from = {}
    best_g = {source: 0.0}
    closed = set()

    while open_heap:
        _, _, g_u, u = heapq.heappop(open_heap)
        if u in closed or g_u > best_g.get(u, float("inf")):
            continue
        if u == target:
            path = [u]
            while u in came_from:
                u = came_from[u]
                path.append(u)
            path.reverse()
            return path, g_u
        closed.add(u)

        for v, edge_data in G[u].items():
            # DiGraph: edge_data is a dict of attrs; MultiDiGraph: dict keyed by edge key.
            if G.is_multigraph():
                edge_cost = min(float(attrs.get(weight, float("inf"))) for attrs in edge_data.values())
            else:
                edge_cost = float(edge_data.get(weight, float("inf")))
            if edge_cost < 0:
                raise ValueError("A* requires non-negative edge costs.")
            if edge_cost == float("inf"):
                continue
            candidate = g_u + edge_cost
            if candidate < best_g.get(v, float("inf")):
                best_g[v] = candidate
                came_from[v] = u
                heapq.heappush(
                    open_heap,
                    (candidate + float(heuristic(v, target)), next(counter), candidate, v)
                )
    return None, float("inf")

def nx_node_error(source, target):
    return ValueError(f"source/target must exist in graph: {source!r}, {target!r}")
