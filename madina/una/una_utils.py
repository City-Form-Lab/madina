import math
from heapq import heappush, heappop

from madina.zonal.network import Network

def turn_o_scope(network: Network,
                  o_idx,
                  search_radius: float,
                  detour_ratio: float,
                  turn_penalty=True,
                  o_graph=None,
                  return_paths=True):
    """
    TODO: fill out the spec
    o_idx: origin index, integer, coming from the node_gdf
    o_graph: reusing updated graphs (e. g. doing inelastic after elastic), optional
    """
    node_gdf = network.nodes
    destinations = node_gdf[node_gdf["type"] == "destination"].index
    # print(f"turn_o_scope: {o_idx = }")

    if o_graph is None:
        graph = network.d_graph
        network.update_light_graph(graph, add_nodes=[o_idx])
    else:
        graph = o_graph

    # visualize_graph(self, graph)
    o_scope = {o_idx: 0}
    d_idxs = {}
    o_scope_paths = {}

    forward_q = [(0, o_idx, [o_idx])]

    furthest_dest_weight = 0

    while forward_q:
        weight, node, visited = heappop(forward_q)
        for neighbor in list(graph.neighbors(node)):

            turn_cost = 0
            if turn_penalty:
                if len(visited) < 2:
                    turn_cost = 0
                else:
                    turn_cost = turn_penalty_value(network, visited[-2], node, neighbor)
                # Need to keep track of visited.

            neighbor_weight = weight + graph.edges[(node, neighbor)]["weight"] + turn_cost
            if (neighbor in o_scope) and (neighbor_weight >= o_scope[neighbor]):  # equivalent to if in seen
                # current_is_better += 1
                continue
            if neighbor in o_scope:  # equivalent to if in seen
                o_scope[neighbor] = neighbor_weight
                if return_paths:
                    o_scope_paths[neighbor] = visited + [neighbor]
                if (neighbor in destinations) and (neighbor_weight <= search_radius):
                    furthest_dest_weight = max(furthest_dest_weight, neighbor_weight)
                    d_idxs[neighbor] = neighbor_weight
                # found_better_updates += 1
                heappush(forward_q, (neighbor_weight, neighbor, visited + [neighbor]))
                continue

            if len(list(graph.neighbors(neighbor))) == 1:
                continue

            if neighbor_weight > max(search_radius, furthest_dest_weight * detour_ratio / 2):
                continue

            if (neighbor in destinations) and (neighbor_weight <= search_radius):
                furthest_dest_weight = max(furthest_dest_weight, neighbor_weight)
                d_idxs[neighbor] = neighbor_weight
            o_scope[neighbor] = neighbor_weight
            if return_paths:
                o_scope_paths[neighbor] = visited + [neighbor]
            heappush(forward_q, (neighbor_weight, neighbor, visited + [neighbor]))
    if o_graph is None:
        network.update_light_graph(graph, remove_nodes=[o_idx])
    return d_idxs, o_scope, o_scope_paths


def turn_penalty_value(network: Network, previous_node, current_node, next_node):
    """
    TODO: fill out the spec
    """
    node_gdf = network.nodes
    edge_gdf = network.edges
    angle = angle_deviation_between_two_lines(
        [
            node_gdf.at[previous_node, "geometry"],
            node_gdf.at[current_node, "geometry"],
            node_gdf.at[next_node, "geometry"]
        ]
    )
    angle = min(angle, abs(angle - 180))
    if angle > network.turn_threshold_degree:
        return network.turn_penalty_amount
    else:
        return 0
    

def angle_deviation_between_two_lines(point_sequence, raw_angle=False):
    a = point_sequence[0].coords[0]
    b = point_sequence[1].coords[0]
    c = point_sequence[2].coords[0]

    ang = math.degrees(
        math.atan2(c[1] - b[1], c[0] - b[0]) - math.atan2(a[1] - b[1], a[0] - b[0]))
    if raw_angle:
        return ang
    else:
        ang = ang + 360 if ang < 0 else ang
        # how far is this turn from being a 180?
        ang = abs(round(ang) - 180)

        return ang