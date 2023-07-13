from collections import deque
from heapq import heappush, heappop

import shapely
import shapely.geometry as geo
from shapely.ops import unary_union

from madina.zonal.network import Network
from madina.una.una_utils import \
    turn_o_scope, \
    turn_penalty_value, \
    angle_deviation_between_two_lines


def path_generator(network: Network, o_idx, search_radius=800, detour_ratio=1.15, turn_penalty=False):
    """
    Generate all possible paths, their weights, and the destination nodes that can be
    reached from the given origin.
    """

    # Adding the origin to the destination graph
    graph = network.d_graph
    network.update_light_graph(graph, add_nodes=[o_idx])

    o_graph, d_idxs, distance_matrix, scope_nodes = get_od_subgraph(
        network,
        o_idx,
        search_radius=search_radius,
        detour_ratio=detour_ratio,
        output_map=False,
        graph_type="trail_blazer",
        turn_penalty=turn_penalty,
        o_graph=graph
    )


    d_allowed_distances = {}
    for d_idx in d_idxs.keys():
        d_allowed_distances[d_idx] = d_idxs[d_idx] * detour_ratio

    paths, distances = bfs_paths_many_targets_iterative(network,
                                                         graph,
                                                         o_idx,
                                                         d_allowed_distances,
                                                         distance_matrix=distance_matrix,
                                                         od_scope=scope_nodes,
                                                         turn_penalty=turn_penalty
                                                         )
    

    # Removing the origin from the destination graph to restore it
    network.update_light_graph(graph, remove_nodes=[o_idx])
    return paths, distances, d_idxs


def get_od_subgraph(network: Network, o_idx, search_radius=800, detour_ratio=1.15,
                      output_map=False, graph_type="geometric", turn_penalty=False, o_graph=None):
    
    if o_graph is None:
        # In case the origin graph (one origin + all destinations) is not created
        graph = network.d_graph.copy()
        graph.graph["added_nodes"] = network.d_graph.graph["added_nodes"].copy()
        network.update_light_graph(graph, add_nodes=[o_idx])
    else:
        graph = o_graph

    if graph_type == "full":
        pass
    elif graph_type == "geometric":
        scope_nodes = geometric_scope(network, o_idx, search_radius, detour_ratio, turn_penalty=False, batched=True,
                                      o_graph=graph)
    elif graph_type == "network":
        pass
    elif graph_type == "double_sided_dijkstra":
        scope_nodes, distance_matrix, d_idxs = explore_exploit_graph(network, o_idx, search_radius, detour_ratio,
                                                                     turn_penalty=turn_penalty)
    elif graph_type == "trail_blazer":
        # scope_nodes, distance_matrix, d_idxs = bfs_subgraph_generation(self, o_idx, search_radius, detour_ratio,
        #                                                               turn_penalty=turn_penalty, o_graph=graph)
        scope_nodes, distance_matrix, d_idxs = bfs_subgraph_generation(network,
                                                                       o_idx,
                                                                       search_radius=search_radius,
                                                                       detour_ratio=detour_ratio,
                                                                       turn_penalty=turn_penalty,
                                                                       o_graph=graph
                                                                       )

    return graph, d_idxs, distance_matrix, scope_nodes

def geometric_scope(network: Network, o_idx, search_radius, detour_ratio, turn_penalty=False, batched=True, o_graph=None):
    node_gdf = network.nodes
    scopes = []
    d_idxs, o_scope, o_scope_paths = turn_o_scope(network, o_idx, search_radius, detour_ratio, turn_penalty=turn_penalty,
                                                  o_graph=o_graph)

    for d_idx in d_idxs.keys():
        od_shortest_distance = d_idxs[d_idx]
        # Get scope: intersection of o_buffer and d_buffer
        od_geometric_distance = node_gdf.at[o_idx, "geometry"].distance(node_gdf.at[d_idx, "geometry"])
        o_scope = node_gdf.at[o_idx, "geometry"].buffer((detour_ratio * od_shortest_distance) / 2)
        d_scope = node_gdf.at[d_idx, "geometry"].buffer((detour_ratio * od_shortest_distance) / 2)

        intersection_points = o_scope.exterior.intersection(d_scope.exterior)

        if len(intersection_points) < 2:
            scope = node_gdf.at[d_idx, "geometry"].buffer((detour_ratio * od_shortest_distance))

        else:
            mid_point = geo.LineString([node_gdf.at[o_idx, "geometry"], node_gdf.at[d_idx, "geometry"]]).centroid
            major_axis = shapely.affinity.scale(
                geo.LineString([node_gdf.at[o_idx, "geometry"], node_gdf.at[d_idx, "geometry"]])
                , xfact=(detour_ratio * od_shortest_distance / od_geometric_distance),
                yfact=(detour_ratio * od_shortest_distance / od_geometric_distance), zfact=1.0, origin='center'
            )

            minor_axis = geo.LineString([intersection_points[0], intersection_points[1]])
            center_unit_circle = mid_point.buffer(1)
            unaligned_ellipse = shapely.affinity.scale(
                center_unit_circle,
                xfact=major_axis.length / 2,
                yfact=minor_axis.length / 2
            )

            point_list = [
                geo.Point(major_axis.coords[0]),
                geo.Point(major_axis.coords[1]),
                geo.Point((major_axis.coords[1][0], major_axis.coords[0][1])),
            ]

            alignment_angle = 90 - angle_deviation_between_two_lines(
                point_list,
                raw_angle=True
            )

            aligned_ellipse = shapely.affinity.rotate(unaligned_ellipse, alignment_angle)
            scope = aligned_ellipse

        scopes.append(scope)

    if batched:
        scopes = unary_union(scopes)
        scope_nodes = list(node_gdf[node_gdf.intersects(scopes)].index)
        return scope_nodes
    else:
        scope_nodes = {}
        for d_idx, scope in zip(d_idxs.keys(), scopes):
            scope_nodes[d_idx] = list(node_gdf[node_gdf.intersects(scope)].index)
        return scope_nodes


def explore_exploit_graph(network: Network, o_idx:int, search_radius, detour_ratio, turn_penalty=False):
    od_scope = set()
    distance_matrix = {}

    d_idxs, o_scope, o_scope_paths = turn_o_scope(network, o_idx, search_radius, detour_ratio, turn_penalty=turn_penalty)
    graph = network.d_graph
    network.update_light_graph(graph, add_nodes=[o_idx])
    if len(d_idxs) == 0:
        network.update_light_graph(graph=graph, remove_nodes=[o_idx])
        return od_scope, distance_matrix, d_idxs

    ## start ellipse search from all nodes.
    ## look at all d_idxs. find thier neighbors, insert into queue.
    # count = 0
    for d_idx in d_idxs:
        best_weight_node_queue = []
        heappush(best_weight_node_queue, (0, d_idx))
        d_scope = {d_idx: 0}
        while best_weight_node_queue:
            weight, node = heappop(best_weight_node_queue)
            for neighbor in list(graph.neighbors(node)):
                # termination conditions...
                spent_weight = graph.edges[(node, neighbor)]["weight"]

                # the given graph have 2 neighbors for origins and destinations.
                # if a node has only one neighbor, its a deadend.
                if len(list(graph.neighbors(neighbor))) == 1:
                    # print(f"{node = }\tSIngle neighbor termination")
                    continue

                if neighbor not in o_scope:
                    # print(f"{node = }\tOut of O Scope termination")
                    continue

                if (spent_weight + weight + o_scope[neighbor]) > o_scope[d_idx] * detour_ratio:
                    # print(f"{node = }\t{spent_weight = }\t{weight = }\t{o_scope[d_idx] = }
                    # \t{o_scope[d_idx] * detour_ratio = }network distance termination termination")
                    continue

                # keep track of discovered nodes and their shortest distance.
                if neighbor in d_scope:  # equivalent to if in seen
                    if (spent_weight + weight) < d_scope[neighbor]:
                        d_scope[neighbor] = (spent_weight + weight)
                    else:
                        # print(f"{node = }\t{d_scope = }\tAlready found shorter distance termination")
                        continue
                else:
                    d_scope[neighbor] = (spent_weight + weight)
                    # print(f"{neighbor = } added to scope")

                # launching new search

                heappush(best_weight_node_queue, (spent_weight + weight, neighbor))
        distance_matrix[d_idx] = d_scope
        od_scope = od_scope.union(set(d_scope.keys()))
    '''
    WHat do we want to achieve?
        @ Starting from a certain origin node
        @ Find all destinations that are within "search radius weight". We need to kmow their minimum distance.
        @ Once all destinations are found:
            @ from each destination, we want to find all edges that connects back to "Origin", that
                are within a "detour ratio from the shortest path" from O to D. to build a search space.
            @ for each node found in the search space, we need to know all destinations
                it connects to, and the minimum distance.
    '''
    network.update_light_graph(
        graph=graph,
        remove_nodes=[o_idx]
    )
    return od_scope, distance_matrix, d_idxs


def bfs_subgraph_generation(network, o_idx, search_radius=800, detour_ratio=1.15, turn_penalty=False, o_graph=None):

    '''
    Suggested Pseudocode:
    @ run O dijkstra fo find all dests and paths just like before
    @ for all dests, and all neighbors to dijkstra queue (dist, d_idx, current_node)
        @ pop node from queue
        @ if node is not in dist_matrix[d_idx]:
            DO all network termination checks, single neighbor term, ...
            @ add it, and add all nodes in shortest path from the origin backwards to maintain
            weight dist_matrix[d_idx][node] = current weight+accumilated backwards weight
            @ to avoid traversing backwards multiple time, just mark the "trailhead" by each destinations that
             gets to it, and add each destination as key in the dist_matrix[d_idx][node]
            @ keep a list of trailheads for later update..
        @ if node is already in and current weight is better:
            @ for all nodes on o_shortest_path, update dist_matrix[node][d_idx] = current weight+accumilated
            backwards weight
        # if already in and weight is worse, continue.
    '''
    # TODO: should we only run destination search to half radius and use origin's shortest paths to complete the rest?
    # TODO: should we use origin shortest paths to quickly terminate search from dest?traverse path to update
    #  distance from destination to the dist_matrix
    # TODO: should we batch search for all dests? it would be hard as we only keep track of one weight? maybe
    #  keep track of multiple weights one for each source.
    # TODO: should we use shortest_path_to_o while we search? probably yes.
    od_scope = set()
    distance_matrix = {}
    trailheads = {}

    if o_graph is None:
        # d_idxs, o_scope, o_scope_paths = get_o_scope(self, graph, o_idx, search_radius, detour_ratio, get_paths=True)
        d_idxs, o_scope, o_scope_paths = turn_o_scope(network, o_idx, search_radius, detour_ratio,
                                                      turn_penalty=turn_penalty)
        graph = network.d_graph
        network.update_light_graph(graph, add_nodes=[o_idx])
    else:
        d_idxs, o_scope, o_scope_paths = turn_o_scope(network, o_idx, search_radius, detour_ratio,
                                                      turn_penalty=turn_penalty, o_graph=o_graph)
        graph = o_graph

    # visualize_graph(self, graph)

    if (len(d_idxs) == 0):
        if (o_graph is None):
            network.update_light_graph(
                graph=graph,
                remove_nodes=[o_idx]
            )
        return od_scope, distance_matrix, d_idxs

    '''
    queue_counter = 0
    queue_neighbor_counter = 0
    current_is_better = 0
    found_better_updates = 0
    found_new_node = 0
    new_node_is_deadend = 0
    new_node_out_of_scope = 0
    new_node_cant_reach_o = 0
    new_node_passed_filters = 0
    #global node_is_new_trailhead
    node_is_new_trailhead = 0
    new_node_is_o_idx = 0
    max_queue_length = 0
    #global trailblazer_early_termination
    trailblazer_early_termination = 0
    better_update_difference_total = 0
    '''

    impossible_high_cost = max(d_idxs.values()) + 1

    def trailblazer(d_idx, source):
        # nonlocal node_is_new_trailhead, trailblazer_early_termination
        # node_is_new_trailhead += 1
        for trailblazer_node in o_scope_paths[source][-2::-1]:
            if trailblazer_node in distance_matrix[d_idx]:
                # trailblazer_early_termination += 1
                break
            distance_matrix[d_idx][trailblazer_node] = impossible_high_cost

    best_weight_node_queue = []
    # TODO: using a hashable priority queue might be something to try... as it offer a fast way to update a
    #  value which might make trailblazing work effeciently with distances than just passing screening.

    for d_idx in d_idxs:
        distance_matrix[d_idx] = {d_idx: 0}
        trailblazer(d_idx, d_idx)
        heappush(best_weight_node_queue, (0, d_idx))

        while best_weight_node_queue:
            # if len(best_weight_node_queue) > max_queue_length:
            #    max_queue_length = len(best_weight_node_queue)

            # queue_counter += 1
            weight, node = heappop(best_weight_node_queue)
            # for neighbor in list(graph.neighbors(node)):
            # if o_graph is None:
            #    scope_neighbors = list(graph.neighbors(node))
            # else:
            #    scope_neighbors = [neighbor for neighbor in list(graph.neighbors(node)) if neighbor in od_scope]
            for neighbor in list(graph.neighbors(node)):
                # queue_neighbor_counter += 1
                spent_weight = graph.edges[(node, neighbor)]["weight"]
                if neighbor in distance_matrix[d_idx]:  # equivalent to if in seen
                    if (spent_weight + weight) >= distance_matrix[d_idx][neighbor]:
                        # current_is_better += 1
                        # print(f"{node = }\t{d_scope = }\tAlready found shorter distance termination")
                        continue
                    else:
                        # better_update_difference_total += distance_matrix[d_idx][neighbor] - (spent_weight + weight)
                        distance_matrix[d_idx][neighbor] = (spent_weight + weight)
                        # found_better_updates += 1
                        # TODO: Here, we also don't need to recheck. just insert to the heap again..
                        heappush(best_weight_node_queue, (spent_weight + weight, neighbor))
                        continue
                # found_new_node += 1

                if neighbor not in o_scope:
                    # print(f"{node = }\tOut of O Scope termination")
                    # new_node_out_of_scope += 1
                    continue

                if (spent_weight + weight + o_scope[neighbor]) > (o_scope[d_idx] * detour_ratio):
                    # new_node_cant_reach_o += 1
                    # print(f"{node = }\t{spent_weight = }\t{weight = }
                    # \t{o_scope[d_idx] = }\t{o_scope[d_idx]
                    # * detour_ratio = }network distance termination termination")
                    continue

                if len(list(graph.neighbors(neighbor))) == 1:
                    # print(f"{node = }\tSIngle neighbor termination")
                    # new_node_is_deadend += 1
                    continue
                # new_node_passed_filters += 1
                # TODO: Mark as trailhead, and also insert o_shortest route elements into distance_matrix[d_idx]
                distance_matrix[d_idx][neighbor] = (spent_weight + weight)

                if neighbor == o_idx:
                    # new_node_is_o_idx += 1
                    # print("got home")
                    continue
                # new_node_passed_filters +=1
                trailblazer(d_idx, neighbor)
                heappush(best_weight_node_queue, (spent_weight + weight, neighbor))

    # start = time.time()
    for d_idx in distance_matrix.keys():
        od_scope = od_scope.union(set(distance_matrix[d_idx].keys()))
    # combining_sets = time.time() - start
    if o_graph is None:
        network.update_light_graph(
            graph=graph,
            remove_nodes=[o_idx]
        )
    '''
    print(f"{queue_counter = }\n"
          f"{queue_neighbor_counter = }\n"
          f"{current_is_better = }\n"
          f"{found_better_updates = }\n"
          f"{found_new_node = }\n"
          f"{new_node_is_deadend = }\n"
          f"{new_node_out_of_scope = }\n"
          f"{new_node_cant_reach_o = }\n"
          f"{new_node_passed_filters = }\n"
          f"{node_is_new_trailhead = }\n"
          f"{new_node_is_o_idx = }\n"
          f"{max_queue_length = }\n"
          f"{trailblazer_early_termination = }\n"
          f"{len(od_scope) = }\n"
          f"{len(d_idxs) = }\n"
          f"{combining_sets = }\n"
          f"{better_update_difference_total/found_better_updates = }\n"
          )
      '''

    return od_scope, distance_matrix, d_idxs


def bfs_paths_many_targets_iterative(network: Network, graph, o_idx, d_idxs, distance_matrix=None, turn_penalty=False,
                                      od_scope=None):
    '''
    Generate all the paths from the given origin to all destinations, given the distance matrix and
    the nodes it can reach (the od_scope)8
    '''
    # TODO: implement this as an iterative function with a queue

    # remove any unnecccisary checks for other algorithms and methods. fucus on network distance.

    # Think if there is value for the queue to be priority (Neighbor with most dests, neighbor
    # with least neighbors, ...)

    # Rethink the remaining dests list. should it be additive (no copy?) or should
    # it copy previous and be subtractvd?

    # collect stats about checks so they're ordered effeciently (hard checks first to narrow funnel.)

    # check if using a generic graph is okay instead of needing to implement a
    # narrow graph in previous step. This cuts many previous preprocessinggs.

    # see if checks could be moved to earlier steps, or brought from previous steps. for less checks.

    # TODO: for turn implementation, make sure to account for turns whenever distance_matrix is updated. Twice in the current version.
    #  better yet, make sure to include turns as part of spent_weight so it runs through all the termination conditions.

    # TODO: should this also be done in graph generation to limit scope? prob yes but need to keep track of predecessor.

    node_gdf = network.nodes
    edge_gdf = network.edges

    paths = {}
    distances = {}
    for d_idx in d_idxs:
        paths[d_idx] = []
        distances[d_idx] = []

    # queue initialization
    q = deque([])
    # for neighbor in list(graph.neighbors(o_idx)):
    # q.append(([], o_idx, list(d_idxs.keys()), 0))
    q.appendleft(([o_idx], o_idx, list(d_idxs.keys()), 0))

    while q:
        visited, source, targets_remaining, current_weight = q.pop()

        if od_scope is None:
            scope_neighbors = list(graph.neighbors(source))
        else:
            scope_neighbors = [neighbor for neighbor in list(graph.neighbors(source)) if neighbor in od_scope]
        for neighbor in scope_neighbors:
            if neighbor in visited:
                continue
            # the given graph have 2 neighbors for origins and destinations. if a node has only one
            # neighbor, its a deadend.
            if len(list(graph.neighbors(neighbor))) == 1:
                continue

            # print(f"__________{neighbor = }________________")
            turn_cost = 0
            if turn_penalty and len(visited) >= 2:
                turn_cost = turn_penalty_value(network, visited[-2], source, neighbor)

            spent_weight = graph.edges[(source, neighbor)]["weight"]
            neighbor_current_weight = current_weight + spent_weight + turn_cost
            neighbor_targets_remaining = []
            '''
            for target in targets_remaining:
                if neighbor_current_weight <= d_idxs[target]:
                    if neighbor in distance_matrix[target]:
                        point_distance = distance_matrix[target][neighbor]
                        if (point_distance + neighbor_current_weight) <= d_idxs[target]:
                            neighbor_targets_remaining.append(target)
            '''

            neighbor_targets_remaining = []
            # TODO think if flipping distance_matrix indexing is better for this
            for target in targets_remaining:
                if neighbor in distance_matrix[target]:
                    if round(distance_matrix[target][neighbor] + neighbor_current_weight, 6) <= round(d_idxs[target],6):
                        neighbor_targets_remaining.append(target)


            if neighbor in neighbor_targets_remaining:
                paths[neighbor].append([x for x in visited if x not in d_idxs] + [neighbor])
                # paths[neighbor] = paths[neighbor] + [[x for x in visited if x not in d_idxs] + [neighbor]]
                distances[neighbor].append(neighbor_current_weight)
                # distances[neighbor] = distances[neighbor] + [neighbor_current_weight]
                neighbor_targets_remaining.remove(neighbor)

            if len(neighbor_targets_remaining) == 0:
                continue
            # visited, source, targets_remaining, current_weight
            # q.append((visited + [neighbor], neighbor, neighbor_targets_remaining, neighbor_current_weight))
            q.appendleft((visited + [neighbor], neighbor, neighbor_targets_remaining, neighbor_current_weight))
    return paths, distances