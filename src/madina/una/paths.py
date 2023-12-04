import math
from collections import deque
from heapq import heappush, heappop
from ..zonal import Network


def path_generator(network: Network, o_idx, search_radius=800, detour_ratio=1.15, turn_penalty=False):
    """
    to be filled in.
"""
    o_graph = network.d_graph
    network.add_node_to_graph(o_graph, o_idx)
    #network.update_light_graph(o_graph, add_nodes=[o_idx])

    d_idxs, o_scope, o_scope_paths = turn_o_scope(
        network=network,
        o_idx=o_idx,
        search_radius=search_radius,
        detour_ratio=detour_ratio,
        turn_penalty=turn_penalty,
        o_graph=o_graph,
        return_paths=True
    )

    #if len(d_idxs) == 0:
    #    # no destinations reachable, return empty dests, paths, weights.
    #    return {}, {}, {}


    scope_nodes, distance_matrix, d_idxs = bfs_subgraph_generation(
        o_idx=o_idx,
        detour_ratio=detour_ratio,
        o_graph=o_graph,
        d_idxs=d_idxs,
        o_scope=o_scope,
        o_scope_paths=o_scope_paths,
    )

    d_allowed_distances = {}
    for d_idx in d_idxs.keys():
        d_allowed_distances[d_idx] =  d_idxs[d_idx] * detour_ratio

    #paths, distances = bfs_paths_many_targets_iterative(
    path_edges, distances = bfs_path_edges_many_targets_iterative(
        network=network,
        o_graph=o_graph,
        o_idx=o_idx,
        d_idxs=d_allowed_distances,
        distance_matrix=distance_matrix,
        turn_penalty=turn_penalty,
        od_scope=scope_nodes
    )
    
    #network.update_light_graph(o_graph, remove_nodes=[o_idx])
    network.remove_node_to_graph(o_graph, o_idx)
    #return paths, distances, d_idxs
    return path_edges, distances, d_idxs



def bfs_subgraph_generation(
    o_idx,
    detour_ratio=1.15,
    o_graph=None,
    d_idxs=None,
    o_scope=None,
    o_scope_paths=None,
    ):
    # TODO: should we only run destination search to half radius and use origin's shortest paths to complete the rest?
    # TODO: should we use origin shortest paths to quickly terminate search from dest?traverse path to update
    #  distance from destination to the dist_matrix
    # TODO: should we batch search for all dests? it would be hard as we only keep track of one weight? maybe
    #  keep track of multiple weights one for each source.
    # TODO: should we use shortest_path_to_o while we search? probably yes.
    od_scope = set()
    distance_matrix = {}

    if (len(d_idxs) == 0):
        return od_scope, distance_matrix, d_idxs

    distance_matrix = {node_idx:{} for node_idx in o_scope.keys()}


    impossible_high_cost = max(d_idxs.values()) + 1

    def trailblazer(d_idx, source):
        # nonlocal node_is_new_trailhead, trailblazer_early_termination
        # node_is_new_trailhead += 1
        for trailblazer_node in o_scope_paths[source][-2::-1]:
            #if trailblazer_node in distance_matrix[d_idx]:
            if d_idx in distance_matrix[trailblazer_node]:
                # trailblazer_early_termination += 1
                break
            #distance_matrix[d_idx][trailblazer_node] = impossible_high_cost
            distance_matrix[trailblazer_node][d_idx] = impossible_high_cost

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
            for neighbor in list(o_graph.neighbors(node)):
                if neighbor not in o_scope:
                    # print(f"{node = }\tOut of O Scope termination")
                    # new_node_out_of_scope += 1
                    continue
                # queue_neighbor_counter += 1
                # TODO: consolidate the addinion of 'weight'
                spent_weight =  o_graph.edges[(node, neighbor)]["weight"] + weight
                #if neighbor in distance_matrix[d_idx]:  # equivalent to if in seen
                if d_idx in distance_matrix[neighbor]:
                    #if spent_weight >= distance_matrix[d_idx][neighbor]:
                    if spent_weight >= distance_matrix[neighbor][d_idx]:
                        # current_is_better += 1
                        # print(f"{node = }\t{d_scope = }\tAlready found shorter distance termination")
                        continue
                    else:
                        # better_update_difference_total += distance_matrix[d_idx][neighbor] - (spent_weight + weight)
                        #distance_matrix[d_idx][neighbor] = spent_weight
                        distance_matrix[neighbor][d_idx] = spent_weight 
                        # found_better_updates += 1
                        # TODO: Here, we also don't need to recheck. just insert to the heap again..
                        heappush(best_weight_node_queue, (spent_weight, neighbor))
                        continue
                # found_new_node += 1



                if (spent_weight + o_scope[neighbor]) > (o_scope[d_idx] * detour_ratio):
                    # new_node_cant_reach_o += 1
                    # print(f"{node = }\t{spent_weight = }\t{weight = }
                    # \t{o_scope[d_idx] = }\t{o_scope[d_idx]
                    # * detour_ratio = }network distance termination termination")
                    continue

                if len(list(o_graph.neighbors(neighbor))) == 1:
                    # print(f"{node = }\tSIngle neighbor termination")
                    # new_node_is_deadend += 1
                    continue
                # new_node_passed_filters += 1
                # TODO: Mark as trailhead, and also insert o_shortest route elements into distance_matrix[d_idx]
                #distance_matrix[d_idx][neighbor] = spent_weight
                distance_matrix[neighbor][d_idx] = spent_weight

                if neighbor == o_idx:
                    # new_node_is_o_idx += 1
                    # print("got home")
                    continue
                # new_node_passed_filters +=1
                trailblazer(d_idx, neighbor)
                heappush(best_weight_node_queue, (spent_weight, neighbor))

    #for d_idx in distance_matrix.keys():
    #    od_scope = od_scope.union(set(distance_matrix[d_idx].keys()))
    

    od_scope = {node_idx for node_idx in distance_matrix.keys() if  len(distance_matrix[node_idx]) > 0}

    return od_scope, distance_matrix, d_idxs


def bfs_paths_many_targets_iterative(
    network: Network,
    o_graph,
    o_idx,
    d_idxs,
    distance_matrix=None,
    turn_penalty=False,
    od_scope=None
    ):
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

    allowed_path_nodes = set(network.street_node_ids)
    allowed_path_nodes.add(o_idx)
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
            scope_neighbors = list(o_graph.neighbors(source))
        else:
            scope_neighbors = [neighbor for neighbor in list(o_graph.neighbors(source)) if neighbor in od_scope]
        for neighbor in scope_neighbors:
            if neighbor in visited:
                continue
            # the given graph have 2 neighbors for origins and destinations. if a node has only one
            # neighbor, its a deadend.
            if len(list(o_graph.neighbors(neighbor))) == 1:
                continue

            turn_cost = 0
            if turn_penalty and len(visited) >= 2:
                turn_cost = turn_penalty_value(network, visited[-2], source, neighbor)

            spent_weight = o_graph.edges[(source, neighbor)]["weight"]
            neighbor_current_weight =  current_weight + spent_weight + turn_cost
            neighbor_targets_remaining = []

            neighbor_targets_remaining = []
            for target in targets_remaining:
                if neighbor in distance_matrix[target]:
                    # equality with small tolerance to allow numerical error in case there was no detour ratio
                    if distance_matrix[target][neighbor] + neighbor_current_weight - d_idxs[target] <=  0.00001:
                        neighbor_targets_remaining.append(target)


            if neighbor in neighbor_targets_remaining:
                paths[neighbor].append([x for x in visited if x in allowed_path_nodes] + [neighbor])  
                # paths[neighbor].append([x for x in visited if x not in d_idxs] + [neighbor])
                distances[neighbor].append(neighbor_current_weight)
                neighbor_targets_remaining.remove(neighbor)

            if len(neighbor_targets_remaining) == 0:
                continue

            q.appendleft((visited + [neighbor], neighbor, neighbor_targets_remaining, neighbor_current_weight))
    return paths, distances

import networkx as nx
def wandering_messenger(
    network: Network,
    o_graph,
    o_idx,
    d_idxs,
    distance_matrix=None,
    turn_penalty=False,
    od_scope=None
):
    graph_dict = nx.to_dict_of_dicts(o_graph, nodelist=od_scope)

    path_edges = {d_idx: deque([]) for d_idx in d_idxs}
    distances = {d_idx: deque([]) for d_idx in d_idxs}

    path_diary =  [(o_idx, [], [], [], 0)] # (source, source_visited, source_visited_targets, source_visited_edges, source_weight)
    path_tree = deque([(o_neighbor, 0) for o_neighbor in graph_dict[o_idx]])


    while path_tree:
        node, source_diary_page = path_tree.pop()
        (source, source_visited, source_visited_targets, source_visited_edges, source_weight) = path_diary[source_diary_page]

        turn_cost = turn_penalty_value(network, source_visited[-1], source, node) if (turn_penalty and len(source_visited) > 0) else 0

        node_weight =  source_weight + graph_dict[source][node]["weight"] + turn_cost
        node_edge_id = graph_dict[source][node]["id"] 

        node_visited_edges = source_visited_edges if node_edge_id in source_visited_edges else source_visited_edges+[node_edge_id]

        if (node in d_idxs)  and (node_weight - d_idxs[node] <=  0.00001):
            path_edges[node].append(node_visited_edges) #### ADD PATH (of step ids...) TO TREE if compression is needed.
            distances[node].append(node_weight) ##### For step ID node, NEIGHBOR AS NODE ATTRIBUTE FOR neighbor. ALSO, Neighbor ID Aattribute
            node_visited_targets = source_visited_targets + [node]
        else:
            node_visited_targets = source_visited_targets

        for target in distance_matrix[node]:
            if (target not in node_visited_targets) and (distance_matrix[node][target] + node_weight - d_idxs[target] <=  0.00001):
                # there is at least one more reachible unvisited target: keep going
                # update diary here
                
                node_visited = source_visited + [source]
                # (source, source_visited, source_visited_targets, source_visited_edges, source_weight)
                path_diary  =  path_diary[:source_diary_page + 1] + [(node, node_visited, node_visited_targets, node_visited_edges, node_weight)]
                    
                ## This intersection could have been eliminated if the o_graph was as tight as the o_scope
                #for neighbor in set(o_graph.neighbors(node)).intersection(od_scope):
                for neighbor in graph_dict[node]:
                    if neighbor not in node_visited:
                        path_tree.append((neighbor, source_diary_page + 1))
                
                break # break after finding one remaining target and doing neibor queuing





    return path_edges, distances






def bfs_path_edges_many_targets_iterative(
    network: Network,
    o_graph,
    o_idx,
    d_idxs,
    distance_matrix=None,
    turn_penalty=False,
    od_scope=None
    ):

    path_edges = {}
    distances = {}
    for d_idx in d_idxs:
        path_edges[d_idx] = deque([])#[]
        distances[d_idx] = deque([])#[]

    # queue initialization
    q = deque([])
    q.append(([o_idx], [],  o_idx, 0))
    #q.append(([o_idx], [], [],  o_idx, 0))

    while q:
        visited, edges, source, current_weight = q.pop()
        #visited, visited_targets, edges, source, current_weight = q.pop()

        scope_neighbors = set(o_graph.neighbors(source)).intersection(od_scope)

        for neighbor in scope_neighbors:
            if neighbor in visited:
                continue

            turn_cost = 0
            if turn_penalty and len(visited) >= 2:
                turn_cost = turn_penalty_value(network, visited[-2], source, neighbor)



            edge_data = o_graph.edges[(source, neighbor)]
            neighbor_current_weight =  current_weight + edge_data["weight"] + turn_cost
            ## create a list of edges visited, avoid adding an edge twice (when passing a destination, two segments have the same edg id for instance)
            


            # neighbor_edges = np.array(edges) if edge_id in edges else np.append(edges, edge_id)

            #neighbor_visited_targets = visited_targets.copy()

            if (neighbor in d_idxs)  and (neighbor_current_weight - d_idxs[neighbor] <=  0.00001):
                #path_edges[neighbor].append(neighbor_edges)
                path_edges[neighbor].append(edges)
                distances[neighbor].append(neighbor_current_weight)
                #neighbor_visited_targets.append(neighbor)



            neighbor_visited = visited + [neighbor]
            #neighbor_visited = np.append(visited, neighbor)
            #neighbor_targets_remaining = [target for target in distance_matrix[neighbor] if (target not in neighbor_visited) and (distance_matrix[neighbor][target] + neighbor_current_weight - d_idxs[target] <=  0.00001)]
            #neighbor_targets_remaining = [target for target in distance_matrix[neighbor] if (target not in neighbor_visited_targets) and (distance_matrix[neighbor][target] + neighbor_current_weight - d_idxs[target] <=  0.00001)]


            for target in distance_matrix[neighbor]:
                if (target not in neighbor_visited) and (distance_matrix[neighbor][target] + neighbor_current_weight - d_idxs[target] <=  0.00001):
                #if (target not in neighbor_visited_targets) and (distance_matrix[neighbor][target] + neighbor_current_weight - d_idxs[target] <=  0.00001):
                    # there is at least one more reachible unvisited target: keep going

                    ## handling edghe ids after appending resuls makes us ignore storing destination segment, reducing memory footpring.
                    if len(visited) == 1:
                        neighbor_edges = []
                    #igniring the first segment, storing it is memory overhead
                    else:
                        edge_id = edge_data["id"] 
                        neighbor_edges = edges.copy() if edge_id in edges else edges + [edge_id]
                    
                    q.append((neighbor_visited, neighbor_edges, neighbor, neighbor_current_weight))
                    #q.append((neighbor_visited, neighbor_visited_targets, neighbor_edges, neighbor, neighbor_current_weight))

                    break
            #otherwise, continue



            #if len(neighbor_targets_remaining) == 0:
            #    continue




    return path_edges, distances

def turn_o_scope(
    network: Network,
    o_idx,
    search_radius: float,
    detour_ratio: float, 
    turn_penalty=True,
    o_graph=None,
    return_paths=True
    ):
    """
    TODO: fill out the spec
    o_idx: origin index, integer, coming from the node_gdf
    o_graph: reusing updated graphs (e. g. doing inelastic after elastic), optional
    """
    node_gdf = network.nodes
    destinations = node_gdf[node_gdf["type"] == "destination"].index
    # print(f"turn_o_scope: {o_idx = }")


    # visualize_graph(self, graph)
    o_scope = {o_idx: 0}
    d_idxs = {}
    o_scope_paths = {}

    forward_q = [(0, o_idx, [o_idx])]

    furthest_dest_weight = 0

    while forward_q:
        weight, node, visited = heappop(forward_q)
        for neighbor in list(o_graph.neighbors(node)):

            turn_cost = 0
            if turn_penalty and len(visited) >= 2 :
                turn_cost = turn_penalty_value(network, visited[-2], node, neighbor)

            # TODO: remove duplicate checking of condition
            neighbor_weight = weight + o_graph.edges[(node, neighbor)]["weight"] + turn_cost
            if neighbor in o_scope :  # equivalent to if in seen
                if (neighbor_weight >= o_scope[neighbor]):
                    #current_is_better += 1
                    continue
                o_scope[neighbor] = neighbor_weight
                if return_paths:
                    o_scope_paths[neighbor] = visited + [neighbor]
                if (neighbor in destinations) and (neighbor_weight <= search_radius):
                    furthest_dest_weight = max(furthest_dest_weight, neighbor_weight)
                    d_idxs[neighbor] = neighbor_weight
                # found_better_updates += 1
                heappush(forward_q, (neighbor_weight, neighbor, visited + [neighbor]))
                continue
                
            
            #not seen, check eligibility
            if neighbor_weight > max(search_radius, furthest_dest_weight * (0.5+detour_ratio*0.5) ):
                continue

            if len(list(o_graph.neighbors(neighbor))) == 1:
                continue


            if (neighbor in destinations) and (neighbor_weight <= search_radius):
                furthest_dest_weight = max(furthest_dest_weight, neighbor_weight)
                d_idxs[neighbor] = neighbor_weight
            o_scope[neighbor] = neighbor_weight
            if return_paths:
                o_scope_paths[neighbor] = visited + [neighbor]
            heappush(forward_q, (neighbor_weight, neighbor, visited + [neighbor]))
    return d_idxs, o_scope, o_scope_paths

def turn_penalty_value(network: Network, previous_node, current_node, next_node):
    """
    TODO: fill out the spec
    """
    node_gdf = network.nodes
    edge_gdf = network.edges
    previous_segment = edge_gdf[
        (edge_gdf['start'] == previous_node) & (edge_gdf['end'] == current_node) | 
        (edge_gdf['end'] == previous_node) & (edge_gdf['start'] == current_node) 
    ]

    next_segment = edge_gdf[
        (edge_gdf['start'] == current_node) & (edge_gdf['end'] == next_node) | 
        (edge_gdf['end'] == current_node) & (edge_gdf['start'] == next_node) 
    ]

    
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
