import numpy as np
import concurrent.futures
import time
from networkx.classes.function import path_weight
import networkx as nx
from shapely.ops import split
import shapely.geometry as geo
import geopandas as gpd
import pandas as pd
import math
from tqdm import tqdm

import shapely

# def dfs_shortest_paths(visited=[], path_so_far = [], start, finish, limit)
'''
Good performance code:



    elif distance_method == "network":
        # graph = self.od_graph.copy()
        graph = self.d_graph
        update_light_graph(self, graph, add_nodes=[o_idx])
        o_network_scope = nx.single_source_dijkstra_path_length(graph, source=o_idx,
                                                                cutoff=max(d_idxs.values()) * detour_ratio,
                                                                weight="weight")
        scope_nodes = set()
        distance_matrix = {}
        for d_idx in d_idxs.keys():
            distance_matrix[d_idx] = {}
            d_network_scope = nx.single_source_dijkstra_path_length(graph, source=d_idx,
                                                                    cutoff=o_network_scope[d_idx] * detour_ratio,
                                                                    weight="weight")
            [node for node in d_network_scope if
             (node in o_network_scope) and (d_network_scope[node] + o_network_scope[node]) <= (
                         o_network_scope[d_idx] * detour_ratio)]
            d_ellipse_scope = set()
            for node in d_network_scope:
                if node in o_network_scope:
                    if (d_network_scope[node] + o_network_scope[node]) <= (o_network_scope[d_idx] * detour_ratio):
                        d_ellipse_scope.add(node)
                        distance_matrix[d_idx][node] = d_network_scope[node]
            scope_nodes = scope_nodes.union(d_ellipse_scope)
        scope_nodes = node_gdf[node_gdf.index.isin(scope_nodes)]
        update_light_graph(self, graph, remove_nodes=[o_idx])




'''

'''
backup of the network section of get_od_subgraph()



    elif distance_method == "network":
        start = time.time()
        #graph = self.G.copy()
        #destination_ids = list(node_gdf[node_gdf["type"] == "destination"].index)
        #destination_ids = list(d_idxs.keys())
        #update_light_graph(
        #    self,
        #    graph=graph,
        #    add_nodes=[o_idx] + destination_ids
        #)
        graph = self.od_graph.copy()
        # TODO: check if narrowing the search radius to max(d_idxs.values()) won't cause issues.
        #print(f"it took {time.time() - start} prepared light graph...")
        o_network_scope = nx.single_source_dijkstra_path_length(graph, source=o_idx,
                                                                cutoff=search_radius * detour_ratio, weight="weight")

        #reachable_destinations = node_gdf[node_gdf.index.isin(list(o_network_scope.keys()))]
        ## filter reachable_destinations by search radis=us. detour ratio was meant to keep the search space large to iclude routing possibilities
        scope_nodes = []
        distance_matrix = {}
        # for d_idx in reachable_destinations.index:
        for d_idx in d_idxs.keys():
            distance_matrix[d_idx] = {}
            # TODO: This could be eliminated by using the sorted results of "o_network_scope" with the appropriate cutoff
            #od_network_scope = nx.single_source_dijkstra_path_length(graph, source=o_idx,
            #                                                         cutoff=o_network_scope[d_idx] * detour_ratio, weight="weight")

            # todo: cHECK IF THIS LOOP IS ACTUALLY NECISSARY WHEN YOU CAN JUST REFERENCE THINGS BY INDES FROM
            od_network_scope = {}
            

            for node in o_network_scope:
                if o_network_scope[node] <= o_network_scope[d_idx] * detour_ratio:
                    od_network_scope[node] = o_network_scope[node]
            d_network_scope = nx.single_source_dijkstra_path_length(graph, source=d_idx,
                                                                    cutoff=od_network_scope[d_idx] * detour_ratio, weight="weight")
            d_ellipse_scope = []
            search_scope = set(d_network_scope.keys()).intersection(set(od_network_scope.keys()))
            # search_scope =  search_scope.remove(d_idx)
            # print (search_scope)
            # TODO: Check if iterating over d_network_scope could eliminate needing to calculate the union
            for node in search_scope: 
                if (d_network_scope[node] + od_network_scope[node]) <= od_network_scope[d_idx] * detour_ratio:
                    d_ellipse_scope.append(node)
                    distance_matrix[d_idx][node] = d_network_scope[node]
            scope_nodes = set(scope_nodes).union(set(d_ellipse_scope))
            # scope_nodes = set(scope_nodes).union(set(od_network_scope.keys()))
            #print(f"{o_idx = }\t{d_idx = }\t{len(d_network_scope) = }\t{len(od_network_scope) = }\t{len(search_scope) = }\t{len(scope_nodes) = }")
        scope_nodes = node_gdf[node_gdf.index.isin(list(scope_nodes))]
        #print(f"it took {time.time() - start} to run single_source_dijkstra_path_length on search_radius * detour_ratio ")

'''


def visualize_graph(self, graph):
    node_gdf = self.layers["network_nodes"]["gdf"]
    edge_gdf = self.layers["network_edges"]["gdf"]
    graph_nodes = node_gdf[node_gdf.index.isin(list(graph.nodes))]
    edge_geometries = []
    edge_weights = []
    edge_ids = []
    start_nodes = []
    end_nodes = []
    for edge in list(graph.edges):
        edge_ids.append(graph.edges[edge]["id"])
        edge_weights.append(graph.edges[edge]["weight"])
        edge_geometries.append(geo.LineString([graph_nodes.at[edge[0], "geometry"], graph_nodes.at[edge[1], "geometry"]]))
        start_nodes.append(edge[0])
        end_nodes.append(edge[1])
    graph_edges = gpd.GeoDataFrame(
        {
            "parent_edge_id": edge_ids,
            "weight": edge_weights,
            "geometry": edge_geometries,
            "start_node": start_nodes,
            "end_node": end_nodes,
        }, crs=node_gdf.crs)
    self.create_deck_map(
        [
            {"gdf": edge_gdf, "color": [125, 125, 125], "opacity": 0.05},
            {"gdf": graph_nodes[graph_nodes["type"] == "street_node"].reset_index(), "color": [0, 255, 125], "opacity": 0.05, "text": "id"},
            {"gdf": graph_nodes[graph_nodes["type"] == "origin"], "opacity": 0.25, "color": [255, 0, 125]},
            {"gdf": graph_nodes[graph_nodes["type"] == "destination"], "opacity": 0.05, "color": [0, 125, 255],
             "text": "type"},
            {"gdf": graph_edges, "color": [0, 255, 125], "opacity": 0.10, "text": "weight"},
        ],
        basemap=False,
        save_as="graph.html"
    )
    return



def get_od_subgraph(self, o_idx, d_idxs, search_radius, detour_ratio, shortest_distance=0, output_map=False, trim=False,
                    distance_method="geometric"):
    #print (f"{detour_ratio = }")
    # The function "destinations_accessible_from_origin" has already been called
    # Assumption: destination is already reachable from origin
    start = time.time()
    if isinstance(d_idxs, int):
        d_idxs = {d_idxs: shortest_distance}

    edge_gdf = self.layers["network_edges"]["gdf"]
    node_gdf = self.layers["network_nodes"]["gdf"]
    counter = 0
    if distance_method == "geometric":
        from shapely.ops import unary_union
        scopes = []
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
        # scope = o_scope.union(d_scope)
        scope = unary_union(scopes)
        scope_nodes = node_gdf[node_gdf.intersects(scope)]
        # print(f"{time.time() - start}\tGeometric Scope Done")
    elif distance_method == "network":
        # graph = self.od_graph.copy()
        graph = self.d_graph
        update_light_graph(self, graph, add_nodes=[o_idx])
        o_network_scope = nx.single_source_dijkstra_path_length(
            graph, source=o_idx,
            cutoff=max(d_idxs.values()) * detour_ratio,
            weight="weight"
        )
        print(f"{time.time() - start}\tO SIngle source path length done")
        #o_graph = graph.subgraph(list(o_network_scope.keys()))
        # TODO: using nodes in o_network_scope to construct a subgraph for the inner loop might be better....
        scope_nodes = set()
        distance_matrix = {}
        #d_graph = self.G
        #update_light_graph(self, d_graph, add_nodes=[o_idx])
        for d_idx in d_idxs.keys():
            #update_light_graph(self, d_graph, add_nodes=[d_idx])
            d_network_scope = nx.single_source_dijkstra_path_length(
                graph,
                source=d_idx,
                cutoff=d_idxs[d_idx] * detour_ratio,
                weight="weight"
            )
            #update_light_graph(self, d_graph, remove_nodes=[d_idx])
            d_ellipse_scope = set()
            #print (f"before inner loop {scope_nodes = }\t{d_ellipse_scope = }")
            distance_matrix[d_idx] = {}
            for node in d_network_scope:
                if node in o_network_scope:
                    if (d_network_scope[node] + o_network_scope[node]) <= (d_idxs[d_idx] * detour_ratio):
                        d_ellipse_scope.add(node)
                        distance_matrix[d_idx][node] = d_network_scope[node]
            #print (f"before union {scope_nodes = }\t{d_ellipse_scope = }")
            scope_nodes = scope_nodes.union(d_ellipse_scope)
            #print (f"after union {scope_nodes = }\t{d_ellipse_scope = }")
        #update_light_graph(self, d_graph, remove_nodes=[o_idx])
        update_light_graph(self, graph, remove_nodes=[o_idx])
        scope_nodes = node_gdf[node_gdf.index.isin(scope_nodes)]
        print(f"{time.time() - start}\tNetwork Scope Done")
    else:
        print(f"distance method {distance_method} not implemented. options are: ['geometric', 'network']")
    while True:
        # get edges and construct a network
        node_list = list(scope_nodes.index)
        # print (edge_gdf["start"].isin(node_list))

        od_edge_list = [
                           node_gdf.at[o_idx, "nearest_street_id"]
                       ] + list(node_gdf.loc[d_idxs]["nearest_street_id"])
        # print(od_edge_list)
        scope_edges = edge_gdf[
            (edge_gdf["start"].isin(node_list) & edge_gdf["end"].isin(node_list)) |
            edge_gdf.index.isin(od_edge_list)
            ]

        od_edge_nodes = list(
            set(list(scope_edges.loc[od_edge_list]["start"]) + list(scope_edges.loc[od_edge_list]["end"])))
        # print(f"{od_edge_nodes = }")
        # inclode od edges..

        # scope_edges = edge_gdf.loc[list(set(list(scope_edges.index) + [node_gdf.at[o_idx, "nearest_street_id"],
        #                                                                       node_gdf.at[d_idx, "nearest_street_id"]]))]

        # This eliminate dother destinations that got included from node_gdf, better to filter node_gdf..
        scope_node_ids = list(set(list(scope_edges["start"].unique()) + list(scope_edges["end"].unique())))
        scope_nodes = node_gdf.loc[scope_node_ids]

        if not trim:
            break

        # scope_node_ids = list(set(list(scope_edges["start"].unique()) + list(scope_edges["end"].unique())))
        # scope_nodes = node_gdf.loc[scope_node_ids]
        # print(f"{~scope_edges.index.isin(od_edge_list) = }")
        # inner_edges = scope_edges[~scope_edges.index.isin(od_edge_list)]
        node_degree = pd.concat([scope_edges["start"], scope_edges["end"]]).value_counts()
        # print(f"node degree{node_degree}")
        # print(f"outer degree mask {~node_degree.index.isin(od_edge_nodes)}")
        inner_nodes_degrees = node_degree[~node_degree.index.isin(od_edge_nodes)]
        # print (f"{inner_nodes_degrees = }")
        if inner_nodes_degrees[inner_nodes_degrees == 1].shape[0] == 0:
            break
        # print (f"{inner_nodes_degrees[inner_nodes_degrees == 1].shape[0] = }, {inner_nodes_degrees[inner_nodes_degrees == 1] = }")
        if counter >= 10:
            break
        counter += 1
        # break
        # print(f"keep these nodes: {inner_nodes_degrees[inner_nodes_degrees != 1].index}")
        scope_nodes = scope_nodes.loc[list(inner_nodes_degrees[inner_nodes_degrees != 1].index) + od_edge_nodes]
    print(f"{time.time() - start}\tScope nodes and edges prepared")
    # print (f"{counter} trims" )
    if output_map:
        if (len(d_idxs) == 1) and (distance_method == "geometric"):
            context_gdf = gpd.GeoDataFrame(
                {
                    "name": ["major_axis", "minor_axis", "o_scope", "d_scope", "scope", "origin", "destination"],
                    "geometry": [major_axis, minor_axis, o_scope, d_scope, scope, node_gdf.at[o_idx, "geometry"],
                                 node_gdf.at[d_idx, "geometry"]]
                },
                crs=edge_gdf.crs
            )
        elif (distance_method == "geometric"):
            context_gdf = gpd.GeoDataFrame(
                {
                    "name": ["scope"] + ["origin"] + ["destination"] * len(list(node_gdf.loc[d_idxs]["geometry"])),
                    "geometry": [scope] + [node_gdf.at[o_idx, "geometry"]] + list(node_gdf.loc[d_idxs]["geometry"])
                },
                crs=edge_gdf.crs
            )
        elif (len(d_idxs) == 1) and (distance_method == "network"):

            context_gdf = gpd.GeoDataFrame(
                {
                    "name": ["o_network_scope"] * len(o_network_scope) + ["d_network_scope"] * len(d_network_scope),
                    "geometry": list(node_gdf.loc[o_network_scope]["geometry"]) + list(
                        node_gdf.loc[d_network_scope]["geometry"])
                },
                crs=edge_gdf.crs
            )

        else:
            context_gdf = gpd.GeoDataFrame(
                {
                    "name": ["origin"],
                    "geometry": [node_gdf.at[o_idx, "geometry"]]
                },
                crs=edge_gdf.crs
            )
        self.create_deck_map(
            [
                {"gdf": context_gdf, "color_by_attribute": "name", "color_method": "categorical", "opacity": 0.10},
                {"gdf": edge_gdf, "color": [125, 125, 125], "opacity": 0.3},
                {"gdf": scope_edges, "color": [0, 255, 125], "opacity": 1},
                {"gdf": scope_nodes.reset_index(), "color": [0, 255, 125], "opacity": 0.10, "text": "id"},
                {"gdf": node_gdf.loc[d_idxs], "color": [255, 0, 125], "opacity": 0.25},
                {"gdf": node_gdf.loc[[o_idx]], "opacity": 1, "color": [0, 125, 255],
                 "text": "type"}
            ],
            basemap=False,
            save_as="small_graph_vis.html"
        )
        print(f"{time.time() - start}\tMap saved")
    G = nx.Graph()
    for idx in scope_edges.index:
        G.add_edge(
            int(scope_edges.at[idx, "start"]),
            int(scope_edges.at[idx, "end"]),
            # TODO: change this to either defaults to distance or a specified column for a weight..
            weight=scope_edges.at[idx, "weight"],
            type=scope_edges.at[idx, "type"],
            id=idx
        )

    for idx in scope_nodes.index:
        G.nodes[int(idx)]['type'] = scope_nodes.at[idx, "type"]

    ##insert o and d as nodes in graph
     print(f"{time.time() - start}\tGraph street edges added")
    update_light_graph(
        self,
        graph=G,
        add_nodes=[o_idx] + list(d_idxs.keys())
    )
    print(f"{time.time() - start}\tGraph origins and destinations added")
    # print(f"it took {time.time() - start} to create second light graph ")
    # print(f"small graph\to:{o_idx}\td:{d_idx}\tedge_count:{scope_edges.shape[0]}\ttime:{round(time.time() - start, 2)}")
    if distance_method == "network":
        return G, distance_matrix
    else:
        return G


def bfs_graph(self, o_idx, d_idxs, search_radius, detour_ratio):
    # TODO: start from origin, discover all destinations and keep track of all nodes, edges visited within search_radius * Detour ratio

    # TODO: For each destination found, run a bfs search to find EDGES IT COULD visit within ()


    return


def iterative_dfs_paths(G, source, targets, cutoff):
    visited = dict.fromkeys([source])
    stack = [iter(G[source])]
    while stack:
        print(visited)
        children = stack[-1]
        child = next(children, None)
        if child is None:
            stack.pop()
            visited.popitem()
        elif len(visited) < cutoff:
            if child in visited:
                continue
            if child in targets:
                yield list(visited) + [child]
            visited[child] = None
            if targets - set(visited.keys()):  # expand stack until find all targets
                stack.append(iter(G[child]))
            else:
                visited.popitem()  # maybe other ways to child
        else:  # len(visited) == cutoff:
            for target in (targets & (set(children) | {child})) - set(visited.keys()):
                yield list(visited) + [target]
            stack.pop()
            visited.popitem()


def bfs_paths(self, graph, source, target, weight_limit=0, distance_termination="geometric", batching=True,
              other_targets=[], distance_matrix=None):
    if distance_termination == "geometric":
        node_gdf = self.layers["network_nodes"]["gdf"]
        graph_nodes = node_gdf[node_gdf.index.isin(list(graph.nodes))]

        if isinstance(target, int):
            t = [target]
        else:
            t = list(target.keys())
        destination_nodes = node_gdf.loc[t]
        distance_matrix = graph_nodes["geometry"].apply(lambda x: destination_nodes["geometry"].distance(x))
        # print(distance_matrix)
    elif distance_termination == "network":
        # distance_matrix = dict(nx.all_pairs_dijkstra_path_length(graph))
        distance_matrix = distance_matrix
        # print (distance_matrix)
    elif distance_termination == "none":
        distance_matrix = None
    else:
        raise ValueError(
            f"parameter 'distance_termination' in function 'bfs_paths' can be ['geometric', 'network', 'none'], {distance_termination} was given.")
    # TODO: probably be more explicit that int means no batching, and dict means batching...
    if isinstance(target, int):
        paths, distances = _bfs_paths(graph, source, target, weight_remaining=weight_limit, visited=[source],
                                      distance_termination=distance_termination, distance_matrix=distance_matrix,
                                      paths=[], distances=[], other_targets=other_targets)
        return paths, distances
    elif isinstance(target, dict):
        ## Think about destination clustering hwere.....
        destinations = list(target.keys())
        try:
            paths = {}
            distances = {}
            for d_idx in destinations:
                paths[d_idx] = []
                distances[d_idx] = []
            paths, distances = _bfs_paths_many_targets(graph, source, targets=target,
                                                       targets_remaining=destinations, visited=[source],
                                                       distance_termination=distance_termination,
                                                       distance_matrix=distance_matrix, start_time=time.time(),
                                                       paths=paths, distances=distances, current_weight=0)
        except TimeoutError:
            # print("TImeout, switching to no batching ... ")
            paths = {}
            distances = {}
            for d_idx in destinations:
                d_paths, d_distances = _bfs_paths(graph, source, d_idx, weight_remaining=target[d_idx],
                                                  visited=[source],
                                                  distance_termination=distance_termination,
                                                  distance_matrix=distance_matrix, paths=[],
                                                  distances=[], other_targets=destinations)
                paths[d_idx] = d_paths
                distances[d_idx] = d_distances

        return paths, distances


'''
            paths = {}
            distances = {}
            for d_idx in destinations:
                paths[d_idx] = []
                distances[d_idx] = []
            paths, distances = _bfs_paths_many_targets(graph, source, targets=destinations,
                                                       targets_remaining=target.copy(), visited=[source],
                                                       distance_termination="none",
                                                       distance_matrix=distance_matrix, start_time=time.time(),
                                                       paths=paths, distances=distances)
'''


def _bfs_paths(graph, source, target, weight_remaining=-1, visited=[], distance_termination="geometric",
               distance_matrix=None, paths=[], distances=[], other_targets=[]):
    # print (f"{visited = }\t{source = }\t{weight_remaining}")
    neighbors = list(graph.neighbors(source))
    # print(len(neighbors))
    # print (neighbors)
    # print (visited)
    if len(list(neighbors)) == 0:
        # print ("no neighbors")
        return []
    for neighbor in neighbors:
        if neighbor in visited:
            # print(f"{source}\tAlready visited ")
            continue
        # print(f"__________{neighbor = }________________")
        spent_weight = graph.edges[(source, neighbor)]["weight"]

        if spent_weight > weight_remaining:
            # print(f"{neighbor}\tweight exceeded {spent_weight = }\t{weight_remaining = }\t{weight_limit = }")
            # print(f"{neighbor}\tweight exceeded")
            continue
        if distance_termination == "geometric":
            point_distance = distance_matrix[target][neighbor]
            if (point_distance + spent_weight) > weight_remaining:
                # print(f"Geometric Termination\t{source = }\t{neighbors = }\t{neighbor = }\t{target = }")
                # print(f"\t{point_distance = }\t{weight_remaining = }\t{weight_limit = }")
                continue
        elif distance_termination == "network":
            if neighbor in distance_matrix[target].keys():
                point_distance = distance_matrix[target][neighbor]
                if (point_distance + spent_weight) > weight_remaining:
                    continue
            else:
                continue

        if neighbor == target:
            # print(f"Arrived {[visited+[target]]}\t{weight_limit = }\t{weight_remaining = }\t{spent_weight = }")
            path = visited + [target]
            paths.append([x for x in visited if x not in other_targets] + [target])
            distances.append(path_weight(graph, path, "weight"))
            # return [visited + [target]]
            # next_visited.append(source)
            # next_visited.append(target)
        else:
            # print(f"{source}\tbranching to {neighbor}")
            _bfs_paths(graph, neighbor, target, weight_remaining - spent_weight,
                       visited=visited + [neighbor], distance_termination=distance_termination,
                       distance_matrix=distance_matrix, paths=paths, distances=distances, other_targets=other_targets)
            # print (f"{source}\tdead end, ready to go to next neighbor")
    return paths, distances


'''
2022-12-01 11:06
Backup of _bfs_paths_many_targets before changing weight calculation method....

def _bfs_paths_many_targets(graph, source, targets=[], targets_remaining=None, visited=[],
                            distance_termination="geometric", distance_matrix=None, start_time=0, paths=None,
                            distances=None):
    # print (f"{visited}\t{source}")
    # print (f"{visited}\t{source}\t{targets_remaining}")
    if time.time() - start_time >= 10:
        raise TimeoutError("_bfs_paths_many_targets timed out...")
    for neighbor in list(graph.neighbors(source)):
        if neighbor in visited:
            continue
        if len(list(graph.neighbors(neighbor))) == 0:
            continue
        neighbor_targets_remaining = targets_remaining.copy()

        # print(f"__________{neighbor = }________________")
        spent_weight = graph.edges[(source, neighbor)]["weight"]
        # print (f"{source = }\t{neighbor = }{spent_weight = }")
        for target in list(neighbor_targets_remaining.keys()):
            if spent_weight > neighbor_targets_remaining[target]:
                del neighbor_targets_remaining[target]
        if distance_termination in ["geometric", "network"]:
            for target in list(neighbor_targets_remaining.keys()):
                if distance_termination == "geometric":
                    point_distance = distance_matrix[neighbor][target]
                    if (point_distance + spent_weight) > neighbor_targets_remaining[target]:
                        del neighbor_targets_remaining[target]
                elif distance_termination == "network":
                    if neighbor in distance_matrix[target].keys():
                        point_distance = distance_matrix[target][neighbor]
                        if (point_distance + spent_weight) > neighbor_targets_remaining[target]:
                            del neighbor_targets_remaining[target]
                    else:
                        del neighbor_targets_remaining[target]
        if len(neighbor_targets_remaining) == 0:
            continue

        if neighbor in list(neighbor_targets_remaining.keys()):
            # paths = paths + [list(set(visited) - set(targets)) + [neighbor]]
            path = [x for x in visited[:] if x not in targets] + [neighbor]
            # print(f"{paths = }, {neighbor = }, {path = }")
            paths[neighbor] = paths[neighbor] + [path]
            distances[neighbor] = distances[neighbor] + [path_weight(graph, visited[:] + [neighbor], "weight")]
            # paths = paths + [visited + [neighbor]]
            del neighbor_targets_remaining[neighbor]
        for target in neighbor_targets_remaining:
            neighbor_targets_remaining[target] = neighbor_targets_remaining[target] - spent_weight
        _bfs_paths_many_targets(
            graph,
            neighbor,
            targets=targets,
            targets_remaining=neighbor_targets_remaining,
            visited=visited + [neighbor],
            distance_termination=distance_termination,
            distance_matrix=distance_matrix,
            start_time=start_time,
            paths=paths,
            distances=distances
        )
    return paths, distances




Backup of _bfs_paths_many_targets before changing remaining destination copying and deletion....
def _bfs_paths_many_targets(graph, source, targets=[], targets_remaining=None, visited=[],
                            distance_termination="geometric", distance_matrix=None, start_time=0, paths=None,
                            distances=None, current_weight=0):
    # print (f"{visited}\t{source}")
    # print (f"{visited}\t{source}\t{current_weight}")
    if time.time() - start_time >= 10:
        raise TimeoutError("_bfs_paths_many_targets timed out...")
    for neighbor in list(graph.neighbors(source)):
        if neighbor in visited:
            continue
        # the given graph have 2 neighbors for origins and destinations. if a node has only one neighbor, its a deadend.
        if len(list(graph.neighbors(neighbor))) == 1:
            continue
        neighbor_targets_remaining = targets_remaining.copy()

        # print(f"__________{neighbor = }________________")
        spent_weight = graph.edges[(source, neighbor)]["weight"]
        neighbor_current_weight = current_weight + spent_weight
        # print (f"{source = }\t{neighbor = }{spent_weight = }")
        for target in list(neighbor_targets_remaining.keys()):
            if neighbor_current_weight > neighbor_targets_remaining[target]:
                del neighbor_targets_remaining[target]
                continue
            if distance_termination == "geometric":
                point_distance = distance_matrix[neighbor][target]
                if (point_distance + neighbor_current_weight) > neighbor_targets_remaining[target]:
                    del neighbor_targets_remaining[target]
                    continue
            elif distance_termination == "network":
                if neighbor in distance_matrix[target]:
                    point_distance = distance_matrix[target][neighbor]
                    if (point_distance + neighbor_current_weight) > neighbor_targets_remaining[target]:
                        del neighbor_targets_remaining[target]
                        continue
                else: # if its not in the distance matrix, its outside the od_ellipse_scope and can't be reached by the current path.
                    del neighbor_targets_remaining[target]
                    continue
        if len(neighbor_targets_remaining) == 0:
            continue

        if neighbor in neighbor_targets_remaining:
            path = [x for x in visited if x not in targets] + [neighbor]
            paths[neighbor] = paths[neighbor] + [path]
            distances[neighbor] = distances[neighbor] + [neighbor_current_weight]
            del neighbor_targets_remaining[neighbor]

        _bfs_paths_many_targets(
            graph,
            neighbor,
            targets=targets,
            targets_remaining=neighbor_targets_remaining,
            visited=visited + [neighbor],
            distance_termination=distance_termination,
            distance_matrix=distance_matrix,
            start_time=start_time,
            paths=paths,
            distances=distances,
            current_weight=neighbor_current_weight
        )
    return paths, distances








'''


def _bfs_paths_many_targets(graph, source, targets=None, targets_remaining=[], visited=[],
                            distance_termination="geometric", distance_matrix=None, start_time=0, paths=None,
                            distances=None, current_weight=0):
    # TODO: rearrnging checks will likely yield a better execution time
    # print (f"{visited}\t{source}")
    # print (f"{visited}\t{source}\t{current_weight}")
    if time.time() - start_time >= 30:
        raise TimeoutError("_bfs_paths_many_targets timed out...")
    for neighbor in list(graph.neighbors(source)):
        if neighbor in visited:
            continue
        # the given graph have 2 neighbors for origins and destinations. if a node has only one neighbor, its a deadend.
        if len(list(graph.neighbors(neighbor))) == 1:
            continue

        # print(f"__________{neighbor = }________________")
        spent_weight = graph.edges[(source, neighbor)]["weight"]
        neighbor_current_weight = current_weight + spent_weight
        neighbor_targets_remaining = []
        # print (f"{source = }\t{neighbor = }{spent_weight = }")
        for target in targets_remaining:
            if neighbor_current_weight <= targets[target]:
                if distance_termination == "geometric":
                    point_distance = distance_matrix[target][neighbor]
                    if (point_distance + neighbor_current_weight) <= targets[target]:
                        neighbor_targets_remaining.append(target)
                elif distance_termination == "network":
                    if neighbor in distance_matrix[target]:
                        point_distance = distance_matrix[target][neighbor]
                        if (point_distance + neighbor_current_weight) <= targets[target]:
                            neighbor_targets_remaining.append(target)
                else:
                    neighbor_targets_remaining.append(target)

        if len(neighbor_targets_remaining) == 0:
            continue

        if neighbor in neighbor_targets_remaining:
            path = [x for x in visited if x not in targets] + [neighbor]
            paths[neighbor] = paths[neighbor] + [path]
            distances[neighbor] = distances[neighbor] + [neighbor_current_weight]
            neighbor_targets_remaining.remove(neighbor)

        _bfs_paths_many_targets(
            graph,
            neighbor,
            targets=targets,
            targets_remaining=neighbor_targets_remaining,
            visited=visited + [neighbor],
            distance_termination=distance_termination,
            distance_matrix=distance_matrix,
            start_time=start_time,
            paths=paths,
            distances=distances,
            current_weight=neighbor_current_weight
        )
    return paths, distances


def get_od_paths(self, o_idx, search_radius, detour_ratio, output_map=False, algorithm="shortest_simple_paths",
                 graph_method="o_light", trim=False, distance_termination="network", batching=True, result="paths"):
    start = time.time()
    paths = {}
    distances = {}
    graph = nx.Graph()
    graph_construction_time = 0
    d_idxs = destinations_accessible_from_origin(self, o_idx, search_radius=search_radius, light_graph=True)
    # print(f"it took {time.time() - start} to identify destinations... ")
    if len(d_idxs) > 0:

        if graph_method == "o_light":
            if distance_termination in ["network", "none"]:
                graph, distance_matrix = get_od_subgraph(self, o_idx, d_idxs, search_radius, detour_ratio,
                                                         output_map=output_map, trim=trim,
                                                         distance_method="network")
            else:
                graph = get_od_subgraph(self, o_idx, d_idxs, search_radius, detour_ratio, output_map=output_map,
                                        trim=trim,
                                        distance_method="geometric")
                distance_matrix = None
        elif graph_method == "od_light":
            graphs = []
            distance_matrices = []
            for d_idx in d_idxs.keys():
                if distance_termination in ["network", "none"]:
                    graph, distance_matrix = get_od_subgraph(self, o_idx, d_idx, search_radius, detour_ratio,
                                                             shortest_distance=d_idxs[d_idx], output_map=output_map,
                                                             trim=trim,
                                                             distance_method="network")
                    distance_matrices.append(distance_matrix)
                else:
                    graph = get_od_subgraph(self, o_idx, d_idx, search_radius, detour_ratio,
                                            shortest_distance=d_idxs[d_idx], output_map=output_map, trim=trim,
                                            distance_method="geometric")
                    distance_matrices = None
                graphs.append(graph)
        elif graph_method == "full":
            if trim:
                raise ValueError(
                    f"parameter 'trim' in function 'get_od_paths' can't be 'True' when parameter 'graph_method' is 'full'.")
            graph = self.G.copy()
            update_light_graph(
                self,
                graph=graph,
                add_nodes=[o_idx] + list(d_idxs.keys())
            )
            distance_matrix = None
        else:
            raise ValueError(
                f"parameter 'graph_method' in function 'get_od_paths' can be ['full', 'od_light', 'o_light'], '{graph_method}' was given")

        graph_construction_time = time.time() - start

        if algorithm == "shortest_simple_paths":
            paths = {}
            distances = {}
            if batching:
                raise ValueError(
                    f"parameter 'batching' in function 'get_od_paths' can't be 'True' when parameter 'algorithm' is 'shortest_simple_paths', {batching} was given.")
            if distance_termination != "none":
                raise ValueError(
                    f"parameter 'distance_termination' in function 'get_od_paths' can only be 'none' when parameter 'algorithm' is 'shortest_simple_paths', {distance_termination} was given.")

            for seq, d_idx in enumerate(d_idxs.keys()):
                if graph_method in ["o_light", 'full']:
                    use_graph = graph
                elif graph_method == "od_light":
                    use_graph = graphs[seq]
                d_paths = nx.shortest_simple_paths(
                    use_graph,
                    source=int(o_idx),
                    target=int(d_idx),
                    weight="weight",  # TODO: use this to reflect a weight paramnetewr or default to distance
                )
                bad_paths = 0
                good_paths = []
                good_paths_weight = []
                for path in d_paths:
                    this_path_weight = path_weight(use_graph, path, "weight")
                    if this_path_weight > d_idxs[d_idx] * detour_ratio:
                        break
                    good_paths.append(path)
                    good_paths_weight.append(this_path_weight)
                paths[d_idx] = good_paths
                distances[d_idx] = good_paths_weight
        elif algorithm == "all_simple_paths":
            paths = {}
            distances = {}
            if batching:
                raise ValueError(
                    f"parameter 'batching' in function 'get_od_paths' can't be 'True' when parameter 'algorithm' is 'all_simple_paths'.")
            if distance_termination != "none":
                raise ValueError(
                    f"parameter 'distance_termination' in function 'get_od_paths' can only be 'none' when parameter 'algorithm' is 'all_simple_paths', {distance_termination} was given.")

            for seq, d_idx in enumerate(d_idxs.keys()):
                if graph_method in ['full', "o_light"]:
                    raise ValueError(
                        f"parameter 'graph_method' in function 'get_od_paths' can only be 'od_light' when parameter 'algorithm' is 'all_simple_paths'.")
                elif graph_method == "o_light":
                    use_graph = graph
                elif graph_method == "od_light":
                    use_graph = graphs[seq]
                d_paths = nx.all_simple_paths(use_graph, source=o_idx, target=d_idx)
                bad_paths = 0
                good_paths = []
                good_paths_weight = []
                for path in d_paths:
                    this_path_weight = path_weight(use_graph, path, "weight")
                    if this_path_weight > d_idxs[d_idx] * detour_ratio:
                        bad_paths += 1
                        continue
                    good_paths.append(path)
                    good_paths_weight.append(this_path_weight)
                paths[d_idx] = good_paths
                distances[d_idx] = good_paths_weight
        elif algorithm == "bfs":
            if (graph_method == "full") and (distance_termination in ["geometric", "network"]):
                raise ValueError(
                    f"parameter 'distance_termination' in function 'get_od_paths' can't be 'geometric' when parameter 'graph_method' is 'full' and 'algorithm' is 'bfs , {graph_method} was given.")
            if (graph_method == "full") and (not batching):
                raise ValueError(
                    f"parameter 'batching' in function 'get_od_paths' should be 'True' when parameter 'graph_method' is 'full' and 'algorithm' is 'bfs , {batching} was given.")

            d_allowed_distances = {}
            for d_idx in d_idxs.keys():
                d_allowed_distances[d_idx] = d_idxs[d_idx] * detour_ratio

            if batching:
                if graph_method == "od_light":
                    raise ValueError(
                        f"parameter 'graph_method' in function 'get_od_paths' can't be 'od_light' when parameter 'batching' is 'True', {graph_method} was given.")

                use_graph = graph
                paths, distances = bfs_paths(self, use_graph, o_idx, d_allowed_distances,
                                             distance_termination=distance_termination, batching=batching, distance_matrix=distance_matrix)
            else:
                paths = {}
                distances = {}
                for seq, d_idx in enumerate(d_idxs.keys()):
                    if graph_method in ["o_light", 'full']:
                        use_graph = graph
                        use_distance_matrix = distance_matrix
                    elif graph_method == "od_light":
                        use_graph = graphs[seq]
                        if distance_termination == "geometric":
                            use_distance_matrix = None
                        else:
                            use_distance_matrix = distance_matrices[seq]
                    d_paths, d_distances = bfs_paths(self, use_graph, o_idx, d_idx, d_idxs[d_idx] * detour_ratio,
                                                     distance_termination=distance_termination,
                                                     other_targets=d_idxs.keys(),
                                                     distance_matrix=use_distance_matrix)
                    paths[d_idx] = d_paths
                    distances[d_idx] = d_distances
        else:
            raise ValueError(
                f"parameter 'algorithm' in function 'get_od_paths' can be ['shortest_simple_paths', 'all_simple_paths', 'bfs'], '{algorithm}' was given")

    if result == "paths":
        return paths, distances
    elif result == "diagnostics":
        return paths, distances, graph, {
            "o_idx": o_idx,
            "time": time.time() - start,
            "graph_construction_time": graph_construction_time,
            "path_generation_time": (time.time() - start) - graph_construction_time,
            "path_count": sum([len(d_paths) for d_paths in list(paths.values())]),
            "total_path_nodes": sum(
                [sum([len(path) for path in paths]) for paths in [paths[index] for index in paths]]),
            "n_nodes": len(graph.nodes),
            "n_edges": len(graph.edges),
            "destination_count": len(d_idxs),
            "average_distance": 0 if len(d_idxs) == 0 else sum(list(d_idxs.values())) / len(d_idxs),
            "algorithm": algorithm,
            "graph_method": graph_method,
            "trim": trim,
            "distance_termination": distance_termination,
            "batching": batching

        }
    else:
        raise ValueError(
            f"parameter 'result' in function 'get_od_paths' can be ['paths', 'diagnostics'], '{result}' was given")


def parallel_betweenness_2(self,
                           search_radius=1000,
                           detour_ratio=1.05,
                           decay=True,
                           decay_method="exponent",
                           beta=0.003,
                           num_cores=4,
                           path_detour_penalty="equal",
                           origin_weights=False,
                           closest_destination=True,
                           destination_weights=False,
                           perceived_distance=False,
                           light_graph=True,
                           ):
    node_gdf = self.layers['network_nodes']['gdf']
    edge_gdf = self.layers['network_edges']['gdf']

    origins = node_gdf[
        (node_gdf["type"] == "origin")
    ]

    # create a new column in edges with an initial 0 betweenness.
    # results should be aggregated  into this column
    edge_gdf['betweenness'] = 0.0

    if num_cores == 1:
        batch_results = one_betweenness_2(
            self,
            search_radius=search_radius,
            origins=origins,
            detour_ratio=detour_ratio,
            decay=decay,
            decay_method=decay_method,
            beta=beta,
            path_detour_penalty=path_detour_penalty,
            origin_weights=origin_weights,
            closest_destination=closest_destination,
            destination_weights=destination_weights,
            perceived_distance=perceived_distance,
            light_graph=light_graph
        )
        batch_results = [batch_results]
    else:
        # Parallel ###
        batch_results = []
        diagnostics_gdfs = []
        num_procs = num_cores  # psutil.cpu_count(logical=logical)

        origins = origins.sample(frac=1)
        # TODO: this is a temporary check, need to see why indices are becoming floats
        origins.index = origins.index.astype("int")
        splitted_origins = np.array_split(origins, num_procs)
        # print("done with filtering, randomizing and splitting: " + str(time.time() - timer))
        start = time.time()
        with concurrent.futures.ProcessPoolExecutor(max_workers=num_procs) as executor:
            # print("entered parallel execution")
            execution_results = [
                executor.submit(
                    one_betweenness_2,
                    self,
                    origins=df,
                    search_radius=search_radius,
                    detour_ratio=detour_ratio,
                    decay=decay,
                    decay_method=decay_method,
                    beta=beta,
                    path_detour_penalty=path_detour_penalty,
                    origin_weights=origin_weights,
                    closest_destination=closest_destination,
                    destination_weights=destination_weights,
                    perceived_distance=perceived_distance,
                    light_graph=light_graph
                ) for df in splitted_origins]
            for result in concurrent.futures.as_completed(execution_results):
                try:
                    batch_result = result.result()
                    batch_results.append(batch_result)
                except Exception as ex:
                    print(str(ex))
                    pass
        end = time.time()
        print("-------------------------------------------")
        print(f"All cores done in {round(end - start, 2)}")
        print("-------------------------------------------")

    for idx in edge_gdf.index:
        edge_gdf.at[idx, "betweenness"] = sum([batch[idx] for batch in batch_results])

    # not sure if this assignment is necessary,
    self.layers['network_nodes']['gdf'] = node_gdf
    self.layers['network_edges']['gdf'] = edge_gdf
    return edge_gdf


def update_light_graph(self, graph, add_nodes=[], remove_nodes=[]):
    #print(f"{str(graph)}\t\t{add_nodes = }\t{remove_nodes = }")
    timer = time.time()
    # print(f"{timer = }")
    node_gdf = self.layers["network_nodes"]["gdf"]
    edge_gdf = self.layers["network_edges"]["gdf"]
    search_timer = 0
    regular_edge_timer = 0
    neighbor_edge_timer = 0
    if "added_nodes" not in graph.graph:
        graph.graph["added_nodes"] = []

    if len(add_nodes) > 0:
        existing_nodes = graph.graph["added_nodes"]
        for node_idx in add_nodes:
            if node_idx in existing_nodes:
                print(f'node ({node_idx}) is already added...{add_nodes = }\t{existing_nodes = }')
                print(graph)
                add_nodes.remove(node_idx)
            else:
                graph.graph["added_nodes"].append(node_idx)
        edge_nodes = {}
        for key, value in node_gdf.loc[graph.graph["added_nodes"]].groupby("nearest_street_id"):
            edge_nodes[int(key)] = list(value.index)
        search_timer += time.time() - timer
        timer = time.time()
        for edge_id in edge_nodes:
            #print(f"{edge_id = }\t{edge_nodes = }")
            neighbors = edge_nodes[edge_id]
            insert_neighbors = set(add_nodes).intersection(set(neighbors))
            existing_neighbors = set(neighbors) - insert_neighbors
            if len(insert_neighbors) == 0:
                continue
            if len(neighbors) == 1:
                # print(f"{add_nodes = }\n"
                #      f"{existing_nodes = }\n"
                #      f"{edge_nodes = }\n"
                #      f"{edge_id = }\n"
                #      f"{neighbors = }")
                node_idx = neighbors[0]
                # print(f"{}\t{}\t{}\t{}")
                left_edge = node_gdf.at[node_idx, "nearest_street_node_distance"]["left"]
                right_edge = node_gdf.at[node_idx, "nearest_street_node_distance"]["right"]
                graph.add_edge(
                    int(left_edge["node_id"]),
                    int(node_idx),
                    weight=left_edge["weight"],
                    id=edge_id
                )
                graph.add_edge(
                    int(node_idx),
                    int(right_edge["node_id"]),
                    weight=right_edge["weight"],
                    id=edge_id
                )
                graph.remove_edge(int(left_edge["node_id"]), int(right_edge["node_id"]))
                regular_edge_timer += time.time() - timer
                timer = time.time()
            else:
                # start a chain addition of neighbors, starting from the 'left',
                # so, need to sort based on distance from left
                segment_weight = edge_gdf.at[edge_id, "weight"]

                chain_start = node_gdf.at[neighbors[0], "nearest_street_node_distance"]["left"]["node_id"]
                chain_end = node_gdf.at[neighbors[0], "nearest_street_node_distance"]["right"]["node_id"]

                chain_distances = [node_gdf.at[node, "nearest_street_node_distance"]["left"]["weight"] for node in
                                   neighbors]

                if len(existing_neighbors) == 0:  # if there are no existing neighbors, remove the original edge
                    graph.remove_edge(int(chain_start), int(chain_end))
                else:  # if there are existing neighbors, remove them. This would also remove their associated edges
                    for node in existing_neighbors:
                        graph.remove_node(node)

                chain_nodes = [chain_start] + neighbors + [chain_end]
                chain_distances = [0] + chain_distances + [segment_weight]

                chain_nodes = np.array(chain_nodes)
                chain_distances = np.array(chain_distances)
                sorting_index = np.argsort(chain_distances)

                # np arrays allows calling them by a list index
                chain_nodes = chain_nodes[sorting_index]
                chain_distances = chain_distances[sorting_index]
                #print(f"{chain_nodes = }\t{chain_distances}")
                accumilated_weight = 0
                for seq in range(len(chain_nodes) - 1):
                    graph.add_edge(
                        int(chain_nodes[seq]),
                        int(chain_nodes[seq + 1]),
                        # TODO: change this to either defaults to distance or a specified column for a weight..
                        weight=chain_distances[seq + 1] - chain_distances[seq],
                        id=edge_id
                    )
                    accumilated_weight += (chain_distances[seq + 1] - chain_distances[seq])
                neighbor_edge_timer += time.time() - timer
                timer = time.time()
    for node_idx in remove_nodes:
        node_idx = int(node_idx)
        if node_idx not in graph.nodes:
            print(f"attempting to remove node {node_idx} that's not in graph {str(graph)}")
            continue

        if len(graph.adj[node_idx]) != 2:
            print(f"attempting to remove a node {node_idx = } that's not degree 2, adjacent to: {graph.adj[node_idx]}")
            continue

        neighbors = list(graph.adj[node_idx])

        start = int(neighbors[0])
        end = int(neighbors[1])
        weight = graph.adj[node_idx][start]["weight"] \
                 + graph.adj[node_idx][end]["weight"]

        original_edge_id = node_gdf.at[node_idx, "nearest_street_id"]

        # remove node after we got the attributes we needed..
        graph.remove_node(node_idx)
        graph.graph["added_nodes"].remove(node_idx)
        #print(f"{neighbors = }\t{start = }\t{end = }\t{weight = }\t{original_edge_id = }")
        # re-stitching the graph
        graph.add_edge(
            start,
            end,
            weight=weight,
            id=original_edge_id
        )
    # print (f"{search_timer = }\t{regular_edge_timer = }\t{neighbor_edge_timer = }")
    return graph


def destinations_accessible_from_origin(self, origin_idx, search_radius=800, light_graph=True):
    # graph = self.G.copy()
    node_gdf = self.layers['network_nodes']['gdf']
    # if "accissible_destinations" not in node_gdf.columns:
    #    node_gdf["accissible_destinations"] = None
    destinations = node_gdf[node_gdf["type"] == "destination"]
    destination_ids = list(destinations.index)
    # print(f"updating light graph, adding {[origin_idx] + destination_ids}")
    # if light_graph:
    #    update_light_graph(
    #        self,
    #        graph=graph,
    #        add_nodes=[origin_idx] + destination_ids
    #    )
    graph = self.d_graph
    update_light_graph(
        self,
        graph=graph,
        add_nodes=[origin_idx]
    )
    length, path = nx.single_source_dijkstra(
        graph,
        origin_idx,
        target=None,
        cutoff=search_radius,
        weight='weight'
    )
    update_light_graph(
        self,
        graph=graph,
        remove_nodes=[origin_idx]
    )
    accessible_dest = {}
    for dest_idx in length:
        if dest_idx in destination_ids:
            accessible_dest[dest_idx] = length[dest_idx]
    #        # print(f"{dest_idx}\t{length[dest_idx]}\t{path[dest_idx]}")
    # node_gdf.at[origin_idx, 'destination_access'] = len(accessible_dest)
    # node_gdf.at[origin_idx, 'accissible_destinations'] = accessible_dest
    return accessible_dest


def one_betweenness_2(
        self,
        search_radius=1000,
        origins=None,
        detour_ratio=1.05,
        decay=True,
        beta=0.003,
        decay_method="exponent",
        path_detour_penalty="equal",  # "exponent" | "power"
        origin_weights=False,
        closest_destination=False,
        destination_weights=False,
        perceived_distance=False,
        light_graph=True
):
    edge_gdf = self.layers["network_edges"]["gdf"]
    node_gdf = self.layers["network_nodes"]["gdf"]

    # graph = self.G.copy()

    batch_betweenness_tracker = {edge_id: 0 for edge_id in list(edge_gdf.index)}

    counter = 0
    for origin_idx in tqdm(origins.index):
        counter += 1
        # if counter % 10 == 0:
        # print(f'core {origins.iloc[0].name}\t{counter = }, progress = {counter / len(origins.index) * 100:5.2f}')

        # TODO: destination IDs and distances can be retrieved from paths, distances instead of re-calculating them
        od_sum_gravities = 0
        if closest_destination:
            destination_ids = [int(origins.at[origin_idx, "closest_destination"])]
        else:
            accissible_destinations = destinations_accessible_from_origin(self, origin_idx, search_radius=search_radius)
            for accissible_destination_idx, od_shortest_distance in accissible_destinations.items():
                od_gravity = 1 / pow(math.e, (beta * od_shortest_distance))
                if destination_weights:
                    od_gravity *= node_gdf.at[accissible_destination_idx, "weight"]
                od_sum_gravities += od_gravity
            destination_ids = list(accissible_destinations.keys())
        destinations = node_gdf.loc[destination_ids]
        '''
        if light_graph:
            graph = self.G.copy()
            update_light_graph(
                self,
                graph=graph,
                add_nodes=[origin_idx] + destination_ids
            )
        '''

        ###
        paths, weights = get_od_paths(self, origin_idx, search_radius, detour_ratio, output_map=False, algorithm="bfs",
                                      graph_method="o_light", trim=True, distance_termination="network", batching=True,
                                      result="paths")
        destination_count = len(list(paths.keys()))

        if destination_count == 0:
            # print(f"o:{origin_idx} has no destinations")
            continue
        ###
        for destination_idx in destination_ids:  # destinations.index:
            try:
                destination_count += 1

                this_od_paths = {
                    "path_length": [],
                    "path_edges": [],
                    "one_over_squared_length": [],
                    'one_over_e_beta_length': []
                }
                '''
                shortest_path_distance = node_gdf.at[origin_idx, "accissible_destinations"][destination_idx]
                graph = get_od_subgraph(self, origin_idx, destination_idx, search_radius, detour_ratio, output_map=False,
                                        trim=False)
                paths = bfs_paths(self, graph, origin_idx, destination_idx, shortest_path_distance * detour_ratio,
                                  geometric_distance_termination=True)
                '''
                '''
                # paths = nx.all_simple_paths(graph, source=origin_idx, target=destination_idx)
                if len(graph.nodes) < 100:
                    paths = bfs_paths(graph, origin_idx, destination_idx, shortest_path_distance * detour_ratio)
                else:
                    paths = nx.shortest_simple_paths(
                        graph,
                        source=int(origin_idx),
                        target=int(destination_idx),
                        weight="weight",  # TODO: use this to reflect a weight paramnetewr or default to distance
                    )
                '''

                path_count = len(paths[destination_idx])
                # shortest_path_distance = 0
                if path_count == 0:
                    print(f"o:{origin_idx}\td:{destination_idx} have no paths...?")
                    continue
                shortest_path_distance = min(weights[destination_idx])
                for path, this_path_weight in zip(paths[destination_idx], weights[destination_idx]):
                    inner_path_edges = list(nx.utils.pairwise(path[1:-1]))
                    inner_edge_ids = [self.G.edges[edge]["id"] for edge in inner_path_edges]
                    edge_ids = [node_gdf.at[path[0], 'nearest_street_id']] + inner_edge_ids + [
                        node_gdf.at[path[-1], 'nearest_street_id']]
                    # this_path_weight = path_weight(graph, path, weight="weight")

                    if this_path_weight > shortest_path_distance * detour_ratio:
                        print(
                            f"o: {origin_idx}\td:{destination_idx}\t{path}\{this_path_weight}\t exceeded limit {shortest_path_distance * detour_ratio}")
                    '''
                    if path_count == 0:  # The shortest path.
                        shortest_path_distance = this_path_weight
    
                    if this_path_weight > shortest_path_distance * detour_ratio:
                        break
                    
    
                    in_path_edges = list(nx.utils.pairwise(path))
                    edge_ids = [graph.edges[edge]["id"] for edge in in_path_edges]
                    # TODO: we're removing duplicates here, check to see if there a deeper issue
                    edge_ids = list(set(edge_ids))
    
                    path_count = path_count + 1
                    '''
                    if this_path_weight < 1:
                        # to solve issue with short paths, or if o/d were on the same location.
                        this_path_weight = 1
                    this_od_paths["path_length"].append(this_path_weight)
                    this_od_paths["one_over_squared_length"].append(1 / (this_path_weight ** 2))
                    this_od_paths["one_over_e_beta_length"].append(1 / pow(math.e, (beta * this_path_weight)))
                    this_od_paths["path_edges"].append(edge_ids)
                # TODO:  change of name to match the attribute. gotta fix later..
                if decay_method == "exponent":
                    decay_method = "one_over_e_beta_length"
                if decay_method == "power":
                    decay_method = "one_over_squared_length"
                sum_one_over_squared_length = 0

                if path_detour_penalty == "exponent":
                    path_detour_penalty = "one_over_e_beta_length"
                    sum_one_over_squared_length = sum(this_od_paths[path_detour_penalty])
                if path_detour_penalty == "power":
                    path_detour_penalty = "one_over_squared_length"
                    sum_one_over_squared_length = sum(this_od_paths[path_detour_penalty])

                this_od_paths["probability"] = []

                # for one_over_squared_length in this_od_paths[path_detour_penalty]:
                for seq, path_edges in enumerate(this_od_paths["path_edges"]):

                    if path_detour_penalty == "equal":
                        this_od_paths["probability"].append(
                            1 / len(this_od_paths["path_edges"])
                        )
                    else:
                        this_od_paths["probability"].append(
                            this_od_paths[path_detour_penalty][seq] / sum_one_over_squared_length
                        )

                for seq, path_probability in enumerate(this_od_paths["probability"]):
                    betweennes_contribution = path_probability
                    if origin_weights:
                        betweennes_contribution *= origins.at[origin_idx, "weight"]
                    if decay:
                        betweennes_contribution *= this_od_paths[decay_method][0]
                    if not closest_destination:
                        this_d_gravity = 1 / pow(math.e, (beta * shortest_path_distance))
                        if destination_weights:
                            this_d_gravity *= destinations.at[destination_idx, "weight"]
                        trip_probability = this_d_gravity / od_sum_gravities
                        betweennes_contribution *= trip_probability

                    for edge_id in this_od_paths["path_edges"][seq]:
                        batch_betweenness_tracker[edge_id] += betweennes_contribution
            except Exception as e:

                print(f"................o: {origin_idx}\td: {destination_idx} faced an error........")
                print(path)
                print(e.__doc__)

    print(f"core {origins.iloc[0].name} done.")
    return batch_betweenness_tracker


def empty_network_dicts():
    node_dict = {
        "id": [],
        "geometry": [],
        "source_layer": [],
        "source_id": [],
        "type": [],
        "weight": [],
        "degree": [],
        "nearest_street_id": [],
        "nearest_street_node_distance": [],
    }
    edge_dict = {
        "id": [],
        "start": [],
        "end": [],
        "length": [],
        "weight": [],
        "type": [],
        "geometry": [],
        "parent_street_id": []
    }
    return node_dict, edge_dict


def flatten_multi_edge_segments(source_gdf):
    ls_dict = {"source_index": [], "geometry": []}

    for source_idx in source_gdf.index:
        geometry = source_gdf.at[source_idx, "geometry"]
        geom_type = geometry.geom_type
        if geom_type == 'LineString':
            ls_dict["source_index"].append(source_idx)
            ls_dict["geometry"].append(geometry)
        elif geom_type == 'MultiLineString':
            for ls in list(geometry):
                ls_dict["source_index"].append(source_idx)
                ls_dict["geometry"].append(ls)
        else:
            raise TypeError(f"geometries could only be of types 'LineString' and 'MultiLineString', {geom_type} is "
                            f"not valid")

    good_lines = {"source_index": [], "geometry": []}
    for geometry_seq, geometry in enumerate(ls_dict["geometry"]):
        number_of_nodes = len(geometry.coords)
        if number_of_nodes != 2:
            # bad segment, segmenting to individual segments
            for i in range(number_of_nodes - 1):
                # create new segment
                start_point = geo.Point(geometry.coords[i])
                end_point = geo.Point(geometry.coords[i + 1])
                line_segment = geo.LineString([start_point, end_point])
                good_lines["source_index"].append(ls_dict["source_index"][geometry_seq])
                good_lines["geometry"].append(line_segment)
        else:
            # good segment, add as is
            good_lines["source_index"].append(ls_dict["source_index"][geometry_seq])
            good_lines["geometry"].append(geometry)
    flat = gpd.GeoDataFrame(good_lines, crs=source_gdf["geometry"].crs)
    flat["length"] = flat["geometry"].length
    return flat


# possible methods: sort by segment length, minimize number of node-moves. minimize number of nodes by fusing to highest degree within tolerance
def insert_nodes_edges(
        node_dict=None,
        edge_dict=None,
        gdf=None,
        node_snapping_tolerance=0,
        return_dict=False,
        discard_redundant_edges=False,
        weight_attribute=None,
):
    if gdf is not None:
        if weight_attribute is not None:
            sorted_gdf = gdf.sort_values(weight_attribute, ascending=False)
        else:
            sorted_gdf = gdf.sort_values("length", ascending=False)
        street_geometry = geo.MultiLineString(
            list(sorted_gdf["geometry"])
        )

    # find proper index for new nodes and edges
    if len(node_dict["id"]) == 0:
        new_node_id = 0
    else:
        new_node_id = node_dict["id"][-1] + 1

    if len(edge_dict["id"]) == 0:
        new_edge_id = 0
    else:
        new_edge_id = edge_dict["id"][-1] + 1

    counter = 0
    zero_length_edges = 0
    redundant_edges = 0
    list_len = len(street_geometry)
    # TODO: Change this to use indexes instead of enumirate
    for street_iloc, street in enumerate(list(street_geometry)):
        counter += 1
        if counter % 100 == 0:
            print(f'{counter = }, progress = {counter / list_len * 100:5.2f}')
        # get this segment's nodes, if segment is more than one segment, get only beginning node and end node, trat as a single sigment
        start_point_index = None
        end_point_index = None

        start_point_geometry = None
        end_point_geometry = None

        start_point = geo.Point(street.coords[0])
        end_point = geo.Point(street.coords[-1])
        # if len(street.coords)>2:
        # print (f"street {street_iloc = } had {len(street.coords)} points")

        # FInd nearest points to start and end, reject if greater than tolerance.
        smallest_distance_to_start = 9999999999999999
        for node_seq, node_point in enumerate(node_dict["geometry"]):
            distance = node_point.distance(start_point)
            if smallest_distance_to_start > distance:
                smallest_distance_to_start = distance
                if smallest_distance_to_start <= node_snapping_tolerance:
                    start_point_index = node_dict["id"][node_seq]
                    start_point_geometry = node_dict["geometry"][node_seq]
                if smallest_distance_to_start == 0:
                    break

        smallest_distance_to_end = 9999999999999999
        for node_seq, node_point in enumerate(node_dict["geometry"]):
            distance = node_point.distance(end_point)
            if smallest_distance_to_end > distance:
                smallest_distance_to_end = distance
                if smallest_distance_to_end <= node_snapping_tolerance:
                    end_point_index = node_dict["id"][node_seq]
                    end_point_geometry = node_dict["geometry"][node_seq]
                if smallest_distance_to_end == 0:
                    break

        if discard_redundant_edges:
            if (start_point_index is not None) and (end_point_index is not None):
                # Check and discard if this starts and ends at the same node
                if start_point_index == end_point_index:
                    zero_length_edges += 1
                    continue
                # CHeck and discard if a link already exist between start and end.
                found_redundant = False
                for starts, ends in zip(edge_dict["start"], edge_dict["end"]):
                    if (
                            (starts == start_point_index) and (ends == end_point_index)
                    ) or (
                            (ends == start_point_index) and (starts == end_point_index)
                    ):
                        redundant_edges += 1
                        found_redundant = True
                        break
                if found_redundant:
                    continue

        if start_point_index is not None:
            node_dict["degree"][start_point_index] += 1
        else:
            start_point_index = new_node_id
            new_node_id += 1
            node_dict["id"].append(start_point_index)
            node_dict["geometry"].append(start_point)
            node_dict["source_layer"].append("streets")
            node_dict["source_id"].append(None)
            node_dict["type"].append("street_node")
            node_dict["weight"].append(0)
            node_dict["degree"].append(1)
            node_dict["nearest_street_id"].append(None)
            node_dict["nearest_street_node_distance"].append(None)

        if end_point_index is not None:
            node_dict["degree"][end_point_index] += 1
        else:
            end_point_index = new_node_id
            new_node_id += 1

            node_dict["id"].append(end_point_index)
            node_dict["geometry"].append(end_point)
            node_dict["source_layer"].append("streets")
            node_dict["source_id"].append(None)
            node_dict["type"].append("street_node")
            node_dict["weight"].append(0)
            node_dict["degree"].append(1)
            node_dict["nearest_street_id"].append(None)
            node_dict["nearest_street_node_distance"].append(None)

        # construct a proper geometry that have the start and end nodes.
        if start_point_geometry:
            use_start = start_point_geometry
        else:
            use_start = start_point

        if end_point_geometry:
            use_end = end_point_geometry
        else:
            use_end = end_point
        new_street = geo.linestring.LineString([
            use_start,
            use_end
        ])

        # Need to add a segment
        edge_dict["id"].append(new_edge_id)
        edge_dict["start"].append(start_point_index)
        edge_dict["end"].append(end_point_index)
        edge_dict["length"].append(street.length)
        if weight_attribute is None:
            edge_dict["weight"].append(street.length)
        else:
            edge_dict["weight"].append(sorted_gdf.iloc[street_iloc][weight_attribute])
        edge_dict["type"].append("street")
        # edge_dict["geometry"].append(new_street)
        edge_dict["geometry"].append(street)
        edge_dict["parent_street_id"].append(sorted_gdf.iloc[street_iloc].name)
        new_edge_id += 1
    if return_dict:
        return node_dict, edge_dict
    else:
        node_gdf = gpd.GeoDataFrame(node_dict, crs=gdf["geometry"].crs).set_index("id")
        edge_gdf = gpd.GeoDataFrame(edge_dict, crs=gdf["geometry"].crs).set_index("id")
        # print(f"Node/Edge table constructed. {zero_length_edges = }, {redundant_edges = }")
    if discard_redundant_edges:
        print(f"redundant edge report: {zero_length_edges = }, {redundant_edges = }")
    return node_gdf, edge_gdf


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


def fuse_degree_2_nodes(node_gdf=None, edge_gdf=None, tolerance_angle=10):
    node_gdf = node_gdf.copy()
    edge_gdf = edge_gdf.copy()
    degree_2_nodes = node_gdf[node_gdf["degree"] == 2]
    fused = 0
    for idx in degree_2_nodes.index:
        fused = fused + 1
        edges = edge_gdf[
            (edge_gdf["start"] == idx) |
            (edge_gdf["end"] == idx)
            ]

        node_list = list(pd.concat([edges["start"], edges["end"]]).unique())
        outer_nodes = [value for value in node_list if value != idx]
        if len(outer_nodes) == 1:
            # these are two parallel lines sharing the same two nodes..
            # They form a 'stray' loop, dropping both is likely a goop option
            # As dropping one would create a stray edge.
            print(f"{outer_nodes = }")
            print(f'{idx = }')
            edge_gdf.drop(index=edges.iloc[0].name, inplace=True)
            edge_gdf.drop(index=edges.iloc[1].name, inplace=True)
            node_gdf.drop(index=idx, inplace=True)
            node_gdf.at[outer_nodes[0], "degree"] -= 2
            # TODO: trace back the source of such 'loops' and see if there is a cause for them
        else:
            if node_gdf.at[idx, "degree"] != 2:
                print(f'{node_gdf.at[idx, "degree"] = }')
                print(f'{idx = }')
                continue

            first_node = node_gdf.loc[outer_nodes[0]]
            last_node = node_gdf.loc[outer_nodes[1]]

            preserved_edge_id = edges.iloc[0].name
            drop_edge_id = edges.iloc[1].name

            start_point = first_node["geometry"]
            end_point = last_node["geometry"]

            line = geo.LineString([start_point, end_point])

            angle = angle_deviation_between_two_lines([start_point, node_gdf.at[idx, "geometry"], end_point])
            # print(f'{angle = }')
            if angle < tolerance_angle:
                edge_gdf.at[preserved_edge_id, "geometry"] = line

                edge_gdf.at[preserved_edge_id, "length"] = edge_gdf.at[preserved_edge_id, "geometry"].length
                edge_gdf.at[preserved_edge_id, "start"] = first_node.name
                edge_gdf.at[preserved_edge_id, "end"] = last_node.name

                # Drop end edge
                edge_gdf.drop(index=drop_edge_id, inplace=True)

                # drop node
                node_gdf.drop(index=idx, inplace=True)
    # print("Fused segments: " + str(fused))
    return node_gdf, edge_gdf


def scan_for_intersections(node_gdf=None, edge_gdf=None, node_snapping_tolerance=1):
    node_gdf = node_gdf.copy()
    edge_gdf = edge_gdf.copy()
    intersections = 0
    for edge_id, edge in edge_gdf.iterrows():
        start_id = edge["start"]
        end_id = edge["end"]
        known_neighbors = edge_gdf[
            (edge_gdf["start"] == start_id) |
            (edge_gdf["end"] == start_id) |
            (edge_gdf["start"] == end_id) |
            (edge_gdf["end"] == end_id)
            ]
        neighbor_ids = list(known_neighbors["id"].unique())

        intersection_filter = edge_gdf["p_geometry"].intersects(edge["p_geometry"])
        # intersection_filter = edge_gdf["p_geometry"].buffer(node_snapping_tolerance/2).intersects(edge["p_geometry"].buffer(node_snapping_tolerance/2))
        intersecting_edges = edge_gdf[intersection_filter]
        intersecting_edge_ids = list(intersecting_edges[~intersecting_edges["id"].isin(neighbor_ids)]['id'])
        if len(intersecting_edge_ids) > 0:
            for intersecting_edge_id in intersecting_edge_ids:
                intersections = intersections + 1
                # find interection point
                intersecting_edge = edge_gdf[edge_gdf["id"] == intersecting_edge_id].iloc[0]
                intersection_point = edge[measurement_geometry].intersection(intersecting_edge[measurement_geometry])
                # intersection_point = edge[measurement_geometry].buffer(node_snapping_tolerance).intersection(intersecting_edge[measurement_geometry])
                print(intersection_point)
                segments = split(
                    geo.MultiLineString([edge[measurement_geometry], intersecting_edge[measurement_geometry]]),
                    intersection_point
                )
                print(segments)
                insert_nodes_edges(self, geometry=segments, node_snapping_tolerance=node_snapping_tolerance)
                edge_gdf.drop(intersecting_edge_id, inplace=True)
            edge_gdf.drop(edge_id, inplace=True)
    print("Solved intersections: " + str(intersections))
    return node_gdf, edge_gdf


def create_street_nodes_edges(self,
                              source_layer="streets",
                              flatten_polylines=True,
                              node_snapping_tolerance=1,
                              fuse_2_degree_edges=True,
                              tolerance_angle=10,
                              solve_intersections=True,
                              loose_edge_trim_tolerance=0.001,
                              weight_attribute=None,
                              discard_redundant_edges=False,
                              ):
    line_geometry_gdf = self.layers[source_layer]["gdf"].copy()
    line_geometry_gdf["length"] = line_geometry_gdf["geometry"].length
    line_geometry_gdf = line_geometry_gdf[line_geometry_gdf["length"] > 0]

    if "network_nodes" not in self.layers:
        node_dict, edge_dict = empty_network_dicts()
    else:
        node_dict = self.layers["network_nodes"]['gdf'].to_dict()
        edge_dict = self.layers["network_edges"]['gdf'].to_dict()

    if flatten_polylines:
        line_geometry_gdf = flatten_multi_edge_segments(line_geometry_gdf)
    node_gdf, edge_gdf = insert_nodes_edges(node_dict=node_dict,
                                            edge_dict=edge_dict,
                                            gdf=line_geometry_gdf,
                                            node_snapping_tolerance=node_snapping_tolerance,
                                            weight_attribute=weight_attribute,
                                            discard_redundant_edges=discard_redundant_edges
                                            )
    if fuse_2_degree_edges:
        node_gdf, edge_gdf = fuse_degree_2_nodes(
            node_gdf=node_gdf,
            edge_gdf=edge_gdf,
            tolerance_angle=tolerance_angle
        )
    if solve_intersections:
        node_gdf, edge_gdf = scan_for_intersections(
            node_gdf=node_gdf,
            edge_gdf=edge_gdf,
            node_snapping_tolerance=1
        )

    self.layers['network_nodes'] = {'gdf': node_gdf, 'show': True}
    self.layers['network_edges'] = {'gdf': edge_gdf, 'show': True}
    self.color_layer('network_nodes')
    self.color_layer('network_edges')
    return
