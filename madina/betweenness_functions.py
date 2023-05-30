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
from heapq import heappush, heappop

import shapely
from shapely import affinity
from shapely.ops import unary_union

# cURRENT_BEST



def best_path_generator_so_far(self, o_idx, search_radius=800, detour_ratio=1.15, turn_penalty=False):

    graph = self.d_graph
    update_light_graph(self, graph, add_nodes=[o_idx])
    o_graph, d_idxs, distance_matrix, scope_nodes = get_od_subgraph_2(
        self,
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
    paths, distances = _bfs_paths_many_targets_iterative(self,
                                                         graph,
                                                         o_idx,
                                                         d_allowed_distances,
                                                         distance_matrix=distance_matrix,
                                                         turn_penalty=turn_penalty,
                                                         od_scope=scope_nodes,
                                                         )
    


    update_light_graph(self, graph, remove_nodes=[o_idx])
    return paths, distances, d_idxs
'''
def best_path_generator_so_far(self, o_idx, search_radius=800, detour_ratio=1.15, turn_penalty=False):
    #print ("using double sided dijkstra and recursive")
    # print("got in")

    
    #od_scope, distance_matrix, d_idxs = explore_exploit_graph(self, o_idx, search_radius, detour_ratio, turn_penalty=False)
    # print("updated light_graph")
    graph, d_idxs, distance_matrix, scope_nodes = get_od_subgraph_2(
        self,
        o_idx,
        search_radius=search_radius,
        detour_ratio=detour_ratio,
        output_map=False,
        graph_type="double_sided_dijkstra",
        turn_penalty=turn_penalty,
        o_graph=None
    )

    # print("generated d_idxs and scope")
    #print ("subgraph generated")

    graph = self.d_graph
    update_light_graph(self, graph, add_nodes=[o_idx])
    #print ("updated light graph by adding origin")

    d_allowed_distances = {}
    for d_idx in d_idxs.keys():
        d_allowed_distances[d_idx] = d_idxs[d_idx] * detour_ratio
    # print(f"{d_allowed_distances = }")

    paths, distances = bfs_paths(self,
            graph,
            o_idx,
            d_allowed_distances,
            #weight_limit=search_radius * detour_ratio,
            distance_termination="network",
            batching=True,
            #other_targets=[],
            distance_matrix=distance_matrix,
            turn_penalty=turn_penalty
    )
    update_light_graph(self, graph, remove_nodes=[o_idx])
    return paths, distances, d_idxs
'''



# cURRENT_BEST, SHOULD MOVE TO una/madina?
def insert_nodes_v2(self, label, layer_name, weight_attribute):
    start = time.time()
    node_gdf = self.layers["network_nodes"]["gdf"]
    edge_gdf = self.layers["network_edges"]["gdf"]
    source_gdf = self.layers[layer_name]["gdf"]

    node_dict = node_gdf.reset_index().to_dict()

    match = edge_gdf["geometry"].sindex.nearest(source_gdf["geometry"])

    def cut(line, distance):
        # Cuts a line in two at a distance from its starting point
        if distance <= 0.0:
            return [
                geo.LineString([line.coords[0], line.coords[0]]),
                geo.LineString(line)]
        elif distance >= line.length:
            return [
                geo.LineString(line),
                geo.LineString([line.coords[-1], line.coords[-1]])
            ]

        coords = list(line.coords)
        for i, p in enumerate(coords):
            pd = line.project(geo.Point(p))
            if pd == distance:
                return [
                    geo.LineString(coords[:i + 1]),
                    geo.LineString(coords[i:])]
            if pd > distance:
                cp = line.interpolate(distance)
                return [
                    geo.LineString(coords[:i] + [(cp.x, cp.y)]),
                    geo.LineString([(cp.x, cp.y)] + coords[i:])]

    new_node_id = int(node_gdf.index[-1])  # increment before use.

    sum_search_time = 0
    sum_cutting_time = 0

    pre_processing_time = time.time() - start
    for source_iloc, source_id in enumerate(source_gdf.index):
        # TODO: Potential speedup if these operations are victorized outside the loop
        start = time.time()
        source_representative_point = source_gdf.at[source_id, "geometry"].centroid
        # closest_edge_id = edge_gdf["geometry"].distance(source_representative_point).idxmin()
        # closest_edge_id = edge_gdf["geometry"].sindex.nearest(source_representative_point)[1][0]
        closest_edge_id = match[1][source_iloc]
        closest_edge_geometry = edge_gdf.at[closest_edge_id, "geometry"]
        distance_along_closest_edge = closest_edge_geometry.project(source_representative_point)
        point_on_nearest_edge = closest_edge_geometry.interpolate(
            distance_along_closest_edge)  ## gives a point on te street where the source point is projected
        sum_search_time += time.time() - start

        start = time.time()
        try:
            cut_lines = cut(closest_edge_geometry, distance_along_closest_edge)
            start_segment = cut_lines[0]
            end_segment = cut_lines[1]
        except:
            # TODO: test cases where this exception occurs.
            continue
        sum_cutting_time += time.time() - start

        new_node_id += 1
        project_point_id = new_node_id
        node_dict["id"][new_node_id] = new_node_id
        node_dict["geometry"][new_node_id] = point_on_nearest_edge
        node_dict["source_layer"][new_node_id] = layer_name
        node_dict["source_id"][new_node_id] = source_id
        node_dict["type"][new_node_id] = label
        node_dict["degree"][new_node_id] = 0
        node_dict["weight"][new_node_id] = \
            1.0 if weight_attribute is None else source_gdf.at[source_id, weight_attribute]

        left_edge_weight = \
            edge_gdf.at[closest_edge_id, "weight"] * start_segment.length / closest_edge_geometry.length
        right_edge_weight = edge_gdf.at[closest_edge_id, "weight"] - left_edge_weight
        node_dict["nearest_street_id"][new_node_id] = closest_edge_id
        node_dict["nearest_street_node_distance"][new_node_id] = \
            {
                "left":
                    {
                        "node_id": edge_gdf.at[closest_edge_id, "start"],
                        # "distance": start_segment.length,
                        "weight": left_edge_weight,
                        "geometry": start_segment
                    },
                "right":
                    {
                        "node_id": edge_gdf.at[closest_edge_id, "end"],
                        # "distance": closest_edge_geometry.length - start_segment.length,
                        "weight": right_edge_weight,
                        "geometry": end_segment
                    }
            }

    start = time.time()
    node_gdf = gpd.GeoDataFrame(node_dict, crs=self.default_projected_crs)
    node_gdf = node_gdf.set_index("id")
    self.layers['network_nodes']['gdf'] = node_gdf
    self.color_layer('network_nodes')
    post_processing_time = time.time() - start

    # print(
    #    f"{sum_cutting_time = :6.3f}s\t{sum_search_time = :6.3f}s\t{post_processing_time = :6.3f}s\t{pre_processing_time = :6.3f}s")
    return node_gdf

# dEBUGGING fUNCTION/ mOVE TO una?
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
        edge_geometries.append(
            geo.LineString([graph_nodes.at[edge[0], "geometry"], graph_nodes.at[edge[1], "geometry"]]))
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

    layers = []
    if edge_gdf.shape[0] > 0:
        layers.append({"gdf": edge_gdf, "color": [125, 125, 125], "opacity": 0.05})

    if graph_nodes[graph_nodes["type"] == "origin"].shape[0] > 0:
        layers.append({"gdf": graph_nodes[graph_nodes["type"] == "street_node"].reset_index(), "color": [0, 255, 125],
                       "opacity": 0.05, "text": "id"})

    if graph_nodes[graph_nodes["type"] == "origin"].shape[0] > 0:
        layers.append({"gdf": graph_nodes[graph_nodes["type"] == "origin"], "opacity": 0.25, "color": [255, 0, 125]})

    if graph_nodes[graph_nodes["type"] == "destination"].shape[0] > 0:
        layers.append(
            {"gdf": graph_nodes[graph_nodes["type"] == "destination"], "opacity": 0.05, "color": [0, 125, 255],
             "text": "type"})

    if graph_edges.shape[0] > 0:
        layers.append({"gdf": graph_edges, "color": [0, 255, 125], "opacity": 0.10, "text": "weight"})

    self.create_deck_map(
        layers,
        basemap=False,
        save_as="graph.html"
    )

    return edge_gdf, graph_nodes, graph_edges

# dEPRECIATED
def     get_od_subgraph_2(self, o_idx, search_radius=800, detour_ratio=1.15,
                      output_map=False, graph_type="geometric", turn_penalty=False, o_graph=None):
    # print (f"got to get_od_subgraph_2, {o_idx = }\t {search_radius = }\t {graph_type = }")
    if o_graph is None:
        graph = self.d_graph.copy()
        graph.graph["added_nodes"] = self.d_graph.graph["added_nodes"].copy()
        update_light_graph(self, graph, add_nodes=[o_idx])
    else:
        graph = o_graph

    if graph_type == "full":
        pass
    elif graph_type == "geometric":
        scope_nodes = geometric_scope(self, o_idx, search_radius, detour_ratio, turn_penalty=False, batched=True,
                                      o_graph=graph)
    elif graph_type == "network":
        pass
    elif graph_type == "double_sided_dijkstra":
        scope_nodes, distance_matrix, d_idxs = explore_exploit_graph(self, o_idx, search_radius, detour_ratio,
                                                                     turn_penalty=turn_penalty)
    elif graph_type == "trail_blazer":
        # scope_nodes, distance_matrix, d_idxs = bfs_subgraph_generation(self, o_idx, search_radius, detour_ratio,
        #                                                               turn_penalty=turn_penalty, o_graph=graph)
        scope_nodes, distance_matrix, d_idxs = bfs_subgraph_generation(self,
                                                                       o_idx,
                                                                       search_radius=search_radius,
                                                                       detour_ratio=detour_ratio,
                                                                       turn_penalty=turn_penalty,
                                                                       o_graph=graph
                                                                       )

    return graph, d_idxs, distance_matrix, scope_nodes

# dEPRECIATED
def geometric_scope(self, o_idx, search_radius, detour_ratio, turn_penalty=False, batched=True, o_graph=None):
    node_gdf = self.layers["network_nodes"]["gdf"]
    scopes = []
    d_idxs, o_scope, o_scope_paths = turn_o_scope(self, o_idx, search_radius, detour_ratio, turn_penalty=turn_penalty,
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

# dEPRECIATED
def network_scope(self, o_idx, search_radius, detour_ratio, turn_penalty=False, batched=True):
    node_gdf = self.layers["network_nodes"]["gdf"]

    d_idxs, o_scope, o_scope_paths = turn_o_scope(self, o_idx, search_radius, detour_ratio, turn_penalty=turn_penalty)

    graph = self.d_graph
    update_light_graph(self, graph, add_nodes=[o_idx])

    scope_nodes = set()
    distance_matrix = {}
    for d_idx in d_idxs.keys():
        d_network_scope = nx.single_source_dijkstra_path_length(
            graph,
            source=d_idx,
            cutoff=d_idxs[d_idx] * detour_ratio,
            weight="weight"
        )
        d_ellipse_scope = set()
        distance_matrix[d_idx] = {}
        for node in d_network_scope:
            if node in o_scope:
                if (d_network_scope[node] + o_scope[node]) <= (d_idxs[d_idx] * detour_ratio):
                    d_ellipse_scope.add(node)
                    distance_matrix[d_idx][node] = d_network_scope[node]
        scope_nodes = scope_nodes.union(d_ellipse_scope)

    update_light_graph(self, graph, remove_nodes=[o_idx])
    scope_nodes = node_gdf[node_gdf.index.isin(scope_nodes)]
    return scope_nodes

# dEPRECIATED
def scope_to_graph(self, od_scope, o_idx, d_idxs):
    edge_gdf = self.layers["network_edges"]["gdf"]
    node_gdf = self.layers["network_nodes"]["gdf"]

    scope_nodes = node_gdf[node_gdf.intersects(od_scope)]
    node_list = list(scope_nodes.index)

    od_edge_list = [
                       node_gdf.at[o_idx, "nearest_street_id"]
                   ] + list(node_gdf.loc[d_idxs]["nearest_street_id"])
    scope_edges = edge_gdf[
        (edge_gdf["start"].isin(node_list) & edge_gdf["end"].isin(node_list)) |
        edge_gdf.index.isin(od_edge_list)
        ]

    od_edge_nodes = list(
        set(list(scope_edges.loc[od_edge_list]["start"]) + list(scope_edges.loc[od_edge_list]["end"])))

    # This eliminate other destinations that got included from node_gdf, better to filter node_gdf..
    scope_node_ids = list(set(list(scope_edges["start"].unique()) + list(scope_edges["end"].unique())))
    scope_nodes = node_gdf.loc[scope_node_ids]
    '''
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
   '''

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

    update_light_graph(
        self,
        graph=G,
        add_nodes=[o_idx] + list(d_idxs.keys())
    )

    return G

# mOVE TO UNA?
def get_elastic_weight(self, search_radius, detour_ratio, beta, decay=False, turn_penalty=False, retained_d_idxs=None):


    node_gdf = self.layers["network_nodes"]["gdf"]
    origins = node_gdf[node_gdf["type"] == "origin"]

    o_reach = {}
    o_gravity = {}

    for o_idx in origins.index:
        if retained_d_idxs is None:
            d_idxs, _, _ = turn_o_scope(self, o_idx, search_radius, detour_ratio,
                                        turn_penalty=turn_penalty, o_graph=None, return_paths=False)
        else:
            d_idxs = retained_d_idxs[o_idx]
        o_reach[o_idx] = int(len(d_idxs))

        # d_gravity = {}
        # for d_idx  in d_idxs:
        #    d_gravity[d_idx] = 1 / pow(math.e, (beta * d_idxs[d_idx]))
        # o_gravity[o_idx] = sum(d_gravity.values())

        o_gravity[o_idx] = sum(1.0 / pow(math.e, (beta * np.array(list(d_idxs.values())))))

    a = 0.5
    b = 1

    if decay:
        access = o_gravity
    else:
        access = o_reach

    min_access = min(access.values())
    max_access = max(access.values())
    for o_idx in origins.index:
        scaled_access = (b - a) * ((access[o_idx] - min_access) / (max_access - min_access)) + a
        scaled_weight = origins.at[o_idx, "weight"] * scaled_access

        # TODO: This overrides original weights by elastic weights. should think of a way to pass this as options for una algorithms.
        node_gdf.at[o_idx, "elastic_weight"] = scaled_weight
        node_gdf.at[o_idx, "gravity"] = o_gravity[o_idx]
        node_gdf.at[o_idx, "reach"] = o_reach[o_idx]

    return

# mOVE TO UNA?
def get_od_subgraph(self, o_idx, d_idxs=None, search_radius=800, detour_ratio=1.15,
                    shortest_distance=0, output_map=False, trim=False,
                    distance_method="geometric", turn_penalty=False):
    # print (f"{detour_ratio = }")
    # The function "destinations_accessible_from_origin" has already been called
    # Assumption: destination is already reachable from origin
    start = time.time()
    if distance_method not in ["double_sided_dijkstra", "trail_blazer"]:
        if isinstance(d_idxs, int):
            d_idxs = {d_idxs: shortest_distance}

    edge_gdf = self.layers["network_edges"]["gdf"]
    node_gdf = self.layers["network_nodes"]["gdf"]
    counter = 0
    if distance_method == "geometric":
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
        # print(f"{time.time() - start}\tO Single source path length done")
        # o_graph = graph.subgraph(list(o_network_scope.keys()))
        # TODO: using nodes in o_network_scope to construct a subgraph for the inner loop might be better....
        scope_nodes = set()
        distance_matrix = {}
        # d_graph = self.G
        # update_light_graph(self, d_graph, add_nodes=[o_idx])
        for d_idx in d_idxs.keys():
            # update_light_graph(self, d_graph, add_nodes=[d_idx])
            d_network_scope = nx.single_source_dijkstra_path_length(
                graph,
                source=d_idx,
                cutoff=d_idxs[d_idx] * detour_ratio,
                weight="weight"
            )
            # update_light_graph(self, d_graph, remove_nodes=[d_idx])
            d_ellipse_scope = set()
            # print (f"before inner loop {scope_nodes = }\t{d_ellipse_scope = }")
            distance_matrix[d_idx] = {}
            for node in d_network_scope:
                if node in o_network_scope:
                    if (d_network_scope[node] + o_network_scope[node]) <= (d_idxs[d_idx] * detour_ratio):
                        d_ellipse_scope.add(node)
                        distance_matrix[d_idx][node] = d_network_scope[node]
            # print (f"before union {scope_nodes = }\t{d_ellipse_scope = }")
            scope_nodes = scope_nodes.union(d_ellipse_scope)
            # print (f"after union {scope_nodes = }\t{d_ellipse_scope = }")
        # update_light_graph(self, d_graph, remove_nodes=[o_idx])
        update_light_graph(self, graph, remove_nodes=[o_idx])
        scope_nodes = node_gdf[node_gdf.index.isin(scope_nodes)]
        # print(f"{time.time() - start}\tNetwork Scope Done")
    elif distance_method == "double_sided_dijkstra":
        scope_nodes, distance_matrix, d_idxs = explore_exploit_graph(self, o_idx, search_radius, detour_ratio,
                                                                     turn_penalty=turn_penalty)
        scope_nodes = node_gdf[node_gdf.index.isin(scope_nodes)]
        # print(f"{time.time() - start}\tdouble_sided_dijkstra Scope Done")
    elif distance_method == "trail_blazer":
        start = time.time()
        scope_nodes, distance_matrix, d_idxs = bfs_subgraph_generation(self, o_idx, search_radius, detour_ratio,
                                                                       turn_penalty=turn_penalty)
        # print(f"{time.time() - start}\ttrail_blazer Scope Done")
        scope_nodes = node_gdf[node_gdf.index.isin(scope_nodes)]
    else:
        print(f"distance method {distance_method} not implemented. options are: ['geometric', 'network']")
    while True:
        # get edges and construct a network
        node_list = list(scope_nodes.index)
        # print (edge_gdf["start"].isin(node_list))

        od_edge_list = [
                           node_gdf.at[o_idx, "nearest_street_id"]
                       ] + list(node_gdf.loc[d_idxs]["nearest_street_id"])
        scope_edges = edge_gdf[
            (edge_gdf["start"].isin(node_list) & edge_gdf["end"].isin(node_list)) |
            edge_gdf.index.isin(od_edge_list)
            ]

        od_edge_nodes = list(
            set(list(scope_edges.loc[od_edge_list]["start"]) + list(scope_edges.loc[od_edge_list]["end"])))
        # print(f"{od_edge_nodes = }")
        # include od edges..

        # scope_edges = edge_gdf.loc[list(set(list(scope_edges.index) + [node_gdf.at[o_idx, "nearest_street_id"],
        #                                                           node_gdf.at[d_idx, "nearest_street_id"]]))]

        # This eliminate other destinations that got included from node_gdf, better to filter node_gdf..
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
        # print(f"{inner_nodes_degrees[inner_nodes_degrees == 1].shape[0] = }"
        #       f", {inner_nodes_degrees[inner_nodes_degrees == 1] = }")
        if counter >= 10:
            break
        counter += 1
        # break
        # print(f"keep these nodes: {inner_nodes_degrees[inner_nodes_degrees != 1].index}")
        scope_nodes = scope_nodes.loc[list(inner_nodes_degrees[inner_nodes_degrees != 1].index) + od_edge_nodes]
    # print(f"{time.time() - start}\tScope nodes and edges prepared")
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

    # insert o and d as nodes in graph
    # print(f"{time.time() - start}\tGraph street edges added")
    update_light_graph(
        self,
        graph=G,
        add_nodes=[o_idx] + list(d_idxs.keys())
    )
    # print(f"{time.time() - start}\tGraph origins and destinations added")
    # print(f"it took {time.time() - start} to create second light graph ")
    # print(f"small graph\to:{o_idx}\td:{d_idx}\tedge_count:{scope_edges.shape[0]}\ttime:{round(time.time()
    # - start, 2)}")
    if distance_method in ["network"]:
        return G, distance_matrix
    elif distance_method in ["double_sided_dijkstra", "trail_blazer"]:
        return G, distance_matrix, d_idxs
    else:
        return G

# mOVE TO UNA?
def get_o_scope(self, graph, o_idx, search_radius, detour_ratio, get_paths=False):
    # lengths, paths = ___
    if get_paths:
        lengths, paths = nx.single_source_dijkstra(
            graph, source=o_idx,
            cutoff=search_radius * detour_ratio,
            weight="weight"
        )
    else:
        lengths = nx.single_source_dijkstra_path_length(
            graph, source=o_idx,
            cutoff=search_radius * detour_ratio,
            weight="weight"
        )
    ## extract destinations within search radius
    ## extract nodes within max(d_idx)* detour ratio
    node_gdf = self.layers['network_nodes']['gdf']
    destinations = list(node_gdf[node_gdf["type"] == "destination"].index)
    d_idxs = {}
    o_scope = {}
    o_scope_paths = {}
    for node_idx in lengths:
        if (node_idx in destinations) and (lengths[node_idx] <= search_radius):
            d_idxs[node_idx] = lengths[node_idx]

    if len(d_idxs) == 0:
        if get_paths:
            return d_idxs, o_scope, o_scope_paths
        else:
            return d_idxs, o_scope

    max_scope_distance = max(d_idxs.values()) * detour_ratio
    # print (f"{max_scope_distance = }\t{d_idxs = }")
    if get_paths:
        for node_idx in lengths:
            if lengths[node_idx] <= max_scope_distance:
                o_scope[node_idx] = lengths[node_idx]
                o_scope_paths[node_idx] = paths[node_idx]

        return d_idxs, o_scope, o_scope_paths
    else:
        for node_idx in lengths:
            if lengths[node_idx] <= max_scope_distance:
                o_scope[node_idx] = lengths[node_idx]
                # o_scope_lengths_paths[node_idx] = lengths[node_idx]
        return d_idxs, o_scope

# dEORECIATED..


def explore_exploit_graph(self, o_idx, search_radius, detour_ratio, turn_penalty=False):
    od_scope = set()
    distance_matrix = {}

    d_idxs, o_scope, o_scope_paths = turn_o_scope(self, o_idx, search_radius, detour_ratio, turn_penalty=turn_penalty)
    graph = self.d_graph
    update_light_graph(self, graph, add_nodes=[o_idx])
    # d_idxs, o_scope = get_o_scope(self, graph, o_idx, search_radius, detour_ratio)
    if len(d_idxs) == 0:
        update_light_graph(
            self,
            graph=graph,
            remove_nodes=[o_idx]
        )
        return od_scope, distance_matrix, d_idxs

    ## start ellipse search from all nodes.
    ## look at all d_idxs. find thier neighbors, insert into queue.
    # count = 0
    for d_idx in d_idxs:
        best_weight_node_queue = []
        heappush(best_weight_node_queue, (0, d_idx))
        # node_queue = [d_idx]
        # weight_queue = [0]
        d_scope = {d_idx: 0}
        # while len(node_queue) > 0:
        while best_weight_node_queue:
            # print (f"{weight_queue = }")
            # print(f"{d_idx = }\t{node_queue}")
            # print(f"{node_queue = }\n{weight_queue = }\n{d_scope}\n")
            ##convert into heappop
            weight, node = heappop(best_weight_node_queue)
            # node = node_queue.pop(0)
            # weight = weight_queue.pop(0)
            # print (f"{weight = }")
            for neighbor in list(graph.neighbors(node)):
                # print(f"{d_idx = }\t{neighbor = }\t neighbors = {list(graph.neighbors(node))}")
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

                        ### update the queue.................
                        # old_loc = node_queue.index(neighbor)
                        # node_queue.pop(old_loc)
                        # weight_queue.pop(old_loc)
                        ### end queue
                        # update...............
                        # print(f"{neighbor = } found shorter path")
                    else:
                        # print(f"{node = }\t{d_scope = }\tAlready found shorter distance termination")
                        continue
                else:
                    d_scope[neighbor] = (spent_weight + weight)
                    # print(f"{neighbor = } added to scope")

                # launching new search

                ### convert into heappush
                heappush(best_weight_node_queue, (spent_weight + weight, neighbor))
                # insert_loc = bisect(weight_queue, spent_weight+weight)
                # weight_queue.insert(insert_loc, spent_weight+weight)
                # node_queue.insert(insert_loc, neighbor)
        distance_matrix[d_idx] = d_scope
        od_scope = od_scope.union(set(d_scope.keys()))
        # count += 1
        # if count == 10:
        # break
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
    update_light_graph(
        self,
        graph=graph,
        remove_nodes=[o_idx]
    )
    return od_scope, distance_matrix, d_idxs

# cURRENT_BEST
def bfs_subgraph_generation(self, o_idx, search_radius=800, detour_ratio=1.15, turn_penalty=False, o_graph=None):
    # print (f"got into bfs_subgraph_generation, {o_idx = }\t{search_radius = }\t {detour_ratio = }\t {turn_penalty}\t {o_graph = }")

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
        d_idxs, o_scope, o_scope_paths = turn_o_scope(self, o_idx, search_radius, detour_ratio,
                                                      turn_penalty=turn_penalty)
        graph = self.d_graph
        update_light_graph(self, graph, add_nodes=[o_idx])
    else:
        d_idxs, o_scope, o_scope_paths = turn_o_scope(self, o_idx, search_radius, detour_ratio,
                                                      turn_penalty=turn_penalty, o_graph=o_graph)
        graph = o_graph

    # visualize_graph(self, graph)

    if (len(d_idxs) == 0):
        if (o_graph is None):
            update_light_graph(
                self,
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
                '''
                if neighbor not in trailheads:
                    o_scope_paths_to_neighbor = o_scope_paths[neighbor]
                    node_is_new_trailhead += 1
                    if len(o_scope_paths_to_neighbor) > 20 and (d_idx not in o_scope_paths_to_neighbor[:-1]):
                        node_new_trailhead_is_long_enough += 1
                        previous_mode = neighbor
                        accumulated_path_weight = 0
                        th_o_path_weight = []
                        th_o_path = o_scope_paths_to_neighbor[-2::-1]
                        early_termination = False
                        # print(f"{d_idx = }\t{node = }\t{neighbor = }\t{spent_weight = }
                        # \t{weight = }\t{distance_matrix[d_idx] = }")
                        for path_node in th_o_path:
                            accumulated_path_weight += graph.edges[(path_node, previous_mode)]["weight"]
                            if (path_node in distance_matrix[d_idx]) \
                                    and (distance_matrix[d_idx][path_node]
                                         <= (accumulated_path_weight + spent_weight + weight)):
                                early_termination = True
                                trailblazer_early_termination += 1
                                break
                            # print(f"{d_idx = }\t{neighbor = }\t{o_idx = }\t{th_o_path = }\n{(path_node
                            # , previous_mode) = }")
                            distance_matrix[d_idx][path_node] = accumulated_path_weight + (spent_weight + weight)
                            heappush(best_weight_node_queue,
                                     (accumulated_path_weight + (spent_weight + weight), path_node))
                            th_o_path_weight.append(accumulated_path_weight)
                            previous_mode = path_node
                        if not early_termination:
                            trailheads[neighbor] = {"path": th_o_path, "weights": th_o_path_weight}
                            # print(f"{distance_matrix[d_idx] = }\n{trailheads[neighbor]}")
                else:
                    node_is_existing_trailhead += 1
                    for path_node, accumulated_path_weight in \
                            zip(trailheads[neighbor]["path"], trailheads[neighbor]["weights"]):

                        if (path_node in distance_matrix[d_idx]) \
                                and (distance_matrix[d_idx][path_node]
                                     <= (accumulated_path_weight + spent_weight + weight)):
                            trailblazer_early_termination += 1
                            break
                        distance_matrix[d_idx][path_node] = accumulated_path_weight + (spent_weight + weight)
                        heappush(best_weight_node_queue,
                                 (accumulated_path_weight + (spent_weight + weight), path_node))
            # break'''

    # start = time.time()
    for d_idx in distance_matrix.keys():
        od_scope = od_scope.union(set(distance_matrix[d_idx].keys()))
    # combining_sets = time.time() - start
    if o_graph is None:
        update_light_graph(
            self,
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


# dEORECIATED..
def bfs_paths(self, graph, source, target, weight_limit=0, distance_termination="geometric", batching=True,
              other_targets=[], distance_matrix=None, turn_penalty=False):
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
            f"parameter 'distance_termination' in function 'bfs_paths' can be ['geometric', 'network', 'none'],"
            f" {distance_termination} was given.")
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
            # paths, distances = _bfs_paths_many_targets_iterative(self, graph, source, target,
            #                                                     distance_matrix=distance_matrix,
            #                                                     turn_penalty=turn_penalty)
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
        # the given graph have 2 neighbors for origins and destinations. if a node has only one neighbor
        , its a deadend.
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
                else: # if its not in the distance matrix, its outside the od_ellipse_scope 
                and can't be reached by the current path.
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

# dEORECIATED..
def _bfs_paths_many_targets(graph, source, targets=None, targets_remaining=[], visited=[],
                            distance_termination="geometric", distance_matrix=None, start_time=0.0, paths=None,
                            distances=None, current_weight=0):
    # TODO: rearrnging checks will likely yield a better execution time
    # print (f"{visited}\t{source}")
    # print (f"{visited}\t{source}\t{current_weight}")
    if time.time() - start_time >= 30:
        raise TimeoutError("_bfs_paths_many_targets timed out...")
    for neighbor in list(graph.neighbors(source)):
        if neighbor in visited:
            continue
        # the given graph have 2 neighbors for origins and destinations. if a node has only one
        # neighbor, its a deadend.
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


from collections import deque

# dEORECIATED..
def turn_penalty_value(self, previous_node, current_node, next_node):
    node_gdf = self.layers["network_nodes"]["gdf"]
    edge_gdf = self.layers["network_edges"]["gdf"]
    '''
    # TODO: This is a lazty implementation that ignore polylines. need to retrieve edges and
    #  extract appropriate points. start of last seg, common point, end of first subsequent seg.
    print (f"{previous_node = }\t{current_node = }")
    start_edge_geometry = edge_gdf[
        ((edge_gdf["start"] == previous_node) & (edge_gdf["end"] == current_node))
        |
        ((edge_gdf["end"] == previous_node) & (edge_gdf["start"] == current_node))
        ]#.iloc[0]["geometry"]
    print (f"{start_edge_geometry = }")
    end_edge_geometry = edge_gdf[
        ((edge_gdf["start"] == current_node) & (edge_gdf["end"] == next_node))
        |
        ((edge_gdf["end"] == current_node) & (edge_gdf["start"] == next_node))
        ].iloc[0]["geometry"]

    if (len(start_edge_geometry.coords) < 2) or (len(end_edge_geometry.coords) < 2):
        print(f"Just letting you know.. turn_penalty had an issue. {len(start_edge_geometry.coords) =}"
              f"\t{len(end_edge_geometry.coords)}")

    print(edge_gdf.iloc[0]["geometry"].coords[-1])
    print(edge_gdf.iloc[0]["geometry"].coords[-2])
    '''
    angle = angle_deviation_between_two_lines(
        [
            node_gdf.at[previous_node, "geometry"],
            node_gdf.at[current_node, "geometry"],
            node_gdf.at[next_node, "geometry"]
        ]
    )
    angle = min(angle, abs(angle - 180))
    # print (f"{previous_node = }\t{current_node = }\t{next_node = }\t{angle = }")
    if angle > 45:
        # print(f"{previous_node = }\t{current_node = }\t{next_node = }\t{angle = }")
        return 62.3 # enable this as a parameter
    else:
        return 0

# cURRENT bESR
def _bfs_paths_many_targets_iterative(self, graph, o_idx, d_idxs, distance_matrix=None, turn_penalty=False,
                                      od_scope=None):
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

    node_gdf = self.layers["network_nodes"]["gdf"]
    edge_gdf = self.layers["network_edges"]["gdf"]

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
                turn_cost = turn_penalty_value(self, visited[-2], source, neighbor)

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

# cURRENT bESR
def turn_o_scope(self, o_idx, search_radius, detour_ratio, turn_penalty=True, o_graph=None, return_paths=True):
    node_gdf = self.layers["network_nodes"]["gdf"]
    destinations = node_gdf[node_gdf["type"] == "destination"].index
    # print(f"turn_o_scope: {o_idx = }")

    if o_graph is None:
        graph = self.d_graph
        update_light_graph(self, graph, add_nodes=[o_idx])
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
                    turn_cost = turn_penalty_value(self, visited[-2], node, neighbor)
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
        update_light_graph(self, graph=graph, remove_nodes=[o_idx])
    return d_idxs, o_scope, o_scope_paths


def get_od_paths(self, o_idx, search_radius, detour_ratio, output_map=False, algorithm="shortest_simple_paths",
                 graph_method="o_light", trim=False, distance_termination="network",
                 batching=True, result="paths", turn_penalty=False):
    start = time.time()
    paths = {}
    distances = {}
    graph = nx.Graph()
    graph_construction_time = 0
    if graph_method == "double_sided_dijkstra":
        graph, distance_matrix, d_idxs = get_od_subgraph(self, o_idx, d_idxs=None, search_radius=search_radius,
                                                         detour_ratio=detour_ratio,
                                                         output_map=output_map, trim=trim,
                                                         distance_method="double_sided_dijkstra",
                                                         turn_penalty=turn_penalty)
    else:
        d_idxs = destinations_accessible_from_origin(self, o_idx, search_radius=search_radius, light_graph=True)
        # print(f"it took {time.time() - start} to identify destinations... ")

    if len(d_idxs) > 0:
        if graph_method == "double_sided_dijkstra":
            pass
        elif graph_method == "o_light":
            if distance_termination in ["network", "none"]:
                graph, distance_matrix = get_od_subgraph(self, o_idx, d_idxs, search_radius, detour_ratio,
                                                         output_map=output_map, trim=trim,
                                                         distance_method="network",
                                                         turn_penalty=turn_penalty)
            else:
                graph = get_od_subgraph(self, o_idx, d_idxs, search_radius, detour_ratio, output_map=output_map,
                                        trim=trim,
                                        distance_method="geometric", turn_penalty=turn_penalty)
                distance_matrix = None
        elif graph_method == "od_light":
            graphs = []
            distance_matrices = []
            for d_idx in d_idxs.keys():
                if distance_termination in ["network", "none"]:
                    graph, distance_matrix = get_od_subgraph(self, o_idx, d_idx, search_radius, detour_ratio,
                                                             shortest_distance=d_idxs[d_idx], output_map=output_map,
                                                             trim=trim,
                                                             distance_method="network",
                                                             turn_penalty=turn_penalty)
                    distance_matrices.append(distance_matrix)
                else:
                    graph = get_od_subgraph(self, o_idx, d_idx, search_radius, detour_ratio,
                                            shortest_distance=d_idxs[d_idx], output_map=output_map, trim=trim,
                                            distance_method="geometric",
                                            turn_penalty=turn_penalty)
                    distance_matrices = None
                graphs.append(graph)
        elif graph_method == "full":
            if trim:
                raise ValueError(
                    f"parameter 'trim' in function 'get_od_paths' can't be 'True'"
                    f" when parameter 'graph_method' is 'full'.")
            graph = self.G.copy()
            update_light_graph(
                self,
                graph=graph,
                add_nodes=[o_idx] + list(d_idxs.keys())
            )
            distance_matrix = None
        else:
            raise ValueError(
                f"parameter 'graph_method' in function 'get_od_paths' can"
                f" be ['full', 'od_light', 'o_light'], '{graph_method}' was given")

        graph_construction_time = time.time() - start

        if algorithm == "shortest_simple_paths":
            paths = {}
            distances = {}
            if batching:
                raise ValueError(
                    f"parameter 'batching' in function 'get_od_paths' can't be 'True' when parameter 'algorithm' is "
                    f"'shortest_simple_paths', {batching} was given.")
            if distance_termination != "none":
                raise ValueError(
                    f"parameter 'distance_termination' in function 'get_od_paths' can only be 'none' when parameter "
                    f"'algorithm' is 'shortest_simple_paths', {distance_termination} was given.")

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
                    f"parameter 'batching' in function 'get_od_paths' can't be 'True' when parameter "
                    f"'algorithm' is 'all_simple_paths'.")
            if distance_termination != "none":
                raise ValueError(
                    f"parameter 'distance_termination' in function 'get_od_paths' can only be 'none' when parameter"
                    f" 'algorithm' is 'all_simple_paths', {distance_termination} was given.")

            for seq, d_idx in enumerate(d_idxs.keys()):
                if graph_method in ['full', "o_light"]:
                    raise ValueError(
                        f"parameter 'graph_method' in function 'get_od_paths' can only be 'od_light' when parameter"
                        f" 'algorithm' is 'all_simple_paths'.")
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
                    f"parameter 'distance_termination' in function 'get_od_paths' can't be 'geometric' when parameter "
                    f"'graph_method' is 'full' and 'algorithm' is 'bfs , {graph_method} was given.")
            if (graph_method == "full") and (not batching):
                raise ValueError(
                    f"parameter 'batching' in function 'get_od_paths' should be 'True' when parameter "
                    f"'graph_method' is'full' and 'algorithm' is 'bfs , {batching} was given.")

            d_allowed_distances = {}
            for d_idx in d_idxs.keys():
                d_allowed_distances[d_idx] = d_idxs[d_idx] * detour_ratio

            if batching:
                if graph_method == "od_light":
                    raise ValueError(
                        f"parameter 'graph_method' in function 'get_od_paths' can't be 'od_light' when parameter "
                        f"'batching' is 'True', {graph_method} was given.")

                use_graph = graph
                paths, distances = bfs_paths(self, use_graph, o_idx, d_allowed_distances,
                                             distance_termination=distance_termination, batching=batching,
                                             distance_matrix=distance_matrix, turn_penalty=turn_penalty)
            else:
                paths = {}
                distances = {}
                for seq, d_idx in enumerate(d_idxs.keys()):
                    if graph_method in ["double_sided_dijkstra", "o_light", 'full']:
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
                                                     distance_matrix=use_distance_matrix,
                                                     turn_penalty=turn_penalty)
                    paths[d_idx] = d_paths
                    distances[d_idx] = d_distances
        else:
            raise ValueError(
                f"parameter 'algorithm' in function 'get_od_paths' can be ['shortest_simple_paths'"
                f", 'all_simple_paths', 'bfs'], '{algorithm}' was given")

    if result == "paths":
        return paths, distances, d_idxs
    elif result == "diagnostics":
        return paths, distances, d_idxs, graph, {
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
                           origin_weight_attribute = None,
                           closest_destination=True,
                           destination_weights=False,
                           destination_weight_attribute = None,
                           perceived_distance=False,
                           light_graph=True,
                           turn_penalty=False,
                           retained_d_idxs=None,
                           retained_paths=None,
                           retained_distances=None,
                           rertain_expensive_data=False
                           ):
    node_gdf = self.layers['network_nodes']['gdf']
    edge_gdf = self.layers['network_edges']['gdf']

    origins = node_gdf[
        (node_gdf["type"] == "origin")
    ]

    retain_paths = {}
    retain_distances = {}
    retain_d_idxs = {}

    # create a new column in edges with an initial 0 betweenness.
    # results should be aggregated  into this column
    edge_gdf['betweenness'] = 0.0

    if num_cores == 1:
        betweennes_results = one_betweenness_2(
            self,
            search_radius=search_radius,
            origins=origins,
            detour_ratio=detour_ratio,
            decay=decay,
            decay_method=decay_method,
            beta=beta,
            path_detour_penalty=path_detour_penalty,
            origin_weights=origin_weights,
            origin_weight_attribute=origin_weight_attribute,
            closest_destination=closest_destination,
            destination_weights=destination_weights,
            destination_weight_attribute=destination_weight_attribute,
            perceived_distance=perceived_distance,
            light_graph=light_graph,
            turn_penalty=turn_penalty,
            retained_d_idxs=retained_d_idxs,
            retained_paths=retained_paths,
            retained_distances=retained_distances,
            rertain_expensive_data=rertain_expensive_data
        )
        batch_results = betweennes_results["batch_betweenness_tracker"]
        batch_results = [batch_results]
        if rertain_expensive_data:
            retain_paths = betweennes_results["retained_paths"]
            retain_distances = betweennes_results["retained_distances"]
            retain_d_idxs = betweennes_results["retained_d_idxs"]

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
                    origin_weight_attribute=origin_weight_attribute,
                    closest_destination=closest_destination,
                    destination_weights=destination_weights,
                    destination_weight_attribute=destination_weight_attribute,
                    perceived_distance=perceived_distance,
                    light_graph=light_graph,
                    turn_penalty=turn_penalty,
                    retained_d_idxs=retained_d_idxs,
                    retained_paths=retained_paths,
                    retained_distances=retained_distances,
                    rertain_expensive_data=rertain_expensive_data
                ) for df in splitted_origins]
            for result in concurrent.futures.as_completed(execution_results):
                try:
                    one_betweennes_output = result.result()
                    batch_results.append(one_betweennes_output["batch_betweenness_tracker"])
                    if rertain_expensive_data:
                        retain_paths = {**retain_paths, **one_betweennes_output["retained_paths"]}
                        retain_distances = {**retain_distances, **one_betweennes_output["retained_distances"]}
                        retain_d_idxs = {**retain_d_idxs, **one_betweennes_output["retained_d_idxs"]}
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
    return_dict = {"edge_gdf": edge_gdf}
    if rertain_expensive_data:
        return_dict["retained_paths"] = retain_paths
        return_dict["retained_distances"] = retain_distances
        return_dict["retained_d_idxs"] = retain_d_idxs
    return return_dict


def update_light_graph(self, graph, add_nodes=[], remove_nodes=[]):
    # print(f"{str(graph)}\t\t{add_nodes = }\t{remove_nodes = }")
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
            # print(f"{edge_id = }\t{edge_nodes = }")
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
                    weight=max(left_edge["weight"], 0),
                    id=edge_id
                )
                graph.add_edge(
                    int(node_idx),
                    int(right_edge["node_id"]),
                    weight=max(right_edge["weight"], 0),
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
                # print(f"{chain_nodes = }\t{chain_distances}")
                accumilated_weight = 0
                for seq in range(len(chain_nodes) - 1):
                    graph.add_edge(
                        int(chain_nodes[seq]),
                        int(chain_nodes[seq + 1]),
                        # TODO: change this to either defaults to distance or a specified column for a weight..
                        weight=max(chain_distances[seq + 1] - chain_distances[seq], 0),
                        ##avoiding small negative numbers due to numerical error when two nodes are superimposed.
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
        # print(f"{neighbors = }\t{start = }\t{end = }\t{weight = }\t{original_edge_id = }")
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


def one_betweenness_3(
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
        light_graph=True,
        turn_penalty=False,

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
            # accissible_destinations = destinations_accessible_from_origin(self, origin_idx, search_radius=search_radius)
            d_idxs, o_scope, _ = turn_o_scope(self, origin_idx, search_radius, detour_ratio, turn_penalty=turn_penalty,
                                              return_paths=False)
            # accissible_destinations = list(d_idxs.keys())
            # for accissible_destination_idx, od_shortest_distance in accissible_destinations.items():
            for accissible_destination_idx, od_shortest_distance in d_idxs.items():
                od_gravity = 1 / pow(math.e, (beta * od_shortest_distance))
                if destination_weights:
                    od_gravity *= node_gdf.at[accissible_destination_idx, "weight"]
                od_sum_gravities += od_gravity
            # destination_ids = list(accissible_destinations.keys())
            destination_ids = list(d_idxs.keys())
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
        paths, weights, d_idxs = get_od_paths(self, origin_idx, search_radius, detour_ratio, output_map=False,
                                              algorithm="bfs",
                                              graph_method="double_sided_dijkstra", trim=False,
                                              distance_termination="network",
                                              batching=True,
                                              result="paths",
                                              turn_penalty=turn_penalty)

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
                graph = get_od_subgraph(self, origin_idx, destination_idx, search_radius, detour_ratio,
                 output_map=False,
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
                        # TODO: trailblazer algorithm sometimes produces paths that exceeds this limin. revisit
                        pass
                    # print(
                    #    f"o: {origin_idx}\td:{destination_idx}\t{path}\{this_path_weight}\t "
                    #    f"exceeded limit {shortest_path_distance * detour_ratio}")
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
    return_dict = {"batch_betweenness_tracker": batch_betweenness_tracker}

    return return_dict


## This is a development version....
def     one_betweenness_2(
        self,
        search_radius=1000,
        origins=None,
        detour_ratio=1.05,
        decay=True,
        beta=0.003,
        decay_method="exponent",
        path_detour_penalty="equal",  # "exponent" | "power"
        origin_weights=False,
        origin_weight_attribute=None,
        closest_destination=False,
        destination_weights=False,
        destination_weight_attribute=None,
        perceived_distance=False,
        light_graph=True,
        turn_penalty=False,
        retained_d_idxs=None,
        retained_paths=None,
        retained_distances=None,
        rertain_expensive_data=False
):
    edge_gdf = self.layers["network_edges"]["gdf"]
    node_gdf = self.layers["network_nodes"]["gdf"]

    # graph = self.G.copy()

    batch_betweenness_tracker = {edge_id: 0 for edge_id in list(edge_gdf.index)}

    counter = 0
    retain_paths = {}
    retain_distances = {}
    retain_d_idxs = {}
    for origin_idx in tqdm(origins.index):
        counter += 1
        '''
        paths, weights, d_idxs = best_path_generator_so_far(self, origin_idx, search_radius=search_radius,
                                                            detour_ratio=detour_ratio, turn_penalty=turn_penalty)
        
        paths, weights, d_idxs = get_od_paths(self, origin_idx, search_radius, detour_ratio, output_map=False,
                                              algorithm="bfs",
                                              graph_method="double_sided_dijkstra", trim=False,
                                              distance_termination="network",
                                              batching=True,
                                              result="paths",
                                              turn_penalty=turn_penalty)
        '''

        if (retained_d_idxs is None) or (retained_paths is None) or (retained_distances is None):
            # print(f"{origin_idx  = } is not retreaved, generating it right now")
            '''
            paths, weights, d_idxs = get_od_paths(self, origin_idx, search_radius, detour_ratio, output_map=False,
                                                  algorithm="bfs",
                                                  graph_method="double_sided_dijkstra", trim=False,
                                                  distance_termination="network",
                                                  batching=True,
                                                  result="paths",
                                                  turn_penalty=turn_penalty)
                                                  '''

            paths, weights, d_idxs = best_path_generator_so_far(self, origin_idx, search_radius=search_radius,
                                                                detour_ratio=detour_ratio, turn_penalty=turn_penalty)
            if rertain_expensive_data:
                # print(f"{origin_idx  = } is retaining generated data")
                # retain_paths[origin_idx] = paths
                # retain_distances[origin_idx] = weights
                retain_d_idxs[origin_idx] = d_idxs
        else:
            # print(f"{origin_idx  = } is using retained data")
            paths = retained_paths[origin_idx]
            weights = retained_distances[origin_idx]
            d_idxs = retained_d_idxs[origin_idx]

        destination_count = len(d_idxs)

        if destination_count == 0:
            # print(f"o:{origin_idx} has no destinations")
            continue

        od_sum_gravities = 0
        if closest_destination:
            destination_ids = [int(origins.at[origin_idx, "closest_destination"])]
        else:
            for accissible_destination_idx, od_shortest_distance in d_idxs.items():
                od_gravity = 1 / pow(math.e, (beta * od_shortest_distance))
                if destination_weights:
                    od_gravity *= node_gdf.at[accissible_destination_idx, "weight"]
                od_sum_gravities += od_gravity
            destination_ids = list(d_idxs.keys())
        destinations = node_gdf.loc[destination_ids]

        for destination_idx in destination_ids:  # destinations.index:
            try:
                destination_count += 1

                this_od_paths = {
                    "path_length": [],
                    "path_edges": [],
                    "one_over_squared_length": [],
                    'one_over_e_beta_length': []
                }

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
                        # TODO: trailblazer algorithm sometimes produces paths that exceeds this limin. revisit
                        pass
                        # print(
                        #    f"o: {origin_idx}\td:{destination_idx}\t{path}\{this_path_weight}\t "
                        #    f"exceeded limit {shortest_path_distance * detour_ratio}")

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
                        if origin_weight_attribute:
                            betweennes_contribution *= origins.at[origin_idx, origin_weight_attribute]
                        else:
                            betweennes_contribution *= origins.at[origin_idx, "weight"]
                    if decay:
                        betweennes_contribution *= this_od_paths[decay_method][0]
                    if not closest_destination:
                        this_d_gravity = 1 / pow(math.e, (beta * shortest_path_distance))
                        if destination_weights:
                            if destination_weight_attribute:
                                this_d_gravity *= destinations.at[destination_idx, destination_weight_attribute]
                            else:
                                this_d_gravity *= destinations.at[destination_idx, "weight"]
                        if (this_d_gravity == 0) and (od_sum_gravities == 0):
                            trip_probability = 0  # to cover a case where ypu can only access destinations of weight 0
                        else:
                            trip_probability = this_d_gravity / od_sum_gravities
                        betweennes_contribution *= trip_probability

                    for edge_id in this_od_paths["path_edges"][seq]:
                        batch_betweenness_tracker[edge_id] += betweennes_contribution
            except Exception as e:

                print(f"................o: {origin_idx}\td: {destination_idx} faced an error........")
                print(path)
                print(e.__doc__)

    print(f"core {origins.iloc[0].name} done.")

    return_dict = {"batch_betweenness_tracker": batch_betweenness_tracker}
    if rertain_expensive_data:
        return_dict["retained_paths"] = retain_paths
        return_dict["retained_distances"] = retain_distances
        return_dict["retained_d_idxs"] = retain_d_idxs
    return return_dict


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


def flatten_multi_edge_segments(source_gdf, flatten_multiLineStrings=True, flatten_LineStrings=False):
    ls_dict = {"source_index": [], "geometry": []}

    if flatten_multiLineStrings:
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
        if (number_of_nodes != 2) and flatten_LineStrings:
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


# possible methods: sort by segment length, minimize number of node-moves. minimize number of nodes by
# fusing to highest degree within tolerance
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
        #street_geometry = geo.MultiLineString(
        #    list(sorted_gdf["geometry"])
        #)
        street_geometry = list(sorted_gdf["geometry"])


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
    #for street_iloc, street in enumerate(list(street_geometry)):
    for street_iloc, street in enumerate(street_geometry):
        counter += 1
        if counter % 100 == 0:
            print(f'{counter = }, progress = {counter / list_len * 100:5.2f}')
        # get this segment's nodes, if segment is more than one segment, get only beginning node and end node
        # , trat as a single sigment
        start_point_index = None
        end_point_index = None

        start_point_geometry = None
        end_point_geometry = None

        start_point = geo.Point(street.coords[0])
        end_point = geo.Point(street.coords[-1])
        # if len(street.coords)>2:
        # print (f"street {street_iloc = } had {len(street.coords)} points")

        # FInd nearest points to start and end, reject if greater than tolerance.

        # todo: use a spatial index here to eliminate the two loops
        # create a dataframe containing all start end end points. keep a reference of what segment they come from.
        #point_gdf["nearest_point_id"] = point_gdf.apply(
            #lambda x: point_gdf["geometry"].sindex.nearest(point_gdf["geometry"])
        #)
        #match = point_gdf["geometry"].sindex.nearest(point_gdf["geometry"])
        # add a column call node assignment, where we start from the longest street, look in the spatial index
        # to find all "unassigned" points within the proximity

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
        # intersection_filter = edge_gdf["p_geometry"].buffer(node_snapping_tolerance/2).intersects(edge["p_geometry"]
        # .buffer(node_snapping_tolerance/2))
        intersecting_edges = edge_gdf[intersection_filter]
        intersecting_edge_ids = list(intersecting_edges[~intersecting_edges["id"].isin(neighbor_ids)]['id'])
        if len(intersecting_edge_ids) > 0:
            for intersecting_edge_id in intersecting_edge_ids:
                intersections = intersections + 1
                # find interection point
                intersecting_edge = edge_gdf[edge_gdf["id"] == intersecting_edge_id].iloc[0]
                intersection_point = edge[measurement_geometry].intersection(intersecting_edge[measurement_geometry])
                # intersection_point = edge[measurement_geometry].buffer(node_snapping_tolerance)
                # .intersection(intersecting_edge[measurement_geometry])
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
