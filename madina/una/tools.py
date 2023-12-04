from ..zonal import Zonal
from .paths import turn_o_scope, wandering_messenger
import math
import numpy as np
import geopandas as gpd
import networkx as nx
from shapely.geometry import Polygon

def accessibility(
    zonal: Zonal,
    reach: bool = False,
    gravity: bool =False,
    closest_facility: bool =False,
    weight: str = None,
    search_radius: float=None,
    alpha:float=1,
    beta:float=None
):
    if gravity and (beta is None):
        raise ValueError("Please specify parameter 'beta' when 'gravity' is True")

    node_gdf = zonal.network.nodes
    origin_gdf = node_gdf[node_gdf["type"] == "origin"]

    reaches = {}
    gravities = {}

    for o_idx in origin_gdf.index:
        reaches[o_idx] = {}
        gravities[o_idx] = {}

        o_graph = zonal.network.d_graph
        zonal.network.add_node_to_graph(o_graph, o_idx)

        d_idxs, o_scope, o_scope_paths = turn_o_scope(
            network=zonal.network,
            o_idx=o_idx,
            search_radius=search_radius,
            detour_ratio=1.00, 
            turn_penalty=False,
            o_graph=o_graph,
            return_paths=False
        )

        zonal.network.remove_node_to_graph(o_graph, o_idx)


        for d_idx in d_idxs:
            source_id = int(node_gdf.at[d_idx, "source_id"])
            source_layer = node_gdf.at[d_idx, "source_layer"]
            d_weight = 1 if weight is None else zonal.layers[source_layer].gdf.at[source_id, weight]

            if reach:
                reaches[o_idx][d_idx] = d_weight
            if gravity:
                gravities[o_idx][d_idx] = pow(d_weight, alpha) / (pow(math.e, (beta * d_idxs[d_idx])))

            if closest_facility:
                if ("una_closest_destination" not in node_gdf.loc[d_idx]) or (np.isnan(node_gdf.loc[d_idx]["una_closest_destination"])):
                    node_gdf.at[d_idx, "una_closest_destination"] = o_idx
                    node_gdf.at[d_idx, "una_closest_destination_distance"] = d_idxs[d_idx]
                elif d_idxs[d_idx] < node_gdf.at[d_idx, "una_closest_destination_distance"]:
                    node_gdf.at[d_idx, "una_closest_destination"] = o_idx
                    node_gdf.at[d_idx, "una_closest_destination_distance"] = d_idxs[d_idx]

        if reach:
            node_gdf.at[o_idx, "una_reach"] = sum([value for value in reaches[o_idx].values() if not np.isnan(value)])

        if gravity:
            node_gdf.at[o_idx, "una_gravity"] = sum([value for value in gravities[o_idx].values() if not np.isnan(value)])

    if closest_facility:
        for o_idx in origin_gdf.index:
            o_closest_destinations = list(node_gdf[node_gdf["una_closest_destination"] == o_idx].index)
            if reach:
                node_gdf.at[o_idx, "una_reach"] = sum([reaches[o_idx][d_idx] for d_idx in o_closest_destinations])
            if gravity:
                node_gdf.at[o_idx, "una_gravity"] = sum([gravities[o_idx][d_idx] for d_idx in o_closest_destinations])
    return

def service_area(
    zonal: Zonal,
    origin_ids: list=None,
    search_radius: float=None
):
    node_gdf = zonal.network.nodes
    edge_gdf = zonal.network.edges
    origins = node_gdf[node_gdf["type"] == "origin"]

    if origin_ids is None:
        origin_ids = list(origins.index)
    elif not isinstance(origin_ids, list):
        raise ValueError("the 'origin_ids' parameter expects a list, for single origins, set it to [11] for example")

    if len(set(origin_ids) - set(origins.index)) != 0:
        raise ValueError(f"some of the indices given are not for an origin {set(origin_ids) - set(origins.index)}")

    scope_names = []
    scope_geometries = []
    scope_origin_ids = []

    network_edges = []
    destination_ids = set()
    source_layer = None

    # TODO: This loop assumes all destinations are the same layer, generalize to all layers.
    for origin_id in origin_ids:

        o_graph = zonal.network.d_graph
        zonal.network.add_node_to_graph(o_graph, origin_id)


        d_idxs, o_scope, o_scope_paths = turn_o_scope(
            network=zonal.network,
            o_idx=origin_id,
            search_radius=search_radius,
            detour_ratio=1.00, 
            turn_penalty=False,
            o_graph=o_graph,
            return_paths=False
        )

        zonal.network.remove_node_to_graph(o_graph, origin_id)

        if (len(d_idxs)) == 0:
            continue

        destination_original_ids = node_gdf.loc[list(d_idxs.keys())]["source_id"]
        source_layer = node_gdf.at[list(d_idxs.keys())[0], "source_layer"]
        destination_ids = destination_ids.union(destination_original_ids)
        

        destinations = zonal.layers[source_layer].gdf.loc[destination_original_ids]
        # destination_scope = destinations.dissolve().convex_hull.exterior
        destination_scope = destinations["geometry"].unary_union.convex_hull


        network_scope = node_gdf.loc[list(o_scope.keys())]["geometry"].unary_union.convex_hull
        scope = destination_scope.union(network_scope)
        # network_edges.append(edge_gdf.clip(scope).dissolve().iloc[0]["geometry"].geoms)
        network_edges.append(edge_gdf.clip(scope)["geometry"].unary_union)

        origin_geom = node_gdf.at[origin_id, "geometry"]

        scope_names.append("service area border")
        scope_names.append("origin")
        scope_geometries.append(scope)
        scope_geometries.append(origin_geom)
        scope_origin_ids.append(origin_id)
        scope_origin_ids.append(origin_id)

    scope_gdf = gpd.GeoDataFrame(
        {
            "name": scope_names,
            "origin_id": scope_origin_ids,
        }
    ).set_geometry(scope_geometries).set_crs(zonal.layers[source_layer].gdf.crs)

    destinations = zonal.layers[source_layer].gdf.loc[list(destination_ids)]
    network_edges = gpd.GeoDataFrame({}).set_geometry(network_edges).set_crs(zonal.layers[source_layer].gdf.crs).explode(index_parts=False)
    return destinations, network_edges, scope_gdf

def closest_facility(
    zonal:Zonal,
    weight:str=None,
    reach:bool=False,
    gravity:bool=False,
    search_radius:float=None,
    beta:float=None,
    alpha:float=1
    ):
    if gravity and (beta is None):
        raise ValueError("Please specify parameter 'beta' when 'gravity' is True")
    accessibility(
        zonal=zonal,
        reach=reach,
        gravity=gravity,
        closest_facility=True,
        weight=weight,
        search_radius=search_radius,
        alpha=alpha,
        beta=beta
    )

    return

def closest_destination(
    zonal:Zonal,
    beta:float=0.003,
    light_graph=True
):
    node_gdf = zonal.network.nodes
    origins = node_gdf[node_gdf["type"] == "origin"]
    destinations = node_gdf[node_gdf["type"] == "destination"]
    distance, path = nx.multi_source_dijkstra(
        zonal.G,
        sources=list(destinations.index),
        weight='weight'
    )
    for idx in origins.index:
        node_gdf.at[idx, 'closest_destination'] = path[idx][0]
        node_gdf.at[idx, 'closest_destination_distance'] = distance[idx]
        node_gdf.at[idx, 'closest_destination_gravity'] = 1 / pow(math.e, (beta * distance[idx]))

    zonal.network.nodes = node_gdf
    return


def alternative_paths(
    zonal:Zonal,
    search_radius:float, 
    detour_ratio:float, 
    o_idx: int, 
    d_idx: int
):

    o_graph = zonal.network.d_graph
    zonal.network.add_node_to_graph(o_graph, origin_id)
    
    scope_nodes, distance_matrix, _ = bfs_subgraph_generation(
        o_idx=origin_idx,
        detour_ratio=detour_ratio,
        o_graph=o_graph,
        d_idxs=dict(sorted(d_idx_chunck.items(), key=lambda item: item[1])),  ## Seems like this expects destinations to be sorted by distance?
        #d_idxs=d_idx_chunck,
        o_scope=o_scope,
        o_scope_paths=o_scope_paths,
    )

    d_allowed_distances = {}
    for d_idx in d_idx_chunck.keys():
        d_allowed_distances[d_idx] =  d_idx_chunck[d_idx] * detour_ratio

    path_edges, weights = wandering_messenger(
    #path_edges, weights = bfs_path_edges_many_targets_iterative(
        network=self.network,
        o_graph=o_graph,
        o_idx=origin_idx,
        d_idxs=d_allowed_distances,
        distance_matrix=distance_matrix,
        turn_penalty=turn_penalty,
        od_scope=scope_nodes
    )

    zonal.network.remove_node_to_graph(o_graph, origin_id)