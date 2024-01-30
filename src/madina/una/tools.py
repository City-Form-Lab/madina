import math
import numpy as np
import geopandas as gpd

from shapely import GeometryCollection
from .paths import turn_o_scope, wandering_messenger, path_generator
from .betweenness import paralell_betweenness_exposure
from ..zonal import Zonal







def accessibility(
    zonal: Zonal,
    reach: bool = False,
    gravity: bool =False,
    closest_facility: bool =False,
    weight: str = None,
    search_radius: float=None,
    alpha:float=1,
    beta:float=None, 
    turn_penalty: bool = False,
    turn_penalty_amount: float = 0, 
    turn_threshold_degree: float = 0,
    save_reach_as: str = None, 
    save_gravity_as: str = None,
    save_closest_facility_as: str = None, 
    save_closest_facility_distance_as: str = None, 
):
    """
    Modifies the input zonal with accessibility metrics such as `reach`, `gravity`, and `closest_facility` analysis.
        Equivalent of 'una_accessibility' function in madina.py

        Returns:
            A modified Zonal object

        Raises:
            ValueError if `gravity` is True but beta is None/unspecified.
    """
    if turn_penalty:
        zonal.network.turn_penalty_amount = turn_penalty_amount
        zonal.network.turn_threshold_degree = turn_threshold_degree
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
            turn_penalty=turn_penalty,
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
                if ("closest_facility" not in node_gdf.loc[d_idx]) or (np.isnan(node_gdf.loc[d_idx]["closest_facility"])):
                    node_gdf.at[d_idx, "closest_facility"] = o_idx
                    node_gdf.at[d_idx, "closest_facility_distance"] = d_idxs[d_idx]
                elif d_idxs[d_idx] < node_gdf.at[d_idx, "closest_facility_distance"]:
                    node_gdf.at[d_idx, "closest_facility"] = o_idx
                    node_gdf.at[d_idx, "closest_facility_distance"] = d_idxs[d_idx]

        if reach:
            node_gdf.at[o_idx, "reach"] = sum([value for value in reaches[o_idx].values() if not np.isnan(value)])

        if gravity:
            node_gdf.at[o_idx, "gravity"] = sum([value for value in gravities[o_idx].values() if not np.isnan(value)])

    if closest_facility:
        for o_idx in origin_gdf.index:
            o_closest_destinations = list(node_gdf[node_gdf["closest_facility"] == o_idx].index)
            if reach:
                node_gdf.at[o_idx, "reach"] = sum([reaches[o_idx][d_idx] for d_idx in o_closest_destinations])
            if gravity:
                node_gdf.at[o_idx, "gravity"] = sum([gravities[o_idx][d_idx] for d_idx in o_closest_destinations])
    




    if (save_reach_as is not None) or (save_gravity_as is not None): 
        origin_gdf = node_gdf[node_gdf["type"] == "origin"]
        origin_layer = origin_gdf.iloc[0]['source_layer']

        saved_attributes = {}
        if save_reach_as is not None:
            saved_attributes['reach'] = save_reach_as

        if save_gravity_as is not None:
            saved_attributes['gravity'] = save_gravity_as

        for key, value in saved_attributes.items():
            node_gdf.loc[origin_gdf.index, key] = node_gdf.loc[origin_gdf.index, key].fillna(0)
            if value in zonal[origin_layer].gdf.columns:
                zonal[origin_layer].gdf.drop(columns=[value], inplace=True)

        zonal[origin_layer].gdf = zonal[origin_layer].gdf.join(
            node_gdf.loc[origin_gdf.index,['source_id'] + list(saved_attributes.keys()) ].set_index("source_id").rename(columns=saved_attributes)
        )
        zonal[origin_layer].gdf.index = zonal[origin_layer].gdf.index.astype(int)



    if (save_closest_facility_as is not None) or (save_closest_facility_distance_as is not None): 
        destination_gdf = node_gdf[node_gdf["type"] == "destination"]
        destination_layer = destination_gdf.iloc[0]['source_layer']

        saved_attributes = {}
        if save_closest_facility_as is not None:
            saved_attributes['closest_facility'] = save_closest_facility_as

        if save_closest_facility_distance_as is not None:
            saved_attributes['closest_facility_distance'] = save_closest_facility_distance_as

        for key, value in saved_attributes.items():
            node_gdf.loc[destination_gdf.index, key] = node_gdf.loc[destination_gdf.index, key].fillna(0)
            if value in zonal[destination_layer].gdf.columns:
                zonal[destination_layer].gdf.drop(columns=[value], inplace=True)

        zonal[destination_layer].gdf = zonal[destination_layer].gdf.join(
            node_gdf.loc[destination_gdf.index,['source_id'] + list(saved_attributes.keys()) ].set_index("source_id").rename(columns=saved_attributes)
        )
        zonal[destination_layer].gdf.index = zonal[destination_layer].gdf.index.astype(int)

    return


def service_area(
    zonal: Zonal,
    origin_ids: list = None,
    search_radius: float = None, 
    turn_penalty: bool = False,
    turn_penalty_amount: float = 0, 
    turn_threshold_degree: float = 0,

):
    
    """
     Calculates the service area of the origins in `origin_ids`.

        Returns:
            a tuple of destinations accessible from the origins, edges traversed, and a pandas GeoDataFrame of their scope
    """
    if turn_penalty:
        zonal.network.turn_penalty_amount = turn_penalty_amount
        zonal.network.turn_threshold_degree = turn_threshold_degree

    node_gdf = zonal.network.nodes
    edge_gdf = zonal.network.edges

    origin_gdf = node_gdf[node_gdf["type"] == "origin"]
    origin_layer = origin_gdf.iloc[0]['source_layer']

    destination_gdf = node_gdf[node_gdf["type"] == "destination"]
    destination_layer = destination_gdf.iloc[0]['source_layer']
    

    if origin_ids is None:
        o_idxs = list(origin_gdf.index)
    elif not isinstance(origin_ids, list):
        raise ValueError("the 'origin_ids' parameter expects a list, for single origins, set it to [11] for example")
    elif len(set(origin_ids) - set(zonal[origin_layer].gdf.index)) != 0:
        raise ValueError(f"some of the indices given are not for an origin {set(origin_ids) - set(zonal[origin_layer].gdf.index)}")
    else:
        o_idxs = list(origin_gdf[origin_gdf['source_id'].isin(origin_ids)].index)

    scope_names = []
    scope_geometries = []
    scope_origin_ids = []

    all_network_edges = []
    all_destination_ids = set()
            

    # TODO: This loop assumes all destinations are the same layer, generalize to all layers.
    o_graph = zonal.network.d_graph
    for o_idx in o_idxs:

        
        zonal.network.add_node_to_graph(o_graph, o_idx)


        d_idxs, o_scope, o_scope_paths = turn_o_scope(
            network=zonal.network,
            o_idx=o_idx,
            search_radius=search_radius,
            detour_ratio=1.00, 
            turn_penalty=turn_penalty,
            o_graph=o_graph,
            return_paths=False
        )

        zonal.network.remove_node_to_graph(o_graph, o_idx)

        if (len(d_idxs)) == 0:
            continue

        destination_ids = node_gdf.loc[list(d_idxs.keys())]["source_id"]
        all_destination_ids = all_destination_ids.union(destination_ids)
        

        destination_scope = zonal[destination_layer].gdf.loc[destination_ids]["geometry"].unary_union.convex_hull
        network_scope = node_gdf.loc[list(o_scope.keys())]["geometry"].unary_union.convex_hull
        scope = destination_scope.union(network_scope)

        all_network_edges.append(edge_gdf.clip(scope)["geometry"].unary_union)

        origin_id = node_gdf.at[o_idx, "source_id"]
        origin_geom = zonal[origin_layer].gdf.at[origin_id, 'geometry']

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
    ).set_geometry(scope_geometries).set_crs(zonal[destination_layer].gdf.crs)

    destinations = zonal[destination_layer].gdf.loc[list(all_destination_ids)]
    network_edges = gpd.GeoDataFrame({}).set_geometry(all_network_edges).set_crs(zonal.layers[destination_layer].gdf.crs).explode(index_parts=False)
    return destinations, network_edges, scope_gdf

def closest_facility(
    zonal:Zonal,
    weight:str=None,
    reach:bool=False,
    gravity:bool=False,
    search_radius:float=None,
    beta:float=None,
    alpha:float=1,
    turn_penalty: bool = False,
    turn_penalty_amount: float = 0, 
    turn_threshold_degree: float = 0, 
    save_reach_as: str = None, 
    save_gravity_as: str = None,
    save_closest_facility_as: str = None, 
    save_closest_facility_distance_as: str = None, 
    ):

    accessibility(
        zonal=zonal,
        reach=reach,
        gravity=gravity,
        closest_facility=True,
        weight=weight,
        search_radius=search_radius,
        alpha=alpha,
        beta=beta, 
        turn_penalty=turn_penalty,
        turn_penalty_amount=turn_penalty_amount, 
        turn_threshold_degree=turn_threshold_degree,
        save_reach_as=save_reach_as, 
        save_gravity_as=save_gravity_as,
        save_closest_facility_as=save_closest_facility_as, 
        save_closest_facility_distance_as=save_closest_facility_distance_as, 
    )

    return

def alternative_paths(
    zonal: Zonal,
    origin_id: int,
    search_radius: float,
    #destination_id: int,
    detour_ratio: float = 1,
    turn_penalty: bool = False,
    turn_penalty_amount: float = 0, 
    turn_threshold_degree: float = 0
):
    if turn_penalty:
        zonal.network.turn_penalty_amount = turn_penalty_amount
        zonal.network.turn_threshold_degree = turn_threshold_degree

    origin_gdf = zonal.network.nodes[zonal.network.nodes['type'] == 'origin']
    o_idx = origin_gdf[origin_gdf['source_id'] == origin_id].iloc[0].name

    path_edges, distances, d_idxs = path_generator(
        network=zonal.network,
        o_idx=o_idx,
        search_radius=search_radius,
        detour_ratio=detour_ratio,
        turn_penalty=turn_penalty
    )

    destination_list = []
    distance_list = []
    path_geometries = []

    for d_idx, destination_distances in distances.items():
        destination_id = zonal.network.nodes.at[d_idx, 'source_id']
        destination_list = destination_list + [destination_id] * len(destination_distances)
        distance_list = distance_list + list(destination_distances)
        for segment_list in path_edges[d_idx]:
            origin_segment_id = int(zonal.network.nodes.at[o_idx, 'nearest_edge_id'])
            destination_segment_id = int(zonal.network.nodes.at[d_idx, 'nearest_edge_id'])
            path_segments = [zonal.network.edges.at[origin_segment_id, 'geometry']] + list(zonal.network.edges.loc[segment_list]['geometry']) + [zonal.network.edges.at[destination_segment_id, 'geometry']]

            path_geometries.append(
                GeometryCollection(
                    path_segments
                )
            )
        #print (f"destination {destinatio_idx} done")


    destination_gdf = gpd.GeoDataFrame({'destination': destination_list, 'distance': distance_list, 'geometry': path_geometries}, crs = zonal.network.nodes.crs)
    #destination_gdf['width'] = (destination_gdf['distance'] - destination_gdf['distance'].min())/(destination_gdf['distance'].max() - destination_gdf['distance'].min()) * 5
    destination_gdf = destination_gdf.sort_values("distance").reset_index(drop=True)
    return destination_gdf

def betweenness(
    zonal: Zonal,
    search_radius: float,
    detour_ratio: float = 1,
    decay: bool = False,
    decay_method: str = "exponent",
    beta: float = 0.003,
    num_cores: int = 1,
    closest_destination: bool = True,
    elastic_weight: bool = False,
    knn_weight: str | list = None,
    knn_plateau: float = 0, 
    turn_penalty: bool = False,
    turn_penalty_amount: float = 0, 
    turn_threshold_degree: float = 0,
    save_betweenness_as: str = None, 
    save_reach_as: str = None, 
    save_gravity_as: str = None,
    save_elastic_weight_as: str = None,
    keep_diagnostics: bool = False, 
    path_exposure_attribute: str = None,
    save_path_exposure_as: str = None,
):
    zonal.network.turn_penalty_amount = turn_penalty_amount
    zonal.network.turn_threshold_degree = turn_threshold_degree
    zonal.network.knn_weight = knn_weight
    zonal.network.knn_plateau = knn_plateau

    betweenness_output = paralell_betweenness_exposure(
        zonal,
        search_radius=search_radius,
        detour_ratio=detour_ratio,
        decay=decay,
        decay_method=decay_method,
        beta=beta,
        num_cores=num_cores,
        path_detour_penalty='equal', # "power" | "exponent" | "equal"
        closest_destination=closest_destination,
        elastic_weight=elastic_weight,
        turn_penalty=turn_penalty,
        path_exposure_attribute=path_exposure_attribute,
        return_path_record=False, 
        destniation_cap=None, 
    )

    if save_betweenness_as is not None:
        edge_layer_name = zonal.network.edge_source_layer
        edge_gdf = zonal.network.edges

        # if the name is already a column, drop it to avoid 'GeoDataFrame cannot contain duplicated column names' error.
        if save_betweenness_as in zonal[edge_layer_name].gdf.columns:
            zonal[edge_layer_name].gdf.drop(columns=[save_betweenness_as], inplace=True)

        zonal[edge_layer_name].gdf = zonal[edge_layer_name].gdf.join(
        edge_gdf[['parent_street_id', 'betweenness']].drop_duplicates(subset='parent_street_id').set_index('parent_street_id')).rename(
        columns={"betweenness": save_betweenness_as})

        #TODO: The index became a float after te join, probably due to the type of 'parent_street_id' in the edge gdf.
        zonal[edge_layer_name].gdf.index = zonal[edge_layer_name].gdf.index.astype(int)


    if (save_reach_as is not None) or (save_gravity_as is not None) or (save_elastic_weight_as is not None): 
        origin_gdf = betweenness_output['origin_gdf']
        origin_layer = origin_gdf.iloc[0]['source_layer']

        saved_attributes = {}
        if save_reach_as is not None:
            saved_attributes['reach'] = save_reach_as

        if save_gravity_as is not None:
            saved_attributes['gravity'] = save_gravity_as

        if (save_elastic_weight_as is not None) and (elastic_weight):
            saved_attributes['knn_weight'] = save_elastic_weight_as
        
        if (save_path_exposure_as is not None) and (path_exposure_attribute is not None):
            saved_attributes['expected_hazzard_meters'] = save_path_exposure_as

        for key, value in saved_attributes.items():
            origin_gdf[key] = origin_gdf[key].fillna(0)
            if value in zonal[origin_layer].gdf.columns:
                zonal[origin_layer].gdf.drop(columns=[value], inplace=True)


        
        if keep_diagnostics:
            for column_name in origin_gdf.columns:
                if (column_name not in saved_attributes.keys()) and (column_name not in ["source_id"]):
                    saved_attributes[column_name] = save_betweenness_as + "_" + column_name
                    if saved_attributes[column_name] in zonal[origin_layer].gdf.columns:
                        zonal[origin_layer].gdf.drop(columns=[saved_attributes[column_name]], inplace=True)
            zonal[origin_layer].gdf = zonal[origin_layer].gdf.join(origin_gdf.drop(columns=['geometry']).set_index("source_id").rename(columns=saved_attributes))
        else:
            zonal[origin_layer].gdf = zonal[origin_layer].gdf.join(origin_gdf[['source_id'] + list(saved_attributes.keys()) ].set_index("source_id").rename(columns=saved_attributes))

        zonal[origin_layer].gdf.index = zonal[origin_layer].gdf.index.astype(int)
    return


    
import psutil
import time
import concurrent
import multiprocessing as mp
import pandas as pd



def parallel_accessibility(
    zonal: Zonal,
    reach: bool = False,
    gravity: bool =False,
    closest_facility: bool =False,
    weight: str = None,
    search_radius: float=None,
    alpha:float=1,
    beta:float=None
    ):
    node_gdf = zonal.network.nodes
    origins = node_gdf[node_gdf["type"] == "origin"]


    num_procs = psutil.cpu_count(logical=True)

    with concurrent.futures.ProcessPoolExecutor(max_workers=num_procs) as executor:
        with mp.Manager() as manager:
            origin_queue = manager.Queue()

            for o_idx in list(origins.index):
                origin_queue.put(o_idx)
            for core_index in range(num_procs):
                origin_queue.put("done")

            execution_results = []
            for core_index in range(num_procs):
                execution_results.append(
                    executor.submit(
                        single_accessibility,
                        zonal=zonal,
                        reach=reach,
                        gravity=gravity,
                        closest_facility=closest_facility,
                        weight=weight,
                        search_radius=search_radius,
                        alpha=alpha,
                        beta=beta,
                        origin_queue=origin_queue,
                        )
                    )

            # Trackinmg queue progress
            start = time.time()
            while not all([future.done() for future in execution_results]):
                time.sleep(0.5)
                print (f"Time spent: {round(time.time()-start):,}s [Done {max(origins.shape[0] - origin_queue.qsize(), 0):,} of {origins.shape[0]:,} origins ({max(origins.shape[0] - origin_queue.qsize(), 0)/origins.shape[0] * 100:4.2f}%)]",  end='\r')
                for future in [f for f in execution_results if f.done() and (f.exception() is not None)]: # if a process is done and have an exception, raise it
                    raise (future.exception())
           
   
            origin_returns = [result.result() for result in concurrent.futures.as_completed(execution_results)]


    ## find a way to join results back to both the network and original layer?
    x = pd.concat(origin_returns)
    return x



def single_accessibility(
    zonal: Zonal,
    reach: bool = False,
    gravity: bool =False,
    closest_facility: bool =False,
    weight: str = None,
    search_radius: float=None,
    alpha:float=1,
    beta:float=None, 
    origin_queue=None,
    ):
    """
    Modifies the input zonal with accessibility metrics such as `reach`, `gravity`, and `closest_facility` analysis.
        Equivalent of 'una_accessibility' function in madina.py

        Returns:
            A modified Zonal object

        Raises:
            ValueError if `gravity` is True but beta is None/unspecified.
    """
    if gravity and (beta is None):
        raise ValueError("Please specify parameter 'beta' when 'gravity' is True")

    node_gdf = zonal.network.nodes
    origin_gdf = node_gdf[node_gdf["type"] == "origin"]

    reaches = {}
    gravities = {}
    processed_origins = []
    while True:
        o_idx = origin_queue.get()
        if o_idx == "done":
            origin_queue.task_done()
            break
        processed_origins.append(o_idx)
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
    return node_gdf.loc[processed_origins]


