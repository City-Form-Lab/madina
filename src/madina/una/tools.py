import math
import numpy as np
import geopandas as gpd

from shapely import GeometryCollection
from .paths import turn_o_scope, path_generator
from .betweenness import paralell_betweenness_exposure
from ..zonal import Zonal

def validate_zonal_ready(zonal: Zonal):
    if not isinstance(zonal, Zonal):
        raise TypeError(f"Parameter 'zonal' must be {Zonal}. {type(zonal)} was given.")
    
    if len(zonal.layers.layers) == 0:
        raise ValueError("Zonal object does not have layers, call zonal.load_layer(name='layer_name', source='layer_source' to add layers")

    
    node_gdf = zonal.network.nodes

    if (node_gdf is None) or (node_gdf[node_gdf["type"] == "street_node"].shape[0] == 0):
        raise ValueError("Zonal object does not have network nodes and edges, call zonal.create_street_network() first.")

    if node_gdf[node_gdf["type"] == "origin"].shape[0] == 0:
        raise ValueError("Zonal object does not have origin nodes, call zonal.insert_node(layer_name, label='origin') to add origins from layers.")
    
    if node_gdf[node_gdf["type"] == "destination"].shape[0] == 0:
        raise ValueError("Zonal object does not have destination nodes, call zonal.insert_node(layer_name, label='destination') to add destinations from layers.")


    if zonal.network.d_graph is None:
        raise ValueError("Zonal object does not have a d_graph, call zonal..create_graph() to create graphs first.")
    return



def accessibility(
    zonal: Zonal,
    search_radius: float | int,
    destination_weight: str = None,
    alpha:float=1,
    beta:float=None, 
    save_reach_as: str = None, 
    save_gravity_as: str = None,
    closest_facility: bool = False,
    save_closest_facility_as: str = None, 
    save_closest_facility_distance_as: str = None, 
    turn_penalty: bool = False,
) -> None:
    """Measures accessibility metrics like reach and gravity to reachable destinations within a search radius to all origins in the network. 

    :param zonal: A zonal object populated with a network, origins, destinations and a graph
    :type zonal: Zonal
    :param search_radius: the maximum search distance for accessible destinations from an origin, measured as a network distance in the same units as the network's CRS, defaults to None
    :type search_radius: float, optional
    :param destination_weight: destination weight, must be an attribute in the destination layer, defaults to None: fall back to the destination weight specified during network construction which is 1 for all destinations by default.
    :type weight: str, optional
    :param alpha: in gravity calculations, the alpha term increases the importance of destination weight by applying a power. default is 1, destination weight is not adjusted, defaults to 1
    :type alpha: float, optional
    :param beta: In gravity calculations, the beta parameter represent the sensitivity to walk. a smaller beta value means that people are less sensitive to walking. When units are in meters, a typical beta value ranges between 0.001 (Low sensitivity) and 0.004 (High sensitivity), defaults to None
    :type beta: float, optional
    :param save_reach_as: Save the reach metric back to the origin layer as a column with this name, defaults to None
    :type save_reach_as: str, optional
    :param save_gravity_as: Save the gravity metric back to the origin layer as a column with this name, defaults to None
    :type save_gravity_as: str, optional
    :param closest_facility: restrict reach and access such that a destination is assigned its closest origin, defaults to False
    :type closest_facility: bool, optional
    :param save_closest_facility_as: if closest_facility=True, save the closest origin ID as a column in the destination layer with this name, defaults to None
    :type save_closest_facility_as: str, optional
    :param save_closest_facility_distance_as: if closest_facility=True, save thw distance to the closest origin as a column in the destination layer with this name, defaults to None
    :type save_closest_facility_distance_as: str, optional
    :param turn_penalty: If True, turn penalty is enabled, defaults to False. Uses the turn penalty amount and turn degree threshold specified in the network
    :type turn_penalty: bool, optional
    """    
    node_gdf = zonal.network.nodes
    origin_gdf = node_gdf[node_gdf["type"] == "origin"]
    origin_layer = origin_gdf.iloc[0]['source_layer']
    destination_gdf = node_gdf[node_gdf["type"] == "destination"]
    destination_layer = destination_gdf.iloc[0]['source_layer']

    validate_zonal_ready(zonal)

    if not isinstance(search_radius, (int, float)):
        raise TypeError(f"Parameter 'search_radius' must be either {int, float}. {type(search_radius)} was given.")
    elif search_radius < 0:
        raise ValueError(f"Parameter 'search_radius': Cannot be negative. search_radius={search_radius} was given.")


    if destination_weight is not None:
        if not isinstance(destination_weight, str):
            raise TypeError(f"Parameter 'destination_weight' must be {str}. {type(destination_weight)} was given.")
        elif destination_weight not in zonal[destination_layer].gdf.columns:
            raise ValueError(f"Parameter 'destination_weight': {destination_weight} not in layer {destination_layer}. Available attributes are: {list(zonal[destination_layer].gdf.columns)}")

    if not isinstance(alpha, (int, float)):
        raise TypeError(f"Parameter 'alpha' must be either {int, float}. {type(alpha)} was given.")
    
    if (beta is not None) and (not isinstance(beta, (int, float))):
        raise TypeError(f"Parameter 'beta' must be either {int, float}. {type(beta)} was given.")

    if (save_reach_as is not None) and (not isinstance(save_gravity_as, str)):
        raise TypeError(f"Parameter 'save_reach_as' must be a string. {type(save_reach_as)} was given.")

    if (save_gravity_as is not None): 
        if not isinstance(save_gravity_as, str):
            raise TypeError(f"Parameter 'save_gravity_as' must be a string. {type(save_gravity_as)} was given.")
        if (beta is None):
            raise ValueError("Please specify parameter 'beta' when 'save_gravity_as' is provided")
         
    if not isinstance(closest_facility, bool):
        raise TypeError(f"Parameter 'closest_facility' must either be a boolean True or False, {type(closest_facility)} was given.")
    elif not closest_facility:
        if save_closest_facility_as is not None:
            raise ValueError("Please set parameter 'closest_facility=True' when 'save_closest_facility_as' is provided")
        if save_closest_facility_distance_as is not None:
            raise ValueError("Please set parameter 'closest_facility=True' when 'save_closest_facility_distance_as' is provided")
    else:
        if not isinstance(save_closest_facility_as, str):
            raise TypeError(f"Parameter 'save_closest_facility_as' must be a string. {type(save_closest_facility_as)} was given.")
        
        if not isinstance(save_closest_facility_distance_as, str):
            raise TypeError(f"Parameter 'save_closest_facility_distance_as' must be a string. {type(save_closest_facility_distance_as)} was given.")

    if not isinstance(turn_penalty, bool):
        raise TypeError(f"Parameter 'turn_penalty' must either be a boolean True or False, {type(turn_penalty)} was given.")

    
    
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
            d_weight = 1 if destination_weight is None else zonal.layers[source_layer].gdf.at[source_id, destination_weight]

            reaches[o_idx][d_idx] = d_weight
            if beta is not None:
                gravities[o_idx][d_idx] = pow(d_weight, alpha) / (pow(math.e, (beta * d_idxs[d_idx])))

            if closest_facility:
                if ("closest_facility" not in node_gdf.loc[d_idx]) or (np.isnan(node_gdf.loc[d_idx]["closest_facility"])):
                    node_gdf.at[d_idx, "closest_facility"] = o_idx
                    node_gdf.at[d_idx, "closest_facility_distance"] = d_idxs[d_idx]
                elif d_idxs[d_idx] < node_gdf.at[d_idx, "closest_facility_distance"]:
                    node_gdf.at[d_idx, "closest_facility"] = o_idx
                    node_gdf.at[d_idx, "closest_facility_distance"] = d_idxs[d_idx]


        node_gdf.at[o_idx, "reach"] = sum([value for value in reaches[o_idx].values() if not np.isnan(value)])

        if beta is not None:
            node_gdf.at[o_idx, "gravity"] = sum([value for value in gravities[o_idx].values() if not np.isnan(value)])

    if closest_facility:
        for o_idx in origin_gdf.index:
            o_closest_destinations = list(node_gdf[node_gdf["closest_facility"] == o_idx].index)
            node_gdf.at[o_idx, "reach"] = sum([reaches[o_idx][d_idx] for d_idx in o_closest_destinations])
            if beta is not None:
                node_gdf.at[o_idx, "gravity"] = sum([gravities[o_idx][d_idx] for d_idx in o_closest_destinations])
    



    # todo: isolate this into network.save_from_nodes_to_layer(zonal, layer_name, name_map={result:param_name..})
    if (save_reach_as is not None) or (save_gravity_as is not None): 

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
    search_radius: float, 
    origin_ids: int | list = None,
    turn_penalty: bool = False,
):
    """For each origin, generate a polygon of its service area, defined by the boundary around the destinations it could reach within a specified searchj radius. A list of destinations is returned, as well as the network geometry inside the service area

    :param zonal: A zonal object populated with a network, origins, destinations and a graph
    :type zonal: Zonal
    :param origin_ids: If not provided, the service area for all origins is generated. This parameter can either be the id of an origin as an integer, or a list of origin IDs
    :type origin_ids: int | list
    :param search_radius: the maximum search distance for accessible destinations from an origin, measured as a network distance in the same units as the network's CRS, defaults to None
    :type search_radius: float | int
    :param turn_penalty: If True, turn penalty is enabled, defaults to False
    :type turn_penalty: bool, optional
    :raises ValueError: if an origin id is not for an origin in the origin layer.
    :return: 
        - `destinations`: A dataframe howing a list of destinations for each origin that falls within the search radius
        - `network_edges`: The network geometry that falls withn the service area of all origins
        - `scope_gdf`: for each origin, a polygon of its service ares
    :rtype: GeoDataFrame
    """

    validate_zonal_ready(zonal)

    if not isinstance(search_radius, (int, float)):
        raise TypeError(f"Parameter 'search_radius' must be either {int, float}. {type(search_radius)} was given.")
    elif search_radius < 0:
        raise ValueError(f"Parameter 'search_radius': Cannot be negative. search_radius={search_radius} was given.")

    if (origin_ids is not None) and not isinstance(origin_ids, (int, list)):
        raise TypeError(f"Parameter 'origin_ids' must be either {int, list} representing an origin id or a list of origin ids. {type(origin_ids)} was given.")


    if not isinstance(turn_penalty, bool):
        raise TypeError(f"Parameter 'turn_penalty' must either be a boolean True or False, {type(turn_penalty)} was given.")


    node_gdf = zonal.network.nodes
    edge_gdf = zonal.network.edges

    origin_gdf = node_gdf[node_gdf["type"] == "origin"]
    origin_layer = origin_gdf.iloc[0]['source_layer']

    destination_gdf = node_gdf[node_gdf["type"] == "destination"]
    destination_layer = destination_gdf.iloc[0]['source_layer']
    

    if origin_ids is None:
        o_idxs = list(origin_gdf.index)
    elif isinstance(origin_ids, int):
        # put user input in a list
        o_idxs = list(origin_gdf[origin_gdf['source_id'].isin([origin_ids])].index)
    elif len(set(origin_ids) - set(zonal[origin_layer].gdf.index)) != 0:
        raise ValueError(f"some of the indices given in `origin_ids` are not for an origin {set(origin_ids) - set(zonal[origin_layer].gdf.index)}")
    else: 
        ## convert user input to network IDs
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



def alternative_paths(
    zonal: Zonal,
    origin_id: int,
    search_radius: float | int,
    detour_ratio: float | int = 1,
    turn_penalty: bool = False,
):
    """Generates all alternative patha between an origin and all reachable destinations within a detour from the shortest path

    :param zonal: A zonal object populated with a network, origins, destinations and a graph
    :type zonal: Zonal
    :param origin_id: an ID for the origin used as a start. Must be an ID from the origin layer. 
    :type origin_id: int
    :param search_radius: The maximum distance to search for reachable destinatations. In the same unit as the network CRS.
    :type search_radius: float | int
    :param detour_ratio: A percentage of detour over the shortest path between an origin and a destination when generating alternative paths. , defaults to 1 and only generates the shortest path. Must be greater than or equal to one. if set to a large number, could result in severe performance issues and memory overflow
    :type detour_ratio: float | int, optional
    :param turn_penalty: If True, turn penalty is enabled, defaults to False
    :type turn_penalty: bool, optional
    :return: This function returns a GeoDataFrame of all paths generated to all reachable destinations. The GeoDataFrame has three columns: 
        - `destination`: the destination ID where a path ends
        - `distance`: the weight of this path: reflecting the network settings for network weight, and turn penalty.
        ` `geometry` a column of Shapely GeometryCollection containing all network segments along the path. origin and destination segments are not trimmed but returned whole.
    :rtype: GeoDataFrame
    """
    origin_gdf = zonal.network.nodes[zonal.network.nodes['type'] == 'origin']


    if not isinstance(origin_id, (int, list)):
        raise TypeError(f"Parameter 'origin_id' must be {int} representing an origin id. {type(origin_id)} was given.")
    if origin_id not in origin_gdf['source_id']:
        raise ValueError(f"Parameter 'origin_id': is not for an origin included in the network.")

    if not isinstance(search_radius, (int, float)):
        raise TypeError(f"Parameter 'search_radius' must be either {int, float}. {type(search_radius)} was given.")
    elif search_radius < 0:
        raise ValueError(f"Parameter 'search_radius': Cannot be negative. search_radius={search_radius} was given.")

    if not isinstance(detour_ratio, (int, float)):
        raise TypeError(f"Parameter 'detour_ratio' must be either {int, float}. {type(detour_ratio)} was given.")
    elif detour_ratio < 1:
        raise ValueError(f"Parameter 'detour_ratio': Cannot be less than 1. detour_ratio={detour_ratio} was given.")

    if not isinstance(turn_penalty, bool):
        raise TypeError(f"Parameter 'turn_penalty' must either be a boolean True or False, {type(turn_penalty)} was given.")



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


    destination_gdf = gpd.GeoDataFrame({'destination': destination_list, 'distance': distance_list, 'geometry': path_geometries}, crs = zonal.network.nodes.crs)
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
    knn_plateau: float | int = 0, 
    turn_penalty: bool = False,
    save_betweenness_as: str = None, 
    save_reach_as: str = None, 
    save_gravity_as: str = None,
    save_elastic_weight_as: str = None,
    keep_diagnostics: bool = False, 
    path_exposure_attribute: str = None,
    save_path_exposure_as: str = None,
):
    """Generate trips between origins and destinations along network segment, accounting for a search radius, decay, detour, destination competition, turn penalty and elastic trip generation.

    :param zonal: A zonal object populated with a network, origins, destinations and a graph
    :type zonal: Zonal
    :param search_radius: The maximum distance to search for reachable destinatations. In the same unit as the network CRS.
    :type search_radius: float
    :param detour_ratio: A percentage of detour over the shortest path between an origin and a destination when allocating trips across alternative paths. Defaults to 1 and only allocate trips along the shortest path. Must be greater than or equal to one. if set to a large number, could result in severe performance issues and memory overflow
    :type detour_ratio: float, optional
    :param decay: If ennabled, trip generation is decayed according to the chosen decay function and beta parameter, defaults to False
    :type decay: bool, optional
    :param decay_method: the function that applies distance decay to trips. could be one of ['exponent', 'power'], defaults to "exponent"
    :type decay_method: str, optional
    :param beta: When applying decay to trip generation, the beta parameter represent the sensitivity to walk. a smaller beta value means that people are less sensitive to walking. When units are in meters, a typical beta value ranges between 0.001 (Low sensitivity) and 0.004 (High sensitivity), defaults to 0.003
    :type beta: float, optional
    :param num_cores: By default, only use a single core, set to as many cores as you want to use for running parallel calculations., defaults to 1
    :type num_cores: int, optional
    :param closest_destination: If set to true, trips are only routed to the closest destination. if set to false, destinations compete to attract trips based on the Huff model that factors in destination attractivenes and distance, defaults to True
    :type closest_destination: bool, optional
    :param elastic_weight: If set to false, origins generate theur full trip potential irrespective of how many destinations they can access. if set to true, origins generate more trips as they have access to more destinations, as defined by the K-nearest neighbor access, defaults to False
    :type elastic_weight: bool, optional
    :param knn_weight: The K-nesrest neighbor access array, should be of the form [0.5, 0.25, ..., 0.01] to assign partial score foe each reachable destination. Should add up to one so it controls trip generation appropriatly, defaults to None
    :type knn_weight: str | list, optional
    :param knn_plateau: A distance penalty could be applied to the KNN access score depending on how close the destination is. the KNN plateau gives a penalty-free score sccumilation for destinations that are closer than the plateau, and applies penalty on the distance that exceeds the plateau, defaults to 0
    :type knn_plateau: float | int, optional
    :param turn_penalty: _description_, defaults to False
    :type turn_penalty: bool, optional
    :param save_betweenness_as: Specify a name for the column in the network layer where the betweenness flow is stored, defaults to None
    :type save_betweenness_as: str, optional
    :param save_reach_as: Specify a name for thecolumn in the origin layer where the reach accessibility score is stored , defaults to None
    :type save_reach_as: str, optional
    :param save_gravity_as: Specify a name for the column in the origin layer where the gravity score is stored, defaults to None
    :type save_gravity_as: str, optional
    :param save_elastic_weight_as: specify a name for the column in the origin layer where the KNN-adjusted origin weight is stored, defaults to None
    :type save_elastic_weight_as: str, optional
    :param keep_diagnostics: If set to true, store performance and memory statistics in the network, defaults to False
    :type keep_diagnostics: bool, optional
    :param path_exposure_attribute: If provided, calculates an exposure to a network value for trips originatinbg from an origin en route to destinations, defaults to None
    :type path_exposure_attribute: str, optional 
    :param save_path_exposure_as: if path exposure attribute is proviided, this is a name for a column in the origin layer that captures origin's exposure to the network exposure attribute, defaults to None
    :type save_path_exposure_as: str, optional
    """

    validate_zonal_ready(zonal)

    if not isinstance(search_radius, (int, float)):
        raise TypeError(f"Parameter 'search_radius' must be either {int, float}. {type(search_radius)} was given.")
    elif search_radius < 0:
        raise ValueError(f"Parameter 'search_radius': Cannot be negative. search_radius={search_radius} was given.")

    if not isinstance(detour_ratio, (int, float)):
        raise TypeError(f"Parameter 'detour_ratio' must be either {int, float}. {type(detour_ratio)} was given.")
    elif detour_ratio < 1:
        raise ValueError(f"Parameter 'detour_ratio': Cannot be less than 1. detour_ratio={detour_ratio} was given.")

    if not isinstance(decay, bool):
        raise TypeError(f"Parameter 'decay' must either be a boolean True or False, {type(decay)} was given.")
    
    if decay and (decay_method not in ['exponent', 'power']):
        if not isinstance(decay_method, str):
            raise TypeError(f"Parameter 'decay_method' must be a string. {type(decay_method)} was given.")
        else: 
            raise ValueError(f"Parameter 'decay_method': must be one of ['exponent', 'power']. node_snapping_tolerance={decay_method} was given.")

    if (decay or save_gravity_as is not None) and (not isinstance(beta, (int, float))):
        raise TypeError(f"Parameter 'beta' must be either {int, float}. {type(beta)} was given.")

    if not isinstance(num_cores, int):
        raise TypeError(f"Parameter 'num_cores' must be {int}. {type(num_cores)} was given.")
    elif num_cores < 1:
        raise ValueError(f"Parameter 'num_cores': Cannot be less than 1. num_cores={num_cores} was given.")
    
    if not isinstance(closest_destination, bool):
        raise TypeError(f"Parameter 'closest_destination' must either be a boolean True or False, {type(closest_destination)} was given.")
    
    if not isinstance(elastic_weight, bool):
        raise TypeError(f"Parameter 'elastic_weight' must either be a boolean True or False, {type(elastic_weight)} was given.")


    if elastic_weight:
        if knn_weight is None:
            raise ValueError(f"Parameter 'elastic_weight': must be provided if `elastic_weight=True`")
        if not isinstance(knn_weight, (str, list)):
            raise TypeError(f"Parameter 'beta' must be either {int, float}. {type(beta)} was given.")

        if not isinstance(knn_plateau, (int, float)):
            raise TypeError(f"Parameter 'knn_plateau' must be either {int, float}. {type(knn_plateau)} was given.")
        if knn_plateau < 0:
            raise ValueError(f"Parameter 'knn_plateau': Cannot be negative. knn_plateau={knn_plateau} was given.")


    if not isinstance(turn_penalty, bool):
        raise TypeError(f"Parameter 'turn_penalty' must either be a boolean True or False, {type(turn_penalty)} was given.")

    if (save_betweenness_as is not None) and not isinstance(save_betweenness_as, str):
        raise TypeError(f"Parameter 'save_betweenness_as' must be a string. {type(save_betweenness_as)} was given.")
    
    if (save_reach_as is not None) and not isinstance(save_reach_as, str):
        raise TypeError(f"Parameter 'save_reach_as' must be a string. {type(save_reach_as)} was given.")

    if (save_gravity_as is not None) and not isinstance(save_gravity_as, str):
        raise TypeError(f"Parameter 'save_gravity_as' must be a string. {type(save_gravity_as)} was given.")
    
    if (save_gravity_as is not None) and not isinstance(save_gravity_as, str):
        raise TypeError(f"Parameter 'save_gravity_as' must be a string. {type(save_gravity_as)} was given.")
    
    if save_elastic_weight_as is not None:
        if not isinstance(save_elastic_weight_as, str):
            raise TypeError(f"Parameter 'save_elastic_weight_as' must be a string. {type(save_elastic_weight_as)} was given.")
        if not elastic_weight:
            raise ValueError(f"Parameter 'elastic_weight': must be set to True if `save_elastic_weight_as` is provided")


    if not isinstance(keep_diagnostics, bool):
        raise TypeError(f"Parameter 'keep_diagnostics' must either be a boolean True or False, {type(keep_diagnostics)} was given.")


    if path_exposure_attribute is not None:
        if not isinstance(path_exposure_attribute, str):
            raise TypeError(f"Parameter 'path_exposure_attribute' must be a string. {type(path_exposure_attribute)} was given.")
        
        if path_exposure_attribute not in zonal[zonal.network.edge_source_layer].gdf.columns:
            raise ValueError(f"Parameter 'path_exposure_attribute' not in layer {zonal.network.edge_source_layer}'s columns. options are: {list(zonal[zonal.network.edge_source_layer].gdf.columns)}")

    if save_path_exposure_as is not None:
        if not isinstance(save_path_exposure_as, str):
            raise TypeError(f"Parameter 'save_path_exposure_as' must be a string. {type(save_path_exposure_as)} was given.")
        if path_exposure_attribute is None:
            raise ValueError(f"Parameter 'path_exposure_attribute' must be provided if `save_path_exposure_as` is provided")

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


