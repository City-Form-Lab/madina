import math
import time
import concurrent
import numpy as np
import networkx as nx
from tqdm import tqdm

from madina.zonal.network import Network
from madina.una.paths import path_generator

def parallel_betweenness(network: Network,
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
                         turn_penalty=False,
                         retained_d_idxs=None,
                         retained_paths=None,
                         retained_distances=None,
                         retain_expensive_data=False
                         ):
    """
    Calculates the betweenness value on the designated network between the given 
    Origin-Destination pair

    Parameters
    ----------
    network: Network
        The network object to run betweenness analysis on
    search_radius: int, defaults to 1000
        The length within which trips will be made from an origin to a destination
    detour_ratio: float, defaults to 1.15
        The ratio between the longest path that will be considered between an O-D pair and the
        shortest path between that O-D pair
    decay: boolean, defaults to ``True``
        Whether or not the total number of trips generated decay over the distance of trips
    decay_method: str in ["exponent", "power"], defaults to "exponent"
        The method in which the trip number decay will take place
    beta: float, defaults to 0.003
        The parameter that controls the speed of the decay over distance
    num_cores: int, defaults to 4
        The number of threads to split the calculation into
    path_detour_penalty: str in ["equal", "exponent", "power"], defaults to "equal"
        The penalty for paths that are longer than the shortest path between an O-D pair
    origin_weights: boolean, defaults to ``False``
        Whether or not origins have individual weights; origins will all be considered equal if
        set to ``False``
    origin_weight_attribute: str, defaults to None
        Which attribute to use as the weight of origins. If None, use the "weight" attribute
    closest_destination: boolean, defaults to ``True``
        Whether or not an origin only generates trips to the closest destination
    destination_weights: boolean, defaults to ``False``
        Whether or not destinations have individual weights; destinations will all be considered
        equal if set to ``False``
    destination_weight_attribute: str, defaults to None
        Which attribute to use as the weight of destinations. If None, use the "weight" attribute
    turn_penalty: boolean, defaults to ``False``
        Whether or not to include turn penalty when calculating the length of a path

    Examples
    --------
    >>> from madina.zonal.zonal import Zonal
    >>> city = Zonal(projected_crs = "EPSG:4326")
    >>> city.load_layer("pedestrian_network", "../data/my_city_pedestrian_network.geojson")
    >>> city.create_street_network(
    ...     source_layer="pedestrian_network",
    ...     node_snapping_tolerance=1.5
    ...     discard_redundant_edges=True,
    ...     turn_threshold_degree=60,
    ...     turn_penalty_amount=42.3
    ... )
    >>> city.load_layer("residential_units", "../data/my_city_residential_units.geojson")
    >>> city.insert_node("residential_units", "origin", weight_attribute="population_census_2020_adjusted")
    >>> city.load_layer("subway_stops", "../data/City_Transit_Stations.geojson")
    >>> city.insert_node("subway_stops", "origin", weight_attribute="ridership_2019")
    >>> betweenness_output = parallel_betweenness(
    ...     city.network,
    ...     search_radius=800,
    ...     decay=True,
    ...     decay_method="exponent",
    ...     beta=0.001,
    ...     path_detour_penalty="equal"
    ...     origin_weights=True,
    ...     closest_destination=False,
    ...     destination_weights=True,
    ...     num_cores=8,
    ...     turn_penalty=True,
    ... )

    Warnings
    --------
    Avoid using the 4 parameters related to retaining elastic weight data as they have not yet
    been thoroughly tested.
    """
    node_gdf = network.nodes
    edge_gdf = network.edges

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
            network,
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
            turn_penalty=turn_penalty,
            retained_d_idxs=retained_d_idxs,
            retained_paths=retained_paths,
            retained_distances=retained_distances,
            rertain_expensive_data=retain_expensive_data
        )
        batch_results = betweennes_results["batch_betweenness_tracker"]
        batch_results = [batch_results]
        if retain_expensive_data:
            retain_paths = betweennes_results["retained_paths"]
            retain_distances = betweennes_results["retained_distances"]
            retain_d_idxs = betweennes_results["retained_d_idxs"]

    else:
        # Parallel ###
        batch_results = []
        diagnostics_gdfs = []
        num_procs = num_cores  # psutil.cpu_count(logical=logical)

        # origins = origins.sample(frac=1)
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
                    network,
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
                    turn_penalty=turn_penalty,
                    retained_d_idxs=retained_d_idxs,
                    retained_paths=retained_paths,
                    retained_distances=retained_distances,
                    rertain_expensive_data=retain_expensive_data
                ) for df in splitted_origins]
            for result in concurrent.futures.as_completed(execution_results):
                try:
                    one_betweennes_output = result.result()
                    batch_results.append(one_betweennes_output["batch_betweenness_tracker"])
                    if retain_expensive_data:
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
    network.nodes = node_gdf
    network.edges = edge_gdf
    return_dict = {"edge_gdf": edge_gdf}
    if retain_expensive_data:
        return_dict["retained_paths"] = retain_paths
        return_dict["retained_distances"] = retain_distances
        return_dict["retained_d_idxs"] = retain_d_idxs
    return return_dict


def one_betweenness_2(
        network: Network,
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
        turn_penalty=False,
        retained_d_idxs=None,
        retained_paths=None,
        retained_distances=None,
        retain_expensive_data=False
):
    """
    TODO: fill out the spec
    """
    
    node_gdf = network.nodes
    edge_gdf = network.edges

    batch_betweenness_tracker = {edge_id: 0 for edge_id in list(edge_gdf.index)}

    counter = 0
    retain_paths = {}
    retain_distances = {}
    retain_d_idxs = {}
    for origin_idx in tqdm(origins.index):
        counter += 1

        if (retained_d_idxs is None) or (retained_paths is None) or (retained_distances is None):
            # paths and weights are dicts: d_idx -> list[path/weight]
            paths, weights, d_idxs = path_generator(network, origin_idx, search_radius=search_radius,
                                                    detour_ratio=detour_ratio, turn_penalty=turn_penalty)
            if retain_expensive_data:
                # print(f"{origin_idx  = } is retaining generated data")
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
            destination_ids = [min(d_idxs, key=d_idxs.get)]
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
                    inner_edge_ids = [network.light_graph.edges[edge]["id"] for edge in inner_path_edges]
                    edge_ids = [node_gdf.at[path[0], 'nearest_edge_id']] + inner_edge_ids + [
                        node_gdf.at[path[-1], 'nearest_edge_id']]
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
                pass
                #print(f"................o: {origin_idx}\td: {destination_idx} faced an error........")
                #print(path)
                #print(e.__doc__)

    print(f"core {origins.iloc[0].name} done.")

    return_dict = {"batch_betweenness_tracker": batch_betweenness_tracker}
    if retain_expensive_data:
        return_dict["retained_paths"] = retain_paths
        return_dict["retained_distances"] = retain_distances
        return_dict["retained_d_idxs"] = retain_d_idxs
    return return_dict