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
                         perceived_distance=False,
                         light_graph=True,
                         turn_penalty=False,
                         retained_d_idxs=None,
                         retained_paths=None,
                         retained_distances=None,
                         rertain_expensive_data=False
                         ):
    """
    TODO: fill out the spec
    """
    node_gdf = network.nodes['gdf']
    edge_gdf = network.edges['gdf']

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
    network.nodes['gdf'] = node_gdf
    network.edges['gdf'] = edge_gdf
    return_dict = {"edge_gdf": edge_gdf}
    if rertain_expensive_data:
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
        perceived_distance=False,
        light_graph=True,
        turn_penalty=False,
        retained_d_idxs=None,
        retained_paths=None,
        retained_distances=None,
        rertain_expensive_data=False
):
    """
    TODO: fill out the spec
    """
    
    node_gdf = network.nodes["gdf"]
    edge_gdf = network.edges["gdf"]

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

            paths, weights, d_idxs = path_generator(network, origin_idx, search_radius=search_radius,
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
            #destination_ids = [int(origins.at[origin_idx, "closest_destination"])]
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
                pass
                #print(f"................o: {origin_idx}\td: {destination_idx} faced an error........")
                #print(path)
                #print(e.__doc__)

    print(f"core {origins.iloc[0].name} done.")

    return_dict = {"batch_betweenness_tracker": batch_betweenness_tracker}
    if rertain_expensive_data:
        return_dict["retained_paths"] = retain_paths
        return_dict["retained_distances"] = retain_distances
        return_dict["retained_d_idxs"] = retain_d_idxs
    return return_dict