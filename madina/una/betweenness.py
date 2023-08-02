import math
import time
import concurrent
import numpy as np
import networkx as nx
from tqdm import tqdm

from madina.zonal.network import Network
from madina.zonal.zonal import Zonal
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
    TODO: fill out the specE
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
        #print("-------------------------------------------")
        #print(f"All cores done in {round(end - start, 2)}")
        #print("-------------------------------------------")

    for idx in edge_gdf.index:
        edge_gdf.at[idx, "betweenness"] = sum([batch[idx] for batch in batch_results])

    # not sure if this assignment is necessary,
    network.nodes = node_gdf
    network.edges = edge_gdf
    return_dict = {"edge_gdf": edge_gdf}
    if rertain_expensive_data:
        return_dict["retained_paths"] = retain_paths
        return_dict["retained_distances"] = retain_distances
        return_dict["retained_d_idxs"] = retain_d_idxs
    return return_dict



import warnings 
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
    
    node_gdf = network.nodes
    edge_gdf = network.edges

    # graph = self.G.copy()

    batch_betweenness_tracker = {edge_id: 0 for edge_id in list(edge_gdf.index)}

    counter = 0
    retain_paths = {}
    retain_distances = {}
    retain_d_idxs = {}
    #for origin_idx in tqdm(origins.index):
    for origin_idx in origins.index:
        counter += 1

        if origins.at[origin_idx, 'weight'] == 0:
            #print (f"{origin_idx = } has weight of 0, skipping." )
            continue

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
            #print(f"o:{origin_idx} has no destinations")
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
                    edge_ids = [node_gdf.at[path[0], 'nearest_edge_id']] + inner_edge_ids + [
                        node_gdf.at[path[-1], 'nearest_edge_id']]
                    # this_path_weight = path_weight(graph, path, weight="weight")

                    if this_path_weight > shortest_path_distance * detour_ratio:
                        # TODO: trailblazer algorithm sometimes produces paths that exceeds this limin. revisit
                        #print(
                        #   f"o: {origin_idx}\td:{destination_idx}\t{path}\{this_path_weight}\t "
                        #   f"exceeded limit {shortest_path_distance * detour_ratio}")
                        continue

                    if this_path_weight < 0.01:
                        # to solve issue with short paths, or if o/d were on the same location.
                            this_path_weight = 0.01
                    this_od_paths["path_length"].append(this_path_weight)
                    this_od_paths["one_over_squared_length"].append(1 / (this_path_weight ** 2))
                    this_od_paths["one_over_e_beta_length"].append(1 / pow(math.e, (beta * this_path_weight)))
                    this_od_paths["path_edges"].append(edge_ids)
                # TODO:  change of name to match the attribute. gotta fix later..
                if decay_method == "exponent":
                    decay_method = "one_over_e_beta_length"
                elif decay_method == "power":
                    decay_method = "one_over_squared_length"
                
                sum_detour_penalty = 0
                if path_detour_penalty == "exponent":
                    sum_detour_penalty = sum(this_od_paths["one_over_e_beta_length"])
                elif path_detour_penalty == "power":
                    sum_detour_penalty = sum(this_od_paths["one_over_squared_length"])

                this_od_paths["probability"] = []

                # for one_over_squared_length in this_od_paths[path_detour_penalty]:
                for seq, path_edges in enumerate(this_od_paths["path_edges"]):

                    if path_detour_penalty == "equal":
                        this_od_paths["probability"].append(
                            1 / len(this_od_paths["path_edges"])
                        )
                    elif path_detour_penalty == "exponent":
                        this_od_paths["probability"].append(
                            this_od_paths['one_over_e_beta_length'][seq] / sum_detour_penalty
                        )
                    elif path_detour_penalty == "power":
                        this_od_paths["probability"].append(
                            this_od_paths['one_over_squared_length'][seq] / sum_detour_penalty
                        )

                        #this_od_paths["probability"].append(
                        #    this_od_paths[path_detour_penalty][seq] / sum_one_over_squared_length
                        #)
                        #warnings.filterwarnings("error")
                        #try:

                        #except:
                        #    print (this_od_paths[path_detour_penalty])
                        #    print (this_od_paths[path_detour_penalty][seq])
                        #    print (sum_one_over_squared_length)
                        #    print (f"{sum(this_od_paths[path_detour_penalty]) = }")
                        '''
                        with warnings.catch_warnings():
                            warnings.filterwarnings("ignore")
                            this_od_paths["probability"].append(
                                this_od_paths[path_detour_penalty][seq] / sum_one_over_squared_length
                            )
                            for warning in warnings:
                                print ("here")
                                print(warning)
                                #print (f"{this_od_paths[path_detour_penalty][seq] = }\t{sum_one_over_squared_length = }")
                                '''


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
                pass



    #print(f"core {origins.iloc[0].name} done.")

    return_dict = {"batch_betweenness_tracker": batch_betweenness_tracker}
    if rertain_expensive_data:
        return_dict["retained_paths"] = retain_paths
        return_dict["retained_distances"] = retain_distances
        return_dict["retained_d_idxs"] = retain_d_idxs
    return return_dict

def betweenness_exposure(
        self: Zonal,
        origins=None,
        search_radius=1000,
        detour_ratio=1.05,
        decay=True,
        beta=0.003,
        decay_method="exponent",  # "exponent" | "power"
        path_detour_penalty="equal",  # "exponent" | "power" | "equal"
        closest_destination=False,
        elastic_weight=False,
        turn_penalty=False,
        path_exposure_attribute=None,
        return_path_record=False, 
        destniation_cap=None
):
    edge_gdf = self.network.edges
    node_gdf = self.network.nodes

    batch_betweenness_tracker = {edge_id: 0 for edge_id in list(edge_gdf.index)}

    if return_path_record:
        path_record = {
            'origin_id': [],
            'destination_id': [],
            'destination_probability': [],
            'path_id': [],
            'path_weight': [],
            'path_decay': [],
            'path_probability': [],
            'path_betweenness': [],
        }
        if path_exposure_attribute is not None:
            path_record['mean_path_exposure'] = []

    origin_count = 0
    for origin_idx in origins.index:

        origin_mean_hazzard = 0
        origin_decayed_mean_hazzard = 0
        expected_hazzard_meters = 0
        probable_travel_distance_weighted_hazzard = 0

        origin_mean_path_length = 0
        probable_travel_distance = 0

        origin_count += 1
        if (origin_count % 100 == 0):
            print(f"core {origins.iloc[0].name}\t {origin_count = }]\t{origin_count / len(origins.index) * 100:6.4f}")


        if origins.at[origin_idx, "weight"] == 0:
            continue

        # finding all destination reachible from this origin within a search radius, and finding all paths within a derour ration from the shortest path
        try:
            paths, weights, d_idxs = best_path_generator_so_far(
                self,
                origin_idx,
                search_radius=search_radius,
                detour_ratio=detour_ratio,
                turn_penalty=turn_penalty
            )
        except:
            continue
        
        # this makes sure destinations are sorted based on distance. (sorting the dictionary nased on its values.)
        d_idxs = dict(sorted(d_idxs.items(), key=lambda item: item[1]))



        # skip this origin if cannot reach any destination
        if len(d_idxs) == 0:
            continue

        if elastic_weight:
            origin_weight_attribute = 'knn_weight'
            get_origin_properties(
                self,
                search_radius=800,
                beta=0.001,
                turn_penalty=False,
                o_idx=None,
                o_graph=None,
                d_idxs=None,
                knn_weights=None, 
                knn_plateau=400
            )


        if closest_destination:
            # just use the closest destination.
            destination_probabilities = [1]
            destination_ids = [min(d_idxs, key=d_idxs.get)]
        else:
            # probability of choosing a destination using the huff model
            eligible_destinations = d_idxs if destniation_cap is None else dict(list(d_idxs.items())[:destniation_cap])
            destination_gravities = [node_gdf.at[d_idx, 'weight'] / pow(math.e, (beta * od_shortest_distance)) for d_idx, od_shortest_distance in eligible_destinations.items()]

            destination_probabilities = np.array(destination_gravities) / sum(destination_gravities)
            destination_ids = list(eligible_destinations.keys())

        if sum(destination_gravities) == 0:
            # This covers a case where all reachible destinations have a weight of 0, this would causea a 0 denominator. assune equal probability instread
            destination_probabilities = np.ones(len(eligible_destinations)) / len(eligible_destinations)

        
        for destination_idx, this_destination_probability in zip(node_gdf.loc[destination_ids].index, destination_probabilities):

            # skip this destination if cannot find paths from origin
            if len(paths[destination_idx]) == 0:
                print(f"o:{origin_idx}\td:{destination_idx} have no paths...")
                continue

            # finding path probabilities given different penalty settings.
            path_detour_penalties = np.zeros(len(weights[destination_idx]))
            d_path_weights = np.array(weights[destination_idx])

            if path_detour_penalty == "exponent":
                path_detour_penalties = 1 / pow(np.e, 0.025 * d_path_weights)
            elif path_detour_penalty == "power":
                path_detour_penalties = 1 / (d_path_weights ** 2)
            elif path_detour_penalty == "equal":
                path_detour_penalties = np.ones(len(weights[destination_idx]))
            else:
                raise ValueError(
                    f"parameter 'path_detour_penalty' should be one of ['equal', 'power', 'exponent'], '{path_detour_penalty}' was given")
            path_probabilities = path_detour_penalties / sum(path_detour_penalties)

            od_shortest_path_distance = min(weights[destination_idx])

            for path_id, (path, this_path_weight, this_path_probability) in enumerate(
                    zip(paths[destination_idx], weights[destination_idx], path_probabilities)):
                # trailblazer algorithm sometimes produces paths that exceeds this limin. revisit later, for now, skip those paths exceeding limit
                if this_path_weight > od_shortest_path_distance * detour_ratio:
                    pass

                # to solve numeric issues with short paths, for example, if o/d were on the same location
                if this_path_weight < 1:
                    this_path_weight = 1


                # constructing this path's betweenness contribution. Accounting for closest destination or compitition, origin weight and decay based on path weight
                
                path_decay = 1
                if decay:
                    if decay_method == "exponent":
                        path_decay = (1 / pow(math.e, (beta * this_path_weight)))
                    elif decay_method == "power":
                        path_decay = 1 / (this_path_weight ** 2)
                    else:
                        raise ValueError(
                            f"parameter 'decay_method' should be one of ['exponent', 'power'], '{decay_method}' was given")

                betweennes_contribution = this_path_probability * this_destination_probability * path_decay * origins.at[origin_idx, origin_weight_attribute]

                # paths are returned as a series of nodes, herem, we extract a list of edges along the path: edge_ids
                inner_path_edges = list(nx.utils.pairwise(path[1:-1]))
                inner_edge_ids = [self.G.edges[edge]["id"] for edge in inner_path_edges]
                edge_ids = [node_gdf.at[path[0], 'nearest_street_id']] + inner_edge_ids + [
                    node_gdf.at[path[-1], 'nearest_street_id']]

                path_weight_exposure = 0
                path_weight_sum = 0
                for edge_id in edge_ids:
                    batch_betweenness_tracker[edge_id] += betweennes_contribution
                    if path_exposure_attribute is not None:
                        segment_weight = edge_gdf.at[int(edge_id), 'weight']
                        path_weight_sum += segment_weight
                        path_weight_exposure += segment_weight * edge_gdf.at[int(edge_id), path_exposure_attribute]
                

                destination_path_probability = this_path_probability * this_destination_probability
                
                origin_mean_path_length += destination_path_probability * this_path_weight
                probable_travel_distance += destination_path_probability * path_decay * this_path_weight


                if path_exposure_attribute is not None:
                    path_mean_exposure = path_weight_exposure / path_weight_sum

                    origin_mean_hazzard += destination_path_probability * path_mean_exposure
                    origin_decayed_mean_hazzard += destination_path_probability * path_decay * path_mean_exposure
                    expected_hazzard_meters += path_mean_exposure * destination_path_probability * this_path_weight
                    probable_travel_distance_weighted_hazzard += destination_path_probability * path_decay * path_mean_exposure * this_path_weight

                if return_path_record:
                    path_record['origin_id'].append(origin_idx)
                    path_record['destination_id'].append(destination_idx)
                    path_record['destination_probability'].append(this_destination_probability)
                    path_record['path_id'].append(path_id)
                    path_record['path_weight'].append(this_path_weight)
                    path_record['path_decay'].append(path_decay)
                    path_record['path_probability'].append(this_path_probability)
                    path_record['path_betweenness'].append(betweennes_contribution)
                    if path_exposure_attribute is not None:
                        path_record['mean_path_exposure'].append(path_mean_exposure)


                        
        
        if path_exposure_attribute is not None:
            origins.at[origin_idx, 'mean_hazzard'] = origin_mean_hazzard
            origins.at[origin_idx, 'decayed_mean_hazzad'] = origin_decayed_mean_hazzard
            origins.at[origin_idx, 'expected_hazzard_meters'] = expected_hazzard_meters
            origins.at[origin_idx, 'probable_travel_distance_weighted_hazzard'] = probable_travel_distance_weighted_hazzard
        
        origins.at[origin_idx, 'shortest_path_length'] = list(eligible_destinations.values())[0]
        origins.at[origin_idx, 'mean_path_length'] = origin_mean_path_length
        origins.at[origin_idx, 'probable_travel_distance'] = probable_travel_distance
        origins.at[origin_idx, 'reach'] = len(d_idxs)
        origins.at[origin_idx, 'eligible_destinations'] = len(eligible_destinations)
        
    print(f"core {origins.iloc[0].name} done.")

    return_dict = {"batch_betweenness_tracker": batch_betweenness_tracker, 'origins': origins}
    if return_path_record:
        return_dict["path_record"] = path_record
    return return_dict

def paralell_betweenness_exposure(
    self: Zonal,
    search_radius=1000,
    detour_ratio=1.05,
    decay=True,
    decay_method="exponent",
    beta=0.003,
    num_cores=4,
    path_detour_penalty="equal",
    closest_destination=True,
    elastic_weight=False,
    turn_penalty=False,
    path_exposure_attribute=None,
    return_path_record=False, 
    destniation_cap=None
    ):
    node_gdf = self.netwoek.nodes
    edge_gdf = self.network.edges

    origins = node_gdf[node_gdf["type"] == "origin"]


    list_of_path_dfs = []
    edge_gdf['betweenness'] = 0.0
    batch_results = []
    origin_returns = []
    num_procs = num_cores  # psutil.cpu_count(logical=logical)

    #TODO: investigate if this randomazation is causing any issues downsream.
    origins = origins.sample(frac=1)
    origins.index = origins.index.astype("int")
    splitted_origins = np.array_split(origins, num_procs)





    start = time.time()
    with concurrent.futures.ProcessPoolExecutor(max_workers=num_procs) as executor:
        # print("entered parallel execution")
        execution_results = [
            executor.submit(
                betweenness_exposure,
                self=self,
                origins=df,
                search_radius=search_radius,
                detour_ratio=detour_ratio,
                decay=decay,
                beta=beta,
                decay_method=decay_method,
                path_detour_penalty=path_detour_penalty,
                elastic_weight=elastic_weight,
                closest_destination=closest_destination,
                turn_penalty=turn_penalty,
                path_exposure_attribute=path_exposure_attribute,
                return_path_record=return_path_record, 
                destniation_cap=destniation_cap
            ) for df in splitted_origins]
        for result in concurrent.futures.as_completed(execution_results):
            try:
                one_betweennes_output = result.result()
                batch_results.append(one_betweennes_output["batch_betweenness_tracker"])
                origin_returns.append(pd.DataFrame(one_betweennes_output["origins"]))

                if return_path_record:
                    list_of_path_dfs.append(pd.DataFrame(one_betweennes_output["path_record"]))
            except Exception as ex:
                print(str(ex))
                print(ex.__doc__)
                pass
    end = time.time()
    print("-------------------------------------------")
    print(f"All cores done in {round(end - start, 2)}")
    print("-------------------------------------------")

    for idx in edge_gdf.index:
        edge_gdf.at[idx, "betweenness"] = sum([batch[idx] for batch in batch_results])


    # not sure if this assignment is necessary,
    self.network.nodes = node_gdf
    self.network.edges = edge_gdf
    return_dict = {"edge_gdf": edge_gdf, "origin_gdf": pd.concat(origin_returns)}
    if return_path_record:
        path_record = pd.concat(list_of_path_dfs)
        return_dict["path_record"] =  path_record
    return return_dict

def get_origin_properties(
        self: Zonal,
        search_radius=800,
        beta=0.001,
        turn_penalty=False,
        o_idx=None,
        d_idxs=None,
        ):
    node_gdf = self.network.nodes

    if d_idxs is None:
        d_idxs, _, _ = turn_o_scope(
            self.network,
            o_idx,
            search_radius,
            turn_penalty=turn_penalty,
            o_graph=None,
            return_paths=False
        )
    ## K-neareast Neighbor
    if isinstance(self.network.knn_weight, str):
        knn_weights = self.network.knn_weight[1:-1].split(';')
        knn_weights = [float(x) for x in knn_weights]
        knn_weight = 0
        for neighbor_weight, neighbor_distance in zip(knn_weights, d_idxs.values()):
            if neighbor_distance < self.network.knn_plateau:
                knn_weight += neighbor_weight
            else:
                knn_weight += neighbor_weight / pow(math.e, (beta * (neighbor_distance-self.network.knn_plateau)))
        node_gdf.at[o_idx, "knn_weight"] = knn_weight *  node_gdf.at[o_idx, "weight"]

    node_gdf.at[o_idx, "gravity"] = sum(1.0 / pow(math.e, (beta * np.array(list(d_idxs.values())))))
    node_gdf.at[o_idx, "reach"] = int(len(d_idxs))
    return
