# this lets geopandas exclusively use shapely (not pygeos) silences a warning about depreciating pygeos out of geopandas. This is not needed when geopandas 1.0 is released in the future
import os
os.environ['USE_PYGEOS'] = '0'

import time
import concurrent
from concurrent import futures

import multiprocessing as mp

import networkx as nx  ## when one_betweenness_2 is deleted, this import is no longer needed. 
import pandas as pd
import geopandas as gpd
import psutil

import math
import numpy as np
import random

import os
from sys import getsizeof




from ..zonal import Zonal
from ..zonal import Network
from .paths import path_generator, turn_o_scope, bfs_subgraph_generation, wandering_messenger

def parallel_betweenness(
    network: Network,
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
    To be filled in
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
        if (counter % 100 == 0):
            print(f"{counter = }]\t{counter / len(origins.index) * 100:6.4f}")



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



def clockwiseangle_and_distance(origin, point):
    refvec = [0, 1]
    # Vector between point and the origin: v = p - o
    vector = [point[0]-origin[0], point[1]-origin[1]]
    # Length of vector: ||v||
    lenvector = math.hypot(vector[0], vector[1])
    # If length is zero there is no angle
    if lenvector == 0:
        return 0
    # Normalize vector: v/||v||
    normalized = [vector[0]/lenvector, vector[1]/lenvector]
    dotprod  = normalized[0]*refvec[0] + normalized[1]*refvec[1]     # x1*x2 + y1*y2
    diffprod = refvec[1]*normalized[0] - refvec[0]*normalized[1]     # x1*y2 - y1*x2
    angle = math.atan2(diffprod, dotprod)
    # Negative angles represent counter-clockwise angles so we need to subtract them 
    # from 2*pi (360 degrees)
    if angle < 0:
        return 2*math.pi+angle
    # I return first the angle because that's the primary sorting criterium
    # but if two vectors have the same angle then the shorter distance should come first.
    return angle


def betweenness_exposure(
        self: Zonal,
        core_index=None,
        origin_queue=None,
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
    origin_gdf=node_gdf[node_gdf['type'] == 'origin']
    node_gdf['chunck_time'] = pd.Series(dtype=object)
    node_gdf['chunck_path_count'] = pd.Series(dtype=object)
    node_gdf['chunck_path_segment_count'] = pd.Series(dtype=object)
    node_gdf['chunck_path_segment_memory'] = pd.Series(dtype=object)
    node_gdf['chunk_scope_node_count'] = pd.Series(dtype=object)



    # TODO: convert this to a numby array to simplify calculations.
    batch_betweenness_tracker = {edge_id: 0.0 for edge_id in list(edge_gdf.index)}
    

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
            'geometry': [],
        }
        if path_exposure_attribute is not None:
            path_record['mean_path_exposure'] = []


    processed_origins = []
    while True:
        try: 
            origin_idx = origin_queue.get()
            start = time.time()
            if origin_idx == "done":
                origin_queue.task_done()
                break
            processed_origins.append(origin_idx)

            #if len(processed_origins)%100 == 0:
                #print (f'DOne {len(processed_origins)}')
                
            if origin_gdf.at[origin_idx, "weight"] == 0:
                continue

            #origin_edge = node_gdf.at[origin_idx, "nearest_edge_id"]
            if path_exposure_attribute is not None:
                origin_mean_hazzard = 0
                origin_decayed_mean_hazzard = 0
                expected_hazzard_meters = 0
                probable_travel_distance_weighted_hazzard = 0

            #origin stats
            origin_mean_path_length = 0
            probable_travel_distance = 0
        except Exception as ex:
            print (f"CORE: {core_index}: [betweenness_exposure]: error aquiring new origin, returning what was processed so far {len(processed_origins) = }")
            return {"batch_betweenness_tracker": batch_betweenness_tracker, 'origins': origin_gdf.loc[processed_origins]}






        # finding all destination reachible from this origin within a search radius, and finding all paths within a derour ration from the shortest path
        try:
            #path_edges, weights, d_idxs = path_generator(
            #    self.network,
            #    origin_idx,
            #    search_radius=search_radius,
            #    detour_ratio=detour_ratio,
            #    turn_penalty=turn_penalty
            #)
            o_graph = self.network.d_graph
            self.network.add_node_to_graph(o_graph, origin_idx)

            d_idxs, o_scope, o_scope_paths = turn_o_scope(
                network=self.network,
                o_idx=origin_idx,
                search_radius=search_radius,
                detour_ratio=detour_ratio,
                turn_penalty=turn_penalty,
                o_graph=o_graph,
                return_paths=True
            )
            destination_discovery_time = time.time() - start
            start = time.time()
        except Exception as ex:
            print (f"CORE: {core_index}: [betweenness_exposure]: error generating path for origin {origin_idx = }, {len(processed_origins) = }")
            print(str(ex))
            print(ex.__doc__)
            print (ex.__traceback__)
            import traceback
            traceback.print_exc()
            print (f"CORE: {core_index}: [betweenness_exposure]: skipping origin: {origin_idx = }, {len(processed_origins) = }")
            continue



        try:
            # skip this origin if cannot reach any destination
            if len(d_idxs) == 0:
                self.network.remove_node_to_graph(o_graph, origin_idx)
                origin_queue.task_done()
                continue
            
            # this makes sure destinations are sorted based on distance. (sorting the dictionary nased on its values.)
            d_idxs = dict(sorted(d_idxs.items(), key=lambda item: item[1]))

            origin_weight_attribute = 'weight'
            if elastic_weight:
                origin_weight_attribute = 'knn_weight'
                get_origin_properties(
                    self,
                    search_radius=search_radius,
                    beta=beta,
                    turn_penalty=turn_penalty,
                    o_idx=origin_idx,
                    o_graph=None,
                    d_idxs=d_idxs,
                )
                #TODO: because node_gdf is modified internally, need to update the copy origin_gdf as it doesn't see the new updates. Consider alternatives.
                origin_gdf=node_gdf[node_gdf['type'] == 'origin']
            origin_weight = origin_gdf.at[origin_idx, origin_weight_attribute]

        except Exception as ex:
            print (f"CORE: {core_index}: [betweenness_exposure]: error generating weight for origin {origin_idx = }, {len(processed_origins) = }, skipping origin.")
            traceback.print_exc()
            continue
        try:
            max_chunck_size = 100 
            chunking_method = 'cocentric-chunks'
            if closest_destination:
                # just use the closest destination.
                destination_probabilities = [1]
                destination_ids = [min(d_idxs, key=d_idxs.get)]
                eligible_destinations = {destination_ids[0]:d_idxs[destination_ids[0]]}
            else:
                # probability of choosing a destination using the huff model
                eligible_destinations = d_idxs if destniation_cap   is None else dict(list(d_idxs.items())[:destniation_cap])
                destination_ids = list(eligible_destinations.keys())
                #max_chunck_size = 100
                #chunking_method = self.network.chunking_method
                #max_chunck_size = self.network.max_chunck_size
                if len(destination_ids) > max_chunck_size:
                    if chunking_method == 'no_chunking':
                        pass # do nothing
                    elif chunking_method == 'cocentric-chunks':
                        pass # do nothing
                    elif chunking_method == 'random_chunks':
                        random.shuffle(destination_ids)  # shuffled in place..
                        eligible_destinations = {destination:eligible_destinations[destination] for destination in destination_ids}
                        pass
                    elif chunking_method == 'pizza_chunks':
                        destinations = node_gdf.loc[destination_ids]
                        origin_geom = node_gdf.at[origin_idx,'geometry'].coords
                        destinations['angle'] = destinations['geometry'].apply(lambda x: clockwiseangle_and_distance([origin_geom[0][0], origin_geom[0][1]], [x.coords[0][0], x.coords[0][1]]))

                        destination_ids = list(destinations.sort_values('angle').index)
                        eligible_destinations = {destination:eligible_destinations[destination] for destination in destination_ids}

                eligible_destinations_weight = np.array([float(node_gdf.at[idx, 'weight']) for idx in destination_ids], dtype=np.float64)
                eligible_destinations_shortest_distance = np.array(list(eligible_destinations.values()))
                destination_gravities = eligible_destinations_weight / pow(np.e, (beta * eligible_destinations_shortest_distance))

                if sum(destination_gravities) == 0:
                    continue # This covers a case where all reachible destinations have a weight of 0, if so, no need to generate trips

                destination_probabilities = np.array(destination_gravities) / sum(destination_gravities)
        except Exception as ex:
            print (f"CORE: {core_index}: [betweenness_exposure]: error generating destination probabilities for origin {origin_idx = }, {len(processed_origins) = }, {eligible_destinations_weight = }, {eligible_destinations_shortest_distance = },  {destination_gravities = }, skipping origin")
            import traceback
            traceback.print_exc()
            continue
        
        if (len(destination_ids) > max_chunck_size) and (chunking_method in ['cocentric-chunks', 'random_chunks', 'pizza_chunks']):
            num_chunks = np.ceil(len(destination_ids)/max_chunck_size)
            d_chuncks = np.array_split(destination_ids, num_chunks)
            d_idx_chuncks = [{destination:eligible_destinations[destination] for destination in destination_chunck} for destination_chunck in d_chuncks]
            destination_probabilities_ckuncks = np.array_split(destination_probabilities, num_chunks)
        else:
            d_idx_chuncks =  [eligible_destinations]
            destination_probabilities_ckuncks = [destination_probabilities]

        #print (f"after chuncks {d_idx_chuncks = }\t{destination_probabilities_ckuncks  =}")
        destination_prep_time = time.time() - start
        memory_stalls = 0
        chunck_time = []
        chunck_path_count = []
        chunck_segment_count = []
        chunck_path_segment_memory = []
        chunk_scope_node_count = []



        for chunck_num, (d_idx_chunck, destination_probabilities_ckunck) in enumerate(zip(d_idx_chuncks, destination_probabilities_ckuncks)): 
            try:
                start = time.time()
                # force to wait for at least 2.4GB of available memory, and %15 of memory is available so processes don't cause memory clogging
                memory_data = psutil.virtual_memory()
                available_gb_memory = memory_data[1] /(1024 ** 3)
                available_memory_pct = 100-memory_data[2]

                while (available_gb_memory < 2.4) or (available_memory_pct < 15.0):
                    memory_stalls +=1
                    time.sleep(30)
                    memory_data = psutil.virtual_memory()
                    available_gb_memory = memory_data[1] /(1024 ** 3)
                    available_memory_pct = 100-memory_data[2]
            except Exception as ex:
                print (f"CORE: {core_index}: [betweenness_exposure]: error with memory management mechanisim, skipping memory management and getting a new task")

            try:
                ## get subgraph and and path..
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

                #Diagnostics
                chunck_path_count.append(sum([len(dest_paths) for dest_paths in path_edges.values()]))
                chunck_segment_count.append(sum([sum([len(path_edges) for path_edges in dest_paths]) for dest_paths in path_edges.values()]))
                chunck_path_segment_memory.append(sum([sum([getsizeof(path_edges) for path_edges in dest_paths]) for dest_paths in path_edges.values()]))
                chunk_scope_node_count.append(len(scope_nodes))
            except Exception as ex:
                print (f"CORE: {core_index}: [betweenness_exposure]: error generating paths for origin {origin_idx = } destination chunck {chunck_num = }, {len(processed_origins) = }, skipping destination")
                import traceback
                traceback.print_exc()

                continue

            for destination_idx, this_destination_probability in zip(d_idx_chunck.keys(), destination_probabilities_ckunck):
            #origin_mean_path_length  = (destination_path_probabilies * d_path_weights).sum()
            #probable_travel_distance = (destination_path_probabilies * path_decays * d_path_weights).sum()
                try: 
                    #od_edges = set([node_gdf.at[destination_idx, "nearest_edge_id"], origin_edge])

                    # skip this destination if cannot find paths from origin
                    if len(weights[destination_idx]) == 0:
                        #print(f"o:{origin_idx}\td:{destination_idx} have no paths...")
                        continue

                    # finding path probabilities given different penalty settings.
                    path_detour_penalties = np.ones(len(weights[destination_idx]))
                    d_path_weights = np.array(weights[destination_idx])

                    # This fixes numerical issues  when path length = 0
                    d_path_weights[d_path_weights < 0.01] = 0.01
                    '''
                    d_path_weights [400, 405, 407]

                    num_bins = 5
                    bin_width = (np.max(d_path_weights) - np.min(d_path_weights))/num_bins
                    range_min = np.min(d_path_weights)

                    binned_porbabilities = []
                    for i in range(num_bins):
                        current_bin_weightrs = d_path_weights[(d_path_weights >= (range_min + i*bin_width)) & (d_path_weights < (range_min + (i+1)*bin_width))]

                        ## Equal Probability
                        if path_detour_penalty == "equal":
                            current_bin_probabilities = np.ones(len(current_bin_weightrs))


                        elif path_detour_penalty == "exponent":
                        ## Distance weightred probability
                            current_bin_probabilities = 1.0 / pow(np.e, beta * current_bin_weightrs)


                        current_bin_probabilities = current_bin_probabilities/sum(current_bin_probabilities)


                        binned_porbabilities.append(current_bin_weightrs)
                    path_probabilities = np.concat(binned_porbabilities)
                    ''' 



                    if path_detour_penalty == "exponent":
                        path_detour_penalties = 1.0 / pow(np.e, beta * d_path_weights)
                    elif path_detour_penalty == "power":
                        path_detour_penalties = 1.0 / (d_path_weights ** 2.0)
                    elif path_detour_penalty == "equal":
                        path_detour_penalties = np.ones(len(d_path_weights))
                    else:
                        raise ValueError(
                            f"parameter 'path_detour_penalty' should be one of ['equal', 'power', 'exponent'], '{path_detour_penalty}' was given")
                    path_probabilities = path_detour_penalties / sum(path_detour_penalties)


                    # FInding path decays. TODO: consider taking out of the loop as this 
                    path_decays = np.ones(len(weights[destination_idx]))
                    if decay:
                        if decay_method == "exponent":
                            # path_decays = 1.0 / pow(np.e, beta * d_path_weights)
                            path_decays = 1.0 / pow(np.e, beta * path_decays *  min(list(d_idxs.values())))
                        elif decay_method == "power":
                            ## WAENING: Path weight cannot be zero!! handle properly.
                            path_decays = 1.0 / (d_path_weights ** 2.0)
                        else:
                            raise ValueError(
                                f"parameter 'decay_method' should be one of ['exponent', 'power'], '{decay_method}' was given")


                    # Betweenness attributes
                    destination_path_probabilies = path_probabilities * this_destination_probability
                    betweennes_contributions = destination_path_probabilies * path_decays * origin_weight

                    # origin stats accumulation
                    origin_mean_path_length += (destination_path_probabilies * d_path_weights).sum()
                    probable_travel_distance += (destination_path_probabilies * path_decays * d_path_weights).sum()

                    if len(d_path_weights[d_path_weights > (d_idx_chunck[destination_idx] * detour_ratio)+ 0.01]) > 0:
                        print(f"SOme paths exceeded allowed tolerance: {d_path_weights[d_path_weights > (d_idx_chunck[destination_idx] * detour_ratio)+ 0.01]}")

                except Exception as ex:
                    print (f"CORE: {core_index}: [betweenness_exposure]: error generating path probabilities, decay,  betweenness for origin {origin_idx = } destination {destination_idx = }, {len(processed_origins) = }, skipping destination")
                    import traceback
                    traceback.print_exc()
                    continue


                try:
                    if path_exposure_attribute is None:

                        for this_path_edges, betweennes_contribution in zip (path_edges[destination_idx], betweennes_contributions): 
                            for edge_id in this_path_edges:
                            #for edge_id in set(this_path_edges).union(od_edges):
                                batch_betweenness_tracker[edge_id] += betweennes_contribution

                    else:  # for exposure
                        for this_path_edges, betweennes_contribution, destination_path_probability, path_decay, this_path_weight in zip (path_edges[destination_idx], betweennes_contributions, destination_path_probabilies, path_decays, d_path_weights): 
                            path_weight_exposure = 0       
                            path_weight_sum = 0            


                            for edge_id in this_path_edges:
                                batch_betweenness_tracker[edge_id] += betweennes_contribution
                                segment_weight = edge_gdf.at[int(edge_id), 'weight']
                                path_weight_sum += segment_weight
                                path_weight_exposure += segment_weight * self[self.network.edge_source_layer].gdf.at[edge_gdf.at[edge_id, 'parent_street_id'], path_exposure_attribute]
                                
                            
                        
                            path_mean_exposure = path_weight_exposure / path_weight_sum

                            origin_mean_hazzard += destination_path_probability * path_mean_exposure
                            origin_decayed_mean_hazzard += destination_path_probability * path_decay * path_mean_exposure
                            expected_hazzard_meters += path_mean_exposure * destination_path_probability * this_path_weight
                            probable_travel_distance_weighted_hazzard += destination_path_probability * path_decay * path_mean_exposure * this_path_weight
                except:
                    print (f"CORE: {core_index}: [betweenness_exposure]: error assigning path betweenness to segment {origin_idx = } destination {destination_idx = }, {len(processed_origins) = }, skipping destination")
                    import traceback
                    traceback.print_exc()
                    continue
            chunck_time.append(time.time()-start)
            #done chunck, since there is chance to pause for memory, delete this ieration's variables


        '''
                if return_path_record:
                    path_record['origin_id'].append(origin_idx)
                    path_record['destination_id'].append(destination_idx)
                    path_record['destination_probability'].append(this_destination_probability)
                    path_record['path_id'].append(path_id)
                    path_record['path_weight'].append(this_path_weight)
                    path_record['path_decay'].append(path_decay)
                    path_record['path_probability'].append(destination_path_probability/this_destination_probability)
                    path_record['path_betweenness'].append(betweennes_contribution)
                    path_record['geometry'].append(geo.MultiLineString(edge_gdf.loc[edge_ids]['geometry'].values))
                    if path_exposure_attribute is not None:
                        path_record['mean_path_exposure'].append(path_mean_exposure)

        '''
        try:
            #pass
            node_gdf.at[origin_idx, 'reach'] = len(d_idxs)
            node_gdf.at[origin_idx, "gravity"] = sum(1.0 / pow(math.e, (beta * np.array(list(d_idxs.values())))))

            if elastic_weight:
                node_gdf.at[origin_idx, "knn_weight"] = origin_weight
                    
            if path_exposure_attribute is not None:
                node_gdf.at[origin_idx, 'mean_hazzard'] = origin_mean_hazzard
                node_gdf.at[origin_idx, 'decayed_mean_hazzad'] = origin_decayed_mean_hazzard
                node_gdf.at[origin_idx, 'expected_hazzard_meters'] = expected_hazzard_meters
                node_gdf.at[origin_idx, 'probable_travel_distance_weighted_hazzard'] = probable_travel_distance_weighted_hazzard
            
            node_gdf.at[origin_idx, 'closest_destination_distance'] = eligible_destinations_shortest_distance.min()
            node_gdf.at[origin_idx, 'furthest_destination_distance'] = eligible_destinations_shortest_distance.max()
            node_gdf.at[origin_idx, 'mean_path_length'] = origin_mean_path_length
            node_gdf.at[origin_idx, 'probable_travel_distance'] = probable_travel_distance
            node_gdf.at[origin_idx, 'eligible_destinations'] = len(eligible_destinations)
            node_gdf.at[origin_idx, 'path_count'] = sum(chunck_path_count)
            node_gdf.at[origin_idx, 'path_segment_count'] = sum(chunck_segment_count)
            node_gdf.at[origin_idx, 'path_segment_memory'] = sum(chunck_path_segment_memory)
            node_gdf.at[origin_idx, 'scope_node_count'] = sum(chunk_scope_node_count)
            node_gdf.at[origin_idx, 'destination_discovery_time'] = destination_discovery_time
            node_gdf.at[origin_idx, 'destination_prep_time'] = destination_prep_time
            node_gdf.at[origin_idx, 'path_generation_time'] = sum(chunck_time)
            node_gdf.at[origin_idx, 'chunck_count'] = len(chunck_time)
            node_gdf.at[origin_idx, 'chunck_time'] = chunck_time
            node_gdf.at[origin_idx, 'chunck_path_count'] = chunck_path_count
            node_gdf.at[origin_idx, 'chunck_path_segment_count'] = chunck_segment_count
            node_gdf.at[origin_idx, 'chunck_path_segment_memory'] = chunck_path_segment_memory
            node_gdf.at[origin_idx, 'chunk_scope_node_count'] = sum(chunk_scope_node_count)
            node_gdf.at[origin_idx, 'memory_stalls'] = memory_stalls


            
        except:
            print (f"CORE: {core_index}: [betweenness_exposure]: error collecting origin statistics {origin_idx = } , {len(processed_origins) = }, proceeding to next task")
            import traceback
            traceback.print_exc()
            continue

        try:
            del path_edges, weights
            del path_detour_penalties, d_path_weights, path_probabilities, path_decays, destination_path_probabilies, betweennes_contributions
            self.network.remove_node_to_graph(o_graph, origin_idx)
            origin_queue.task_done()
        except:
            print (f"CORE: {core_index}: [betweenness_exposure]: error marking task done {origin_idx = } , {len(processed_origins) = }, proceeding to next task")
            import traceback
            traceback.print_exc()
            continue


    return_dict = {"batch_betweenness_tracker": batch_betweenness_tracker, 'origins': node_gdf.loc[processed_origins]}
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
    node_gdf = self.network.nodes
    edge_gdf = self.network.edges

    origins = node_gdf[node_gdf["type"] == "origin"]


    list_of_path_dfs = []
    edge_gdf['betweenness'] = 0.0
    batch_results = []
    origin_returns = []
    num_procs = num_cores  # psutil.cpu_count(logical=logical)

    #TODO: investigate if this randomazation is causing any issues downsream.
    #print (f"before 'sample' {origins.index = }")
    origins = origins.sample(frac=1)
    origins.index = origins.index.astype("int")
    #print (f"after 'sample' {origins.index = }")







    with concurrent.futures.ProcessPoolExecutor(max_workers=num_procs) as executor:
        with mp.Manager() as manager:
            #create aNDfill queue
            origin_queue = manager.Queue()
            origin_list = list(origins.index)
            #random.shuffle(origin_list)     #randomize origins in queue so queue won't face bursts of easy origins and bursts of hard ones based on 
            #print (f"after 'shuffle' {origin_list = }")

            for o_idx in origin_list:
                origin_queue.put(o_idx)
            for core_index in range(num_cores):
                origin_queue.put("done")

            execution_results = []
            for core_index in range(num_cores):
                execution_results.append(executor.submit(
                    betweenness_exposure,
                    self=self,
                    core_index=core_index,
                    origin_queue=origin_queue,
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
                ))

            start = time.time()
            while not all([future.done() for future in execution_results]):
                time.sleep(0.5)
                print (f"Time spent: {round(time.time()-start):,}s [Done {max(origins.shape[0] - origin_queue.qsize(), 0):,} of {origins.shape[0]:,} origins ({max(origins.shape[0] - origin_queue.qsize(), 0)/origins.shape[0] * 100:4.2f}%)]",  end='\r')
                for future in [f for f in execution_results if f.done() and (f.exception() is not None)]: # if a process is done and have an exception, raise it
                    raise (future.exception())
           
   

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
    #print("-------------------------------------------")
    #print(f"All cores done in {round(end - start, 2)}")
    #print("-------------------------------------------")

    for idx in edge_gdf.index:
        edge_gdf.at[idx, "betweenness"] = sum([batch[idx] for batch in batch_results])


    # not sure if this assignment is necessary,
    self.network.nodes = node_gdf
    self.network.edges = edge_gdf
    return_dict = {"edge_gdf": edge_gdf, "origin_gdf": pd.concat(origin_returns)}
    if return_path_record:
        path_record = gpd.GeoDataFrame(pd.concat(list_of_path_dfs), crs=edge_gdf.crs)
        return_dict["path_record"] =  path_record
    return return_dict

def get_origin_properties(
        self: Zonal,
        search_radius=800,
        beta=0.001,
        turn_penalty=False,
        o_idx=None,
        d_idxs=None,
        o_graph=None, 
        destination_weight=None,
        alpha=1.0
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
    if self.network.knn_weight is not None:

        
        knn_weight = 0
        for neighbor_weight, neighbor_distance in zip(self.network.knn_weight, d_idxs.values()):
            if neighbor_distance < self.network.knn_plateau:
                knn_weight += neighbor_weight
            else:
                knn_weight += neighbor_weight / pow(math.e, (beta * (neighbor_distance-self.network.knn_plateau)))

        ## Vectorizing this could improve effeciency
        #eligible_neighbors = min (len(d_idxs), len(knn_weights))
        #neighbor_weights  = np.array(knn_weights[:eligible_neighbors])
        #neighbor_distances = np.array(list(d_idxs.values())[:eligible_neighbors])
        #knn_weight = neighbor_weights/ pow(math.e, beta * max(neighbor_distances-self.network.knn_plateau, 0))

        node_gdf.at[o_idx, "knn_weight"] = knn_weight *  node_gdf.at[o_idx, "weight"]
        node_gdf.at[o_idx, "knn_access"] = knn_weight

    
    #node_gdf.at[o_idx, "gravity"] = sum(1.0 / pow(math.e, (beta * np.array(list(d_idxs.values())))))
    #node_gdf.at[o_idx, "reach"] = int(len(d_idxs))
        
    

    o_closest_destinations = list(d_idxs.keys())
    o_destination_distances = np.array(list(d_idxs.values()))
    if destination_weight is None:
        #o_destination_weights = node_gdf.loc[o_closest_destinations, "weight"].values
        o_destination_weights = np.array([node_gdf.at[idx, 'weight'] for idx in o_closest_destinations], dtype=np.float64)
    else:
        ## This case could be optimized. Optimal to determine weights at node insertion..
        destination_gdf=node_gdf[node_gdf['type'] == 'destination']
        destination_layer = destination_gdf.iloc[0]['source_layer']
        source_ids = list(node_gdf.loc[o_closest_destinations, "source_id"])
        o_destination_weights = self[destination_layer].gdf.loc[source_ids, destination_weight].fillna(0).values
    
    node_gdf.at[o_idx, "reach"] = o_destination_weights.sum()
    if beta is not None:
        node_gdf.at[o_idx, "gravity"] = (np.power(o_destination_weights, alpha) / np.power(math.e, beta * o_destination_distances)).sum()


    return






def one_access(
    self: Zonal,
    origin_queue = None,
    search_radius: float = None, 
    destination_weight: str = None,
    alpha: float = None,
    beta: float = None, 
    knn_weights: list = None, 
    knn_plateau: int | float = None,
    closest_facility: bool = False,
    turn_penalty: bool = False, 
    reporting: bool = False, 
    ):

    node_gdf = self.network.nodes
    origin_gdf=node_gdf[node_gdf['type'] == 'origin']
    



    o_graph = self.network.d_graph
    
    start = time.time()
    i = 1
    processed_origins = []
    try: 
        while True:
            origin_idx = origin_queue.get()

            if origin_idx == "done":
                origin_queue.task_done()
                break
            processed_origins.append(origin_idx)



            self.network.add_node_to_graph(o_graph, origin_idx)
            d_idxs, _, _ = turn_o_scope(
                network=self.network,
                o_idx=origin_idx,
                search_radius=search_radius,
                detour_ratio=1,
                turn_penalty=turn_penalty,
                o_graph=o_graph,
                return_paths=False
            )
            self.network.remove_node_to_graph(o_graph, origin_idx)

            if len(d_idxs) == 0:
                continue

            d_idxs = dict(sorted(d_idxs.items(), key=lambda item: item[1]))


            if closest_facility:
                for d_idx in d_idxs:
                    if ("closest_facility" not in node_gdf.loc[d_idx]) or (np.isnan(node_gdf.loc[d_idx]["closest_facility"])):
                        node_gdf.at[d_idx, "closest_facility"] = origin_idx
                        node_gdf.at[d_idx, "closest_facility_distance"] = d_idxs[d_idx]
                    elif d_idxs[d_idx] < node_gdf.at[d_idx, "closest_facility_distance"]:
                        node_gdf.at[d_idx, "closest_facility"] = origin_idx
                        node_gdf.at[d_idx, "closest_facility_distance"] = d_idxs[d_idx]

            else:
                self.network.knn_weight = knn_weights
                self.network.knn_plateau = knn_plateau
                get_origin_properties(
                    self,
                    search_radius=search_radius,
                    beta=beta,
                    turn_penalty=turn_penalty,
                    o_idx=origin_idx,
                    d_idxs=d_idxs,
                    o_graph=o_graph,
                    destination_weight=destination_weight,
                    alpha=alpha
                )

            if reporting:
                print (f"Time spent: {round(time.time()-start):,}s [Done {i:,} of {origin_gdf.shape[0]:,} origins ({i  /origin_gdf.shape[0] * 100:4.2f}%)]",  end='\r')
                i = i + 1

    except Exception as ex:
        print ('Issues in one access...')
        import traceback
        traceback.print_exc()

    return node_gdf.loc[processed_origins], node_gdf[node_gdf['type'] == 'destination']




def parallel_access(
    self: Zonal,
    search_radius: float = None, 
    destination_weight: str = None,
    alpha: float = None,
    beta: float = None, 
    knn_weights: list = None, 
    knn_plateau: int | float = None,
    closest_facility: bool = False,
    turn_penalty: bool = False, 
    num_cores: int = None,
):
    node_gdf = self.network.nodes
    origin_gdf = node_gdf[node_gdf["type"] == "origin"]
    origin_gdf = origin_gdf.sample(frac=1)
    origin_gdf.index = origin_gdf.index.astype("int")

    destination_gdf = node_gdf[node_gdf["type"] == "destination"]
    destination_layer = destination_gdf.iloc[0]['source_layer']


    
    with mp.Manager() as manager:

        origin_queue = manager.Queue()

        for o_idx in origin_gdf.index:
            origin_queue.put(o_idx)

        for core_index in range(num_cores):
            origin_queue.put("done")

        if num_cores == 1:
            origin_gdf, destination_gdf = one_access(
                self=self,
                origin_queue=origin_queue,
                search_radius=search_radius,
                destination_weight=destination_weight,
                alpha=alpha,
                beta=beta,
                knn_weights=knn_weights,
                knn_plateau=knn_plateau,
                closest_facility=closest_facility,
                turn_penalty=turn_penalty,
                reporting=True
            )
        else:
            with concurrent.futures.ProcessPoolExecutor(max_workers=num_cores) as executor:        
                execution_results = [
                    executor.submit(
                        one_access,
                        self=self,
                        origin_queue=origin_queue,
                        search_radius=search_radius,
                        destination_weight=destination_weight,
                        alpha=alpha,
                        beta=beta,
                        knn_weights=knn_weights,
                        knn_plateau=knn_plateau,
                        closest_facility=closest_facility,
                        turn_penalty=turn_penalty,
                        reporting=False
                    ) for core_index in range(num_cores)
                ]

                # Progress update and raising exceptions..
                start = time.time()
                while not all([future.done() for future in execution_results]):
                    time.sleep(0.5)
                    done_so_far = origin_gdf.shape[0] - max(origin_queue.qsize() - num_cores, 0)
                    print (f"Time spent: {round(time.time()-start):,}s [Done {done_so_far:,} of {origin_gdf.shape[0]:,} origins ({done_so_far/origin_gdf.shape[0] * 100:4.2f}%)]",  end='\r')
                    for future in [f for f in execution_results if f.done() and (f.exception() is not None)]: # if a process is done and have an exception, raise it
                        raise (future.exception())
                
                

                ## COnsolidating output from cores
                core_origin_gdfs = []
                core_destination_gdfs = []

                for result in concurrent.futures.as_completed(execution_results):
                   o_gdf, d_gdf =  result.result()
                   core_origin_gdfs.append(o_gdf)
                   core_destination_gdfs.append(d_gdf)

                origin_gdf = pd.concat(core_origin_gdfs)

                node_gdf['reach'] = origin_gdf['reach']
                node_gdf['gravity'] = origin_gdf['gravity']
                if knn_weights is not None:
                    node_gdf["knn_weight"] = origin_gdf["knn_weight"]
                    node_gdf["knn_access"] = origin_gdf["knn_access"]

                if closest_facility:
                    # extracting closest facility over multiple destination_gdf coming from different cores, think about a panda solution instead of nested loops.
                    node_gdf["closest_facility"] = np.nan
                    node_gdf["closest_facility_distance"] = np.nan
                    for core_destination_gdf in core_destination_gdfs:
                        for d_idx in core_destination_gdf.index:
                            if np.isnan(node_gdf.loc[d_idx]["closest_facility"]):
                                node_gdf.at[d_idx, "closest_facility"] = core_destination_gdf.at[d_idx, "closest_facility"]
                                node_gdf.at[d_idx, "closest_facility_distance"] = core_destination_gdf.at[d_idx, "closest_facility_distance"]
                            elif core_destination_gdf.at[d_idx, "closest_facility_distance"] < node_gdf.at[d_idx, "closest_facility_distance"]:
                                node_gdf.at[d_idx, "closest_facility"] = core_destination_gdf.at[d_idx, "closest_facility"]
                                node_gdf.at[d_idx, "closest_facility_distance"] = core_destination_gdf.at[d_idx, "closest_facility_distance"]


                            

                    
                    
    if closest_facility:
        ## adjust origin reach ad gravity based on joint destinations
        for o_idx in origin_gdf.index:
            o_closest_destinations = list(node_gdf[node_gdf["closest_facility"] == o_idx].index)
            o_destination_distances = node_gdf.loc[o_closest_destinations]["closest_facility_distance"].values
            if destination_weight is None:
                o_destination_weights = node_gdf.loc[o_closest_destinations]["weight"].values
            else:
                source_ids = list(node_gdf.loc[o_closest_destinations]["source_id"])
                o_destination_weights = self[destination_layer].gdf.loc[source_ids][destination_weight].fillna(0).values
            
            node_gdf.at[o_idx, "reach"] = sum(o_destination_weights)
            if beta is not None:
                node_gdf.at[o_idx, "gravity"] = sum(pow(o_destination_weights, alpha) / pow(math.e, beta * o_destination_distances))

        ## reverting to layer indexing...
        ## This could be an apply function??
        for d_idx in destination_gdf.index: 
            if not np.isnan(node_gdf.at[d_idx, "closest_facility"]):
                o_idx = int(node_gdf.at[d_idx, "closest_facility"])
                origin_id = node_gdf.at[o_idx, "source_id"]
                node_gdf.at[d_idx, "closest_facility"] = origin_id


    self.network.nodes = node_gdf

    return




