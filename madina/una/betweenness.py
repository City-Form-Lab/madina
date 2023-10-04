# this lets geopandas exclusively use shapely (not pygeos) silences a warning about depreciating pygeos out of geopandas. This is not needed when geopandas 1.0 is released in the future
import os
os.environ['USE_PYGEOS'] = '0'

import time
import concurrent
from concurrent import futures
import multiprocessing as mp
import sys
import networkx as nx
import pandas as pd
import geopandas as gpd
import psutil
import shapely as shp

from datetime import datetime
import os
from pathlib import Path

import shapely.geometry as geo

from madina.zonal import Zonal, VERSION, RELEASE_DATE
from madina.zonal import Network
from madina.una.paths import path_generator, turn_o_scope, bfs_subgraph_generation, wandering_messenger

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


import math
import numpy as np
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

import random
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
            origin_weight = origin_gdf.at[origin_idx, origin_weight_attribute]
        except Exception as ex:
            print (f"CORE: {core_index}: [betweenness_exposure]: error generating weight for origin {origin_idx = }, {len(processed_origins) = }, skipping origin.")
            traceback.print_exc()
            continue

        try:
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
                chunking_method = 'cocentric-chunks'
                #chunking_method = self.network.chunking_method
                max_chunck_size = 100 
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


                    # FInding path decays.
                    path_decays = np.ones(len(weights[destination_idx]))
                    if decay:
                        if decay_method == "exponent":
                            path_decays = 1.0 / pow(np.e, beta * d_path_weights)
                        elif decay_method == "power":
                            ## WAENING: Path weight cannot be zero!! handle properly.
                            path_decays = 1.0 / (d_path_weights ** 2.0)
                        else:
                            raise ValueError(
                                f"parameter 'decay_method' should be one of ['exponent', 'power'], '{decay_method}' was given")


                    # Betweenness attributes
                    destination_path_probabilies = path_probabilities * this_destination_probability
                    betweennes_contributions = destination_path_probabilies * path_decays * origin_weight
                
                    if len(d_path_weights[d_path_weights > (d_idx_chunck[destination_idx] * detour_ratio)+ 0.01]) > 0:
                        print(f"SOme paths exceeded allowed tolerance: {d_path_weights[d_path_weights > (d_idx_chunck[destination_idx] * detour_ratio)+ 0.01]}")

                except Exception as ex:
                    print (f"CORE: {core_index}: [betweenness_exposure]: error generating path probabilities, decay,  betweenness for origin {origin_idx = } destination {destination_idx = }, {len(processed_origins) = }, skipping destination")
                    continue


                try:
                    for this_path_edges, betweennes_contribution in zip (path_edges[destination_idx], betweennes_contributions): 
                        for edge_id in this_path_edges:
                        #for edge_id in set(this_path_edges).union(od_edges):
                            batch_betweenness_tracker[edge_id] += betweennes_contribution
                except:
                    print (f"CORE: {core_index}: [betweenness_exposure]: error assigning path betweenness to segment {origin_idx = } destination {destination_idx = }, {len(processed_origins) = }, skipping destination")
                    continue
            chunck_time.append(time.time()-start)
            #done chunck, since there is chance to pause for memory, delete this ieration's variables
            del path_edges, weights
            del path_detour_penalties, d_path_weights, path_probabilities, path_decays, destination_path_probabilies, betweennes_contributions

        '''
                    if path_exposure_attribute is not None:
                        segment_weight = edge_gdf.at[int(edge_id), 'weight']
                        path_weight_sum += segment_weight
                        path_weight_exposure += segment_weight * edge_gdf.at[int(edge_id), path_exposure_attribute]

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
                    path_record['path_probability'].append(destination_path_probability/this_destination_probability)
                    path_record['path_betweenness'].append(betweennes_contribution)
                    path_record['geometry'].append(geo.MultiLineString(edge_gdf.loc[edge_ids]['geometry'].values))
                    if path_exposure_attribute is not None:
                        path_record['mean_path_exposure'].append(path_mean_exposure)

                

                        
        
        if path_exposure_attribute is not None:
            origins.at[origin_idx, 'mean_hazzard'] = origin_mean_hazzard
            origins.at[origin_idx, 'decayed_mean_hazzad'] = origin_decayed_mean_hazzard
            origins.at[origin_idx, 'expected_hazzard_meters'] = expected_hazzard_meters
            origins.at[origin_idx, 'probable_travel_distance_weighted_hazzard'] = probable_travel_distance_weighted_hazzard
        
        #origins.at[origin_idx, 'shortest_path_length'] = list(eligible_destinations.values())[0]
        #origins.at[origin_idx, 'mean_path_length'] = origin_mean_path_length
        #origins.at[origin_idx, 'probable_travel_distance'] = probable_travel_distance
        #origins.at[origin_idx, 'eligible_destinations'] = len(eligible_destinations)
        origin_mean_hazzard = 0
        origin_decayed_mean_hazzard = 0
        expected_hazzard_meters = 0
        probable_travel_distance_weighted_hazzard = 0

        origin_mean_path_length = 0
        probable_travel_distance = 0
        '''
        try:
            #pass
            node_gdf.at[origin_idx, 'reach'] = len(d_idxs)
            #node_gdf.at[origin_idx, 'path_count'] = sum([len(dest_paths) for dest_paths in path_edges.values()])
            #node_gdf.at[origin_idx, 'path_segment_count'] = sum([sum([len(path_edges) for path_edges in dest_paths]) for dest_paths in path_edges.values()])
            #node_gdf.at[origin_idx, 'path_segment_memory'] = sum([sum([getsizeof(path_edges) for path_edges in dest_paths]) for dest_paths in path_edges.values()])
            node_gdf.at[origin_idx, 'destination_discovery_time'] = destination_discovery_time
            node_gdf.at[origin_idx, 'destination_prep_time'] = destination_prep_time
            node_gdf.at[origin_idx, 'path_generation_time'] = sum(chunck_time)
            node_gdf.at[origin_idx, 'chunck_count'] = len(chunck_time)
            node_gdf.at[origin_idx, 'chunck_time'] = chunck_time
            node_gdf.at[origin_idx, 'memory_stalls'] = memory_stalls

            
        except:
            print (f"CORE: {core_index}: [betweenness_exposure]: error collecting origin statistics {origin_idx = } , {len(processed_origins) = }, proceeding to next task")
            import traceback
            traceback.print_exc()
            continue

        try:
            self.network.remove_node_to_graph(o_graph, origin_idx)
            origin_queue.task_done()
        except:
            print (f"CORE: {core_index}: [betweenness_exposure]: error marking task done {origin_idx = } , {len(processed_origins) = }, proceeding to next task")
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
        o_graph=None
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

class Logger():
    def __init__(self, output_folder):
        self.output_folder = output_folder
        self.start_time = datetime.now()
        self.log_df = pd.DataFrame(
            {
                "time": pd.Series(dtype='datetime64[ns]'),
                "flow_name": pd.Series(dtype="string"),
                "event": pd.Series(dtype="string")
            }
        )

        self.betweenness_record = None
        self.log(f"SIMULATION STARTED: VERSION: {VERSION}, RELEASE DATEL {RELEASE_DATE}")
        self.log(f"{sys.version}")
        self.log(f"Dependencies: Geopandas:{gpd.__version__}, Shapely:{shp.__version__}, Pandas:{pd.__version__}, Numpy:{np.__version__}, NetworkX:{nx.__version__}")

    def log(self, event: str, pairing: pd.Series = None):
        time = datetime.now()

        #printing the log header if this is the first log entry
        if self.log_df.shape[0] == 0:
            print(f"{'total time':^10s} | {'seconds elapsed':^15s} | {'flow_name':^40s} | event")
            seconds_elapsed = 0
            cumulative_seconds = 0
        else:
            cumulative_seconds = (time - self.start_time).total_seconds()
            seconds_elapsed = (cumulative_seconds - self.log_df['seconds_elapsed'].sum())


        log_entry = {
            "time": [time],
            "seconds_elapsed": [seconds_elapsed],
            "cumulative_seconds": [cumulative_seconds],
            'event': [event]
        }
        if pairing is not None:
            log_entry['flow_name'] = [pairing['Flow_Name']]

        self.log_df = pd.concat([self.log_df, pd.DataFrame(log_entry)] ,ignore_index=True)

        print(
            f"{cumulative_seconds:10.4f} | "
            f"{seconds_elapsed:15.6f} | "
            f"{pairing['Flow_Name'] if pairing is not None else '---':^40s} | "
            f"{event}"
        )

    def pairing_end(
            self,
            shaqra: Zonal,
            pairing: pd.Series,
            save_flow_map=True,
            save_flow_geoJSON=True,
            save_flow_csv=False,
            save_origin_geoJSON=True,
            save_origin_csv=False
        ):
        # creating a folder for output

        if self.betweenness_record is None:
            self.betweenness_record = shaqra.layers['streets'].gdf.copy(deep=True)
        
        pairing_folder = os.path.join(self.output_folder, f"{pairing['Flow_Name']}_O({pairing['Origin_Name']})_D({pairing['Destination_Name']})")
        Path(pairing_folder).mkdir(parents=True, exist_ok=True)

        street_gdf = shaqra.layers["streets"].gdf
        node_gdf = shaqra.network.nodes
        origin_gdf = node_gdf[node_gdf["type"] == "origin"]
        destination_gdf = node_gdf[node_gdf["type"] == "destination"]
        edge_gdf = shaqra.network.edges

        self.betweenness_record = self.betweenness_record.join(
            edge_gdf[['parent_street_id', 'betweenness']].set_index('parent_street_id')).rename(
            columns={"betweenness": pairing['Flow_Name']})


        # creating origins and desrinations connector lines
        origin_layer = shaqra.layers[pairing['Origin_Name']].gdf
        origin_joined = origin_layer.join(origin_gdf.set_index('source_id'),lsuffix='_origin')
        origin_joined['geometry'] = origin_joined.apply(lambda x:geo.LineString([x['geometry'], x["geometry_origin"]]), axis=1)


        destination_layer = shaqra.layers[pairing['Destination_Name']].gdf
        destination_joined = destination_layer.join(destination_gdf.set_index('source_id'),lsuffix='_destination')
        destination_joined['geometry'] = destination_joined.apply(lambda x:geo.LineString([x['geometry'], x["geometry_destination"]]), axis=1)

        if save_flow_map:
            # line width is now 0.5 for minimum flow and 5 for maximum flow
            edge_gdf["width"] = ((edge_gdf["betweenness"] - edge_gdf["betweenness"].min()) / (edge_gdf["betweenness"].max() - edge_gdf["betweenness"].min()) + 0.1) * 5

            shaqra.create_map(
                layer_list=[
                    {"gdf": street_gdf, "color": [0, 255, 255], "opacity": 0.1},
                    {
                        "gdf": edge_gdf[edge_gdf["betweenness"] > 0],
                        "color_by_attribute": "betweenness",
                        "opacity": 0.50,
                        "color_method": "quantile",
                        "width": "width", "text": "betweenness"
                    },
                    {"gdf": origin_gdf, "color": [100, 0, 255], "opacity": 0.5},
                    {'gdf': origin_joined[['geometry']], 'color': [100, 0, 255]},
                    {'gdf': origin_layer, 'color': [100, 0, 255]},
                    {"gdf": destination_gdf, "color": [255, 0, 100], "opacity": 0.5},
                    {'gdf': destination_layer, 'color': [255, 0, 100]},
                    {'gdf': destination_joined[['geometry']], 'color': [255, 0, 100]},
                ],
                basemap=False,
                save_as=os.path.join(pairing_folder, "flow_map.html")
            )

        if save_flow_geoJSON:
            self.betweenness_record.to_file(os.path.join(pairing_folder, "betweenness_record_so_far.geoJSON"), driver="GeoJSON",  engine='pyogrio')

        if save_flow_csv:
            self.betweenness_record.to_csv(os.path.join(pairing_folder, "betweenness_record_so_far.csv"))

        if save_origin_geoJSON:
            save_origin = shaqra.layers[pairing["Origin_Name"]].gdf.join(origin_gdf.set_index("source_id").drop(columns=['geometry']))
            save_origin.to_file(os.path.join(f'{pairing_folder}', f'origin_record_({pairing["Origin_Name"]}).geoJSON'), driver="GeoJSON",  engine='pyogrio')

        if save_origin_csv: 
            save_origin = shaqra.layers[pairing["Origin_Name"]].gdf.join(origin_gdf.set_index("source_id").drop(columns=['geometry']))
            save_origin.to_csv(os.path.join(f'{pairing_folder}', f'origin_record_({pairing["Origin_Name"]}).csv'))

        self.log_df.to_csv(os.path.join(pairing_folder, "time_log.csv"))

        self.log("Output saved", pairing)

    def simulation_end(
            self,
        ):
        self.log_df.to_csv(os.path.join(self.output_folder, "time_log.csv"))
        self.betweenness_record.to_file(os.path.join(self.output_folder, "betweenness_record.geoJSON"), driver="GeoJSON",  engine='pyogrio')
        self.betweenness_record.to_csv(os.path.join(self.output_folder, "betweenness_record.csv"))

        self.log("Simulation Output saved: ALL DONE")

def betweenness_flow_simulation(
        city_name=None,
        data_folder=None,
        output_folder=None,
        pairings_file="Pairings.csv",
        num_cores=8,
    ):

    if city_name is None:
        raise ValueError("parameter 'city_name' needs to be specified")

    if data_folder is None:
        data_folder = os.path.join("Cities", city_name, "Data")
    if output_folder is None:
        start_time = datetime.now()
        output_folder = os.path.join("Cities", f"{city_name}", "Simulations", f"{start_time.year}-{start_time.month:02d}-{start_time.day:02d} {start_time.hour:02d}-{start_time.minute:02d}")

    logger=Logger(output_folder)

    pairings = pd.read_csv(os.path.join(data_folder, pairings_file))

    # Shaqra is a town in Saudi Arabia. this name would be used to reference a generic place that we're running a simulation for
    shaqra = Zonal()

    shaqra.load_layer(
        layer_name='streets',
        file_path=os.path.join(data_folder,  pairings.at[0, "Network_File"])
    )

    logger.log(f"Network FIle Loaded, Projection: {shaqra.layers['streets'].gdf.crs}")


    for pairing_idx, pairing in pairings.iterrows():

        # Setting up a street network if this is the first pairing, or if the network weight changed from previous pairing
        if (pairing_idx == 0) or (pairings.at[pairing_idx, 'Network_Cost'] != pairings.at[pairing_idx-1, 'Network_Cost']):
            shaqra.create_street_network(
                source_layer='streets', 
                node_snapping_tolerance=0.00001,  #todo: remove parameter once a finalized default is set.
                weight_attribute=pairings.at[pairing_idx, 'Network_Cost'] if pairings.at[pairing_idx, 'Network_Cost'] != "Geometric" else None
            )
            logger.log("Network topology created", pairing)
            clean_network_nodes = shaqra.network.nodes.copy(deep=True)
        else:
            # either generate a new network, or flush nodes.
            shaqra.network.nodes = clean_network_nodes.copy(deep=True)



        # Loading layers, if they're not already loaded.
        if pairing["Origin_Name"] not in shaqra.layers:
            shaqra.load_layer(
                layer_name=pairing["Origin_Name"],
                file_path=os.path.join(data_folder, pairing["Origin_File"])
            )
            logger.log(f"{pairing['Origin_Name']} file {pairing['Origin_File']} Loaded, Projection: {shaqra.layers[pairing['Origin_Name']].gdf.crs}", pairing)

        if pairing["Destination_Name"] not in shaqra.layers:
            shaqra.load_layer(
                layer_name=pairing["Destination_Name"],
                file_path=os.path.join(data_folder, pairing["Destination_File"])
            )
            logger.log(f"{pairing['Destination_Name']} file {pairing['Destination_File']} Loaded, Projection: {shaqra.layers[pairing['Destination_Name']].gdf.crs}", pairing)

        

        shaqra.insert_node(
            layer_name=pairing['Origin_Name'], 
            label='origin', 
            weight_attribute=pairing['Origin_Weight'] if pairing['Origin_Weight'] != "Count" else None
        )
        shaqra.insert_node(
            layer_name=pairing['Destination_Name'], 
            label='destination', 
            weight_attribute=pairing['Destination_Weight'] if pairing['Destination_Weight'] != "Count" else None
        )

        logger.log("Origins and Destinations Inserted.", pairing)

        shaqra.create_graph()

        logger.log("NetworkX Graphs Created.", pairing)

        #parameter settings for turns and elastic weights
        shaqra.network.turn_penalty_amount = pairing['Turn_Penalty']
        shaqra.network.turn_threshold_degree = pairing['Turn_Threshold']
        shaqra.network.knn_weight = pairing['KNN_Weight']
        shaqra.network.knn_plateau = pairing['Plateau']

        #setting appropriate number of cores
        node_gdf = shaqra.network.nodes
        origin_gdf = node_gdf[node_gdf["type"] == "origin"]
        num_cores = min(origin_gdf.shape[0], num_cores)

        betweenness_output = paralell_betweenness_exposure(
            shaqra,
            search_radius=pairing['Radius'],
            detour_ratio=pairing['Detour'],
            decay=False if pairing['Elastic_Weights'] else pairing['Decay'],  # elastic weight already reduces origin weight factoring in decay. if this pairing uses elastic weights, don't decay again,
            decay_method=pairing['Decay_Mode'],
            beta=pairing['Beta'],
            num_cores=num_cores,
            path_detour_penalty='equal', # "power" | "exponent" | "equal"
            closest_destination=pairing['Closest_destination'],
            elastic_weight=pairing['Elastic_Weights'],
            turn_penalty=pairing['Turns'],
            path_exposure_attribute=None,
            return_path_record=False, 
            destniation_cap=None
        )

        logger.log("Betweenness estimated.", pairing)
        logger.pairing_end(shaqra, pairing)
    logger.simulation_end()
    return 

