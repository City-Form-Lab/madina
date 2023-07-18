import math
import numpy as np

from madina.zonal.network import Network
from madina.una.una_utils import turn_o_scope

import math



def get_elastic_weight(network: Network,
                search_radius: float, 
                detour_ratio: float, 
                beta: float, 
                decay=False, 
                turn_penalty=False, 
                retained_d_idxs=None):
    """
    Assign elastic weights to the origins in the given network
    TODO: fill function spec
    """

    node_gdf = network.nodes
    origins = node_gdf[node_gdf["type"] == "origin"]

    o_reach = {}
    o_gravity = {}

    for o_idx in origins.index:
        if retained_d_idxs is None:
            d_idxs, _, _ = turn_o_scope(network, o_idx, search_radius, detour_ratio,
                                        turn_penalty=turn_penalty, o_graph=None, return_paths=False)
        else:
            d_idxs = retained_d_idxs[o_idx]
        o_reach[o_idx] = int(len(d_idxs))

        destination_weights = network.nodes.loc[list(d_idxs.keys())]["weight"].values
        #print (destination_weights, np.array(list(d_idxs.values())))
        #print (destination_weights / pow(math.e, (beta * np.array(list(d_idxs.values())))))
        o_gravity[o_idx] = sum(destination_weights / pow(math.e, (beta * np.array(list(d_idxs.values())))))

    a = 0.5
    b = 1

    if decay:
        access = o_gravity
    else:
        access = o_reach

    #min_access = min(access.values())
    min_access = min(filter(lambda x: x != 0, access.values()))
    max_access = max(access.values())
    #print (f"{min_access = }\t{max_access = }\t{min_access2 = }")
    for o_idx in origins.index:
        if (max_access - min_access) == 0:
            scaled_access = (a+b)/2
        else:
            scaled_access = (b - a) * ((access[o_idx] - min_access) / (max_access - min_access)) + a



        scaled_weight = origins.at[o_idx, "weight"] * scaled_access

        if math.isnan(scaled_weight):
            print (f"{scaled_weight = }\t{origins.at[o_idx, 'weight'] = }\t {scaled_access}")

        # TODO: This overrides original weights by elastic weights. should think of a way to pass this as options for una algorithms.
        node_gdf.at[o_idx, "scaled_access"] = scaled_access
        node_gdf.at[o_idx, "elastic_weight"] = scaled_weight
        node_gdf.at[o_idx, "gravity"] = o_gravity[o_idx]
        node_gdf.at[o_idx, "reach"] = o_reach[o_idx]

    network.nodes = node_gdf
    return
