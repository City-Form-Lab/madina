from betweenness_functions import create_street_nodes_edges, parallel_betweenness_2, insert_nodes_v2, get_elastic_weight

import pandas as pd
import geopandas as gpd
import madina as md
from datetime import datetime
from pathlib import Path
from shapely.ops import transform


'''
betweenness_output = parallel_betweenness_2(
    beirut,
    search_radius=float(pairing["Radius"]),
    detour_ratio=float(pairing["Detour"]),
    decay=False if elastic_weight else True,
    decay_method="exponent",  # "power", "exponent"
    beta=float(pairing["Beta"]),
    path_detour_penalty="equal",  # "power", "exponent", "equal"
    origin_weights=True,
    closest_destination=False,
    destination_weights=True,  # if pairing["Destination_Name"] != "Mosques" else False,
    # perceived_distance=False, This is handled in network creation, placeholder for future usel.
    num_cores=1,
    light_graph=True,
    turn_penalty=turn_penalty,
    #retained_d_idxs=retained_d_idxs if elastic_weight else None,
    #retained_paths=retained_paths if elastic_weight else None,
    #retained_distances=retained_distances if elastic_weight else None,
    #rertain_expensive_data=False if elastic_weight else True
    retained_d_idxs=None,
    retained_paths=None,
    retained_distances=None,
    rertain_expensive_data=False
)


paths, weights, d_idxs = best_path_generator_so_far(beirut, 6593, search_radius=800, detour_ratio=1.15, turn_penalty=False)

md.Zonal.save_map(
    [

        {"gdf": street_gdf, "color": [0, 255, 255], "opacity": 0.1},
        {"gdf": origin_gdf.drop("nearest_street_node_distance", axis=1), "color": [0, 255, 0], "opacity": 0.1},
        {"gdf": destination_gdf.drop("nearest_street_node_distance", axis=1), "color": [255, 0, 0]},
        {"gdf": edge_gdf[["geometry", "betweenness", "width"]], "color_by_attribute": "betweenness", "opacity": 1.0,
         "color_method": "quantile",
         "width": "width", "text": "betweenness"},
        # {"gdf": context_gdf, "color": [125, 125, 125], "opacity": 0.05}
    ],
    centerX=context_gdf.to_crs(md.Zonal.default_geographic_crs).at[0, "geometry"].coords[0][0],
    centerY=context_gdf.to_crs(md.Zonal.default_geographic_crs).at[0, "geometry"].coords[0][1],
    basemap=False,
    filename=pairing_folder + "flow_maps.html"
)

turn_o_scope(beirut, 14605, 800, 1.15,

min([x[2] for x in list(beirut.d_graph.edges.data("weight"))])
'''


def logger(log_pd, input_dict):
    if log_pd.shape[0] == 0:
        print(f"total time\tseconds elapsed\tdiatance method\t{'origin':^15s}\t{'destination':^15s}\tevent")
        input_dict["seconds_elapsed"] = 0
        input_dict["cumulative_seconds"] = 0
    else:
        time_elapsed = (input_dict["time"] - log_pd.iloc[-1]["time"]).seconds
        input_dict["seconds_elapsed"] = time_elapsed
        input_dict["cumulative_seconds"] = log_pd["seconds_elapsed"].sum() + time_elapsed

    for column_name in log_pd.columns:
        if column_name not in input_dict:
            input_dict[column_name] = "---"

    log_pd = log_pd.append(input_dict, ignore_index=True)
    print(
        f"{input_dict['cumulative_seconds']:6.4f}\t\t"
        f"{input_dict['seconds_elapsed']}\t\t\t\t"
        f"{input_dict['distance']}\t\t"
        f"{input_dict['tune_penalty']}\t\t"
        f"{input_dict['elastic_weight']}\t\t"
        f"{input_dict['origin']:^15s}\t"
        f"{input_dict['destination']:^15s}\t"
        f"{input_dict['event']}"
    )
    return log_pd


if __name__ == "__main__":

    log = pd.DataFrame(
        {
            "time": pd.Series(dtype='datetime64[ns]'),
            "distance": pd.Series(dtype="string"),
            "tune_penalty": pd.Series(dtype="string"),
            "elastic_weight": pd.Series(dtype="string"),
            "origin": pd.Series(dtype="string"),
            "destination": pd.Series(dtype="string"),
            "event": pd.Series(dtype="string")
        }
    )
    start_time = datetime.now()
    log = logger(log, {"time": start_time, "event": "beginning"})

    data_folder = "Cities\\Beirut\\Data\\"
    output_folder = f"Cities\\Beirut\\Simulations\\{start_time.year}-{start_time.month}-{start_time.day} {start_time.hour}-{start_time.minute}\\ "

    pairings = gpd.read_file(data_folder + "Pairings.csv")

    # START ################
    # This code segmewnt is completely unneccissary if we handled center, context internally instead of doing it during
    # madina.__init__()
    street_gdf = gpd.read_file(
        data_folder + pairings.at[0, "Network_File"]
    )


    def _to_2d(x, y, z):
        return tuple(filter(None, [x, y]))

    if street_gdf.has_z.any():
        street_gdf["geometry"] = street_gdf["geometry"].apply(lambda s: transform(_to_2d, s))

    betweenness_record = street_gdf.copy(deep=True)

    separate_simulation_records = {}
    for network_weight in ["Perceived", "Geometric"]:
        separate_simulation_records[network_weight] = {}
        for turn_penalty in [False, True]:
            separate_simulation_records[network_weight][turn_penalty] = {}
            for elastic_weight in [False, True]:
                separate_simulation_records[network_weight][turn_penalty][elastic_weight] = street_gdf.copy(deep=True)

    center = street_gdf["geometry"].unary_union.centroid
    scope = center.buffer(50000)

    context_gdf = gpd.GeoDataFrame({"name": ["center", "scope"], "geometry": [center, scope]},
                                   crs=street_gdf.crs)
    # END ################

    beirut = md.Zonal(scope, projected_crs=street_gdf.crs)
    beirut.load_layer(
        layer_name='streets',
        file_path=data_folder +  pairings.at[0, "Network_File"]  # "Network.geojson"
    )




    if beirut.layers["streets"]["gdf"].has_z.any():
        beirut.layers["streets"]["gdf"]["geometry"] = beirut.layers["streets"]["gdf"][
            "geometry"].apply(lambda s: transform(_to_2d, s))


    create_street_nodes_edges(beirut,
                              source_layer="streets",
                              flatten_polylines=False,
                              node_snapping_tolerance=1,
                              fuse_2_degree_edges=False,
                              tolerance_angle=10,
                              solve_intersections=False,
                              loose_edge_trim_tolerance=0.001,
                              weight_attribute=pairings.at[0, "Network_Cost"],
                              discard_redundant_edges=True
                              )

    perceived_network_weight = beirut.layers["network_edges"]["gdf"]["weight"]
    perceived_network_weight = perceived_network_weight.apply(lambda x: max(1, x))
    geometric_network_weight = beirut.layers["network_edges"]["gdf"]["geometry"].length

    clean_node_gdf = beirut.layers["network_nodes"]["gdf"].copy(deep=True)
    log = logger(log, {"time": datetime.now(), "event": "Network topology created."})

    for idx, pairing in pairings.iterrows():
        if pairing["Origin_Name"] not in beirut.layers:
            beirut.load_layer(
                layer_name=pairing["Origin_Name"],
                file_path=data_folder + pairing["Origin_File"]
            )

            if beirut.layers[pairing["Origin_Name"]]["gdf"].has_z.any():
                beirut.layers[pairing["Origin_Name"]]["gdf"]["geometry"] = beirut.layers[pairing["Origin_Name"]]["gdf"]["geometry"].apply(
                    lambda s: transform(_to_2d, s))
        if pairing["Destination_Name"] not in beirut.layers:
            beirut.load_layer(
                layer_name=pairing["Destination_Name"],
                file_path=data_folder + pairing["Destination_File"]
            )

            if beirut.layers[pairing["Destination_Name"]]["gdf"].has_z.any():
                beirut.layers[pairing["Destination_Name"]]["gdf"]["geometry"] = beirut.layers[pairing["Destination_Name"]]["gdf"][
                    "geometry"].apply(lambda s: transform(_to_2d, s))

        # making sure to clear any existing origins and destinations before adding new ones.

        ###################
        for network_weight in ["Perceived", "Geometric"]:
            if network_weight == "Perceived":
                beirut.layers["network_edges"]["gdf"]["weight"] = perceived_network_weight
            elif network_weight == "Geometric":
                beirut.layers["network_edges"]["gdf"]["weight"] = geometric_network_weight

            # using an effecient insert algorithm, should be built inti the main Madina code.
            beirut.layers["network_nodes"]["gdf"] = clean_node_gdf.copy(deep=True)

            '''
            beirut.insert_nodes(
                label="origin",  # "origin" | "destination" | "observer"
                layer_name=pairing["Origin_Name"],
                filter=None,
                attachment_method="light_insert",  # "project" | "snap" | "light_insert"
                representative_point="centroid",  # "nearest_point" | "centroid"
                node_weight_attribute=pairing["Origin_Weight"],
                projection_edge_cost_attribute=None
            )

            beirut.insert_nodes(
                label="destination",  # "origin" | "destination" | "observer"
                layer_name=pairing["Destination_Name"],
                filter=None,
                attachment_method="light_insert",  # "project" | "snap" | "light_insert"
                representative_point="centroid",  # "nearest_point" | "centroid"
                node_weight_attribute=pairing["Destination_Weight"] if pairing[
                                                                           "Destination_Name"] != "Mosques" else "Field",
                projection_edge_cost_attribute=None
            )
            
            '''
            insert_nodes_v2(
                beirut,
                label="origin",
                layer_name=pairing["Origin_Name"],
                weight_attribute=pairing["Origin_Weight"] if pairing["Origin_Weight"] != "Count" else None,
            )

            insert_nodes_v2(
                beirut,
                label="destination",
                layer_name=pairing["Destination_Name"],
                weight_attribute=pairing["Destination_Weight"] if pairing["Destination_Weight"] != "Count" else None,
            )





            log = logger(log, {"time": datetime.now(), "origin": pairing["Origin_Name"],
                               "destination": pairing["Destination_Name"],
                               "event": "Origins and Destinations prepared."})







            for turn_penalty in [False, True]:
                retained_d_idxs = {}
                retained_paths = {}
                retained_distances = {}
                for elastic_weight in [False,
                                       True]:  # The order of these is important, as the weight is overriden by
                    # elastic weight as there is no clean way to update weight for now.

                    beirut.create_graph(light_graph=True, dense_graph=True, d_graph=True)
                    log = logger(log, {"time": datetime.now(), "origin": pairing["Origin_Name"],
                                       "destination": pairing["Destination_Name"],
                                       "event": "Light and dense graphs prepared."})

                    if elastic_weight:
                        #print(f"Attempting to generate elastic weights...")
                        get_elastic_weight(beirut,
                                           search_radius=800,
                                           detour_ratio=1.15,
                                           beta=0.002,
                                           decay=True,
                                           turn_penalty=turn_penalty,
                                           retained_d_idxs=retained_d_idxs
                                           #retained_d_idxs=None
                                           )

                        log = logger(log, {"time": datetime.now(), "origin": pairing["Origin_Name"],
                                           "destination": pairing["Destination_Name"],
                                           "event": "Elastic Weights generated.",
                                           "distance": network_weight, "tune_penalty": turn_penalty,
                                           "elastic_weight": elastic_weight})

                    node_gdf = beirut.layers["network_nodes"]["gdf"]
                    origin_gdf = node_gdf[node_gdf["type"] == "origin"]
                    num_cores = min(origin_gdf.shape[0], 36) # if not elastic_weight else 1

                    betweenness_output = parallel_betweenness_2(
                        beirut,
                        search_radius=float(pairing["Radius"]),
                        detour_ratio=float(pairing["Detour"]),
                        decay=False, # if elastic_weight else True,
                        decay_method="exponent",  # "power", "exponent"
                        beta=float(pairing["Beta"]),
                        path_detour_penalty="equal",  # "power", "exponent", "equal"
                        origin_weights=True,
                        closest_destination=False,
                        destination_weights=True,  # if pairing["Destination_Name"] != "Mosques" else False,
                        # perceived_distance=False, This is handled in network creation, placeholder for future usel.
                        num_cores=num_cores,
                        light_graph=True,
                        turn_penalty=turn_penalty,
                        #retained_d_idxs=retained_d_idxs if elastic_weight else None,
                        #retained_paths=retained_paths if elastic_weight else None,
                        #retained_distances=retained_distances if elastic_weight else None,
                        rertain_expensive_data=False if elastic_weight else True
                        #retained_d_idxs=None,
                        #retained_paths=None,
                        #retained_distances=None,
                        #rertain_expensive_data=False
                    )

                    if not elastic_weight:
                        retained_d_idxs = betweenness_output["retained_d_idxs"]
                        #retained_paths = betweenness_output["retained_paths"]
                        #retained_distances = betweenness_output["retained_distances"]

                    log = logger(log, {"time": datetime.now(), "origin": pairing["Origin_Name"],
                                       "destination": pairing["Destination_Name"], "event": "Betweenness estimated.",
                                       "distance": network_weight, "tune_penalty": turn_penalty,
                                       "elastic_weight": elastic_weight})

                    # creating a folder for output
                    pairing_folder = output_folder + f"{network_weight}_{'with_turns' if turn_penalty else 'no_turns'}_{'elastic_weight' if elastic_weight else 'unadjusted_weight'}_O({pairing['Origin_Name']})_D({pairing['Destination_Name']})\\"
                    Path(pairing_folder).mkdir(parents=True, exist_ok=True)

                    streets = beirut.layers["streets"]["gdf"]
                    edge_gdf = beirut.layers["network_edges"]["gdf"]

                    betweenness_record = betweenness_record.join(
                        edge_gdf[['parent_street_id', 'betweenness']].set_index('parent_street_id')).rename(
                        columns={
                            "betweenness": f"{network_weight}_{'with_turns' if turn_penalty else 'no_turns'}_{'elastic_weight' if elastic_weight else 'unadjusted_weight'}_{pairing['Between_Name']}"})

                    separate_simulation_records[network_weight][turn_penalty][elastic_weight] = \
                    separate_simulation_records[network_weight][turn_penalty][elastic_weight].join(
                        edge_gdf[['parent_street_id', 'betweenness']].set_index('parent_street_id')).rename(
                        columns={
                            "betweenness": f"{network_weight}_{'with_turns' if turn_penalty else 'no_turns'}_{'elastic_weight' if elastic_weight else 'unadjusted_weight'}_{pairing['Between_Name']}"})

                    save_results = \
                        edge_gdf.set_index('parent_street_id').join(streets, lsuffix='_from_edge')[
                            # , rsuffix='_from_streets'
                            ["betweenness", "__GUID", "geometry"]]
                    save_results = save_results.rename(
                        columns={
                            "betweenness": f"{network_weight}_{'with_turns' if turn_penalty else 'no_turns'}_{'elastic_weight' if elastic_weight else 'unadjusted_weight'}_{pairing['Between_Name']}"})

                    save_results.to_csv(pairing_folder + "flows.csv")
                    save_results.to_file(pairing_folder + "flows.geojson", driver="GeoJSON")
                    betweenness_record.to_csv(pairing_folder + "betweenness_record_so_far.csv")
                    separate_simulation_records[network_weight][turn_penalty][elastic_weight].to_csv(
                        pairing_folder + "simulation_record_so_far.csv")

                    node_gdf = beirut.layers["network_nodes"]["gdf"]
                    origin_gdf = node_gdf[node_gdf["type"] == "origin"]
                    destination_gdf = node_gdf[node_gdf["type"] == "destination"]
                    edge_gdf = beirut.layers["network_edges"]["gdf"]
                    edge_gdf["width"] = edge_gdf["betweenness"] / edge_gdf["betweenness"].mean() + 0.25
                    md.Zonal.save_map(
                        [

                            {"gdf": street_gdf, "color": [0, 255, 255], "opacity": 0.1},
                            {"gdf": origin_gdf.drop("nearest_street_node_distance", axis=1), "color": [0, 255, 0], "opacity": 0.1},
                            {"gdf": destination_gdf.drop("nearest_street_node_distance", axis=1), "color": [255, 0, 0]},
                            {"gdf": edge_gdf, "color_by_attribute": "betweenness", "opacity": 1.0,
                             "color_method": "quantile",
                             "width": "width", "text": "betweenness"},
                            # {"gdf": context_gdf, "color": [125, 125, 125], "opacity": 0.05}
                        ],
                        centerX=context_gdf.to_crs(md.Zonal.default_geographic_crs).at[0, "geometry"].coords[0][0],
                        centerY=context_gdf.to_crs(md.Zonal.default_geographic_crs).at[0, "geometry"].coords[0][1],
                        basemap=False,
                        filename=pairing_folder + "flow_maps.html"
                    )

                    log = logger(log, {"time": datetime.now(), "origin": pairing["Origin_Name"],
                                       "destination": pairing["Destination_Name"], "event": "Output saved",
                                       "distance": network_weight, "tune_penalty": turn_penalty,
                                       "elastic_weight": elastic_weight})

                    log.to_csv(pairing_folder + "time_log.csv")

    betweenness_record.to_csv(output_folder + "betweenness_record.csv")
    betweenness_record.to_file(output_folder + "street_network_betweenness_record.geojson", driver="GeoJSON")
    for network_weight in ["Perceived", "Geometric"]:
        for turn_penalty in [False, True]:
            for elastic_weight in [False, True]:
                separate_simulation_records[network_weight][turn_penalty][elastic_weight].to_csv(
                    output_folder + f"{network_weight}_{'with_turns' if turn_penalty else 'no_turns'}_{'elastic_weight' if elastic_weight else 'unadjusted_weight'}_betweenness_record.csv")
                separate_simulation_records[network_weight][turn_penalty][elastic_weight].to_file(
                    output_folder + f"{network_weight}_{'with_turns' if turn_penalty else 'no_turns'}_{'elastic_weight' if elastic_weight else 'unadjusted_weight'}_street_network_betweenness_record.geojson",
                    driver="GeoJSON")

    log = logger(log, {"time": datetime.now(), "event": "All DONE."})
    log.to_csv(output_folder + "time_log.csv")
