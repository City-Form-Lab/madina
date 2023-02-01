from betweenness_functions import create_street_nodes_edges, parallel_betweenness_2, insert_nodes_v2, get_elastic_weight
import pandas as pd
import geopandas as gpd
import madina as md
from datetime import datetime
from pathlib import Path


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

    bronx_crs = "EPSG:6534"

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

    data_folder = "Cities\\NYC\\Bronx\\Data\\"
    output_folder = f"Cities\\NYC\\Bronx\\Simulations\\{start_time.year}" \
                    f"-{start_time.month}-{start_time.day} {start_time.hour}-{start_time.minute}\\ "

    pairings = gpd.read_file(data_folder + "pairings.csv")
    street_gdf = gpd.read_file(
        data_folder + pairings.at[0, "Network_File"]
    ).set_crs(bronx_crs, allow_override=True)

    betweenness_record = street_gdf.copy(deep=True)

    separate_simulation_records = {}
    network_weight_parameters = ["Geometric"]  # ["Perceived", "Geometric"]
    turn_penalty_parameters = [False, True]
    elastic_weight_parameters = [False, True]
    retain_previous_data = True

    for network_weight in network_weight_parameters:
        separate_simulation_records[network_weight] = {}
        for turn_penalty in turn_penalty_parameters:
            separate_simulation_records[network_weight][turn_penalty] = {}
            for elastic_weight in elastic_weight_parameters:
                separate_simulation_records[network_weight][turn_penalty][elastic_weight] = street_gdf.copy(deep=True)

    NTAs = gpd.read_file(
        "C:/Users/abdul/Dropbox (MIT)/115_NYCWalks/03_Data/01_Raw/NYCOpenData/NeighborhoodTabulationAreas"
        "/geo_export_73f5b4b0-c2ca-4646-9e84-d591f2562f1f.shp"
    )
    BX_NTAs = NTAs[NTAs["nta2020"].isin(["BX0101", "BX0102"])]
    scope = BX_NTAs.to_crs(street_gdf.crs).dissolve().buffer(804.672)
    scope = scope.geometry[0]
    center = scope.centroid
    # center = street_gdf["geometry"].unary_union.centroid
    # scope = center.buffer(50000)

    context_gdf = gpd.GeoDataFrame({"name": ["center", "scope"], "geometry": [center, scope]},
                                   crs=street_gdf.crs)
    street_gdf["width"] = 1
    md.Zonal.save_map(
        [
            {"gdf": street_gdf, "color": [0, 255, 255], "opacity": 1, 'width': "width"},
            {"gdf": context_gdf, "color": [125, 125, 125], "opacity": 0.1}
        ],
        centerX=BX_NTAs.to_crs(street_gdf.crs).dissolve().buffer(804.672).centroid.to_crs(
            md.Zonal.default_geographic_crs).at[0].coords[0][0],
        centerY=BX_NTAs.to_crs(street_gdf.crs).dissolve().buffer(804.672).centroid.to_crs(
            md.Zonal.default_geographic_crs).at[0].coords[0][1],
        basemap=False,
        filename="bronx_basemap.html"
    )

    bronx = md.Zonal(scope, projected_crs=street_gdf.crs)
    bronx.load_layer(
        layer_name='streets',
        file_path=data_folder + pairings.at[0, "Network_File"]
    )

    # make sure if street network contains polygons, they are corrected by using the polygon exterior as a line.
    for street_idx in street_gdf[street_gdf["geometry"].geom_type == "Polygon"].index:
        street_gdf.at[street_idx, "geometry"] = street_gdf.at[street_idx, "geometry"].exterior

    # making sure percieved distance is at least 1. (Negative edge costs are not allowed.)
    if "Perceived" in network_weight_parameters:
        street_gdf[pairings.at[0, "Network_Cost"]] = street_gdf[pairings.at[0, "Network_Cost"]].apply(
            lambda x: max(1, x))

    bronx.layers["streets"]["gdf"] = street_gdf

    create_street_nodes_edges(bronx,
                              source_layer="streets",
                              flatten_polylines=False,
                              node_snapping_tolerance=1,
                              fuse_2_degree_edges=False,
                              tolerance_angle=10,
                              solve_intersections=False,
                              loose_edge_trim_tolerance=0.001,
                              weight_attribute=pairings.at[
                                  0, "Network_Cost"] if "Perceived" in network_weight_parameters else None,
                              discard_redundant_edges=True
                              )
    perceived_network_weight = None
    if "Perceived" in network_weight_parameters:
        perceived_network_weight = bronx.layers["network_edges"]["gdf"]["weight"]

    geometric_network_weight = None
    if "Geometric" in network_weight_parameters:
        geometric_network_weight = bronx.layers["network_edges"]["gdf"]["geometry"].length

    clean_node_gdf = bronx.layers["network_nodes"]["gdf"].copy(deep=True)

    log = logger(log, {"time": datetime.now(), "event": "Network topology created."})
    for idx, pairing in pairings.iterrows():  # .loc[[6,5]]
        if pairing["Origin_Name"] not in bronx.layers:
            bronx.load_layer(
                layer_name=pairing["Origin_Name"],
                file_path=data_folder + pairing["Origin_File"]
            )
            origin_gdf = gpd.read_file(data_folder + pairing["Origin_File"]
                                       )
            if origin_gdf.crs == "EPSG:4326":
                origin_gdf["geometry"] = origin_gdf["geometry"].to_crs(bronx_crs)
            bronx.layers[pairing["Origin_Name"]]["gdf"] = origin_gdf.clip(scope)
            log = logger(log, {"time": datetime.now(), "origin": pairing["Origin_Name"],
                               "event": f"Data file loaded."})

        if pairing["Destination_Name"] not in bronx.layers:
            bronx.load_layer(
                layer_name=pairing["Destination_Name"],
                file_path=data_folder + pairing["Destination_File"]
            )
            destination_gdf = gpd.read_file(data_folder + pairing["Destination_File"]
                                            )
            if destination_gdf.crs == "EPSG:4326":
                destination_gdf["geometry"] = destination_gdf["geometry"].to_crs(bronx_crs)
            bronx.layers[pairing["Destination_Name"]]["gdf"] = destination_gdf.clip(scope)

            log = logger(log, {"time": datetime.now(), "destination": pairing["Destination_Name"],
                               "event": f"Data file loaded."})


        for network_weight in network_weight_parameters:
            if network_weight == "Perceived":
                bronx.layers["network_edges"]["gdf"]["weight"] = perceived_network_weight
            elif network_weight == "Geometric":
                bronx.layers["network_edges"]["gdf"]["weight"] = geometric_network_weight

            # making sure to clear any existing origins and destinations before adding new ones.
            bronx.layers["network_nodes"]["gdf"] = clean_node_gdf.copy(deep=True)

            
            insert_nodes_v2(
                bronx,
                label="origin",
                layer_name=pairing["Origin_Name"],
                weight_attribute=pairing["Origin_Weight"] if pairing["Origin_Weight"] != "Count" else None,
            )

            insert_nodes_v2(
                bronx,
                label="destination",
                layer_name=pairing["Destination_Name"],
                weight_attribute=pairing["Destination_Weight"] if pairing["Destination_Weight"] != "Count" else None,
            )

            log = logger(log, {"time": datetime.now(), "origin": pairing["Origin_Name"],
                               "destination": pairing["Destination_Name"],
                               "event": "Origins and Destinations prepared.", "distance": network_weight})

            for turn_penalty in [False, True]:
                retained_d_idxs = {}
                retained_paths = {}
                retained_distances = {}
                for elastic_weight in [False,
                                       True]:  # The order of these is important, as the weight is overriden by
                    # elastic weight as there is no clean way to update weight for now.

                    bronx.create_graph(light_graph=True, dense_graph=True)

                    log = logger(log, {"time": datetime.now(), "origin": pairing["Origin_Name"],
                                       "destination": pairing["Destination_Name"],
                                       "event": "Light and dense graphs prepared.", "distance": network_weight})

                    if elastic_weight:
                        get_elastic_weight(bronx,
                                           search_radius=800,
                                           detour_ratio=0.002,
                                           beta=0.002,
                                           decay=True,
                                           turn_penalty=turn_penalty,
                                           retained_d_idxs=retained_d_idxs if retain_previous_data else None
                                           )

                        log = logger(log, {"time": datetime.now(), "origin": pairing["Origin_Name"],
                                           "destination": pairing["Destination_Name"],
                                           "event": "Elastic Weights generated.",
                                           "distance": network_weight, "tune_penalty": turn_penalty,
                                           "elastic_weight": elastic_weight})

                    node_gdf = bronx.layers["network_nodes"]["gdf"]
                    origin_gdf = node_gdf[node_gdf["type"] == "origin"]
                    num_cores = min(origin_gdf.shape[0], 8)

                    betweenness_output = parallel_betweenness_2(
                        bronx,
                        search_radius=float(pairing["Radius"]),
                        detour_ratio=float(pairing["Detour"]),
                        decay=False,
                        decay_method="exponent",  # "power", "exponent"
                        beta=float(pairing["Beta"]),
                        path_detour_penalty="equal",  # "power", "exponent", "equal"
                        origin_weights=True,
                        closest_destination=False,
                        destination_weights=True,  # if pairing["Destination_Name"] != "Mosques" else False,
                        perceived_distance=False,  # This is handled in network creation, placeholder for future usel.
                        num_cores=num_cores,
                        light_graph=True,
                        turn_penalty=turn_penalty,
                        #retained_d_idxs=retained_d_idxs if (elastic_weight and retain_previous_data) else None,
                        #retained_paths=retained_paths if (elastic_weight and retain_previous_data) else None,
                        #retained_distances=retained_distances if (elastic_weight and retain_previous_data) else None,
                        rertain_expensive_data=False if (elastic_weight or (not retain_previous_data)) else True
                    )

                    if not elastic_weight and retain_previous_data:
                        retained_d_idxs = betweenness_output["retained_d_idxs"]
                        #retained_paths = betweenness_output["retained_paths"]
                        #retained_distances = betweenness_output["retained_distances"]

                    log = logger(log, {"time": datetime.now(), "origin": pairing["Origin_Name"],
                                       "destination": pairing["Destination_Name"], "event": "Betweenness estimated.",
                                       "distance": network_weight, "tune_penalty": turn_penalty,
                                       "elastic_weight": elastic_weight})

                    # creating a folder for output
                    pairing_folder = output_folder + f"{network_weight}" \
                                                     f"_{'with_turns' if turn_penalty else 'no_turns'}" \
                                                     f"_{'elastic_weight' if elastic_weight else 'unadjusted_weight'}" \
                                                     f"_O({pairing['Origin_Name']})" \
                                                     f"_D({pairing['Destination_Name']})\\"
                    Path(pairing_folder).mkdir(parents=True, exist_ok=True)

                    streets = bronx.layers["streets"]["gdf"]
                    edge_gdf = bronx.layers["network_edges"]["gdf"]

                    betweenness_record = betweenness_record.join(
                        edge_gdf[['parent_street_id', 'betweenness']].set_index('parent_street_id')).rename(
                        columns={
                            "betweenness": f"{network_weight}"
                                           f"_{'with_turns' if turn_penalty else 'no_turns'}"
                                           f"_{'elastic_weight' if elastic_weight else 'unadjusted_weight'}"
                                           f"_{pairing['Between_Name']}"})

                    separate_simulation_records[network_weight][turn_penalty][elastic_weight] = \
                        separate_simulation_records[network_weight][turn_penalty][elastic_weight] \
                        .join(edge_gdf[['parent_street_id', 'betweenness']].set_index('parent_street_id')) \
                        .rename(columns={"betweenness": pairing['Between_Name']})

                    save_results = \
                        edge_gdf.set_index('parent_street_id').join(streets, lsuffix='_from_edge')[
                            ["betweenness", "__GUID", "geometry"]]
                    save_results = save_results.rename(
                        columns={
                            "betweenness": f"{network_weight}"
                                           f"_{'with_turns' if turn_penalty else 'no_turns'}"
                                           f"_{'elastic_weight' if elastic_weight else 'unadjusted_weight'}"
                                           f"_{pairing['Between_Name']}"})

                    save_results.to_csv(pairing_folder + "flows.csv")
                    save_results.to_file(pairing_folder + "flows.geojson", driver="GeoJSON")
                    betweenness_record.to_csv(pairing_folder + "betweenness_record_so_far.csv")
                    separate_simulation_records[network_weight][turn_penalty][elastic_weight].to_csv(
                        pairing_folder + "simulation_record_so_far.csv")

                    node_gdf = bronx.layers["network_nodes"]["gdf"]
                    origin_gdf = node_gdf[node_gdf["type"] == "origin"]
                    destination_gdf = node_gdf[node_gdf["type"] == "destination"]
                    edge_gdf = bronx.layers["network_edges"]["gdf"]
                    edge_gdf["width"] = edge_gdf["betweenness"] / edge_gdf["betweenness"].mean() + 0.25
                    md.Zonal.save_map(
                        [

                            {"gdf": street_gdf, "color": [0, 255, 255], "opacity": 0.1},
                            {"gdf": origin_gdf.drop("nearest_street_node_distance", axis=1), "color": [0, 255, 0],
                             "opacity": 0.1},
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

    for network_weight in network_weight_parameters:
        for turn_penalty in turn_penalty_parameters:
            for elastic_weight in elastic_weight_parameters:
                separate_simulation_records[network_weight][turn_penalty][elastic_weight].to_csv(
                    output_folder + f"{network_weight}"
                                    f"_{'with_turns' if turn_penalty else 'no_turns'}"
                                    f"_{'elastic_weight' if elastic_weight else 'unadjusted_weight'}"
                                    f"_betweenness_record.csv")

                separate_simulation_records[network_weight][turn_penalty][elastic_weight].to_file(
                    output_folder + f"{network_weight}"
                                    f"_{'with_turns' if turn_penalty else 'no_turns'}"
                                    f"_{'elastic_weight' if elastic_weight else 'unadjusted_weight'}"
                                    f"_street_network_betweenness_record.geojson",
                    driver="GeoJSON")

    log = logger(log, {"time": datetime.now(), "event": "All DONE."})
    log.to_csv(output_folder + "time_log.csv")
