from betweenness_functions import create_street_nodes_edges, parallel_betweenness_2
import pandas as pd
import geopandas as gpd
import madina as Madina
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

    if "origin" not in input_dict:
        input_dict['origin'] = "---"
    if "destination" not in input_dict:
        input_dict['destination'] = "---"
    if "distance" not in input_dict:
        input_dict['distance'] = "---"

    log_pd = log_pd.append(input_dict, ignore_index=True)
    print(
        f"{input_dict['cumulative_seconds']:6.4f}\t\t"
        f"{input_dict['seconds_elapsed']}\t\t\t\t"
        f"{input_dict['distance']}\t\t"
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
    street_gdf = gpd.read_file(
        data_folder + pairings.at[0, "Network_File"]
    )

    betweenness_record = street_gdf.copy(deep=True)

    center = street_gdf["geometry"].unary_union.centroid
    scope = center.buffer(50000)

    context_gdf = gpd.GeoDataFrame({"name": ["center", "scope"], "geometry": [center, scope]},
                                   crs=street_gdf.crs)

    for weight_attribute in [None]:  # ["PER_LEN_AM",None]:
        beirut = Madina.Zonal(scope, projected_crs=street_gdf.crs)
        beirut.load_layer(
            layer_name='streets',
            file_path=data_folder + pairings.at[0, "Network_File"]
        )
        if weight_attribute is None:
            distance_name = "geometric_distance"
        else:
            distance_name = weight_attribute

        create_street_nodes_edges(beirut,
                                  source_layer="streets",
                                  flatten_polylines=False,
                                  node_snapping_tolerance=1,
                                  fuse_2_degree_edges=False,
                                  tolerance_angle=10,
                                  solve_intersections=False,
                                  loose_edge_trim_tolerance=0.001,
                                  weight_attribute=weight_attribute,
                                  discard_redundant_edges=True
                                  )
        log = logger(log, {"time": datetime.now(), "event": "Network topology created.", "distance": distance_name})
        clean_node_gdf = beirut.layers["network_nodes"]["gdf"].copy(deep=True)

        for idx, pairing in pairings.loc[[20]].iterrows():
            if pairing["Origin_Name"] not in beirut.layers:
                beirut.load_layer(
                    layer_name=pairing["Origin_Name"],
                    file_path=data_folder + pairing["Origin_File"]
                )
            if pairing["Destination_Name"] not in beirut.layers:
                beirut.load_layer(
                    layer_name=pairing["Destination_Name"],
                    file_path=data_folder + pairing["Destination_File"]
                )

            # making sure to clear any existing origins and destinations before adding new ones.
            beirut.layers["network_nodes"]["gdf"] = clean_node_gdf.copy(deep=True)

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

            log = logger(log, {"time": datetime.now(), "origin": pairing["Origin_Name"],
                               "destination": pairing["Destination_Name"],
                               "event": "Origins and Destinations prepared.", "distance": distance_name})

            beirut.create_graph(light_graph=True, dense_graph=True)

            log = logger(log, {"time": datetime.now(), "origin": pairing["Origin_Name"],
                               "destination": pairing["Destination_Name"],
                               "event": "Light and dense graphs prepared.", "distance": distance_name})


            node_gdf = beirut.layers["network_nodes"]["gdf"]
            origin_gdf = node_gdf[node_gdf["type"] == "origin"]
            num_cores = min(origin_gdf.shape[0], 8)

            parallel_betweenness_2(
                beirut,
                search_radius=float(pairing["Radius"]),
                detour_ratio=float(pairing["Detour"]),
                decay=False,
                decay_method="exponent",                                    # "power", "exponent"
                beta=float(pairing["Beta"]),
                path_detour_penalty="equal",                               # "power", "exponent", "equal"
                origin_weights=True,
                closest_destination=False,
                destination_weights=True,                                   #if pairing["Destination_Name"] != "Mosques" else False,
                perceived_distance=False,
                num_cores=num_cores,
                light_graph=True
            )

            log = logger(log, {"time": datetime.now(), "origin": pairing["Origin_Name"],
                               "destination": pairing["Destination_Name"], "event": "Betweenness estimated.",
                               "distance": distance_name})

            # creating a folder for output
            pairing_folder = output_folder + f"Flows using {distance_name} from {pairing['Origin_Name']} to {pairing['Destination_Name']}\\"
            Path(pairing_folder).mkdir(parents=True, exist_ok=True)

            streets = beirut.layers["streets"]["gdf"]
            edge_gdf = beirut.layers["network_edges"]["gdf"]

            betweenness_record = betweenness_record.join(
                edge_gdf[['parent_street_id', 'betweenness']].set_index('parent_street_id')).rename(
                columns={"betweenness": f"{distance_name}_{pairing['Between_Name']}"})

            save_results = \
            edge_gdf.set_index('parent_street_id').join(streets, lsuffix='_from_edge')[   #, rsuffix='_from_streets'
                ["betweenness", "__GUID", "geometry"]]
            save_results = save_results.rename(columns={"betweenness": f"{distance_name}_{pairing['Between_Name']}"})
            save_results.to_csv(pairing_folder + "flows.csv")
            save_results.to_file(pairing_folder + "flows.geojson", driver="GeoJSON")
            betweenness_record.to_csv(pairing_folder + "betweenness_record_so_far.csv")

            node_gdf = beirut.layers["network_nodes"]["gdf"]
            origin_gdf = node_gdf[node_gdf["type"] == "origin"]
            destination_gdf = node_gdf[node_gdf["type"] == "destination"]
            edge_gdf = beirut.layers["network_edges"]["gdf"]
            edge_gdf["width"] = edge_gdf["betweenness"] / edge_gdf["betweenness"].mean() + 0.25
            Madina.Zonal.save_map(
                [

                    {"gdf": street_gdf, "color": [0, 255, 255], "opacity": 0.1},
                    {"gdf": origin_gdf, "color": [0, 255, 0], "opacity": 0.1},
                    {"gdf": destination_gdf, "color": [255, 0, 0]},
                    {"gdf": edge_gdf, "color_by_attribute": "betweenness", "opacity": 1.0, "color_method": "quantile",
                     "width": "width", "text": "betweenness"},
                    #{"gdf": context_gdf, "color": [125, 125, 125], "opacity": 0.05}
                ],
                centerX=context_gdf.to_crs(Madina.Zonal.default_geographic_crs).at[0, "geometry"].coords[0][0],
                centerY=context_gdf.to_crs(Madina.Zonal.default_geographic_crs).at[0, "geometry"].coords[0][1],
                basemap=False,
                filename=pairing_folder + "flow_maps.html"
            )

            log = logger(log, {"time": datetime.now(), "origin": pairing["Origin_Name"],
                               "destination": pairing["Destination_Name"], "event": "Output saved",
                               "distance": distance_name})

            log.to_csv(pairing_folder + "time_log.csv")

        betweenness_record.to_csv(output_folder + "betweenness_record.csv")
        betweenness_record.to_file(output_folder + "street_network_betweenness_record.geojson", driver="GeoJSON")

        log = logger(log, {"time": datetime.now(), "event": "Distance Method Done.", "distance": distance_name})
    log = logger(log, {"time": datetime.now(), "event": "All DONE."})
    log.to_csv(output_folder + "time_log.csv")
