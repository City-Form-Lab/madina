import os
from datetime import datetime
import pandas as pd
from madina.una.betweenness import Logger, get_origin_properties
from madina.una.betweenness import turn_o_scope
from madina.zonal.zonal import Zonal
from pathlib import Path
import time

def KNN_workflow(
        city_name=None,
        data_folder=None,
        output_folder=None,
        pairings_file="pairing.csv",
        num_cores=8,
    ):
    origin_record =  None

    if city_name is None:
        raise ValueError("parameter 'city_name' needs to be specified")

    if data_folder is None:
        data_folder = os.path.join("Cities", city_name, "Data")
    if output_folder is None:
        start_time = datetime.now()
        output_folder = os.path.join("Cities", f"{city_name}", "KNN_workflow", f"{start_time.year}-{start_time.month:02d}-{start_time.day:02d} {start_time.hour:02d}-{start_time.minute:02d}")

    logger=Logger(output_folder)

    pairings = pd.read_csv(os.path.join(data_folder, pairings_file))



    for pairing_idx, pairing in pairings.iterrows():

        # Shaqra is a town in Saudi Arabia. this name would be used to reference a generic place that we're running a simulation for
        shaqra = Zonal()

        shaqra.load_layer(
            layer_name='streets',
            file_path=os.path.join(data_folder,  pairings.at[pairing_idx, "Network_File"])
        )

        logger.log(f"network FIle Loaded, Projection: {shaqra.layers['streets'].gdf.crs}", pairing)

        shaqra.create_street_network(
            source_layer='streets', 
            node_snapping_tolerance=0.00001,  #todo: remove parameter once a finalized default is set.
            weight_attribute=pairings.at[pairing_idx, 'Network_Cost'] if pairings.at[pairing_idx, 'Network_Cost'] != "Geometric" else None
        )
        logger.log("network topology created", pairing)




        # Loading layers, if they're not already loaded.

        shaqra.load_layer(
            layer_name=pairing["Origin_Name"],
            file_path=os.path.join(data_folder, pairing["Origin_File"])
        )
        logger.log(f"{pairing['Origin_Name']} file {pairing['Origin_File']} Loaded, Projection: {shaqra.layers[pairing['Origin_Name']].gdf.crs}", pairing)


        shaqra.load_layer(
            layer_name=pairing["Destination_Name"],
            file_path=os.path.join(data_folder, pairing["Destination_File"])
        )
        logger.log(f"{pairing['Destination_Name']} file {pairing['Destination_File']} Loaded, Projection: {shaqra.layers[pairing['Destination_Name']].gdf.crs}", pairing)

        

        shaqra.insert_node(
            layer_name=pairing['Origin_Name'], 
            label='origin', 
            weight_attribute=None
        )
        shaqra.insert_node(
            layer_name=pairing['Destination_Name'], 
            label='destination', 
            weight_attribute=pairing['Destination_Weight'] if pairing['Destination_Weight'] != "Count" else None
        )

        logger.log("Origins and Destinations Inserted.", pairing)
        shaqra.create_graph()
        logger.log("NetworkX Graphs Created.", pairing)



        if origin_record is None:
            origin_record = shaqra.layers[pairing['Origin_Name']].gdf.copy(deep=True)

        origins = shaqra.network.nodes[shaqra.network.nodes['type'] == 'origin']
        o_graph = shaqra.network.d_graph
        
        start = time.time()
        i = 0
        for origin_idx in origins.index:
            print (f"Time spent: {round(time.time()-start):,}s [Done {i:,} of {origins.shape[0]:,} origins ({i  /origins.shape[0] * 100:4.2f}%)]",  end='\r')
            i = i + 1

            shaqra.network.add_node_to_graph(o_graph, origin_idx)

            #parameter settings for turns and elastic weights
            shaqra.network.turn_penalty_amount = pairing['Turn_Penalty']
            shaqra.network.turn_threshold_degree = pairing['Turn_Threshold']
            shaqra.network.knn_weight = pairing['KNN_Weight']
            shaqra.network.knn_plateau = pairing['Plateau']

            d_idxs, _, _ = turn_o_scope(
                network=shaqra.network,
                o_idx=origin_idx,
                search_radius=pairing['Radius'],
                detour_ratio=1,
                turn_penalty=pairing['Turns'],
                o_graph=o_graph,
                return_paths=False
            )

            get_origin_properties(
                shaqra,
                search_radius=pairing['Radius'],
                beta=pairing['Beta'],
                turn_penalty=pairing['Turns'],
                o_idx=origin_idx,
                o_graph=o_graph,
                d_idxs=d_idxs,
            )

            shaqra.network.remove_node_to_graph(o_graph, origin_idx)

            origin_source_id = origins.at[origin_idx, 'source_id']
            origin_record.at[origin_source_id, pairing['Destination_Name']+"_KNN"] = shaqra.network.nodes.at[origin_idx, 'knn_weight']
            origin_record.at[origin_source_id, pairing['Destination_Name']+"_gravity"] = shaqra.network.nodes.at[origin_idx, 'gravity']
            origin_record.at[origin_source_id, pairing['Destination_Name']+"_reach"] = shaqra.network.nodes.at[origin_idx, 'reach']
        

        



    
        Path(logger.output_folder).mkdir(parents=True, exist_ok=True)
        origin_record.to_csv(os.path.join(logger.output_folder, "origin_record.csv"))
        logger.log("accissibility calculated.", pairing)    

    origin_record.to_file(os.path.join(logger.output_folder, "origin_record.geoJSON"), driver="GeoJSON",  engine='pyogrio')
    logger.log_df.to_csv(os.path.join(logger.output_folder, "time_log.csv"))
    logger.log("Output saved: ALL DONE")
    return 


if __name__ == '__main__':
    city_name = 'LA'
    KNN_workflow(
        city_name=city_name,
        #data_folder=os.path.join('examples', 'Cities', city_name, 'Data'),
        #output_folder=os.path.join('examples', 'Cities', city_name, 'Simulations', 'Baseline'),
        pairings_file="pairings.csv",       
        num_cores=8,
    )

