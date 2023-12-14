import os
from datetime import datetime
import pandas as pd
from madina.una.betweenness import Logger
from madina.una.tools import accessibility, parallel_accessibility
from madina.zonal.network import Network
from madina.zonal.zonal import Zonal
from pathlib import Path

def fifteen_minute_city(
        city_name=None,
        data_folder=None,
        output_folder=None,
        pairings_file="15_min_city.csv",
        num_cores=8,
    ):
    origin_record =  None

    if city_name is None:
        raise ValueError("parameter 'city_name' needs to be specified")

    if data_folder is None:
        data_folder = os.path.join("Cities", city_name, "Data")
    if output_folder is None:
        start_time = datetime.now()
        output_folder = os.path.join("Cities", f"{city_name}", "Fifteen_min_city", f"{start_time.year}-{start_time.month:02d}-{start_time.day:02d} {start_time.hour:02d}-{start_time.minute:02d}")

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

        #parameter settings for turns and elastic weights
        shaqra.network.turn_penalty_amount = pairing['Turn_Penalty']
        shaqra.network.turn_threshold_degree = pairing['Turn_Threshold']

        if origin_record is None:
            origin_record = shaqra.layers[pairing['Origin_Name']].gdf.copy(deep=True)

        

        '''
        accessibility(
            zonal=shaqra,
            reach=True,
            gravity=True,
            closest_facility=False,
            weight=pairing['Destination_Weight'] if pairing['Destination_Weight'] != "Count" else None,
            search_radius=pairing['Radius'],
            alpha=1,
            beta=pairing['Beta']
        )

        node_gdf = shaqra.network.nodes
        origin_gdf = node_gdf[node_gdf["type"] == "origin"]

        
        origin_record = origin_record.join(
            origin_gdf[['source_id', 'una_reach', 'una_gravity']].set_index('source_id')).rename(
            columns={"una_reach": f"{pairing['Flow_Name']}_reach", "una_gravity": f"{pairing['Flow_Name']}_gravity"})
        logger.log("accissibility calculated.", pairing)
        '''

        returned_origins = parallel_accessibility(
            zonal=shaqra,
            reach=True,
            gravity=True,
            closest_facility=False,
            weight=pairing['Destination_Weight'] if pairing['Destination_Weight'] != "Count" else None,
            search_radius=pairing['Radius'],
            alpha=1,
            beta=pairing['Beta']
        )
        origin_record = origin_record.join(
            returned_origins[['source_id', 'una_reach', 'una_gravity']].set_index('source_id')).rename(
            columns={"una_reach": f"{pairing['Flow_Name']}_reach", "una_gravity": f"{pairing['Flow_Name']}_gravity"})
        logger.log("accissibility calculated.", pairing)
        



    
    Path(logger.output_folder).mkdir(parents=True, exist_ok=True)
    origin_record.to_file(os.path.join(logger.output_folder, "origin_record.geoJSON"), driver="GeoJSON",  engine='pyogrio')
    origin_record.to_csv(os.path.join(logger.output_folder, "origin_record.csv"))
    logger.log_df.to_csv(os.path.join(logger.output_folder, "time_log.csv"))


    
    origin_record["L1_ped"] = origin_record.apply(
        lambda x: ((x['Ped_net_Bus_reach'] >= 1) or (x['Ped_net_Subway_reach'] >= 1) or (x['Ped_net_Train_reach'] >= 1))
        and x['Ped_net_Child_care_reach'] >= 1 
        and x['Ped_net_Grocery_Stores_reach'] >= 1
        and x['Ped_net_Healthcare_reach'] >= 1 
        and x['Ped_net_Leisure_reach'] >= 1
        and x['Ped_net_Local_Services_reach'] >= 1
        and x['Ped_net_Parks_reach'] >= 1
        and x['Ped_net_Public_school_reach'] >= 1
        and x['Ped_net_Sports_Facilities_reach'] >= 1
        , axis=1
    )

    origin_record["L1_street"] = origin_record.apply(
        lambda x: ((x['Street_net_Bus_reach'] >= 1) or (x['Street_net_Subway_reach'] >= 1) or (x['Street_net_Train_reach'] >= 1))
        and x['Street_net_Child_care_reach'] >= 1 
        and x['Street_net_Grocery_Stores_reach'] >= 1
        and x['Street_net_Healthcare_reach'] >= 1 
        and x['Street_net_Leisure_reach'] >= 1
        and x['Street_net_Local_Services_reach'] >= 1
        and x['Street_net_Parks_reach'] >= 1
        and x['Street_net_Public_school_reach'] >= 1
        and x['Street_net_Sports_Facilities_reach'] >= 1
        , axis=1
    )
    origin_record.to_file(os.path.join(logger.output_folder, "origin_record_L1.geoJSON"), driver="GeoJSON",  engine='pyogrio')
    origin_record.to_csv(os.path.join(logger.output_folder, "origin_record_L1.csv"))
    






    logger.log("Output saved: ALL DONE")
    return 


if __name__ == '__main__':
    city_name = 'Milan'

    fifteen_minute_city(
        city_name=city_name,
        #data_folder=os.path.join('examples', 'Cities', city_name, 'Data'),
        #output_folder=os.path.join('examples', 'Cities', city_name, 'Simulations', 'Baseline'),
        pairings_file="15_min_city.csv",       
        num_cores=8,
    )

