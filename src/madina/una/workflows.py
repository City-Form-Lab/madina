import os
import warnings
import sys              ## for version
import time

os.environ['USE_PYGEOS'] = '0'

import shapely as shp ## For version 
import networkx as nx   ## For Version
import geopandas as gpd ## for version
import numpy as np      ## for version

import pydeck as pdk
import pandas as pd
import shapely.geometry as geo


from datetime import datetime
from pydeck.types import String
from pathlib import Path
from .tools import betweenness
from .betweenness import  get_origin_properties
from .paths import turn_o_scope
from ..zonal import Zonal, VERSION, RELEASE_DATE




class Logger():
    def __init__(self, output_folder, pairing_table):
        self.pairing_table = pairing_table
        self.output_folder = output_folder
        self.start_time = datetime.now()
        self.log_df = pd.DataFrame(
            {
                "time": pd.Series(dtype='datetime64[ns]'),
                "flow_name": pd.Series(dtype="string"),
                "event": pd.Series(dtype="string")
            }
        )

        #self.betweenness_record = None
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


        if pairing is not None:
            pairing_name = f"({1+int(pairing.name)}/{int(self.pairing_table.shape[0])}) " + pairing['Flow_Name']
        print(
            f"{cumulative_seconds:10.4f} | "
            f"{seconds_elapsed:15.6f} | "
            f"{pairing_name if pairing is not None else '---':^40s} | "
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
            save_origin_csv=False,
            save_diagnostics_map=False,
        ):
        # creating a folder for output

        #if self.betweenness_record is None:
            #self.betweenness_record = shaqra.layers['streets'].gdf.copy(deep=True)
        
        pairing_folder = os.path.join(self.output_folder, f"{pairing['Flow_Name']}_O({pairing['Origin_Name']})_D({pairing['Destination_Name']})")
        Path(pairing_folder).mkdir(parents=True, exist_ok=True)

        street_gdf = shaqra[shaqra.network.edge_source_layer].gdf
        #street_gdf = shaqra.layers["streets"].gdf
        node_gdf = shaqra.network.nodes
        edge_gdf = shaqra.network.edges

        '''
        self.betweenness_record = self.betweenness_record.join(
            edge_gdf[['parent_street_id', 'betweenness']].drop_duplicates(subset='parent_street_id').set_index('parent_street_id')).rename(
            columns={"betweenness": pairing['Flow_Name']})
        '''
        
        
        # .drop_duplicates(subset='parent_street_id') is needed to handle split parallel edges. this won't be needed if parallel edges were allowed and not needed to be split


        # creating origins and desrinations connector lines
        origin_layer = shaqra.layers[pairing['Origin_Name']].gdf


        if save_flow_map:
            self.flow_map_template_1(
                flow_gdf=edge_gdf,
                flow_parameter='betweenness',
                max_flow=None, 
                dark_mode=True,
                file_name=os.path.join(pairing_folder, "flow_map_dark.html")
            )
            self.flow_map_template_1(
                flow_gdf=edge_gdf,
                flow_parameter='betweenness',
                max_flow=None, 
                dark_mode=False,
                file_name=os.path.join(pairing_folder, "flow_map_light.html")
            )
        if save_diagnostics_map:
            origin_gdf = node_gdf[node_gdf["type"] == "origin"]
            destination_gdf = node_gdf[node_gdf["type"] == "destination"]
            origin_joined = origin_layer.join(origin_gdf.set_index('source_id'),lsuffix='_origin')
            origin_joined['geometry'] = origin_joined.apply(lambda x:geo.LineString([x['geometry'], x["geometry_origin"]]), axis=1)


            destination_layer = shaqra.layers[pairing['Destination_Name']].gdf
            destination_joined = destination_layer.join(destination_gdf.set_index('source_id'),lsuffix='_destination')
            destination_joined['geometry'] = destination_joined.apply(lambda x:geo.LineString([x['geometry'], x["geometry_destination"]]), axis=1)

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
            street_gdf.to_file(os.path.join(pairing_folder, "betweenness_record_so_far.geoJSON"), driver="GeoJSON",  engine='pyogrio')
            #self.betweenness_record.to_file(os.path.join(pairing_folder, "betweenness_record_so_far.geoJSON"), driver="GeoJSON",  engine='pyogrio')

        if save_flow_csv:
            street_gdf.to_csv(os.path.join(pairing_folder, "betweenness_record_so_far.csv"))
            #self.betweenness_record.to_csv(os.path.join(pairing_folder, "betweenness_record_so_far.csv"))

        if save_origin_geoJSON:
            #save_origin = shaqra.layers[pairing["Origin_Name"]].gdf.join(origin_gdf.set_index("source_id").drop(columns=['geometry']))
            #save_origin.to_file(os.path.join(f'{pairing_folder}', f'origin_record_({pairing["Origin_Name"]}).geoJSON'), driver="GeoJSON",  engine='pyogrio')
            origin_layer.to_file(os.path.join(f'{pairing_folder}', f'origin_record_({pairing["Origin_Name"]}).geoJSON'), driver="GeoJSON",  engine='pyogrio')

        if save_origin_csv: 
            #save_origin = shaqra.layers[pairing["Origin_Name"]].gdf.join(origin_gdf.set_index("source_id").drop(columns=['geometry']))
            #save_origin.to_csv(os.path.join(f'{pairing_folder}', f'origin_record_({pairing["Origin_Name"]}).csv'))
            origin_layer.to_file(os.path.join(f'{pairing_folder}', f'origin_record_({pairing["Origin_Name"]}).geoJSON'), driver="GeoJSON",  engine='pyogrio')

        self.log_df.to_csv(os.path.join(pairing_folder, "time_log.csv"))

        self.log("Output saved", pairing)

    def simulation_end(
            self,
            shaqra: Zonal
        ):
        self.log_df.to_csv(os.path.join(self.output_folder, "time_log.csv"))
        #self.betweenness_record.to_file(os.path.join(self.output_folder, "betweenness_record.geoJSON"), driver="GeoJSON",  engine='pyogrio')
        #self.betweenness_record.to_csv(os.path.join(self.output_folder, "betweenness_record.csv"))
        shaqra[shaqra.network.edge_source_layer].gdf.to_file(os.path.join(self.output_folder, "betweenness_record.geoJSON"), driver="GeoJSON",  engine='pyogrio')
        shaqra[shaqra.network.edge_source_layer].gdf.to_csv(os.path.join(self.output_folder, "betweenness_record.csv"))

        self.log("Simulation Output saved: ALL DONE")
    
    def flow_map_template_1(
        self, 
        flow_gdf,
        flow_parameter,
        max_flow=None, 
        dark_mode=True,
        file_name=None
        ):
        predicted_flow_gdf = flow_gdf.copy(deep=True)
        pdk_layers = []


        # preprocessing, probably should be done prior to function calling
        predicted_flow_gdf = predicted_flow_gdf[~predicted_flow_gdf[flow_parameter].isna()]

        if predicted_flow_gdf['geometry'].crs != 'EPSG:4326':
            predicted_flow_gdf['geometry'] = predicted_flow_gdf['geometry'].to_crs('EPSG:4326')


        h_s = predicted_flow_gdf[flow_parameter]
        if max_flow is None:
            max_flow = h_s.max()
        predicted_flow_gdf['__width__'] = (h_s - 0)/ (max_flow-0) * 40 + 0.5

        rbg_color = [249, 245, 10] if dark_mode else [255, 105, 180]
        predicted_flow_gdf['__color__'] = [rbg_color] * predicted_flow_gdf.shape[0] 

        predicted_flow_gdf['__text__'] = predicted_flow_gdf[flow_parameter].apply(lambda x: f"{int(x):,}")

        pdk_layers.append(pdk.Layer(
            'GeoJsonLayer',
            predicted_flow_gdf,
            opacity=0.99,
            stroked=True,
            filled=True,
            wireframe=True,
            get_line_width='__width__',         ## line width attribute
            get_line_color='__color__',             ## line color for line data, or stroke color for points and poygons
            get_fill_color="__color__",             ## fill color for points and polygons
            pickable=True,
        )
        )

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=UserWarning)
            predicted_flow_gdf['geometry']  = predicted_flow_gdf['geometry'].centroid

        pdk_layers.append(pdk.Layer(
            "TextLayer",
            predicted_flow_gdf,
            pickable=False,
            get_position="geometry.coordinates",
            get_text="__text__",
            get_size=3,            ## text font size
            size_units=String('meters'),
            sizeMaxPixels=18,         #  prevent the icon from getting too big when zoomed in.
            opacity=1,
            get_color='__color__',
            get_angle=0,
            background=True,
            get_background_color=[0, 0, 0]if dark_mode else [255, 255, 255],  ## gives a black background box for text with transperancy 0.25
            get_text_anchor=String("middle"),
            get_alignment_baseline=String("center"),
        ))

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=UserWarning)
            initial_view_state = pdk.data_utils.compute_view(
                points=[[p.coords[0][0], p.coords[0][1]] for p in predicted_flow_gdf['geometry'].centroid],
                view_proportion=1
            )
            initial_view_state.zoom = initial_view_state.zoom + 2

        tooltip = {
            "html": F"<b>{flow_parameter}:</b> {{{'__text__'}}}",
            "style": {
                    "backgroundColor": "steelblue",
                    "color": "white"
            }
        }


        r = pdk.Deck(
            layers=pdk_layers,
            initial_view_state=initial_view_state,
            map_style='dark_no_labels' if dark_mode else 'light_no_labels',                                # options are  ‘light’, ‘dark’, ‘road’, ‘satellite’, ‘dark_no_labels’, and ‘light_no_labels’, a URI for a basemap style, which varies by provider, or a dict that follows the Mapbox style specification <https://docs.mapbox.com/mapbox-gl-js/style-spec/
            tooltip=tooltip
        )

        r.to_html(file_name)
        return r

def betweenness_flow_simulation(
        city_name=None,
        data_folder=None,
        output_folder=None,
        pairings_file="pairings.csv",
        num_cores=8,
    ) -> None:
    """A workflow to generate trips between pairs of origins and destonations along a network. for detailed description of the workflow, please reference this page: https://madinadocs.readthedocs.io/en/latest/ped_flow.html to learn more about preparing data and constructing a pairing table needed for this workflow

    :param city_name: a city name that correspond to a folder inside the "Cities" folder in the current working directory, defaults to None
    :type city_name: str, optional
    :param data_folder: If the parameter `city_name` was provided, this parameter is optional if the data and pairing table were stored in a folder "Cities/city_name/Data" relative to the current working directory, defaults to None
    :type data_folder: str, optional
    :param output_folder: if the parameter `city_name` was provided, this prameter is optional, and all output would be stored in a folder inside "Cities\city_name\Simulations" relatie to the current working directory, defaults to None
    :type output_folder: str, optional
    :param pairings_file: the name of the file containing the pairing table inside the data folder, defaults to "pairings.csv"
    :type pairings_file: str, optional
    :param num_cores: the number of cores to be used in multiprocessing to speed up the simulation, defaults to 8
    :type num_cores: int, optional
    """
    

    if (city_name is None) and (data_folder is None) and (output_folder is None):
        raise ValueError("parameter 'city_name' needs to be specified if `data_folder` and `output_folder` are not provided, or provide paths to the `data_folder` and `output_folder`")

    if data_folder is None:
        data_folder = os.path.join("Cities", city_name, "Data")
    if output_folder is None:
        start_time = datetime.now()
        output_folder = os.path.join("Cities", f"{city_name}", "Simulations", f"{start_time.year}-{start_time.month:02d}-{start_time.day:02d} {start_time.hour:02d}-{start_time.minute:02d}")

    pairings = pd.read_csv(os.path.join(data_folder, pairings_file))

    logger=Logger(output_folder, pairings)

    # Shaqra is a town in Saudi Arabia. this name would be used to reference a generic place that we're running a simulation for
    shaqra = Zonal()

    shaqra.load_layer(
        name='streets',
        source=os.path.join(data_folder,  pairings.at[0, "Network_File"])
    )

    logger.log(f"network FIle Loaded, Projection: {shaqra.layers['streets'].gdf.crs}")


    for pairing_idx, pairing in pairings.iterrows():

        # Setting up a street network if this is the first pairing, or if the network weight changed from previous pairing
        if (pairing_idx == 0) or (pairings.at[pairing_idx, 'Network_Cost'] != pairings.at[pairing_idx-1, 'Network_Cost']):
            shaqra.create_street_network(
                source_layer='streets', 
                node_snapping_tolerance=0.00001,  #todo: remove parameter once a finalized default is set.
                weight_attribute=pairings.at[pairing_idx, 'Network_Cost'] if pairings.at[pairing_idx, 'Network_Cost'] != "Geometric" else None
            )
            logger.log("network topology created", pairing)
            clean_network_nodes = shaqra.network.nodes.copy(deep=True)
        else:
            # either generate a new network, or flush nodes.
            shaqra.network.nodes = clean_network_nodes.copy(deep=True)

        shaqra.set_turn_parameters(
            turn_penalty_amount=pairing['Turn_Penalty'], 
            turn_threshold_degree=pairing['Turn_Threshold'],
        )


        # Loading layers, if they're not already loaded.
        if pairing["Origin_Name"] not in shaqra.layers:
            shaqra.load_layer(
                name=pairing["Origin_Name"],
                source=os.path.join(data_folder, pairing["Origin_File"])
            )
            logger.log(f"{pairing['Origin_Name']} file {pairing['Origin_File']} Loaded, Projection: {shaqra.layers[pairing['Origin_Name']].gdf.crs}", pairing)

        if pairing["Destination_Name"] not in shaqra.layers:
            shaqra.load_layer(
                name=pairing["Destination_Name"],
                source=os.path.join(data_folder, pairing["Destination_File"])
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



        betweenness(
            zonal=shaqra,
            search_radius=pairing['Radius'],
            detour_ratio=pairing['Detour'],
            decay=False if pairing['Elastic_Weights'] else pairing['Decay'],  # elastic weight already reduces origin weight factoring in decay. if this pairing uses elastic weights, don't decay again,,
            decay_method=pairing['Decay_Mode'],
            beta=pairing['Beta'],
            num_cores=min(shaqra[pairing['Origin_Name']].gdf.shape[0], num_cores),
            closest_destination=pairing['Closest_destination'],
            elastic_weight=pairing['Elastic_Weights'],
            knn_weight=pairing['KNN_Weight'],
            knn_plateau=pairing['Plateau'], 
            turn_penalty=pairing['Turns'],
            save_betweenness_as=pairing['Flow_Name'], 
            save_reach_as='reach_'+pairing['Flow_Name'], 
            save_gravity_as='gravity_'+pairing['Flow_Name'],
            save_elastic_weight_as='elastic_weight_'+pairing['Flow_Name'] if pairing['Elastic_Weights'] else None,
            keep_diagnostics=True, 
            path_exposure_attribute=pairing['Exposure_Attribute']  if 'Exposure_Attribute' in pairing.index else None,
            save_path_exposure_as="exposure_"+pairing['Flow_Name'] if 'Exposure_Attribute' in pairing.index else None,
        )


        logger.log("Betweenness estimated.", pairing)
        logger.pairing_end(shaqra, pairing)
    logger.simulation_end(shaqra)
    return 



def KNN_accessibility(
        city_name=None,
        data_folder=None,
        output_folder=None,
        pairings_file="pairing.csv",
        num_cores=8,
    ):
    """A workflow to generate accessibility metrics: reach, gravity and KNN access for an origin to all its paired destinations in the pairing table.

    :param city_name: a city name that correspond to a folder inside the "Cities" folder in the current working directory, defaults to None
    :type city_name: str, optional
    :param data_folder: If the parameter `city_name` was provided, this parameter is optional if the data and pairing table were stored in a folder "Cities/city_name/Data" relative to the current working directory, defaults to None
    :type data_folder: str, optional
    :param output_folder: if the parameter `city_name` was provided, this prameter is optional, and all output would be stored in a folder inside "Cities\city_name\Simulations" relatie to the current working directory, defaults to None
    :type output_folder: str, optional
    :param pairings_file: the name of the file containing the pairing table inside the data folder, defaults to "pairings.csv"
    :type pairings_file: str, optional
    :param num_cores: the number of cores to be used in multiprocessing to speed up the simulation, defaults to 8
    :type num_cores: int, optional
    """


    if city_name is None:
        raise ValueError("parameter 'city_name' needs to be specified")

    if data_folder is None:
        data_folder = os.path.join("Cities", city_name, "Data")
    if output_folder is None:
        start_time = datetime.now()
        output_folder = os.path.join("Cities", f"{city_name}", "KNN_workflow", f"{start_time.year}-{start_time.month:02d}-{start_time.day:02d} {start_time.hour:02d}-{start_time.minute:02d}")

    

    pairings = pd.read_csv(os.path.join(data_folder, pairings_file))
    logger=Logger(output_folder, pairings)

    # Shaqra is a town in Saudi Arabia. this name would be used to reference a generic place that we're running a simulation for
    shaqra = Zonal()




    for pairing_idx, pairing in pairings.iterrows():
        if (pairing_idx == 0) or (pairings.at[pairing_idx, 'Network_File'] != pairings.at[pairing_idx-1, 'Network_File']):
            shaqra.load_layer(
                name='streets',
                source=os.path.join(data_folder,  pairings.at[0, "Network_File"])
            )
            logger.log(f"network FIle Loaded, Projection: {shaqra.layers['streets'].gdf.crs}", pairing)


        if (pairing_idx == 0) or (pairings.at[pairing_idx, 'Network_Cost'] != pairings.at[pairing_idx-1, 'Network_Cost']): 
            shaqra.create_street_network(
                source_layer='streets',
                weight_attribute=pairings.at[pairing_idx, 'Network_Cost'] if pairings.at[pairing_idx, 'Network_Cost'] != "Geometric" else None,
                node_snapping_tolerance=0.00001,  #todo: remove parameter once a finalized default is set.
                redundant_edge_treatment='discard',
            )
            logger.log("network topology created", pairing)
        else:
            shaqra.clear_nodes()



        # Loading layers, if they're not already loaded.
        if pairing["Origin_Name"] not in shaqra.layers:
            shaqra.load_layer(
                name=pairing["Origin_Name"],
                source=os.path.join(data_folder, pairing["Origin_File"])
            )
            logger.log(f"{pairing['Origin_Name']} file {pairing['Origin_File']} Loaded, Projection: {shaqra.layers[pairing['Origin_Name']].gdf.crs}", pairing)


        if pairing["Destination_Name"] not in shaqra.layers:
            shaqra.load_layer(
                name=pairing["Destination_Name"],
                source=os.path.join(data_folder, pairing["Destination_File"])
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

        shaqra.set_turn_parameters(
            turn_penalty_amount=pairing['Turn_Penalty'], 
            turn_threshold_degree=pairing['Turn_Threshold'],
        )

        shaqra.network.knn_weight = pairing['KNN_Weight']
        shaqra.network.knn_plateau = pairing['Plateau']

        origin_gdf = parallel_access(
            shaqra,
            num_cores=num_cores,
            search_radius=pairing['Radius'],
            beta=pairing['Beta'],
            turn_penalty=pairing['Turns'],
        )

        # bring back results to layer...

        saved_attributes = {metric: pairing['Flow_Name']+"_"+metric for metric in ['reach', 'gravity', 'knn_access', 'knn_weight']}

        for key, value in saved_attributes.items():
            origin_gdf[key] = origin_gdf[key].fillna(0)
            if value in shaqra[pairing['Origin_Name']].gdf.columns:
                shaqra[pairing['Origin_Name']].gdf.drop(columns=[value], inplace=True)


        shaqra[pairing['Origin_Name']].gdf = shaqra[pairing['Origin_Name']].gdf.join(origin_gdf[['source_id'] + list(saved_attributes.keys()) ].set_index("source_id").rename(columns=saved_attributes))
        shaqra[pairing['Origin_Name']].gdf.index = shaqra[pairing['Origin_Name']].gdf.index.astype(int)
        

    
        Path(logger.output_folder).mkdir(parents=True, exist_ok=True)
        shaqra[pairing['Origin_Name']].gdf.to_csv(os.path.join(logger.output_folder, "origin_record.csv"))
        logger.log("accissibility calculated.", pairing)    

    shaqra[pairing['Origin_Name']].gdf['total_knn_access']= shaqra[pairing['Origin_Name']].gdf[[flow_name+"_knn_access" for flow_name in pairings['Flow_Name']]].sum(axis=1)
    shaqra[pairing['Origin_Name']].gdf.to_file(os.path.join(logger.output_folder, "origin_record.geoJSON"), driver="GeoJSON",  engine='pyogrio')
    logger.log_df.to_csv(os.path.join(logger.output_folder, "time_log.csv"))
    logger.log("Output saved: ALL DONE")
    return 








def one_access(
    self: Zonal,
    origin_queue = None,
    search_radius: float = 0, 
    beta: float = 0, 
    turn_penalty: bool = False, 
    reporting: bool = False, 
    ):

    node_gdf = self.network.nodes
    origin_gdf=node_gdf[node_gdf['type'] == 'origin']

    o_graph = self.network.d_graph
    
    start = time.time()
    i = 0
    processed_origins = []
    try: 
        while True:
            origin_idx = origin_queue.get()

            if origin_idx == "done":
                origin_queue.task_done()
                break
            processed_origins.append(origin_idx)

            if reporting:
                print (f"Time spent: {round(time.time()-start):,}s [Done {i:,} of {origin_gdf.shape[0]:,} origins ({i  /origin_gdf.shape[0] * 100:4.2f}%)]",  end='\r')
                i = i + 1

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

            get_origin_properties(
                self,
                search_radius=search_radius,
                beta=beta,
                turn_penalty=turn_penalty,
                o_idx=origin_idx,
                o_graph=o_graph,
                d_idxs=d_idxs,
            )

            self.network.remove_node_to_graph(o_graph, origin_idx)

    except Exception as ex:
        print ('Issues in one access...')
        import traceback
        traceback.print_exc()

    return node_gdf.loc[processed_origins]


import multiprocessing as mp
import concurrent
#from concurrent import futures

def parallel_access(
    self: Zonal,
    num_cores: int = 1,
    search_radius: float = 0, 
    beta: float = 0, 
    turn_penalty: bool = False, 
):
    node_gdf = self.network.nodes
    origins = node_gdf[node_gdf["type"] == "origin"]
    origins = origins.sample(frac=1)
    origins.index = origins.index.astype("int")

    
    with mp.Manager() as manager:

        origin_queue = manager.Queue()

        for o_idx in origins.index:
            origin_queue.put(o_idx)

        for core_index in range(num_cores):
            origin_queue.put("done")

        ## TODO: possible to implement single core without needing to create external process, use a thread?

        with concurrent.futures.ProcessPoolExecutor(max_workers=num_cores) as executor:        
            execution_results = [
                executor.submit(
                    one_access,
                    self=self,
                    origin_queue=origin_queue,
                    search_radius=search_radius,
                    beta=beta,
                    turn_penalty=turn_penalty,
                ) for core_index in range(num_cores)
            ]

            # Progress update and raising exceptions..
            start = time.time()
            while not all([future.done() for future in execution_results]):
                time.sleep(0.5)
                print (f"Time spent: {round(time.time()-start):,}s [Done {max(origins.shape[0] - origin_queue.qsize(), 0):,} of {origins.shape[0]:,} origins ({max(origins.shape[0] - origin_queue.qsize(), 0)/origins.shape[0] * 100:4.2f}%)]",  end='\r')
                for future in [f for f in execution_results if f.done() and (f.exception() is not None)]: # if a process is done and have an exception, raise it
                    raise (future.exception())
        
            origin_gdf = pd.concat([result.result() for result in concurrent.futures.as_completed(execution_results)])


        

    return origin_gdf




