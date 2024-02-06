# this lets geopandas exclusively use shapely (not pygeos) silences a warning about depreciating pygeos out of geopandas. This is not needed when geopandas 1.0 is released in the future
import os
os.environ['USE_PYGEOS'] = '0'      ## This needs to be done before importing geopandas to prevent warnings
import random
import warnings
import geopandas as gpd
import pandas as pd
import pydeck as pdk

from pathlib import Path
from .network import Network
from .network_utils import node_edge_builder, _discard_redundant_edges, _split_redundant_edges, efficient_node_insertion
from .utils import prepare_geometry, color_gdf, create_deckGL_map, DEFAULT_COLORS
from .layer import Layer, Layers


VERSION = '0.0.12'
RELEASE_DATE = '2023-02-06'


class Zonal:
    """
    A class to manage and organize urban data into layers and networks, acting as a workspace that manages data and facilitate interoperability across tools.
    A Zonal object populated with a veriety of data layeers and network could be used as input toi many urban analysis tools.
    Please look at the examples to see a gallery of use cases.

    :Example:
        >>> shaqra = Zonal()
        >>> shaqra.add_layer(name='streets', source='streets.geojson')
        >>> shaqra.create_map(save_as='street_map.html', basemap=False)
    """
    DEFAULT_PROJECTED_CRS = "EPSG:3857"
    DEFAULT_GEOGRAPHIC_CRS = "EPSG:4326"
    DEFAULT_COLORS = DEFAULT_COLORS

    def __init__(self):
        self.network = None
        self.geo_center = (None, None)
        self.layers = Layers(None)

    def __getitem__(self, item):
        return self.layers[item]

    def load_layer(
            self,
            name: str,
            source: str | Path | gpd.GeoDataFrame,
            pos: int = None,
            first: bool =False,
            before: str = None,
            after: str = None
        ) -> None:
        """ Loads a new layer from the given source with the specified layer name.

        :param name: A name for the layer the new layer.  
        :type name: str
        :param source: The data source of this layer. Could either be an existing GeoDataFrame, or a path to an acceptable file. Supported types are '.geojson', '.shp', or any file accepted by geopanda's read_file()
        :type source: str | Path | gpd.GeoDataFrame
        :param pos: position of this layer in the list of layers, defaults adding a layer at the end of the list.
        :type pos: int, optional
        :param first: if True, inserts this layer at the top of the list, defaults to False
        :type first: bool, optional
        :param before: insert before the specified layer name if specified
        :type before: str, optional
        :param after: insert after the specified layer name if specified
        :type after: str, optional
        :raises TypeError: _description_
        :raises TypeError: _description_
        :Example:

            >>> shaqra = Zonal()  # Create a Zonal object.
            >>> zonal.load_layer("streets", "path/to/streets.geojson", first=True) # Load a new layer at the beginning of the layers list.
        """
        # input validation
        if not isinstance(name, str):
            raise TypeError(f"Parameter 'layer_name' must be a string. {type(name)} was given.")
        
        if not isinstance(source, (Path, str, gpd.GeoDataFrame)): 
            raise TypeError(f"Parameter 'file_path' must be either {Path, str, gpd.GeoDataFrame}. {type(source)} was given.")
        
        if isinstance(source, gpd.GeoDataFrame):
            gdf = source.copy(deep=True)
        else:
            gdf = gpd.read_file(
                source,
                engine='pyogrio'
            )

        gdf['id'] = range(gdf.shape[0])
        gdf = gdf.set_index('id')
        original_crs = gdf.crs

        # perform a standard data cleaning process to ensure compatibility with later processes
        gdf = prepare_geometry(gdf)

        layer = Layer(
            name,
            gdf,
            True,
            original_crs,
            'GeoDataFrame' if isinstance(source, gpd.GeoDataFrame) else source, 
            default_color=[round(random.random() * 255), round(random.random() * 255), round(random.random() * 255)]
        )
        self.layers.add(
            layer,
            pos,
            first,
            before,
            after
        )

        if None in self.geo_center:
            with warnings.catch_warnings():
                # This is to ignore a warning issued for doing calculations in a geographic coordinate system, but that's the needed output:
                # a point in a geographic coordinate system to center the visualization
                warnings.simplefilter("ignore", category=UserWarning)
                # centroid_point = gdf.iloc[[0]].to_crs(self.DEFAULT_GEOGRAPHIC_CRS).centroid.iloc[0]
                centroid_point = gdf['geometry'].to_crs("EPSG:4326").unary_union.centroid
            self.geo_center = centroid_point.coords[0]
        return

    def create_street_network(
            self,
            source_layer: str,
            weight_attribute: str = None,
            node_snapping_tolerance: int | float = 0,
            redundant_edge_treatment: str ='split',
            turn_threshold_degree: int | float = 45,
            turn_penalty_amount: int | float = 30,
    ) -> None:
        """Create a topologically connected street network from a specified layer in the Zonal object.

        :param source_layer: The name of the source layer to create the network from. Layer must be loaded using 'load_layer()' first.
        :type source_layer: str
        :param weight_attribute: Name of the attribute to use as percieved diatance. Given name must exist in layer attributes. Default is None, and the network cost would be calculated using geometric distance.
        :type weight_attribute: str, optional
        :param node_snapping_tolerance: Tolerance for snapping nodes. Default is 0.0 assuming that line geometries that are connected share identical common start/end points, defaults to 0.0
        :type node_snapping_tolerance: int | float, optional
        :param redundant_edge_treatment: Due to current limitations, only one edge can exist between a pair of nodes. Three options: "keep": redundant edges will be kept, posing a risk for error in network construction/calculations. "discard": The shortest edge of redundant edges will be kept. others are discarded. is kept if set to True. "split": Split redundant edges by thier centroids into non-redundant segments. Default "split".
        :type redundant_edge_treatment: str, optional
        :param turn_threshold_degree: Degree threshold for considering a turn. Default is 45. This threshold would only be used whem enabling turn penalty in UNA operations. The angle is measured as the deviation from a straight line. Can only be between 0-180. 90 degrees means a right or left turn, a 180 degree means a full U-turn
        :type turn_threshold_degree: int | float,  optional
        :param turn_penalty_amount: Penalty amount for turns. Default is 30. This penalty (in the units of the layers' CRS) would be used as turn cost when enabling turn penalty in UNA operations. Cannot b e negative.  defaults to 30
        :type turn_penalty_amount: int | float, optional
        :raises ValueError: if the source layer did not exist in the object.
        :Example:
            >>> zonal = Zonal()  # Create a Zonal object.
            >>> zonal.create_street_network(
            ...     source_layer="streets",
            ...     weight_attribute="length",
            ...     node_snapping_tolerance=0.001,
            ... )
            # Create a street network using 'streets' layer and allowing geometries to be atr most 0.001 CRS units apart to form a node.
        """

        # input validation
        if not isinstance(source_layer, str):
            raise TypeError(f"Parameter 'source_layer' must be a string. {type(source_layer)} was given.")

        if source_layer not in self.layers:
            raise ValueError(f"Source layer {source_layer} not in zonal zonal_layers, available layers are: {self.layers.layers}")
        
        if weight_attribute is not None:
            if not isinstance(weight_attribute, str): 
                raise TypeError(f"Parameter 'weight_attribute' must be a string. {type(weight_attribute)} was given.")

            if weight_attribute not in self[source_layer].gdf.columns:
                raise ValueError(f"Parameter 'weight_attribute': {weight_attribute} not in layer {source_layer}. Available attributes are: {list(self[source_layer].gdf.columns)}")
            
        if not isinstance(node_snapping_tolerance, (int, float)): 
            raise TypeError(f"Parameter 'node_snapping_tolerance' must be either {int, float}. {type(node_snapping_tolerance)} was given.")

        if node_snapping_tolerance < 0:
            raise ValueError(f"Parameter 'node_snapping_tolerance': Cannot be negative. node_snapping_tolerance={node_snapping_tolerance} was given.")
        
        if redundant_edge_treatment not in ['keep', 'discard', 'split']:
            if not isinstance(redundant_edge_treatment, str):
                raise TypeError(f"Parameter 'redundant_edge_treatment' must be a string. {type(redundant_edge_treatment)} was given.")
            else: 
                raise ValueError(f"Parameter 'redundant_edge_treatment': must be one of ['keep', 'discard', 'split']. node_snapping_tolerance={redundant_edge_treatment} was given.")
            
        

        geometry_gdf = self.layers[source_layer].gdf

        node_gdf, edge_gdf = node_edge_builder(
            geometry_gdf,
            weight_attribute=weight_attribute,
            tolerance=node_snapping_tolerance, 
            source_layer=source_layer
        )

        if redundant_edge_treatment == 'split':
            node_gdf, edge_gdf = _split_redundant_edges(node_gdf, edge_gdf)
        elif redundant_edge_treatment == 'discard':
            edge_gdf = _discard_redundant_edges(edge_gdf)

        edge_gdf = color_gdf(
            edge_gdf,
            color=[125, 125, 125],
        )

        node_gdf = color_gdf(
            node_gdf,
            color=[125, 125, 125],
        )

        #type is a placeholder for now, for future use when there are multuple network types, like sidewalks, bikepaths, subway links,...
        edge_gdf = edge_gdf.drop(columns=['type'])
        
        self.network = Network(node_gdf, edge_gdf, None, None, weight_attribute, edge_source_layer=source_layer)
        
        self.set_turn_parameters(turn_threshold_degree, turn_penalty_amount)
        return

    def set_turn_parameters(
        self,
        turn_threshold_degree: int | float,
        turn_penalty_amount: int | float,
    ) -> None :
        """Set the turn penalty threshold and turn penalty amount for use when turn penalty is enabled in analysis tools.
        :param turn_threshold_degree: Degree threshold for considering a turn. Default is 45. This threshold would only be used whem enabling turn penalty in UNA operations. The angle is measured as the deviation from a straight line. Can only be between 0-180. 90 degrees means a right or left turn, a 180 degree means a full U-turn
        :type turn_threshold_degree: int | float
        :param turn_penalty_amount: Penalty amount for turns. Default is 30. This penalty (in the units of the layers' CRS) would be used as turn cost when enabling turn penalty in UNA operations. Cannot b e negative.  defaults to 30
        :type turn_penalty_amount: int | float
        :raises TypeError: if parameter `turn_threshold_degree` is not int or float
        :raises ValueError: if parameter `turn_threshold_degree` is not between 0 and 180 degrees.
        :raises TypeError: if parameter `turn_penalty_amount` is not int or float
        :raises ValueError: if parameter `turn_penalty_amount` is negative
        """

        if not isinstance(turn_threshold_degree, (int, float)): 
            raise TypeError(f"Parameter 'turn_threshold_degree' must be either {int, float}. {type(turn_threshold_degree)} was given.")

        if (turn_threshold_degree < 0) or (turn_threshold_degree > 180):
            raise ValueError(f"Parameter 'turn_threshold_degree': Must be between 0 and 180. turn_threshold_degree={turn_threshold_degree} was given.")
        

        if not isinstance(turn_penalty_amount, (int, float)): 
            raise TypeError(f"Parameter 'turn_penalty_amount' must be either {int, float}. {type(turn_penalty_amount)} was given.")

        if turn_penalty_amount < 0:
            raise ValueError(f"Parameter 'turn_penalty_amount': Cannot be negative. turn_penalty_amount={turn_penalty_amount} was given.")

        self.network.turn_threshold_degree = turn_threshold_degree
        self.network.turn_penalty_amount = turn_penalty_amount
        
        return 
        

    def insert_node(
            self,
            layer_name: str,
            label: str,
            weight_attribute: str = None
        ) -> None:
        """Insert "origin" and "destination" nodes into the network. This function must be called aftet the 'create_street_network' function is called, and the corresponding layer have already been loaded by calling 'load_layer'

        :param layer_name: The name of the layer to insert the nodes from.
        :type layer_name: str
        :param label: The label for the new node. Could either be "origin" or "destination".
        :type label: str
        :param weight_attribute: Name of the attribute to use as the node's weight. Default is None. If no weight is given, all nodes are weighted equally (Assigned a weight of 1). The attribute name must exist in the layer.
        :type weight_attribute: str, optional
        :Example:
            >>> shaqra = Zonal()  # Create a Zonal object.
            >>> shaqra.load_layer('streets', 'streets.geojson') # load streets layer
            >>> shaqra.create_street_network("streets")  # Create a street network
            >>> shaqra.load_layer('homes', 'homes.geojson')
            >>> shaqra.insert_node('homes', label="origin", weight_attribute="residents")
            >>> shaqra.load_layer('schools', 'schools.geojson')
            >>> shaqra.insert_node('schools', label="destination", weight_attribute="school_enrollment")
            # Insert a homes as origins, schools as destinations into the 'shgaqra' Zonal object

        """

        # input validation
        if not isinstance(layer_name, str):
            raise TypeError(f"Parameter 'layer_name' must be a string. {type(layer_name)} was given.")

        if layer_name not in self.layers:
            raise ValueError(f"Source layer {layer_name} not in zonal zonal_layers, available layers are: {self.layers.layers}")
        
        if label not in ['origin', 'destination']:
            if not isinstance(label, str):
                raise TypeError(f"Parameter 'label' must be a string. {type(label)} was given.")
            else: 
                raise ValueError(f"Parameter 'label': must be one of ['origin', 'destination']. label={label} was given.")



        source_gdf = self.layers[layer_name].gdf
        inserted_node_gdf = efficient_node_insertion(
            self.network.nodes, 
            self.network.edges, 
            source_gdf, 
            layer_name=layer_name, 
            label=label,
            weight_attribute=weight_attribute
        )


        # these columns are not created during network creation. Adding them now prevent them from being NaN and changing column type to float instead of int. 
        for column in ['nearest_edge_id', 'edge_start_node', 'weight_to_start', 'edge_end_node', 'weight_to_end']:
            if column not in self.network.nodes:
                self.network.nodes[column] = 0

        self.network.nodes = pd.concat([self.network.nodes, inserted_node_gdf])

        self.network.nodes = color_gdf(
            self.network.nodes,
            color_by_attribute='type',
            color_method= 'categorical',
            color= {'origin': [86,5,255], 'destination': [239,89,128], 'street_node': [125, 125, 125]}
        )
        return

    def create_graph(
            self,
            light_graph: bool=True,
            d_graph: bool=True,
            od_graph:bool =False
        ) -> None:
        """After creating a street network, adding origin nodes, and destination nodes, this function must be called to construct a NetworkX object internally. This is needed to run UNA tools. 

        :param light_graph: contains only network nodes and edges, defaults to True
        :type light_graph: bool, optional
        :param d_graph: contains all destination nodes and network intersectionsa. This is needed to run UNA tools, defaults to True
        :type d_graph: bool, optional
        :param od_graph: contains all origin, destination, network, etc. nodes, defaults to False
        :type od_graph: bool, optional
        :Example:
            >>> shaqra = Zonal()  # Create a Zonal object.
            >>> shaqra.load_layer('streets', 'streets.geojson') # load streets layer
            >>> shaqra.create_street_network("streets")  # Create a street network
            >>> shaqra.load_layer('homes', 'homes.geojson')
            >>> shaqra.insert_node('homes', label="origin", weight_attribute="residents")
            >>> shaqra.load_layer('schools', 'schools.geojson')
            >>> shaqra.insert_node('schools', label="destination", weight_attribute="school_enrollment")
            >>> shaqra.create_graph()
            # The zonal object now have everything it needs to be used as input in a UNA tool.
        """
        # input validation
        if not isinstance(light_graph, bool):
            raise TypeError(f"Parameter 'light_graph' must either be a boolean True or False, {type(light_graph)} was given.")
        
        if not isinstance(d_graph, bool):
            raise TypeError(f"Parameter 'd_graph' must either be a boolean True or False, {type(d_graph)} was given.")
        
        if not isinstance(od_graph, bool):
            raise TypeError(f"Parameter 'od_graph' must either be a boolean True or False, {type(od_graph)} was given.")


        self.network.create_graph(light_graph, d_graph, od_graph)

    def describe(self) -> None:
        """prints a textual representation of the zonal objecgt, listing and describing layers

        :Example:
            >>> zshaqra = Zonal()
            >>> shaqra.describe()
            >>> shaqra.load_layer('homes', 'homes.geojson')
            >>> shaqra.describe()
            a string representation of the `Zonal` object, a list of layers if any exists. 

        """

        if len(self.layers.layers) == 0:
            print("No zonal_layers yet, load a layer using 'load_layer(layer_name, file_path)'")
        else:
            print(f"{'Layer name':20} | {'Visible':7} | {'projection':10} | {'rows':5} | {'File path':20}")
            for key in self.layers:
                print(
                    f"{key:20} | {self.layers[key].show:7} | {str(self.layers[key].gdf.crs):10} | {self.layers[key].gdf.shape[0]:5} | {str(self.layers[key].file_path):20}")
                # print(f"\tColumn names: {list(self.layers[key].gdf.columns)}")

        geo_center_x, geo_center_y = self.geo_center

        if self.geo_center is None:
            print(f"No center yet, add a layer or set a scope to define a center")
        else:
            print(f"Geographic center: ({geo_center_x}, {geo_center_y})")

        if self.network is None:
            print(
                f"No network graph yet. First, insert a layer that contains network segments (streets, sidewalks, ..) and call create_street_network(layer_name,  weight_attribute=None)")
            print(f"\tThen,  insert origins and destinations using 'insert_nodes(label, layer_name, weight_attribute)'")
            print(f"\tFinally, when done, create a network by calling 'create_street_network()'")

    def create_map(
            self,
            layer_list: list = None,
            save_as: str=None,
            basemap: bool=False
        ) -> pdk.Deck:
        """ Create a map visualization using the specified layers within the `Zonal` object.


        :param layer_list:  A list of dictionaries, each containing a 'gdf' key with a GeoDataFrame, defaults to None. If None, the method includes all visible layers from the `Zonal` object.
        :type layer_list: list, optional
        :param save_as: The filename to save the map visualization. Default is None (not saved)., defaults to None
        :type save_as: str, optional
        :param basemap: Include a basemap in the map if True. Default is False.
        :type basemap: bool, optional
        :return: returns a PyDeck's Deck object that can be visualized in Jupyter notebooks.
        :rtype: pdk.Deck
        :Example:
            >>> zonal = Zonal()  # Create a Zonal object.
            >>> zonal.load_layer("streets", "streets.geojson")  # load streets layer.
            >>> zonal.load_layer("homes", "homes.geojson")  # load homes layer.
            >>> zonal.create_map(layer_list=[{"gdf": zonal.layers["streets"].gdf}], save_as="map.html", basemap=True)
            # Create a map visualization with a custom layer and a basemap, and save it as an HTML file.

        """



        if layer_list is None:
            layer_list = []
            for layer_name in self.layers.layers:
                if self.layers[layer_name].show:
                    layer_gdf = self.layers[layer_name].gdf.copy(deep=True)
                    layer_gdf = color_gdf(layer_gdf, color=self.layers[layer_name].default_color)
                    layer_list.append({"gdf": layer_gdf})
        else:
            for layer_position, layer_dict in enumerate(layer_list):
                if "layer" in layer_dict:
                    # switch from ysung the keyword layer, into using the keyword 'gdf' by supplying layer's gdf
                    # TODO: here;s a good place to impose default stylings from layer attribute. the layer_dict overrides default layer styling.

                    # color by default layer, would be overriden if different parameters are given..
                    layer_gdf = self.layers[layer_dict["layer"]].gdf.copy(deep=True)
                    layer_dict['gdf'] = color_gdf(layer_gdf, color=self.layers[layer_dict["layer"]].default_color)
                    layer_list[layer_position] = layer_dict
        map = create_deckGL_map(
            gdf_list=layer_list,
            centerX=self.geo_center[0],
            centerY=self.geo_center[1],
            basemap=basemap,
            zoom=17,
            filename=save_as
        )
        return map

    def clear_nodes(self) -> None:
        """Erase the existing origins and destinations from the current network, but retains the network nodes and edges
        """
        self.network.nodes = self.network.nodes[self.network.nodes["type"] == "street_node"]
        return

