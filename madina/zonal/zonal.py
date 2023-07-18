from madina.zonal.layer import *
from madina.zonal.network import Network
from madina.zonal.network_utils import _node_edge_builder,  _discard_redundant_edges, _tag_edges, _effecient_node_insertion
from madina.zonal.zonal_utils import _prepare_geometry, DEFAULT_COLORS


import warnings
from geopandas import GeoDataFrame, GeoSeries
import geopandas as gpd
import pandas as pd


class Zonal:
    """
    Base class for geographic data in madina composed of multiple geographic layers.

    Parameters
    ----------
    scope: GeoSeries or GeoDataFrame, *optional*, defaults to None
        The scope of the Zonal object, determined by the geometry of the
        first element in the GeoSeries or the GeoDataFrame.
    projected_crs: str, *optional*, defaults to None
        The overall CRS of the Zonal object, all layers in the object will 
        be treated as in this CRS. If None, EPSG:4326 (WGS 84) will be used.
    
    Examples
    --------
    >>> from Madina.zonal.zonal import Zonal
    >>> city = Zonal(projected_crs = "EPSG:4326")

    Warnings
    --------
    The ``layers`` argument in the constructor need to be rewritten. Leave it blank for now.
    """

    DEFAULT_PROJECTED_CRS = "EPSG:4326"
    DEFAULT_GEOGRAPHIC_CRS = "EPSG:3857"
    DEFAULT_COLORS = DEFAULT_COLORS

    def __init__(self, scope: GeoSeries | GeoDataFrame = None, projected_crs: str = None,
                 layers: list = None):
        
        self.network = None
        if scope is not None:
            self._set_scope(scope, projected_crs)
        else:
            self.scope = scope
            self.projected_center = (None, None)
            self.geo_center = (None, None)

        if projected_crs is not None:
            self.projected_crs = projected_crs
        else:
            self.projected_crs = self.DEFAULT_PROJECTED_CRS

        self.layers = Layers(layers)

    def load_layer(
            self, 
            layer_name: str, 
            file_path: str, 
            allow_out_of_scope=False, 
            pos=None, 
            first=False, 
            before=None, 
            after=None
        ) -> None:
        """
        Loads a new geographic layer into the Zonal object from local files.

        Parameters
        ----------
        layer_name: str
            The name of the new layer
        file_path: str
            The file path to the file storing the geographic layer
        allow_out_of_scope: bool, *optional*, defaults to False
            If False, clips the layer to the scope of the Zonal object

        Examples
        -------
        >>> from Madina.zonal.zonal import Zonal
        >>> city = Zonal(projected_crs = "EPSG:4326")
        >>> city.load_layer("pedestrian_network", "../data/my_city_pedestrian_network.geojson")

        Warnings
        --------
        Pos, first, before, and after that were used for visualization need documentation.
        Leave them blank for now until visualization is implemented.
        """
        gdf = gpd.read_file(
            file_path,
            engine='pyogrio'
            )
        
        gdf_rows, _ = gdf.shape
        gdf['id'] = range(gdf_rows)
        gdf.set_index('id')
        original_crs = gdf.crs

        layer = Layer(
            layer_name,
            gdf,
            True,
            original_crs,
            file_path
            )
        self.layers.add(layer, pos, first, before, after)

        if None in self.geo_center:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", category=UserWarning)
                centroid_point = gdf.to_crs(self.projected_crs).dissolve().centroid.iloc[0]
            self.geo_center = centroid_point.coords[0]

        if not allow_out_of_scope and self.scope is not None:
            self.layers[layer_name].gdf = gpd.clip(
                self.layers[layer_name].gdf,
                gpd.GeoDataFrame({
                        'name': ['scope'], 
                        'geometry': [self.scope]
                    },
                    crs=self.projected_crs
                )
            )

        return

    def create_street_network(
            self,
            source_layer: str = "streets",
            weight_attribute: str = None,
            node_snapping_tolerance: int | float = 0.0,
            prepare_geometry=False,
            tag_edges=False,
            discard_redundant_edges=True,
            turn_threshold_degree: float = 45,
            turn_penalty_amount: float = 30,
        ) -> None:
        """
        Creates a network from the given layer with the given arguments.

        Parameters
        ----------
        source_layer: str, defaults to "streets"
            The name of the layer used to create the street network. Must be a layer composed of
            lines that has been already loaded into the Zonal object.
        weight_attribute: str, defaults to "None"
            The attribute name of the layer that will be used as the edge weight in the created 
            network. If None, the geometric length of the edges will be used
        node_snapping_tolerance: int, float, defaults to 0
            If greater than 0, snaps nodes that are closer than ``node_snapping_tolerance`` together
        prepare_geometry: bool, defaults to False
            If True, runs simple fixes to the geometry of the sourcelayer
        tag_edges: bool, defaults to False
            If True, attach tags to edges that are problematic
        discard_redundant_edges: bool, defaults to False
            If True, delete repetitive edges connecting the same nodes, preserving the shortest one
        turn_threshold_degree: float, defaults to 45
            The threshold of a turn to be considered penalizable
        turn_penalty_amount: float, defaults to 30
            The penalty added to a turn

        Examples
        --------
        >>> from Madina.zonal.zonal import Zonal
        >>> city = Zonal(projected_crs = "EPSG:4326")
        >>> city.load_layer("pedestrian_network", "../data/my_city_pedestrian_network.geojson")
        >>> city.create_street_network(
        ...     source_layer="pedestrian_network",
        ...     node_snapping_tolerance=1.5
        ...     discard_redundant_edges=True,
        ...     turn_threshold_degree=60,
        ...     turn_penalty_amount=42.3
        ... )
        """

        if source_layer not in self.layers:
            raise ValueError(f"Source layer {source_layer} not in zonal zonal_layers, available layers are: {self.layers.layers}")

        geometry_gdf = self.layers[source_layer].gdf
        if prepare_geometry:
            geometry_gdf = _prepare_geometry(geometry_gdf)

        node_gdf, edge_gdf = _node_edge_builder(
            geometry_gdf,
            weight_attribute=weight_attribute,
            tolerance=node_snapping_tolerance
        )

        if discard_redundant_edges:
            edge_gdf = _discard_redundant_edges(edge_gdf)


        if tag_edges:
            edge_gdf = _tag_edges(edge_gdf, tolerance=node_snapping_tolerance)


        self.network = Network(node_gdf, edge_gdf, self.projected_crs, turn_threshold_degree, turn_penalty_amount, weight_attribute)
        return

    def insert_node(self, layer_name: str, label: str ="origin", weight_attribute: str = None):
        """
        Insert nodes from a given layer to the network

        Parameters
        ----------
        layer_name: str
            The name of the layer from which nodes will be inserted
        label: str, defaults to "origin"
            The label of the inserted nodes in the network
        weight_attribute: str, defaults to None
            The attribute in the layer ``layer_name`` that will be used as the weight of the
            inserted nodes. If None, all weights will be assigned as 1
        """
        n_node_gdf = self.network.nodes
        n_edge_gdf = self.network.edges
        source_gdf = self.layers[layer_name].gdf
        inserted_node_gdf = _effecient_node_insertion(n_node_gdf, n_edge_gdf, source_gdf, layer_name=layer_name, label=label, weight_attribute=weight_attribute)
        self.network.nodes = pd.concat([n_node_gdf, inserted_node_gdf])
        return 

    def create_graph(self, light_graph=False, d_graph=True, od_graph=False):
        """
        Enables the creation of three kinds of graphs.

        Parameters
        ----------
        light_graph: bool, defaults to False
            If true, generate a graph in the network that contains only network nodes and edges.
        d_graph: bool, defaults to True
            If true, generate a graph in the network that contains the network nodes and edges, as
            well as the destination nodes.
        od_graph: bool, defaults to False
            If true, generate a graph in the network thatcontains all origin nodes, destination
            nodes, and network nodes and edges.

        Returns:
            None
        """
        self.network.create_graph(light_graph, d_graph, od_graph)

    def describe(self):
        """
        Prints information about the Zonal Object

        Examples
        --------
        >>> from Madina.zonal.zonal import Zonal
        >>> city = Zonal(projected_crs = "EPSG:4326")
        >>> city.describe()
        No layers yet, load a layer using 'load_layer(layer_name, file_path)'
        No scope yet. If needed (When your zonal_layers contain data that is outside of your analysis scope, setting a scope speeds up the analysis), set a scope using 'set_scope(scope)'
        No center yet, add a layer or set a scope to define a center
        No network graph yet. First, insert a layer that contains network segments (streets, sidewalks, ..) and call create_street_network(layer_name, weight_attribute=None)
        Then, insert origins and destinations using 'insert_nodes(label, layer_name, weight_attribute)'
        Finally, when done, create a network by calling 'create_street_network()'
        """
        if len(self.layers) == 0:
            print("No layers yet, load a layer using 'load_layer(layer_name, file_path)'")
        else:

            for key in self.layers:
                print(f"Layer name: {key}")
                print(f"\tVisible?: {self.layers[key].show}")
                print(f"\tFile path: {self.layers[key].file_path}")
                print(f"\tOriginal projection: {self.layers[key].crs}")
                print(f"\tCurrent projection: {self.layers[key].gdf.crs}")
                print(f"\tColumn names: {list(self.layers[key].gdf.columns)}")
                print(f"\tNumber of rows: {self.layers[key].gdf.shape[0]}")

        geo_center_x, geo_center_y = self.geo_center
        proj_center_x, proj_center_y = self.projected_center
        if self.scope is None:
            print("No scope yet. If needed (When your zonal_layers contain data that is outside of\
                   your analysis scope, setting a scope speeds up the analysis), set a scope using\
                   'set_scope(scope)'")

            if self.geo_center is None:
                print(f"No center yet, add a layer or set a scope to define a center")
            else:

                print(f"Projected center: projected center: ({proj_center_x}, {proj_center_y}), "
                      f"Geographic center: ({geo_center_x}, {geo_center_y})")
        else:
            print(f"Scope area: {self.scope.area}m2, "
                  f"Scope projected center: ({proj_center_x}, {proj_center_y}), "
                  f"Scope geographic center: ({geo_center_x}, {geo_center_y})")
        if self.network is None:
            print(f"No network graph yet. First, insert a layer that contains network segments\
                   (streets, sidewalks, ..) and call create_street_network(layer_name,\
                    weight_attribute=None)")
            print(f"\tThen, insert origins and destinations using 'insert_nodes(label, layer_name,\
                   weight_attribute)'")
            print(f"\tFinally, when done, create a network by calling 'create_street_network()'")

    def _set_scope(self, scope: GeoSeries | GeoDataFrame, projected_crs: str):

        def _get_geographic_scope(scope: GeoSeries | GeoDataFrame):
            if not (isinstance(scope, GeoSeries) or isinstance(scope, GeoDataFrame)):
                raise ValueError("scope must be a geopandas `Geoseries` or `GeoDataFrame` type")
            return scope[0] if isinstance(scope, GeoSeries) else scope.iloc[0]["geometry"]
        
        scope_geometry = _get_geographic_scope(scope)
        self.scope = scope_geometry  # set the Zonal object's scope to the geometry of the parameter

        if isinstance(scope, GeoSeries):
            scope_projected_crs = scope.crs
        else:
            scope_projected_crs = scope["geometry"].crs

        self.projected_crs = projected_crs if projected_crs != None else scope_projected_crs

        geographic_scope_gdf = gpd.GeoDataFrame({"id": [0], "geometry": [self.scope]}, crs=self.DEFAULT_PROJECTED_CRS)
        geographic_scope_gdf = geographic_scope_gdf.to_crs(self.DEFAULT_GEOGRAPHIC_CRS)
        geographic_scope = geographic_scope_gdf.at[0, "geometry"]
        self.projected_center = \
            (geographic_scope_gdf["geometry"].to_crs(self.DEFAULT_PROJECTED_CRS).at[0].centroid.coords[0][0],
             geographic_scope_gdf["geometry"].to_crs(self.DEFAULT_PROJECTED_CRS).at[0].centroid.coords[0][1])
        self.geo_center = (geographic_scope.centroid.coords[0][0], geographic_scope.centroid.coords[0][1])

        return
