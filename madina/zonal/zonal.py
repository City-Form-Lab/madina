import warnings
import geopandas as gpd

from network import Network
from network_utils import empty_network_template, DEFAULT_COLORS
from zonal_utils import flatten_multi_edge_segments, load_nodes_edges_from_gdf
from layer import *

from geopandas import GeoDataFrame, GeoSeries

from typing import Union


class Zonal:
    """
    Represents a zonal map with scope 'scope' and projection 'projected_crs'.
    Composed of zonal_layers, which...
    """
    DEFAULT_PROJECTED_CRS = "EPSG:4326"
    DEFAULT_GEOGRAPHIC_CRS = "EPSG:3857"
    DEFAULT_COLORS = DEFAULT_COLORS

    def __init__(self, scope: Union[GeoSeries, GeoDataFrame] = None, projected_crs: str = None,
                 layers: list = None):

        self.network = None
        if scope:
            self._set_scope(scope, projected_crs)
        else:
            self.scope = scope
            self.projected_center = (None, None)
            self.geo_center = (None, None)

        self.projected_crs = projected_crs

        self.layers = Layers(layers)

    def load_layer(self, layer_name: str, file_path: str, allow_out_of_scope=False, pos=None, first=False, before=None, after=None):
        """
        Loads a new layer from file path `file_path` with the layer name `layer_name`.
        If `allow_out_of_scope` is false, clips the CRS to the scope of the Zonal object

        Returns:
            None
        """
        gdf = gpd.read_file(file_path)
        gdf_rows, _ = gdf.shape
        gdf['id'] = range(gdf_rows)
        gdf.set_index('id')
        original_crs = gdf.crs

        layer = Layer(layer_name, gdf.to_crs(self.DEFAULT_PROJECTED_CRS), True, original_crs, file_path)
        self.layers.add(layer, pos, first, before, after)

        if None in self.geo_center:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", category=UserWarning)
                centroid_point = gdf.to_crs(self.DEFAULT_GEOGRAPHIC_CRS).dissolve().centroid.iloc[0]
            self.geo_center = centroid_point.coords[0]

        if not allow_out_of_scope:
            # TODO: Check scope is not None?
            self.layers[layer_name].gdf = gpd.clip(
                self.layers[layer_name].gdf,
                gpd.GeoDataFrame({
                        'name': ['scope'], 
                        'geometry': [self.scope]
                    },
                    crs=self.DEFAULT_PROJECTED_CRS
                )
            )

        return

    def create_street_network(self, source_layer: str, node_snapping_tolerance=1,
                              weight_attribute=None, discard_redundant_edges=False):
        """
        Creates a street network layer from the `source_layer` with the given arguments.

        Returns:
            None
        """

        if source_layer not in self.layers:
            raise ValueError(f"Source layer {source_layer} not in zonal zonal_layers")

        line_geometry_gdf = self.layers[source_layer].gdf.copy()
        line_geometry_gdf["length"] = line_geometry_gdf["geometry"].length
        line_geometry_gdf = line_geometry_gdf[line_geometry_gdf["length"] > 0]

        if 'network_nodes' not in self.layers:
            node_dict, edge_dict = empty_network_template['node'], empty_network_template['edge']
        else:
            node_dict = self.layers["network_nodes"].gdf.to_dict()
            edge_dict = self.layers["network_edges"].gdf.to_dict()

        nodes, edges = load_nodes_edges_from_gdf(
            node_dict, edge_dict, line_geometry_gdf, node_snapping_tolerance,
            weight_attribute, discard_redundant_edges
        )

        self.network = Network(nodes, edges, self.projected_crs, weight_attribute)

        self.layers['network_nodes'], self.layers['network_edges'] = self.network.network_to_layer()

        return

    def insert_nodes(self, label: str, layer_name: str, weight_attribute: int):
        """
        Inserts a node into a layer within the `Zonal`.

        Returns:
            None
        """
        raise NotImplementedError

    def create_graph(self, light_graph, dense_graph, d_graph):
        """
        Enables the creation of three kinds of graphs.

        Args:
            `light_graph` - contains only network nodes and edges
            `dense_graph` - contains all origin, destination, network, etc. nodes
            `dest_graph` - contains all destination nodes and network intersectionsa

        Returns:
            None
        """
        raise NotImplementedError

    def describe(self):
        """
        Returns:
            a string representation of the `Zonal`
        """
        if len(self.layers) == 0:
            print("No zonal_layers yet, load a layer using 'load_layer(layer_name, file_path)'")
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
            print(
                "No scope yet. If needed (When your zonal_layers contain data that is outside of your analysis scope, setting a scope speeds up the analysis), set a scope using 'set_scope(scope)'")

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
            print(
                f"No network graph yet. First, insert a layer that contains network segments (streets, sidewalks, ..) and call create_street_network(layer_name,  weight_attribute=None)")
            print(f"\tThen,  insert origins and destinations using 'insert_nodes(label, layer_name, weight_attribute)'")
            print(f"\tFinally, when done, create a network by calling 'create_street_network()'")

    def _set_scope(self, scope: Union[GeoSeries, GeoDataFrame], projected_crs: str):
        """
        Sets the `Zonal` object's scope and projection.

        Returns:
            None
        """

        def _get_geographic_scope(scope: Union[GeoSeries, GeoDataFrame]):
            """
            Strips the provided `scope` down to its geometry.

            Returns:
                Geometry of `scope`
            """
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
