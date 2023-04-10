import warnings
import geopandas as gpd
import pydeck as pdk

from network import Network
from network_utils import empty_network_template, DEFAULT_COLORS
from zonal_utils import get_color_column

from geopandas import GeoDataFrame, GeoSeries
from shapely import affinity


def _get_geographic_scope(scope: GeoSeries | GeoDataFrame):
    """
    Strips the provided `scope` down to its geometry.

    Returns:
        Geometry of `scope`
    """
    if not (isinstance(scope, gpd.geoseries.GeoSeries) or isinstance(scope, gpd.GeoDataFrame)):
        raise ValueError("scope must be a geopandas `Geoseries` or `GeoDataFrame` type")

    return scope[0] if isinstance(scope, GeoSeries) else scope.iloc[0]["geometry"]


class Zonal:
    """
    Represents a zonal map with scope 'scope' and projection 'projected_crs'.
    Composed of zonal_layers, which...
    """
    DEFAULT_PROJECTED_CRS = "EPSG:4326"
    DEFAULT_GEOGRAPHIC_CRS = "EPSG:3857"
    DEFAULT_COLORS = DEFAULT_COLORS

    def __init__(self, scope: GeoSeries | GeoDataFrame = None, projected_crs: str = None,
                 layers: dict = None):

        self.network = None
        if scope:
            self._set_scope(scope, projected_crs)
        else:
            self.scope = scope
            self.projected_center = (None, None)
            self.geo_center = (None, None)

        self.projected_crs = projected_crs

        self.layers = {} if layers is None else layers

    def load_layer(self, label, file_path, allow_out_of_scope=False):
        """
        Loads a new layer from file path `file_path` with the label `label`.
        If `allow_out_of_scope` enabled, clips the CRS.

        Returns:
            None
        """
        gdf = gpd.read_file(file_path)
        gdf_rows, _ = gdf.shape
        gdf['id'] = range(gdf_rows)
        gdf.set_index('id')
        original_crs = gdf.crs

        self.layers[label] = {
            'gdf': gdf.to_crs(self.DEFAULT_PROJECTED_CRS),
            'show': True,
            'file_path': file_path,
            'original_crs': original_crs
        }

        # self.color_layer(layer_name)

        if None in self.geo_center:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", category=UserWarning)
                centroid_point = gdf.to_crs(self.DEFAULT_GEOGRAPHIC_CRS).dissolve().centroid.iloc[0]
            self.geo_center = centroid_point.coords[0]

        if not allow_out_of_scope:
            self._geoprocess_layer(label)

        return

    def create_street_network(self, source_layer, flatten_polylines=True, node_snapping_tolerance=1,
                              fuse_2_degree_edges=True, tolerance_angle=10, solve_intersections=True,
                              loose_edge_trim_tolerance=0.001, weight_attribute=None, discard_redundant_edges=False, ):
        """
        Creates a street network layer from the `source_layer`.
        Maps to create_street_nodes_edges in `betweenness functions`

        Returns:
            None
        """

        if source_layer not in self.layers:
            raise ValueError(f"Source layer {source_layer} not in zonal zonal_layers")

        line_geometry_gdf = self.layers[source_layer]["gdf"].copy()
        line_geometry_gdf["length"] = line_geometry_gdf["geometry"].length
        line_geometry_gdf = line_geometry_gdf[line_geometry_gdf["length"] > 0]

        if 'network_nodes' not in self.layers:
            node_dict, edge_dict = empty_network_template['node'], empty_network_template['edge']
        else:
            node_dict = self.layers["network_nodes"]['gdf'].to_dict()
            edge_dict = self.layers["network_edges"]['gdf'].to_dict()

        # if flatten_polylines:
        #     line_geometry_gdf = flatten_multi_edge_segments(line_geometry_gdf)

        nodes, edges = self._load_nodes_and_edges_from_gdf(
            node_dict, edge_dict, line_geometry_gdf, node_snapping_tolerance,
            weight_attribute, discard_redundant_edges
        )

        self.network = Network(nodes, edges, self.projected_crs, weight_attribute)

        if fuse_2_degree_edges:
            self.network.fuse_degree_2_nodes(tolerance_angle)

        if solve_intersections:
            self.network.scan_for_intersections(node_snapping_tolerance)

        # TODO: we need a Layers internal class
        self.layers['network_nodes'], self.layers['network_edges'] = self.network.network_to_layer()

        # self.color_layer('network_nodes')
        # self.color_layer('network_edges')

        return

    def reset_nodes(self):
        """
        Sets `Zonal` nodes to the original's street nodes.

        Returns:
            None
        """
        raise NotImplementedError

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
        `light_graph` - contains only network nodes and edges

        `dense_graph` - contains all origin, destination, network, etc. nodes

        `dest_graph` - contains all destination nodes and network intersectionsa

        Returns:
            None
        """
        raise NotImplementedError

    def color_layer(self, label, by_attribute=None, method="single", color_scheme=None):
        """
        Colors the layer with name `label` using the attribute `by_attribute` with method `method`
        and scheme `color_scheme`.

        Adds a color field to the Zonal layer.

        Returns:
            None
        """
        if label in self.DEFAULT_COLORS.keys() and not by_attribute and not color_scheme:
            color = self.DEFAULT_COLORS[label].copy()

            if type(color_scheme) is dict:
                by_attribute = color["__attribute_name__"]
                method = "categorical"

        self.layers[label]["gdf"] = self._color_layer(
            label,
            by_attribute,
            method,
            color_scheme
        )

        return

    def describe(self):
        """
        Returns:
            a string representation of the `Zonal`
        """
        if len(self.layers) == 0:
            print("No zonal_layers yet, load a layer using 'load_layer(layer_name, file_path)'")
        else:

            for key in self.layers.keys():
                print(f"Layer name: {key}")
                print(f"\tVisible?: {self.layers[key]['show']}")
                print(f"\tFile path: {self.layers[key]['file_path']}")
                print(f"\tOriginal projection: {self.layers[key]['original_crs']}")
                print(f"\tCurrent projection: {self.layers[key]['gdf'].crs}")
                print(f"\tColumn names: {list(self.layers[key]['gdf'].columns)}")
                print(f"\tNumber of rows: {self.layers[key]['gdf'].shape[0]}")

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

    def create_map(self, zonal_layers: list[str], save_as: str, basemap: bool, add_layers: dict[str, dict]):
        """
        Creates a map with specified `zonal_layers` with the filename `save_as`. If `zonal_layers` is left blank, all layers within the Zonal are mapped.

        The optional field `add_layers` allows addition of layers that are not within the `Zonal` object.
        Uses the `Layer` object fields. Only the `gdf` is required.

        Returns:
            An HTML map
        """

        gdf_layers = []

        if not zonal_layers:
            for label in self.layers:
                if self.layers[label]["show"]:
                    gdf_layers.append({"label": label, "gdf": self.layers[label]["gdf"]})
        else:
            for label in zonal_layers:
                gdf_layers.append({"label": label, "gdf": self.layers[label]["gdf"]})
            for label in add_layers:
                add_layer = add_layers[label]

                if 'gdf' not in add_layer:
                    raise ValueError(f"Additional layer {label} is required to have a gdf property.")
                if type(add_layer["gdf"]) != GeoDataFrame:
                    raise TypeError(f"Additional layer {label} is required to be a GeoDataFrame type.")

                gdf_layers.append({"label": label} | add_layer)  # merges the dictionaries

        map = self.save_map(gdf_layers, *self.geo_center, basemap, zoom=17, output_filename=save_as)

        return map

    def save_map(self, layers: list[dict], center_x: float, center_y: float, basemap: bool, zoom: int,
                 output_filename: str):
        """
        Saves a map with zonal_layers `zonal_layers`, center (`center_x`, `center_y`), basemap `basemap` and a `zoom` zoom
        to the filename `output_filename`.

        Returns:
            An HTML map
        """
        raise NotImplementedError

    def _set_scope(self, scope: GeoSeries | GeoDataFrame, projected_crs: str):
        """
        Sets the `Zonal` object's scope and projection.

        Returns:
            None
        """
        scope_geometry = _get_geographic_scope(scope)
        self.scope = scope_geometry  # set the Zonal object's scope to the geometry of the parameter

        if isinstance(scope, gpd.geoseries.GeoSeries):
            scope_projected_crs = scope.crs
        else:
            scope_projected_crs = scope["geometry"].crs

        self.projected_crs = projected_crs

        if not self.projected_crs:
            self.projected_crs = scope_projected_crs

        geo_scope = gpd.GeoDataFrame({"id": [0], "geometry": [scope_geometry]}, crs=self.projected_crs)
        reprojected_geo_scope = geo_scope.to_crs(self.DEFAULT_GEOGRAPHIC_CRS)

        proj_center = geo_scope.at[0, "geometry"]
        geo_center = reprojected_geo_scope.at[0, "geometry"]
        # TODO: Check above

        self.projected_center = (proj_center.centroid.coords[0][0], proj_center.centroid.coords[0][1])
        self.geo_center = (geo_center.centroid.coords[0][0], geo_center.centroid.coords[0][1])

        return

    def _geoprocess_layer(self, label: str, clip=True, x_offset=0.0, y_offset=0.0):
        """
        Sets the CRS of the layer to the default.
        If `clip_by` is passed, it clips the GeoDataFrame. If offsets are defined, it shifts the geometry
        by `x_offset` and `y_offset`

        TODO: Approve change to functionality

        Returns:
            None
        """
        scope_gdf = gpd.GeoDataFrame({
            'name': ['scope'],
            'geometry': [self.scope]
        },
            crs=self.DEFAULT_PROJECTED_CRS)

        layer_gdf = self.layers[label]['gdf']

        if clip:
            self.layers[label]['gdf'] = gpd.clip(layer_gdf, scope_gdf)

        if x_offset:
            for idx in layer_gdf.index:
                layer_gdf.at[idx, "geometry"] = affinity.translate(layer_gdf.at[idx, "geometry"], xoff=x_offset,
                                                                   yoff=y_offset)

        return

    def _color_layer(self, label, by_attribute=None, method=None, color_scheme=None):
        """
        Internal function which colors a layer with label `label` with attribute `by_attribute`
        using the method `method`.

        Returns:
            A GeoDataFrame object with a color column added.

        """
        gdf = self.layers[label]["gdf"]

        if not method:
            if by_attribute:
                method = "single"
            else:
                method = "categorical"

        # if "color_by_attribute" is not given, and its not a default layer, assuming color_method == "single_color"
        if by_attribute is None and color_scheme is None:
            method = "single"

        gdf['color'] = get_color_column(method, gdf, by_attribute, color_scheme)

        return gdf

    def _load_nodes_and_edges_from_gdf(self, node: dict, edge: dict, source: GeoDataFrame, node_snapping_tolerance: int,
                                       weight_attribute: int, discard_redundant_edges: bool):
        """
        Assumes a nodes and edge table has been created from the geometry.
        Takes a layer that's a different geometry and maps them to the closest network edge.
        Creates a table that contains this new snapped geometry with the weight of the node, along with closest nodes.
        """
        raise NotImplementedError
