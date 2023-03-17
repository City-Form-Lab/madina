import warnings
import geopandas as gpd
from network import Network
from network_utils import empty_network_template

from geopandas import GeoDataFrame, GeoSeries
from shapely import affinity


class Zonal:
    """
    Represents a zonal map with scope 'scope' and projection 'projected_crs'.
    Composed of layers, which...
    """
    default_projected_crs = "EPSG:4326"
    default_geographic_crs = "EPSG:3857"

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
            'gdf': gdf.to_crs(self.default_projected_crs),
            'show': True,
            'file_path': file_path,
            'original_crs': original_crs
        }

        # self.color_layer(layer_name)

        if None in self.geo_center:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", category=UserWarning)
                centroid_point = gdf.to_crs(self.default_geographic_crs).dissolve().centroid.iloc[0]
            self.geo_center = centroid_point.coords[0]

        if not allow_out_of_scope:
            self._geoprocess_layer(label)

        return

    def create_street_network(self, source_layer, flatten_polylines=True, node_snapping_tolerance=1,
                              fuse_2_degree_edges=True, tolerance_angle=10, solve_intersections=True,
                              loose_edge_trim_tolerance=0.001, weight_attribute=None, discard_redundant_edges=False,):
        """
        Creates a street network layer from the `source_layer`.

        Returns:
            None
        """

        if source_layer not in self.layers:
            raise ValueError(f"Source layer {source_layer} not in zonal layers")

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

        nodes, edges = self._load_nodes_and_edges(node_dict, edge_dict, line_geometry_gdf, node_snapping_tolerance, weight_attribute, discard_redundant_edges)

        self.network = Network(nodes, edges, self.projected_crs, weight_attribute)

        if fuse_2_degree_edges:
            self.network.fuse_degree_2_nodes(tolerance_angle)

        if solve_intersections:
            self.network.scan_for_intersections(node_snapping_tolerance)

        # we need a Layers class
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
        Inserts a node into the `Zonal`'s network.

        Returns:
            None
        """
        raise NotImplementedError

    def create_graph(self, light_graph, dense_graph, d_graph):
        """
        Enables the creation of three kinds of graphs.
        `light_graph` - contains only network nodes and edges

        `dense_graph` - contains all origin and destination nodes

        `dest_graph` - contains all destination nodes

        Returns:
            A new `Zonal` object with
        """
        raise NotImplementedError

    def describe(self):
        """
        Returns:
            a string representation of the `Zonal`
        """
        raise NotImplementedError

    def create_deck_map(self, layers: list[str], save_as: str, basemap: bool):
        """
        Creates a ... map (what type, what format)?

        Returns:
            A type of map (?)
        """
        raise NotImplementedError

    def save_map(self, data: list[GeoDataFrame], center_x: float, center_y: float, basemap: bool, zoom: int,
                 output_filename: str):
        """
        Saves a map with components `data`, center (`center_x`, `center_y`), basemap `basemap` and a `zoom` zoom
        to the filename `output_filename`

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
        raise NotImplementedError

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
            crs=self.default_projected_crs)

        layer_gdf = self.layers[label]['gdf']

        if clip:
            self.layers[label]['gdf'] = gpd.clip(layer_gdf, scope_gdf)

        if x_offset:
            for idx in layer_gdf.index:
                layer_gdf.at[idx, "geometry"] = affinity.translate(layer_gdf.at[idx, "geometry"], xoff=x_offset,
                                                                   yoff=y_offset)

        return

    def _load_nodes_and_edges(self, node: dict, edge: dict, source: GeoDataFrame, node_snapping_tolerance: int,
                              weight_attribute: int, discard_redundant_edges: bool):
        """
        TODO: Fill out function spec
        """
        raise NotImplementedError
