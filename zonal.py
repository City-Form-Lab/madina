import warnings
import geopandas as gpd

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

        if scope:
            self._set_scope(scope, projected_crs)
        else:
            self.scope = scope
            self.projected_center = (None, None)
            self.geo_center = (None, None)

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
            self._geoprocess_layer(label, clip_by=self.scope)

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

    def _geoprocess_layer(self, label: str, clip_by=None, x_offset=0.0, y_offset=0.0):
        """
        Sets the CRS of the layer to the default.
        If `clip_by` is passed, it clips the GeoDataFrame. If offsets are defined, it shifts the geometry
        by `x_offset` and `y_offset`

        Returns:
            None
        """
        raise NotImplementedError
