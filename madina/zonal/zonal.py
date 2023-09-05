# this lets geopandas exclusively use shapely (not pygeos) silences a warning about depreciating pygeos out of geopandas. This is not needed when geopandas 1.0 is released in the future
import os
os.environ['USE_PYGEOS'] = '0'

from madina.zonal.layer import *
from madina.zonal.network import Network
from madina.zonal.network_utils import _node_edge_builder,  _discard_redundant_edges, _split_redundant_edges, _tag_edges, _effecient_node_insertion
from madina.zonal.zonal_utils import _prepare_geometry, DEFAULT_COLORS



import warnings
import geopandas as gpd
import pandas as pd
from typing import Union

import pydeck as pdk
from pydeck.types import String
import numpy as np
import random
import time


__version__ = '0.0.4'
__release_date__ = '2023-08-13'


class Zonal:
    """
    Represents a zonal map with scope 'scope' and projection 'projected_crs'.
    Composed of zonal_layers, which...
    """
    DEFAULT_PROJECTED_CRS ="EPSG:3857"
    DEFAULT_GEOGRAPHIC_CRS ="EPSG:4326"
    DEFAULT_COLORS = DEFAULT_COLORS

    def __init__(self, layers: list = None):
        self.network = None
        self.geo_center = (None, None)
        self.layers = Layers(layers)

    def load_layer(self, layer_name: str, file_path: str, pos=None, first=False, before=None, after=None):
        """
        Loads a new layer from file path `file_path` with the layer name `layer_name`.
        If `allow_out_of_scope` is false, clips the CRS to the scope of the Zonal object

        Returns:
            None
        """
        gdf = gpd.read_file(
            file_path,
            engine='pyogrio'
            )
        
        gdf['id'] = range(gdf.shape[0])
        gdf.set_index('id')
        original_crs = gdf.crs
        gdf = self.color_gdf(gdf)

        # perform a standard data cleaning process to ensure compatibility with later processes
        gdf = _prepare_geometry(gdf)

        layer = Layer(
            layer_name,
            gdf, 
            True,
            original_crs,
            file_path
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
                # This is to ignore a warning issued for dpoing calculations in a geographic coordinate system, but that's the needed output:
                # a point in a geographic coordinate system to center the visualization
                warnings.simplefilter("ignore", category=UserWarning)
                centroid_point = gdf.iloc[[0]].to_crs(self.DEFAULT_GEOGRAPHIC_CRS).centroid.iloc[0]
            self.geo_center = centroid_point.coords[0]
        return

    def create_street_network(
            self,
            source_layer: str ="streets",
            weight_attribute=None,
            node_snapping_tolerance: Union[int, float] = 0.0,
            prepare_geometry=False,
            tag_edges=False,
            discard_redundant_edges=False,
            split_redundant_edges=True,
            turn_threshold_degree=45,
            turn_penalty_amount=30,
        ) -> None:
        """
        Creates a street network layer from the `source_layer` with the given arguments.

        Returns:
            None
        """

        if source_layer not in self.layers:
            raise ValueError(f"Source layer {source_layer} not in zonal zonal_layers, available layers are: {self.layers.layers}")

        geometry_gdf = self.layers[source_layer].gdf

        #TODO: consider removing this, as preparing geometry is now a standard precedure when loading a new layer
        if prepare_geometry:
            geometry_gdf = _prepare_geometry(geometry_gdf)

        node_gdf, edge_gdf = _node_edge_builder(
            geometry_gdf,
            weight_attribute=weight_attribute,
            tolerance=node_snapping_tolerance
        )

        if split_redundant_edges:
            node_gdf , edge_gdf = _split_redundant_edges(node_gdf ,edge_gdf)
        elif discard_redundant_edges:
            edge_gdf = _discard_redundant_edges(edge_gdf)


        


        if tag_edges:
            edge_gdf = _tag_edges(edge_gdf, tolerance=node_snapping_tolerance)


        self.network = Network(node_gdf, edge_gdf, turn_threshold_degree, turn_penalty_amount, weight_attribute)
        return

    def insert_node(self, layer_name: str, label: str ="origin", weight_attribute: str = None):
        """
        Inserts a node into a layer within the `Zonal`.

        Returns:
            None
        """
        n_node_gdf = self.network.nodes
        n_edge_gdf = self.network.edges
        source_gdf = self.layers[layer_name].gdf
        inserted_node_gdf = _effecient_node_insertion(n_node_gdf, n_edge_gdf, source_gdf, layer_name=layer_name, label=label, weight_attribute=weight_attribute)
        self.network.nodes = pd.concat([n_node_gdf, inserted_node_gdf])
        return 

    def create_graph(self, light_graph=True, d_graph=True, od_graph=False):
        """
        Enables the creation of three kinds of graphs.

        Args:
            `light_graph` - contains only network nodes and edges
            `od_graph` - contains all origin, destination, network, etc. nodes
            `d_graph` - contains all destination nodes and network intersectionsa

        Returns:
            None
        """
        self.network.create_graph(light_graph, d_graph, od_graph)

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

    def create_map(self, layer_list=None, save_as=None, basemap=False):
        if layer_list is None:
            layer_list = []
            for layer_name in self.layers.layers:
                if self.layers[layer_name].show:
                    layer_list.append({"gdf": self.layers[layer_name].gdf})
        else:
            for layer_position, layer_dict in enumerate(layer_list):
                if "layer" in layer_dict:
                    # switch from ysung the keyword layer, into using the keyword 'gdf' by supplying layer's gdf
                    layer_dict['gdf'] = self.layers[layer_dict["layer"]].gdf
                    layer_list[layer_position] = layer_dict
        map = self.create_deckGL_map(
            gdf_list=layer_list,
            centerX=self.geo_center[0],
            centerY=self.geo_center[1],
            basemap=basemap,
            zoom=17,
            filename=save_as
        )
        return map

    @staticmethod
    def create_deckGL_map(gdf_list=[], centerX=46.6725, centerY=24.7425, basemap=False, zoom=17, filename=None):
        start = time.time()
        pdk_layers = []
        for layer_number, gdf_dict in enumerate(gdf_list):
            local_gdf = gdf_dict["gdf"].copy(deep=True)
            local_gdf["geometry"] = local_gdf["geometry"].to_crs("EPSG:4326")
            #print(f"{(time.time()-start)*1000:6.2f}ms\t {layer_number = }, gdf copied")
            start = time.time()

            radius_attribute = 1
            if "radius" in gdf_dict:
                radius_attribute = gdf_dict["radius"]
                r_series = local_gdf[radius_attribute]
                r_series = (r_series - r_series.mean()) / r_series.std()
                r_series = r_series.apply(lambda x: max(1,x) + 3 if not np.isnan(x) else 0.5)
                local_gdf[radius_attribute] = r_series

            width_attribute = 1
            width_scale = 1
            if "width" in gdf_dict:
                width_attribute = gdf_dict["width"]
                if "width_scale" in gdf_dict:
                    width_scale = gdf_dict["width_scale"]
                local_gdf[width_attribute] = local_gdf[width_attribute] * width_scale

            if "opacity" in gdf_dict:
                opacity = gdf_dict["opacity"]
            else:
                opacity = 1

            if ("color_by_attribute" in gdf_dict) or ("color_method" in gdf_dict) or ("color" in gdf_dict):
                args = {arg: gdf_dict[arg] for arg in ['color_by_attribute', 'color_method', 'color'] if arg in gdf_dict}
                local_gdf = Zonal.color_gdf(local_gdf, **args)
                #print (local_gdf['color'])

            pdk_layer = pdk.Layer(
                'GeoJsonLayer',
                local_gdf.reset_index(),
                opacity=opacity,
                stroked=True,
                filled=True,
                wireframe=True,
                get_line_width=width_attribute,
                get_radius=radius_attribute,
                get_line_color='color',
                get_fill_color="color",
                pickable=True,
            )
            pdk_layers.append(pdk_layer)
            #print(f"{(time.time()-start)*1000:6.2f}ms\t {layer_number = }, layer styled and added.")
            start = time.time()

            if "text" in gdf_dict:
                # if numerical, round within four decimals, else, do nothing and treat as string
                try:
                    local_gdf["text"] = round(local_gdf[gdf_dict["text"]], 6).astype('string')
                except TypeError:
                    local_gdf["text"] = local_gdf[gdf_dict["text"]].astype('string')

                # formatting a centroid point to be [lat, long]
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", category=UserWarning)
                    local_gdf["coordinates"] = local_gdf["geometry"].centroid
                local_gdf["coordinates"] = [[p.coords[0][0], p.coords[0][1]] for p in local_gdf["coordinates"]]

                layer = pdk.Layer(
                    "TextLayer",
                    local_gdf.reset_index(),
                    pickable=True,
                    get_position="coordinates",
                    get_text="text",
                    get_size=16,
                    get_color='color',
                    get_angle=0,
                    background=True,
                    get_background_color=[0, 0, 0, 125],
                    # Note that string constants in pydeck are explicitly passed as strings
                    # This distinguishes them from columns in a data set
                    get_text_anchor=String("middle"),
                    get_alignment_baseline=String("center"),
                )
                pdk_layers.append(layer)
                #print(f"{(time.time()-start)*1000:6.2f}ms\t {layer_number = }, text layer created and added.")
                start = time.time()

        initial_view_state = pdk.ViewState(
            latitude=centerY,
            longitude=centerX,
            zoom=zoom,
            max_zoom=20,
            pitch=0,
            bearing=0
        )

        if basemap:
            r = pdk.Deck(
                layers=pdk_layers,
                initial_view_state=initial_view_state,
            )
        else:
            r = pdk.Deck(
                layers=pdk_layers,
                initial_view_state=initial_view_state,
                map_provider=None,
                parameters={
                    "clearColor": [0.00, 0.00, 0.00, 1]
                },
            )

        if filename is not None:
            r.to_html(
                filename,
                css_background_color="cornflowerblue"
            )
        #print(f"{(time.time()-start)*1000:6.2f}ms\t {layer_number = }, map rendered.")
        start = time.time()
        return r

    def color_layer(self, layer_name, color_by_attribute=None, color_method="single_color", color=None):
        if layer_name in self.default_colors.keys() and color_by_attribute is None and color is None:
            # set default colors first. all default layers call without specifying "color_by_attribute"
            # default layer creation always calls self.color_layer(layer_name) without any other parameters
            color = self.default_colors[layer_name].copy()
            color_method = "single_color"
            if type(color) is dict:
                # the default color is categorical..
                color_by_attribute = color["__attribute_name__"]
                color_method = "categorical"
        self.layers[layer_name]["gdf"] = self.color_gdf(
            self.layers[layer_name]["gdf"],
            color_by_attribute=color_by_attribute,
            color_method=color_method,
            color=color
        )
        return
    
    @staticmethod
    def color_gdf(gdf, color_by_attribute=None, color_method=None, color=None):
        """
        A  method to set geometry color

        :param gdf: GeoDataFrame to be colored.
        :param color_by_attribute: string, attribute name, or column name to
        visualize geometry by
        :param color_method: string, "single_color" to color all geometry by the same color.
        "categorical" to use distingt color to distingt value, "gradient": to use a gradient of colors for a
        neumeric, scalar attribute.
        :param color: if color method is single color, expects one color. if categorical,
        expects nothing and would give automatic assignment, or a dict {"val': [0,0,0]}. if color_method is gradient,
        expects nothing for a default color map, or a color map name
        :return: nothing
        """
        if color_method is None:
            if color_by_attribute is not None:
                color_method = "categorical"
            else:
                color_method = "single_color"

        if color_by_attribute is None and color is None:
            # if "color_by_attribute" is not given, and its not a default layer, assuming color_method == "single_color"
            # if no color is given, assign random color, else, color=color
            color = [random.random() * 255, random.random() * 255, random.random() * 255]
            color_method = "single_color"
        elif color is None:


            # color by attribute ia given, but no color is given..
            if color_method == "single_color":
                # if color by attribute is given, and color method is single color, this is redundant but just in case:
                color = [random.random() * 255, random.random() * 255, random.random() * 255]
            if color_method == "categorical":
                color = {"__other__": [255, 255, 255]}
                for distinct_value in gdf[color_by_attribute].unique():
                    color[distinct_value] = [random.random() * 255, random.random() * 255, random.random() * 255]

        # create color column
        if color_method == "single_color":
            color_column = [color] * len(gdf)
        elif color_method == "categorical":
            #color = {"__other__": [255, 255, 255]}
            color_column = []
            for value in gdf[color_by_attribute]:
                if value in color.keys():
                    color_column.append(color[value])
                else:
                    color_column.append(color["__other__"])
        elif color_method == "gradient":
            cbc = gdf[color_by_attribute]  # color by column
            nc = 255 * (cbc - cbc.min()) / (cbc.max() - cbc.min())  # normalized column
            color_column = [[255 - v, 0 + v, 0] if not np.isnan(v) else [255, 255, 255] for v in list(nc)]  # convert normalized values to color spectrom.
            # TODO: insert color map options here..
        elif color_method == 'quantile':
            scaled_percentile_rank = 255 * gdf[color_by_attribute].rank(pct=True)
            color_column = [[255.0 - v, 0.0 + v, 0] if not np.isnan(v) else [255, 255, 255] for v in
                            scaled_percentile_rank]  # convert normalized values to color spectrom.

        gdf["color"] = color_column
        return gdf