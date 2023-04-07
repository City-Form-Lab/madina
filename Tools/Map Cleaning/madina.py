import shapely.geometry as geo
from shapely.ops import split, substring
from shapely import affinity

import warnings
with warnings.catch_warnings():
    warnings.simplefilter("ignore", category=UserWarning)
    import geopandas as gpd
import pandas as pd
import pydeck as pdk
from pydeck.types import String
import numpy as np
import networkx as nx
import time
from networkx.classes.function import path_weight
import math
import random


from betweenness_functions import update_light_graph, turn_o_scope


class Zonal:
    # TODO: the Zonal object could have multiple networks. an implementation could be a dictionary similar to the layers
    default_map_name = "Zonal_map"

    # "EPSG:4326", is WGS 84, the geographic coordinate system used in GPS, and is used to visualize maps using Deck
    default_geographic_crs = "EPSG:4326"

    # "EPSG:3857", is a projected coordinate system used for rendering maps in Google Maps, OpenStreetMap. Used as a
    # default projected coordinate system in case one was not given at initialization. This is a very low accuracy
    # projected coordinate system
    # TODO: implement a function that sets a better accuracy projected crs if one is not
    #  given at initialization. Ideas: use Universal Transverse Mercator (UTM) where the appropriate grid is chosen
    #  based on layer content. another idea: if one (or many) layers have a usable projected coordinate system,
    #  use it to unify all layer into (The most occurring?) one
    default_projected_crs = "EPSG:3857"

    # Default colors applied initially  by the function "color by attribute".
    default_colors = {
        "streets": [150, 150, 150],
        "blocks": [100, 100, 100],
        "parcels": [50, 50, 50],
        "network_edges": {
            "__attribute_name__": "type",
            "street": [0, 150, 150],
            "project_line": [0, 0, 150],
            "__other__": [0, 0, 0]
        },
        "network_nodes": {
            "__attribute_name__": "type",
            "street_node": [0, 255, 255],
            "project_node": [0, 0, 255],
            "destination": [255, 0, 0],
            "origin": [0, 255, 0],
            "__other__": [0, 0, 0]
        }
    }
    STYLE_COLUMNS = ["color"]
    default_layer_names = ['streets', 'blocks', 'parcels', 'network_nodes', 'network_edges']
    ## Currently used
    def __init__(self, scope=None, projected_crs=None):
        if scope is not None:
            self.set_scope(scope, projected_crs)
        else:
            self.scope = None
            self.centerX_projected = None
            self.centerY_projected = None
            self.centerX_geographic = None
            self.centerY_geographic = None
        self.layers = {}
        self.G = None

    ## Currently used
    def set_scope(self, scope, projected_crs=None):
        # if scope was a GeoSeries or a GeoDataFrame, strip down into geometry.
        if isinstance(scope, gpd.geoseries.GeoSeries):
            scope_projected_crs = scope.crs
            scope = scope[0]
        elif isinstance(scope, gpd.GeoDataFrame):
            scope_projected_crs = scope["geometry"].crs
            scope = scope.iloc[0]["geometry"]

        # if projected_crs was given, it would be used. this overrides if a Geoseries or a GeoDataFrame was given
        # with a crs
        if projected_crs is not None:
            self.default_projected_crs = projected_crs
        else:
            self.default_projected_crs = scope_projected_crs

            # raise a TypeError in case scope is not a polygon
        if not isinstance(scope, geo.polygon.Polygon):
            raise TypeError

        geographic_scope_gdf = gpd.GeoDataFrame({"id": [0], "geometry": [scope]}, crs=self.default_projected_crs)
        geographic_scope_gdf = geographic_scope_gdf.to_crs(self.default_geographic_crs)
        geographic_scope = geographic_scope_gdf.at[0, "geometry"]
        self.centerX_projected = \
            geographic_scope_gdf["geometry"].to_crs(self.default_projected_crs).at[0].centroid.coords[0][0]
        self.centerY_projected = \
            geographic_scope_gdf["geometry"].to_crs(self.default_projected_crs).at[0].centroid.coords[0][1]
        self.centerX_geographic = geographic_scope.centroid.coords[0][0]
        self.centerY_geographic = geographic_scope.centroid.coords[0][1]

        self.scope = scope
        return

    ## Currently used
    def describe(self):
        """
        A function to return a string representation of the object for debugging and/or development
        :return:
        """
        if len(self.layers) == 0:
            print("No layers yet, load a layer using 'load_layer(layer_name, file_path)'")
        else:
            for key in self.layers.keys():
                print(f"Layer name: {key}")
                print(f"\tVisible?: {self.layers[key]['show']}")
                print(f"\tFile path: {self.layers[key]['file_path']}")
                print(f"\tOriginal projection: {self.layers[key]['original_crs']}")
                print(f"\tCurrent projection: {self.layers[key]['gdf'].crs}")
                print(f"\tColumn names: {list(self.layers[key]['gdf'].columns)}")
                print(f"\tNUmber of rows: {self.layers[key]['gdf'].shape[0]}")
        if self.scope is None:
            print(
                "No scope yet. If needed (When your layers contain data that is outside of your analysis scope, setting a scope speeds up the analysis), set a scope using 'set_scope(scope)'")
            if self.centerY_geographic is None:
                print(f"No center yet, add a layer or set a scope to define a center")
            else:
                print(f"Projected center: projected center: ({self.centerX_projected}, {self.centerY_projected}), "
                      f"Geographic center: ({self.centerX_geographic}, {self.centerY_geographic})")
        else:
            print(f"Scope area: {self.scope.area}m2, "
                  f"Scope projected center: ({self.centerX_projected}, {self.centerY_projected}), "
                  f"Scope geographic center: ({self.centerX_geographic}, {self.centerY_geographic})")
        if self.G is None:
            print(
                f"No graph yet. FIrst, insert a layer that contains network segments (streets, sidewalks, ..) and call create_network_nodes_edges(layer_name,  weight_attribute=None)")
            print(f"\tThen,  insert origins and destinations using 'insert_nodes(label, layer_name, weight_attribute)'")
            print(f"\tFinally, when done, create a network by calling 'create_graph()'")

        return

    ## Currently used
    def set_attribute(self, layer=None, id=None, attribute=None, value=None):
        self.layers[layer]["gdf"].at[id, attribute] = value
        return

    ## Currently used
    def clear_nodes(self):
        node_gdf = self.layers["network_nodes"]["gdf"]
        node_gdf = node_gdf[node_gdf["type"] == "street_node"]
        self.layers["network_nodes"]["gdf"] = node_gdf
        return

    ## Currently used
    def load_layer(self, layer_name, file_path, do_not_load=False, allow_out_of_scope=False):
        if do_not_load:
            return
        else:
            gdf = gpd.read_file(file_path)
            gdf["id"] = range(gdf.shape[0])
            gdf.set_index('id')
            original_crs = gdf.crs
            self.layers[layer_name] = {'gdf': gdf.to_crs(self.default_projected_crs),
                                       'show': True,
                                       "file_path": file_path,
                                       "original_crs": original_crs
                                       }
            self.color_layer(layer_name)

            if self.centerX_geographic is None:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", category=UserWarning)
                    centroid_point = gdf.to_crs(self.default_geographic_crs).dissolve().centroid.iloc[0]
                self.centerX_geographic = centroid_point.coords[0][0]
                self.centerY_geographic = centroid_point.coords[0][1]

            if allow_out_of_scope:
                return
            else:
                self.geoprocess_layer(layer_name, clip_by=self.scope)
                return

    ## Depreciated
    def filter_layer(self, layer_name, attribute, isin=None):
        filtered_gdf = self.layers[layer_name]["gdf"]

        if isin is not None:
            filtered_gdf = filtered_gdf[filtered_gdf[attribute].isin(isin)]

        self.layers[layer_name]["gdf"] = filtered_gdf
        return

    ## Depreciated
    def geoprocess_layer(self, layer_name, xoff=None, yoff=None, clip_by=None):
        scope_gdf = gpd.GeoDataFrame({
            'name': ['scope'],
            'geometry': [self.scope]
        },
            crs=self.default_projected_crs)
        layer_gdf = self.layers[layer_name]['gdf']

        if clip_by is not None:
            self.layers[layer_name]['gdf'] = gpd.clip(layer_gdf, scope_gdf)

        if xoff is not None:
            for idx in layer_gdf.index:
                layer_gdf.at[idx, "geometry"] = affinity.translate(layer_gdf.at[idx, "geometry"], xoff=xoff, yoff=yoff)

        return


    ## Currently used
    def create_deck_map(self, layer_list=None, save_as=None, basemap=False):
        if layer_list is None:
            layer_list = []
            for layer_name in self.layers:
                if self.layers[layer_name]["show"]:
                    layer_list.append({"gdf": self.layers[layer_name]["gdf"]})
        else:
            for layer_id, layer_dict in enumerate(layer_list):
                if "layer" in layer_dict:
                    layer_dict['gdf'] = self.layers[layer_dict["layer"]]["gdf"]
                    layer_list[layer_id] = layer_dict
        map = self.save_map(
            layer_list,
            centerX=self.centerX_geographic,
            centerY=self.centerY_geographic,
            basemap=basemap,
            filename=save_as
        )
        return map

    ## Currently used
    @staticmethod
    def save_map(gdf_list, centerX=46.6725, centerY=24.7425, basemap=False, zoom=17, filename="output_map.html"):
        pdk_layers = []
        for gdf_dict in gdf_list:
            local_gdf = gdf_dict["gdf"].copy(deep=True)
            local_gdf["geometry"] = local_gdf["geometry"].to_crs("EPSG:4326")

            radius_attribute = 1
            if "radius" in gdf_dict:
                radius_attribute = "radius"
                r_series = local_gdf[gdf_dict["radius"]]
                # normalize between 0-1
                #r_series = (r_series - r_series.min()) / (r_series.max() - r_series.min())

                # mean normalization
                r_series = (r_series - r_series.mean()) / r_series.std()

                r_series = r_series.apply(lambda x: max(1,x) + 3 if not np.isnan(x) else 0.5)
                local_gdf["radius"] = r_series

            width_attribute = 1
            width_scale = 1
            if "width" in gdf_dict:
                width_attribute = "width"
                if "width_scale" in gdf_dict:
                    width_scale = gdf_dict["width_scale"]
                local_gdf["width"] = local_gdf[gdf_dict["width"]] * width_scale

            if "opacity" in gdf_dict:
                opacity = gdf_dict["opacity"]
            else:
                opacity = 1

            if ("color_by_attribute" in gdf_dict) or ("color_method" in gdf_dict) or ("color" in gdf_dict):
                args = {arg: gdf_dict[arg] for arg in ['color_by_attribute', 'color_method', 'color'] if
                        arg in gdf_dict}
                local_gdf = Zonal.color_gdf(local_gdf, **args)

            pdk_layer = pdk.Layer(
                'GeoJsonLayer',
                local_gdf.reset_index(),
                opacity=opacity,
                stroked=True,
                filled=True,
                # extruded=False if "extrude" not in self.layers[layer_name] else self.layers[layer_name]["extrude"],
                # get_elevation=0 if "extrude_attribute" not in self.layers[layer_name] else self.layers[layer_name][
                #    "extrude_attribute"],
                wireframe=True,
                get_line_width=width_attribute,
                get_radius=radius_attribute,
                # get_line_color='[155, 155, 155]',
                get_line_color='color',
                get_fill_color="color",
                pickable=True,
            )
            pdk_layers.append(pdk_layer)

            if "text" in gdf_dict:
                # if numerical, round within four decimals, else, do nothing
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

        initial_view_state = pdk.ViewState(
            # latitude=self.centerY,
            latitude=centerY,
            # longitude=self.centerX,
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
                # map_provider='google_maps',
                # map_style='satellite' # "dark"
            )
        else:
            r = pdk.Deck(
                layers=pdk_layers,
                initial_view_state=initial_view_state,
                map_provider=None,
                parameters={
                    "clearColor": [0.00, 0.00, 0.00, 1]
                    # "clearColor": [0.10, 0.10, 0.10, 1]
                },
                # tooltip={"text": "{name}\n{address}"}
                # width='100%',
                # height=2500,
            )

        if filename is not None:
            r.to_html(
                filename,
                css_background_color="cornflowerblue"
            )

        return r

    ## Currently used
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

    ## Currently used
    @staticmethod
    def color_gdf(gdf, color_by_attribute=None, color_method=None, color=None):
        """
        A static method to set geometry color

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

    # Gerenative Geometry
    def create_layer_from_geometry(self, geometry, geotype, layer_name, visibility=True, color=None):
        d = {'id':
                 list(range(len(geometry))),
             'geometry':
                 list(geometry.geoms)
             }
        # TODO: Add dimension attributes...
        # if geotype == 'line':
        #            d["length"] = [segment.length for segment in geometry],
        #        if geotype == 'polygon':
        #            d["area"] = [segment.area for segment in geometry],

        gdf = gpd.GeoDataFrame(d, crs=self.default_projected_crs)
        gdf = gdf.set_index('id')
        # utm_crs = gdf["geometry"].estimate_utm_crs()
        # print (utm_crs)
        # gdf["geometry"] = gdf["geometry"].to_crs(utm_crs)
        self.layers[layer_name] = {'gdf': gdf, 'show': visibility}
        self.color_layer(layer_name)
        return

    # Gerenative Geometry
    def generate_grid_streets(self, scope, n_rows, n_columns):
        avenue_segments = []
        coords = []
        bottom_avenue = geo.LineString([
            (scope.bounds[0], scope.bounds[1]),
            (scope.bounds[2], scope.bounds[1])
        ])
        street_length = geo.LineString([
            (scope.bounds[0], scope.bounds[1]),
            (scope.bounds[0], scope.bounds[3])
        ]).length

        for j in range(n_columns):
            top_avenue = affinity.translate(bottom_avenue, yoff=1 / n_columns * street_length)
            for i in range(n_rows):
                x1 = substring(bottom_avenue, start_dist=i / n_rows, end_dist=(i + 1) / n_rows, normalized=True)
                x2 = substring(top_avenue, start_dist=i / n_rows, end_dist=(i + 1) / n_rows, normalized=True)
                avenue_segments.append(x1)

                coords.append(
                    (x1.coords[0], x2.coords[0])
                )

                # Close the top on last iteration
                if j + 1 == n_columns:
                    avenue_segments.append(x2)
                if i + 1 == n_rows:
                    coords.append(
                        (x1.coords[1], x2.coords[1])
                    )
            bottom_avenue = top_avenue
        street_segments = geo.MultiLineString(coords)
        all_segments = geo.MultiLineString(avenue_segments + list(street_segments.geoms))
        self.create_layer_from_geometry(all_segments, geotype='line', layer_name='streets', visibility=True)
        return

    # Gerenative Geometry
    def generate_radial_streets(self, radius, nrays, nrows):
        rays = []

        main_ray = geo.LineString([
            (self.centerX_projected, self.centerY_projected),
            (self.centerX_projected, self.centerY_projected + radius)
        ])
        rays.append(main_ray)
        for i in range(nrays - 1):
            rays.append(
                affinity.rotate(main_ray,
                                (i + 1) / nrays * 360,
                                origin=(self.centerX_projected, self.centerY_projected)
                                )
            )

        coords = []
        ray_segments = []
        for j, ray in enumerate(rays):
            next_ray = rays[(j + 1) % len(rays)]
            for i in range(nrows):
                x1 = substring(ray, start_dist=i / nrows, end_dist=(i + 1) / nrows, normalized=True)
                x2 = substring(next_ray, start_dist=i / nrows, end_dist=(i + 1) / nrows, normalized=True)
                coords.append(
                    (x1.coords[1], x2.coords[1])
                )
                ray_segments.append(x1)
                # ray_segments.append(x2)

        # axis_streets = geo.MultiLineString(rays)
        axis_streets = geo.MultiLineString(ray_segments)
        round_streets = geo.MultiLineString(coords)
        all_streets = geo.MultiLineString(list(axis_streets.geoms) + list(round_streets.geoms))
        self.create_layer_from_geometry(all_streets, geotype='line', layer_name='streets', visibility=True)
        return

    # Gerenative Geometry
    def generate_blocks(self, street_width):
        street_geometry = geo.MultiLineString(
            list(self.layers['streets']['gdf']['geometry'])
        )
        buffered_streets = street_geometry.buffer(street_width)
        blocks = geo.MultiPolygon(
            self.scope.difference(buffered_streets)
        )
        self.create_layer_from_geometry(blocks, geotype='polygon', layer_name='blocks', visibility=True)
        return

    # Gerenative Geometry
    def generate_parcels(self, max_parcel_frontage, abutting_parcels=False):
        # TODO: to enble abutting parcels, we need to slice across the short axis.
        parcels = []
        for idx, row in self.layers['blocks']['gdf'].iterrows():
            block = row['geometry']

            ## Make sure all block are four-sided..
            original_block = block  ## preserve original block for clipping, as all non four-sided shapes are sliced from a best-fit rectangule
            block_coords = list(block.exterior.coords)
            if (len(block_coords) != 5):  ## non-rectangular polygons
                block = block.minimum_rotated_rectangle
                block_coords = list(block.exterior.coords)

            ## Identify longest axis as front, and its back.
            segment_length = []
            for i in range(len(block_coords) - 1):
                segment = geo.LineString([block_coords[i], block_coords[i + 1]])
                segment_length.append(segment.length)
            longest_segment = segment_length.index(max(segment_length))

            front = geo.LineString([block_coords[longest_segment], block_coords[longest_segment + 1]])
            back_segment_idx = (longest_segment + 2) % 4
            back = geo.LineString([block_coords[back_segment_idx + 1], block_coords[back_segment_idx]])

            ## Generate parcel splitter lines
            n_parcels = int(np.ceil(front.length / max_parcel_frontage))
            splitters = []
            for i in range(n_parcels):
                x1 = substring(front, start_dist=i / n_parcels, end_dist=(i + 1) / n_parcels, normalized=True)
                x2 = substring(back, start_dist=i / n_parcels, end_dist=(i + 1) / n_parcels, normalized=True)
                splitter = geo.LineString((x1.coords[1], x2.coords[1]))
                splitter = affinity.scale(splitter, xfact=2.00, yfact=2.00,
                                          origin='center')  # extend the spliter to ensure it falls out of the original geometry
                splitters.append(splitter)

            ## split blocks by parcel split lines
            parcels_in_this_block = original_block
            for line in splitters:
                parcels_in_this_block = split(parcels_in_this_block, line)
                parcels_in_this_block = geo.MultiPolygon(list(parcels_in_this_block.geoms))

            # for visuals, scale down parcels and append them into the parcel list.
            for parcel in list(parcels_in_this_block.geoms):
                parcel = affinity.scale(parcel, xfact=0.80, yfact=0.80, origin='center')
                parcels.append(parcel)
        parcels = geo.MultiPolygon(parcels)

        self.create_layer_from_geometry(parcels, geotype='polygon', layer_name='parcels', visibility=True)
        return












    # Depreciated, replace by the one in development..
    # Network/UNA Function
    def create_network_nodes_edges(self, layer_name="streets", weight_attribute=None, temporary_copy=False):
        ## The goal of this function is to ensure topological integrity of the network, provide lists of nodes and edges needed to produce a NetworkX Graph
        ## TODO: Thi function could be faster when avoiding the use of node_gdf = node_gdf.append(). construct a dictionary first, then when finished, make the node_gdf.

        # street_geometry = geo.MultiLineString(
        #            list(self.layers['streets']['gdf']['geometry'])
        #        )
        street_gdf = self.layers[layer_name]['gdf']

        node_dict = {
            "id": [],
            "geometry": [],
            "source_layer": [],
            "source_id": [],
            "type": [],
            "weight": [],
            "nearest_street_id": [],
            "nearest_street_node_distance": [],
        }

        node_gdf = gpd.GeoDataFrame(node_dict, crs=self.default_projected_crs).astype({'id': 'int32'})
        node_gdf = node_gdf.set_index("id")
        edge_dict = {
            "id": [],
            "start": [],
            "end": [],
            "length": [],
            "weight": [],
            "type": [],
            "geometry": [],
            "parent_street_id": [],
        }
        edge_gdf = gpd.GeoDataFrame(edge_dict, crs=self.default_projected_crs).astype({'id': 'int32'})
        edge_gdf = edge_gdf.set_index("id")

        node_id = 0
        edge_id = 0
        for street_idx in street_gdf.index:
            street = street_gdf.at[street_idx, "geometry"]
            node_snapping_tolerance = 0.0005
            ## Add ends to node list, get IDs, make sure to avoid duplicates.
            start_point_index = None
            end_point_index = None
            if len(street.coords) != 2:  ## bad segment, consider segmenting to individual segments for accuracy..
                start_point = geo.Point(street.coords[0])
                end_point = geo.Point(street.coords[-1])
                # street = geo.LineString([start_point, end_point])
            else:  ## Good segment with two ends
                start_point = geo.Point(street.coords[0])
                end_point = geo.Point(street.coords[1])

            for node_idx in node_gdf.index:
                if node_gdf.at[node_idx, "geometry"].almost_equals(start_point, decimal=6):
                    start_point_index = node_idx
                if node_gdf.at[node_idx, "geometry"].almost_equals(end_point, decimal=6):
                    end_point_index = node_idx
                # TODO: this loop could either be iliminated by a better loockup, or terminated earlier

            if start_point_index is None:
                start_point_index = node_id
                node_id += 1

                node_gdf = node_gdf.append(
                    {"id": start_point_index,
                     "geometry": start_point,
                     "source_layer": "streets",
                     "source_id": None,
                     "type": "street_node",
                     "weight": 0,
                     "nearest_street_id": None,
                     "nearest_street_node_distance": 0,
                     }
                    , ignore_index=True)
            if end_point_index is None:
                end_point_index = node_id
                node_id += 1
                node_gdf = node_gdf.append(
                    {"id": end_point_index,
                     "geometry": end_point,
                     "source_layer": "streets",
                     "source_id": None,
                     "type": "street_node",
                     "weight": 0,
                     "nearest_street_id": None,
                     "nearest_street_node_distance": None,
                     }
                    , ignore_index=True,
                )

            ## Need to add a segment
            edge_gdf = edge_gdf.append(
                {
                    "id": edge_id,
                    "start": start_point_index,
                    "end": end_point_index,
                    "length": street.length,
                    "weight": street_gdf.at[
                        street_idx, weight_attribute] if weight_attribute is not None else street.length,
                    "type": "street",
                    "geometry": street,
                    "parent_street_id": street_idx,
                }
                , ignore_index=True)
            edge_id += 1
        node_gdf = node_gdf.set_index('id')
        node_gdf["geometry"] = node_gdf["geometry"].set_crs(self.default_projected_crs)
        edge_gdf = edge_gdf.set_index("id")
        edge_gdf["geometry"] = edge_gdf["geometry"].set_crs(self.default_projected_crs)
        self.layers['network_nodes'] = {'gdf': node_gdf, 'show': True}
        self.color_layer('network_nodes')
        self.layers['network_edges'] = {'gdf': edge_gdf, 'show': True}
        self.color_layer('network_edges')
        return

    # TODO : write a replacement for this function assuming a light graph. This function is now a serious bottleneck in large simulations..

    # Up to date, not completely tested...
    # Network/UNA Function
    def insert_nodes_v2(self, label, layer_name, weight_attribute):
        node_gdf = self.layers["network_nodes"]["gdf"]
        edge_gdf = self.layers["network_edges"]["gdf"]
        source_gdf = self.layers[layer_name]["gdf"]

        node_dict = node_gdf.reset_index().to_dict()

        def cut(line, distance):
            # Cuts a line in two at a distance from its starting point
            if distance <= 0.0:
                return [
                    geo.LineString([line.coords[0], line.coords[0]]),
                    geo.LineString(line)]
            elif distance >= line.length:
                return [
                    geo.LineString(line),
                    geo.LineString([line.coords[-1], line.coords[-1]])
                ]

            coords = list(line.coords)
            for i, p in enumerate(coords):
                pd = line.project(geo.Point(p))
                if pd == distance:
                    return [
                        geo.LineString(coords[:i + 1]),
                        geo.LineString(coords[i:])]
                if pd > distance:
                    cp = line.interpolate(distance)
                    return [
                        geo.LineString(coords[:i] + [(cp.x, cp.y)]),
                        geo.LineString([(cp.x, cp.y)] + coords[i:])]

        new_node_id = int(node_gdf.index[-1])  # increment before use.
        for source_id in source_gdf.index:
            # find nearest edge:
            source_representative_point = source_gdf.at[source_id, "geometry"].centroid
            closest_edge_id = edge_gdf["geometry"].distance(source_representative_point)
            closest_edge_geometry = edge_gdf.at[closest_edge_id, "geometry"]
            point_on_closest_edge = closest_edge_geometry.project(source_representative_point)

            try:
                cut_lines = cut(closest_edge_geometry, point_on_closest_edge)
                start_segment = cut_lines[0]
                end_segment = cut_lines[1]
            except:
                # TODO: test cases where this exception occurs.
                continue

            new_node_id += 1
            project_point_id = new_node_id
            node_dict["id"][new_node_id] = new_node_id
            node_dict["geometry"][new_node_id] = point_on_closest_edge
            node_dict["source_layer"][new_node_id] = layer_name
            node_dict["source_id"][new_node_id] = source_id
            node_dict["type"][new_node_id] = label
            node_dict["weight"][new_node_id] = \
                1.0 if weight_attribute is None else source_gdf.at[source_id, weight_attribute]

            left_edge_weight = \
                edge_gdf.at[closest_edge_id, "weight"] * start_segment.length / closest_edge_geometry.length
            right_edge_weight = edge_gdf.at[closest_edge_id, "weight"] - left_edge_weight
            node_dict["nearest_street_id"][new_node_id] = closest_edge_id
            node_dict["nearest_street_node_distance"][new_node_id] = \
                {
                    "left":
                        {
                            "node_id": edge_gdf.at[closest_edge_id, "start"],
                            # "distance": start_segment.length,
                            "weight": left_edge_weight,
                            "geometry": start_segment
                        },
                    "right":
                        {
                            "node_id": edge_gdf.at[closest_edge_id, "end"],
                            # "distance": closest_edge_geometry.length - start_segment.length,
                            "weight": right_edge_weight,
                            "geometry": end_segment
                        }
                }
        node_gdf = gpd.GeoDataFrame(node_dict, crs=self.default_projected_crs)
        node_gdf = node_gdf.set_index("id")
        self.layers['network_nodes']['gdf'] = node_gdf
        self.color_layer('network_nodes')
        return node_gdf

    #Depreciated, but tested heavily.
    # Network/UNA Function
    def insert_nodes(self, label=None, layer_name=None, filter=None, attachment_method="light_insert",
                     representative_point="nearest_point", node_weight_attribute=None,
                     projection_edge_cost_attribute=None, source_identifier="id",
                     node_gdf=None, edge_gdf=None, source_gdf=None):
        # attachment_method= "project" | "snap" | "light_insert"
        # representative_point="nearest_point" | "centroid"
        # labels = "origin" | "destination" | "observer"

        # TODO: This function should preserve style. style settings should be saved intl the layer, and reapplied after any
        #  layer insert/delete/update operation.
        # TODO : for now, focus on a given layer as input, deal with a filter later..

        node_gdf = self.layers['network_nodes']['gdf'].drop(self.STYLE_COLUMNS, axis=1, errors='ignore')
        edge_gdf = self.layers['network_edges']['gdf'].drop(self.STYLE_COLUMNS, axis=1, errors='ignore')
        source_gdf = self.layers[layer_name]["gdf"]
        # TODO: keep track of a layer's identifier in the Zonal object layers...
        # self.layers[layer_name]["identifier"]

        # counters to keep track of what a new node should have as an id. Increment prior to each use.
        # new_node_id = int(node_gdf.iloc[-1]["id"])
        new_node_id = int(max(node_gdf.index))
        # new_edge_id = int(edge_gdf.iloc[-1]["id"])
        new_edge_id = int(max(edge_gdf.index))

        # dictionaries are much more effecient for insertion than pandas df/gdfs.
        edge_dict = edge_gdf.reset_index().to_dict()
        node_dict = node_gdf.reset_index().to_dict()

        for source_idx in source_gdf.index:
            # TODO: handle variations of geometries and representative point options here.
            source_representative_point = source_gdf.at[source_idx, "geometry"].centroid

            nearest_street_distance = 9999999999999999999
            nearest_street_id = None
            point_on_the_street = None
            street_segment_ids = [edge_id for edge_id in edge_dict["type"] if edge_dict["type"][edge_id] == "street"]
            for street_id in street_segment_ids:
                street_segment = edge_dict["geometry"][street_id]
                distance = street_segment.distance(source_representative_point)
                if distance < nearest_street_distance:
                    nearest_street_distance = distance
                    nearest_street_id = street_id
                    old_street_segment = edge_dict["geometry"][nearest_street_id]  ## needed to be split later..
                    point_on_the_street = street_segment.interpolate(street_segment.project(
                        source_representative_point))  ## gives a point on te street where the source point is projected

            # start_segment = geo.LineString((old_street_segment.coords[0], point_on_the_street))
            # end_segment = geo.LineString((point_on_the_street, old_street_segment.coords[1]))

            def cut(line, distance):
                # Cuts a line in two at a distance from its starting point
                if distance <= 0.0:
                    return [
                        geo.LineString([line.coords[0], line.coords[0]]),
                        geo.LineString(line)]
                elif distance >= line.length:
                    return [
                        geo.LineString(line),
                        geo.LineString([line.coords[-1], line.coords[-1]])
                    ]

                coords = list(line.coords)
                for i, p in enumerate(coords):
                    pd = line.project(geo.Point(p))
                    if pd == distance:
                        return [
                            geo.LineString(coords[:i + 1]),
                            geo.LineString(coords[i:])]
                    if pd > distance:
                        cp = line.interpolate(distance)
                        return [
                            geo.LineString(coords[:i] + [(cp.x, cp.y)]),
                            geo.LineString([(cp.x, cp.y)] + coords[i:])]

            try:
                cut_lines = cut(
                    old_street_segment,
                    old_street_segment.project(source_representative_point)
                )
                start_segment = cut_lines[0]
                end_segment = cut_lines[1]
            except:
                continue

            if attachment_method == "project":
                # need to store representative point (labeled as origin)
                # + a projection line (labeled as "projection_line"
                # + a projection point (point_on_the_street) labeled as "projection_point

                new_node_id += 1
                representative_point_id = new_node_id
                node_dict["id"][representative_point_id] = representative_point_id
                node_dict["geometry"][representative_point_id] = source_representative_point
                node_dict["source_layer"][representative_point_id] = layer_name
                node_dict["source_id"][representative_point_id] = source_idx
                node_dict["type"][representative_point_id] = label
                node_dict["weight"][representative_point_id] = \
                    1 if node_weight_attribute is None else source_gdf.at[source_idx, node_weight_attribute]

                new_node_id += 1
                project_point_id = new_node_id
                node_dict["id"][project_point_id] = project_point_id
                node_dict["geometry"][project_point_id] = point_on_the_street
                node_dict["source_layer"][project_point_id] = layer_name
                node_dict["source_id"][project_point_id] = source_idx
                node_dict["type"][project_point_id] = "project_point"
                node_dict["weight"][project_point_id] = 0

                new_edge_id += 1
                projection_segment_id = new_edge_id
                projection_segment = geo.LineString((source_representative_point, point_on_the_street))

                edge_dict["id"][projection_segment_id] = projection_segment_id
                edge_dict["start"][projection_segment_id] = representative_point_id
                edge_dict["end"][projection_segment_id] = project_point_id
                edge_dict["length"][projection_segment_id] = \
                    0 if projection_edge_cost_attribute is None else source_gdf.at[
                        source_idx, projection_edge_cost_attribute]
                # projection_segment.length
                edge_dict["type"][projection_segment_id] = "project_line"
                edge_dict["geometry"][projection_segment_id] = projection_segment

            elif attachment_method in ["snap", "light_insert"]:
                # need to only store point_on_the_street (labeled as origin)
                new_node_id += 1
                project_point_id = new_node_id
                node_dict["id"][project_point_id] = project_point_id
                node_dict["geometry"][project_point_id] = point_on_the_street
                node_dict["source_layer"][project_point_id] = layer_name
                node_dict["source_id"][project_point_id] = source_idx
                node_dict["type"][project_point_id] = label
                node_dict["weight"][project_point_id] = \
                    0 if node_weight_attribute is None else source_gdf.at[source_idx, node_weight_attribute]

                ## These two are needed for light inserts to be used later in network operations..

                ## this is for numerical accuricy so things add up precisely to the original length/weight values
                left_edge_weight = edge_dict["weight"][
                                       nearest_street_id] * start_segment.length / old_street_segment.length
                right_edge_weight = edge_dict["weight"][
                                        nearest_street_id] - left_edge_weight
                node_dict["nearest_street_id"][project_point_id] = nearest_street_id
                node_dict["nearest_street_node_distance"][project_point_id] = \
                    {
                        "left":
                            {
                                "node_id": edge_dict["start"][nearest_street_id],
                                "distance": start_segment.length,
                                "weight": left_edge_weight,
                                "geometry": start_segment
                                ## TODO: This should be a weight, from the edge layer, not a length... but weighted by length obviosly`
                            },
                        "right":
                            {
                                "node_id": edge_dict["end"][nearest_street_id],
                                "distance": old_street_segment.length - start_segment.length,
                                ## this is for numerical accuricy so things add up precisely to the original length/weight values
                                "weight": right_edge_weight,
                                "geometry": end_segment
                                ## TODD: This too should be a weight, but adjusted to lengrh.
                            }
                    }

            else:
                print("method is incorrect or not implemented.")

            # insert segmented street
            if attachment_method in ["snap", "project"]:
                new_edge_id += 1
                edge_dict["id"][new_edge_id] = new_edge_id
                edge_dict["start"][new_edge_id] = edge_dict["start"][nearest_street_id]
                edge_dict["end"][new_edge_id] = project_point_id
                edge_dict["length"][new_edge_id] = start_segment.length
                edge_dict["weight"][new_edge_id] = edge_dict["weight"][
                                                       nearest_street_id] * start_segment.length / old_street_segment.length
                edge_dict["type"][new_edge_id] = "street"
                edge_dict["geometry"][new_edge_id] = start_segment
                edge_dict["parent_street_id"][new_edge_id] = edge_dict["parent_street_id"][nearest_street_id]

                new_edge_id += 1
                edge_dict["id"][new_edge_id] = new_edge_id
                edge_dict["start"][new_edge_id] = project_point_id
                edge_dict["end"][new_edge_id] = edge_dict["end"][nearest_street_id]
                edge_dict["length"][new_edge_id] = end_segment.length
                edge_dict["weight"][new_edge_id] = edge_dict["weight"][
                                                       nearest_street_id] * end_segment.length / old_street_segment.length
                edge_dict["type"][new_edge_id] = "street"
                edge_dict["geometry"][new_edge_id] = end_segment
                edge_dict["parent_street_id"][new_edge_id] = edge_dict["parent_street_id"][nearest_street_id]

                # remove old street
                [edge_dict[k].pop(nearest_street_id) for k in edge_dict.keys()]

        node_gdf = gpd.GeoDataFrame(node_dict, crs=self.default_projected_crs)
        node_gdf = node_gdf.set_index("id")

        edge_gdf = gpd.GeoDataFrame(edge_dict, crs=self.default_projected_crs)
        edge_gdf = edge_gdf.set_index("id")

        self.layers['network_nodes']['gdf'] = node_gdf
        self.color_layer('network_nodes')
        self.layers['network_edges']['gdf'] = edge_gdf
        self.color_layer('network_edges')

        # TODO: re-apply style with new points.
        return  # node_gdf, edge_gdf

    # Network/UNA Function
    def create_graph(self, light_graph=False, dense_graph=False, d_graph=True):
        edge_gdf = self.layers['network_edges']['gdf']

        if light_graph:
            node_gdf = self.layers['network_nodes']['gdf']
            node_gdf = node_gdf[node_gdf["type"] == "street_node"]
            ## no need to filter edges, because in light graph, they haven't been inseted
            G = nx.Graph()
            for idx in edge_gdf.index:
                G.add_edge(
                    int(edge_gdf.at[idx, "start"]),
                    int(edge_gdf.at[idx, "end"]),
                    weight=max(edge_gdf.at[idx, "weight"], 0),
                    type=edge_gdf.at[idx, "type"],
                    id=idx
                )

            for idx in node_gdf.index:
                G.nodes[int(idx)]['type'] = node_gdf.at[idx, "type"]

            self.G = G
        if dense_graph:
            node_gdf = self.layers['network_nodes']['gdf']
            od_list = list(node_gdf[node_gdf["type"].isin(["origin", "destination"])].index)
            graph = self.G.copy()
            update_light_graph(
                self,
                graph=graph,
                add_nodes=od_list
            )
            self.od_graph = graph

        if d_graph:
            node_gdf = self.layers['network_nodes']['gdf']
            od_list = list(node_gdf[node_gdf["type"] == "destination"].index)
            graph = self.G.copy()
            update_light_graph(
                self,
                graph=graph,
                add_nodes=od_list
            )
            self.d_graph = graph
        return

    # Network/UNA Function
    def una_accessibility(self, reach=False, gravity=False, closest_facility=False, weight=None, search_radius=None,
                          alpha=1, beta=None):
        if gravity and (beta is None):
            raise ValueError("Please specify parameter 'beta' when 'gravity' is True")

        node_gdf = self.layers["network_nodes"]["gdf"]
        origin_gdf = node_gdf[node_gdf["type"] == "origin"]

        reaches = {}
        gravities = {}

        for o_idx in origin_gdf.index:
            reaches[o_idx] = {}
            gravities[o_idx] = {}
            d_idxs, o_scope, o_scope_paths = turn_o_scope(
                self,
                o_idx,
                search_radius,
                detour_ratio=1.00,
                turn_penalty=False,
                o_graph=None,
                return_paths=False
            )

            for d_idx in d_idxs:
                source_id = int(node_gdf.at[d_idx, "source_id"])
                source_layer = node_gdf.at[d_idx, "source_layer"]
                d_weight = 1 if weight is None else self.layers[source_layer]["gdf"].at[source_id, weight]

                if reach:
                    reaches[o_idx][d_idx] = d_weight
                if gravity:
                    gravities[o_idx][d_idx] = pow(d_weight, alpha) / (pow(math.e, (beta * d_idxs[d_idx])))

                if closest_facility:
                    if ("una_closest_destination" not in node_gdf.loc[d_idx]) or (np.isnan(node_gdf.loc[d_idx]["una_closest_destination"])):
                        node_gdf.at[d_idx, "una_closest_destination"] = o_idx
                        node_gdf.at[d_idx, "una_closest_destination_distance"] = d_idxs[d_idx]
                    elif d_idxs[d_idx] < node_gdf.at[d_idx, "una_closest_destination_distance"]:
                        node_gdf.at[d_idx, "una_closest_destination"] = o_idx
                        node_gdf.at[d_idx, "una_closest_destination_distance"] = d_idxs[d_idx]

            if reach:
                node_gdf.at[o_idx, "una_reach"] = sum([value for value in reaches[o_idx].values() if not np.isnan(value)])

            if gravity:
                node_gdf.at[o_idx, "una_gravity"] = sum([value for value in gravities[o_idx].values() if not np.isnan(value)])

        if closest_facility:
            for o_idx in origin_gdf.index:
                o_closest_destinations = list(node_gdf[node_gdf["una_closest_destination"] == o_idx].index)
                if reach:
                    node_gdf.at[o_idx, "una_reach"] = sum([reaches[o_idx][d_idx] for d_idx in o_closest_destinations])
                if gravity:
                    node_gdf.at[o_idx, "una_gravity"] = sum([gravities[o_idx][d_idx] for d_idx in o_closest_destinations])
        return

    # Network/UNA Function
    def una_service_area(self, origin_ids=None, search_radius=None):
        node_gdf = self.layers["network_nodes"]["gdf"]
        edge_gdf = self.layers["network_edges"]["gdf"]
        origins = node_gdf[node_gdf["type"] == "origin"]

        if origin_ids is None:
            origin_ids = list(origins.index)
        elif not isinstance(origin_ids, list):
            raise ValueError("the 'origin_ids' parameter expects a list, for single origins, set it to [11] for example")

        if len(set(origin_ids) - set(origins.index)) != 0:
            raise ValueError(f"some of the indices given are not for an origin {set(origin_ids) - set(origins.index)}")

        scope_names = []
        scope_geometries = []
        scope_origin_ids = []

        network_edges = []
        destination_ids = set()
        source_layer = None

        # TODO: This loop assumes all destinations are the same layer, generalize to all layers.
        for origin_id in origin_ids:
            d_idxs, o_scope, o_scope_paths = turn_o_scope(
                self,
                origin_id,
                search_radius,
                detour_ratio=1.00,
                turn_penalty=False,
                o_graph=None,
                return_paths=False
            ) 
            if (len(d_idxs)) == 0:
                continue
            destination_original_ids = list(node_gdf.loc[list(d_idxs.keys())]["source_id"])
            source_layer = node_gdf.at[list(d_idxs.keys())[0], "source_layer"]
            destination_ids = destination_ids.union(destination_original_ids)

            destinations = self.layers[source_layer]["gdf"].loc[destination_original_ids]
            destination_scope = destinations.dissolve().convex_hull.iloc[0]


            network_scope = node_gdf.loc[set(o_scope.keys())].dissolve().convex_hull.iloc[0]
            scope = destination_scope.union(network_scope)

            network_edges.append(edge_gdf.clip(scope).dissolve().iloc[0]["geometry"])

            origin_geom = node_gdf.at[origin_id, "geometry"]

            scope_names.append("service area border")
            scope_names.append("origin")
            scope_geometries.append(scope.exterior)
            scope_geometries.append(origin_geom)
            scope_origin_ids.append(origin_id)
            scope_origin_ids.append(origin_id)

        scope_gdf = gpd.GeoDataFrame(
            {
                "name": scope_names,
                "origin_id": scope_origin_ids,
                "geometry": scope_geometries
            },
            crs=self.default_projected_crs
        )
        destinations = self.layers[source_layer]["gdf"].loc[destination_ids]
        network_edges = gpd.GeoDataFrame({"geometry": network_edges}, crs=self.default_projected_crs)
        return destinations, network_edges, scope_gdf

    # Network/UNA Function
    def una_closest_facility(self, weight=None, reach=False, gravity=False, search_radius=None, beta=None, alpha=1):
        if gravity and (beta is None):
            raise ValueError("Please specify parameter 'beta' when 'gravity' is True")
        self.una_accessibility(
            reach=reach,
            gravity=gravity,
            closest_facility=True,
            weight=weight,
            search_radius=search_radius,
            alpha=alpha,
            beta=beta)
        return

    # Network/UNA Function
    def closest_destination(self, beta=0.003, light_graph=True):
        node_gdf = self.layers['network_nodes']['gdf']
        origins = node_gdf[node_gdf["type"] == "origin"]
        destinations = node_gdf[node_gdf["type"] == "destination"]
        distance, path = nx.multi_source_dijkstra(
            self.G,
            sources=list(destinations.index),
            weight='weight'
        )
        for idx in origins.index:
            node_gdf.at[idx, 'closest_destination'] = path[idx][0]
            node_gdf.at[idx, 'closest_destination_distance'] = distance[idx]
            node_gdf.at[idx, 'closest_destination_gravity'] = 1 / pow(math.e, (beta * distance[idx]))

        self.layers['network_nodes']['gdf'] = node_gdf
        return

    # Network/UNA Function
    # Depreciated, new
    def edge_betweenness(self, search_radius, detour_ratio=1.05, beta=0.003):
        self.closest_destination(beta=beta)

        node_gdf = self.layers['network_nodes']['gdf']
        edge_gdf = self.layers['network_edges']['gdf']

        origins = node_gdf[
            (node_gdf["type"] == "origin") &
            (node_gdf["closest_destination_distance"] <= search_radius)
            ]
        edge_gdf['betweenness'] = 0.0
        for idx, origin in origins.iterrows():
            edge_counter = {edge_id: 0 for edge_id in list(edge_gdf["id"])}
            paths = nx.shortest_simple_paths(
                self.G,
                source=origin["id"],
                target=origin["closest_destination"],
                weight="weight",
            )
            path_count = 0
            for path in paths:
                if path_count == 0:  # first path is the shortest path
                    shortest_path_length = path_weight(self.G, path, weight="weight")

                # terminate loop if current path is longer than  shortest_path_length * detour_ratio
                if path_weight(self.G, path, weight="weight") > shortest_path_length * detour_ratio:
                    # if path_weight(self.G, path, weight="weight") > search_radius * detour_ratio:
                    break
                else:
                    path_count = path_count + 1
                    edges = list(nx.utils.pairwise(path))
                    for edge in edges:
                        edge_counter[self.G.edges[edge]["id"]] = edge_counter[self.G.edges[edge]["id"]] + 1
            for edge_id, edge_count in edge_counter.items():
                edge_gdf.at[
                    edge_id, "betweenness"] += edge_count / path_count * 1  # origin[weight] * gravity(origin, destination, beta)
        # assign betweenness to origin nodes..
        for idx, origin in origins.iterrows():
            origin_betweennes = edge_gdf[
                (edge_gdf["start"] == origin['id']) |
                (edge_gdf["end"] == origin['id'])
                ]["betweenness"].max()
            node_gdf.at[idx, "betweenness"] = origin_betweennes

        self.layers['network_nodes']['gdf'] = node_gdf
        self.layers['network_edges']['gdf'] = edge_gdf
        return edge_gdf

    # Gerenative Geometry
    def create_building_envelop(self, source_layer="parcels", attribute="zone"):
        self.layers["buildings"] = {'gdf': self.layers[source_layer]["gdf"].copy(),
                                    'show': True,
                                    "extrude": True,
                                    "extrude_attribute": attribute
                                    }
        self.color_layer("buildings")
        return

    # cORE, BUT NEEDS RE-WRITING...
    def save_results_from_network_to_layer(self, result_name):
        # network_nodes | network edges
        # betweenness | closest_destination_distance | closest_destination_gravity
        # origin | destination \ observer | street | projection_line |

        node_gdf = self.layers["network_nodes"]["gdf"]
        edge_gdf = self.layers["network_edges"]["gdf"]

        for result in result_name:
            parameter_name = result_name[result]
            if result in node_gdf.columns.values.tolist():
                # Iterate over nodes and save each result to its original layer
                for node_id, row in node_gdf[node_gdf[result].notnull()].iterrows():  # iterate over all non-NaN nodes
                    # print (row)
                    self.layers[row["source_layer"]]["gdf"].at[int(row["source_id"]), parameter_name] = row[result]

            if result in edge_gdf.columns.values.tolist():
                # TODO : make edges remember their original layer id, then handle them the same way
                pass
        return

    # dEPRECIATED..
    def node_degree(self):
        ## calculate degrees
        node_gdf = self.layers["network_nodes"]['gdf']
        degrees = dict(self.G.degree)

        for i in node_gdf.index:
            if i in degrees:
                node_gdf.loc[i, "degree"] = degrees[i]
            else:
                node_gdf.loc[i, "degree"] = 0

    # dEPRECIATED
    def redundant_paths(self):
        pass
        return

    # dEPRECIATED
    def time_function(self):
        start = time.time()
        ## call input function
        print("street node edge creation: " + str(time.time() - start))

    # dEPRECIATED
    def generate_snowflake(self):
        centerX = 46.6725  # long
        centerY = 24.7425  # lat
        radius = 0.0025
        street_width = radius / 50

        max_parcel_frontage = radius / 10
        nrays = 12
        nrows = 8
        ncolums = 16
        center = geo.Point(centerX, centerY)
        scope = center.buffer(radius)
        square = geo.Polygon([(centerX - radius, centerY - radius), (centerX + radius, centerY - radius),
                              (centerX + radius, centerY + radius), (centerX - radius, centerY + radius)])
        rotated_square = affinity.rotate(square, 45)

        min_street_length = radius / 3
        number_of_branches = 6
        branching_points = [center]
        intersection_points = []
        connections = []

        '''
        end_point = affinity.translate(center, yoff=radius*1.8)
        main_ray = geo.LineString([center, end_point])
        for i in range(number_of_branches):
            current_line = affinity.rotate(main_ray, i / number_of_branches * 360, origin=center)
            new_branch_point = geo.Point(current_line.coords[1])
            branching_points.append(new_branch_point)

        blocked_area =  geo.MultiPoint(branching_points).buffer(min_street_length)
        scope = geo.MultiPoint(branching_points).buffer(radius)
        '''
        blocked_area = geo.MultiPoint(branching_points).buffer(min_street_length)

        start = time.time()
        import random

        idx = 0

        while len(branching_points) > 0:
            current_intersection = branching_points.pop(0)  # take first intersection in queue
            if current_intersection.within(scope):
                intersection_points.append(current_intersection)
                # end_point = affinity.translate(current_intersection, yoff=min_street_length*(0.75+random.random()))
                end_point = affinity.translate(current_intersection, yoff=min_street_length * 1)

                main_ray = geo.LineString([current_intersection, end_point])
                # number_of_branches = round(random.random() * 6) + 1
                for i in range(number_of_branches):
                    i = (i * 2 + round(i / (number_of_branches - 1))) % number_of_branches
                    current_line = affinity.rotate(main_ray, i / number_of_branches * 360, origin=current_intersection)
                    new_branch_point = geo.Point(current_line.coords[1])
                    # if not geo.MultiLineString(connections).crosses(current_line) and
                    # if not new_branch_point.within(blocked_area):
                    # if not current_line.crosses(blocked_area):
                    if not new_branch_point.within(blocked_area):  # and not scope.disjoint(new_branch_point):
                        # if  not current_line.within(blocked_area):

                        #                if not  new_branch_point.within(blocked_area):
                        #    trimmed_lines = split(current_line, geo.MultiLineString(connections))
                        #    if len(trimmed_lines)>1:
                        #        trimmed_line = trimmed_lines[0]
                        #        if trimmed_line.length >  min_street_length*0.5:
                        #            connections.append(trimmed_line)
                        # else:
                        # if blocked_area.disjoint(current_line):
                        # print(new_branch_point)
                        # print("is out of blocked buffer")
                        branching_points.append(new_branch_point)
                        connections.append(current_line)

                # blocked_area = geo.MultiLineString(connections).buffer(min_street_length*(random.random()+0.0))
            blocked_area = geo.MultiLineString(connections).buffer(min_street_length * 0.25)
            # blocked_area = geo.MultiPoint(intersection_points).buffer(min_street_length * 0.25)

            idx = idx + 1
            if (idx == 6000000):
                break

        print("generating snowflake: " + str(time.time() - start))
        street_grid = geo.MultiLineString(connections)
        hex_grid = street_grid
        new_streets = affinity.translate(street_grid, yoff=radius * 2)
        for i in range(number_of_branches):
            rotated_streets = affinity.rotate(new_streets, i / number_of_branches * 360, origin=center)
            hex_grid = geo.MultiLineString(list(hex_grid.geoms) + list(rotated_streets.geoms))

        d = {'id':
            [
                # "scope",
                # "blocked_area",
                # 'intersection points',
                # "branching points",
                #          "connections",
                "hex grid"
            ],
            'geometry':
                [
                    # scope,
                    # blocked_area,
                    # geo.MultiPoint(intersection_points),
                    # geo.MultiPoint(branching_points),
                    #             geo.MultiLineString(connections),
                    hex_grid
                ]
        }
        snowflake_gdf = gpd.GeoDataFrame(d, crs="EPSG:4326")

class Logger:
    pass

class Reporter:
    pass



class Project:
    containers = []
    datagraph = None
    center_lat = 0
    center_long = 0
    name = ""

    def __init__(self, atribute_dict, container_list):
        pass

    def project_page(self):
        return

    def project_map(self):
        return

    def to_json(self):
        return

    def deploy(self):
        return

    def apply_workflow(self, workflow):
        return

    @staticmethod
    def create_new():
        database_folder = "SF_Data//"
        projects = pd.read_csv(database_folder + "projects.csv")


class DataContainer:
    is_spatial = False
    is_timeseries = False
    is_network = False
    is_table = False

    def __init__(self, dict):
        self.id
        self.name = dict["name"]
        self.owner = 'public'  ## associate with user id whemn needed
        self.attribute_names = []
        self.attribute_types = []
        self.attribute_values = []
        self.object_identifier = ""
        self.object_namer = ""
        self.url_tag = ""
        self.source_name = ""
        self.source_url = ""

    pass

    def container_html(self):
        return

    def object_html(self, id):
        return

    def to_json(self):
        return

    def create_Container(self):
        return

    def apply_tool(self, tool):
        # Figure out tool requarements
        # Formal layer to fit requirements
        return


class Workflow:
    reqs = []

    def ececute(self, project):
        return


class TOD_workflow(Workflow):
    pass


class Choice_Set_Workflow(Workflow):
    pass


class Datagraph:
    def datagraph_html(self):
        return

    def perform_join(self):
        return

    def inspect_join(self):
        return


class Filter:
    pass
