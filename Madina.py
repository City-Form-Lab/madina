import shapely.geometry as geo
from shapely.ops import split, substring
from shapely import affinity

import geopandas as gpd
import pandas as pd
import pydeck as pdk
import numpy as np
import networkx as nx
import time
import momepy
from networkx.classes.function import path_weight
import math
import random


'''
    def insert_nodes(self, points, labels):
        node_gdf = self.layers['network_nodes']['gdf'].drop(self.STYLE_COLUMNS, axis=1, errors='ignore')
        edge_gdf = self.layers['network_edges']['gdf'].drop(self.STYLE_COLUMNS, axis=1, errors='ignore')

        node_id = int(node_gdf.iloc[-1]["id"]) + 1
        node_ids = []
        node_types = []
        node_geometries = []
        edge_id = int(edge_gdf.iloc[-1]["id"])

        edge_dict = edge_gdf.to_dict()

        for i, point in enumerate(points):
            nearest_street_distance = 9999999999999999999
            nearest_street_id = None
            point_on_the_street = None
            # TODO: filter street only segmwents
            street_segment_ids = [idx for idx in edge_dict["type"] if edge_dict["type"][idx] == "street"]
            for idx in street_segment_ids:
                street_segment = edge_dict["geometry"][idx]
                distance = street_segment.distance(point)
                if distance < nearest_street_distance:
                    nearest_street_distance = distance
                    nearest_street_id = idx
                    old_street_segment = edge_dict["geometry"][nearest_street_id]
                    point_on_the_street = street_segment.interpolate(street_segment.project(point))

            ## insert project node, and point node
            projected_node_id = node_id
            node_id = node_id + 1

            node_ids.append(projected_node_id)
            node_types.append("project_node")
            node_geometries.append(point_on_the_street)

            point_node_id = node_id
            node_id = node_id + 1
            node_ids.append(point_node_id)
            node_types.append(labels[i])
            node_geometries.append(point)

            ## insert segmented street and projection edge
            start_segment = geo.LineString((old_street_segment.coords[0], point_on_the_street))
            end_segment = geo.LineString((point_on_the_street, old_street_segment.coords[1]))

            edge_id = edge_id + 1
            edge_dict["id"][edge_id] = edge_id
            edge_dict["start"][edge_id] = edge_dict["start"][nearest_street_id]
            edge_dict["end"][edge_id] = projected_node_id
            edge_dict["length"][edge_id] = start_segment.length
            edge_dict["type"][edge_id] = "street"
            edge_dict["geometry"][edge_id] = start_segment

            edge_id = edge_id + 1
            edge_dict["id"][edge_id] = edge_id
            edge_dict["start"][edge_id] = projected_node_id
            edge_dict["end"][edge_id] = edge_dict["end"][nearest_street_id]
            edge_dict["length"][edge_id] = end_segment.length
            edge_dict["type"][edge_id] = "street"
            edge_dict["geometry"][edge_id] = end_segment

            projection_segment = geo.LineString((point, point_on_the_street))

            edge_id = edge_id + 1
            edge_dict["id"][edge_id] = edge_id
            edge_dict["start"][edge_id] = point_node_id
            edge_dict["end"][edge_id] = projected_node_id
            edge_dict["length"][edge_id] = 0  # projection_segment.length
            edge_dict["type"][edge_id] = "project_line"
            edge_dict["geometry"][edge_id] = projection_segment

            ## remove old street
            [edge_dict[k].pop(nearest_street_id) for k in edge_dict.keys()]

        d = {"id": node_ids,
             "type": node_types,
             'geometry': node_geometries
             }
        new_node_gdf = gpd.GeoDataFrame(d, crs="EPSG:4326")
        new_node_gdf.set_index("id")

        node_gdf = pd.concat([node_gdf, new_node_gdf], ignore_index=True)
        node_gdf.set_index("id")
        edge_gdf = gpd.GeoDataFrame(edge_dict, crs="EPSG:4326")
        edge_gdf.set_index("id")

        self.layers['network_nodes']['gdf'] = node_gdf
        self.color_layer('network_nodes')
        self.layers['network_edges']['gdf'] = edge_gdf
        self.color_layer('network_edges')
        return node_gdf, edge_gdf
'''

class Zonal:
    # TODO: Make this a point based system. Points are the base, and each line is a sequence of points. each polygon is a ring. Everything reference the base points. All cad is line-based, lines are disjoint by nature. Here, everything is connected by nature

    # TODO: the Zonal object could have multiple networks. an implimintation could be a dictionsry similar to the layers
    default_map_name = "Zonal_map"
    default_crs = "EPSG:4326" # WGS 84, the geographic coordinate system used in GPS
    # default_crs = "EPSG:3857" # Web Mercator, the geographic coordinate system used Google Maps and most other web-based maps.
    # default_crs = 'EPSG:8836' # Saudi Arabia - onshore and offshore.

    default_geographic_crs = "EPSG:4326"
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

    def __init__(self, centerX, centerY, scope):
        self.centerX = centerX
        self.centerY = centerY
        self.scope = scope
        self.layers = {}
        self.G = None

    def describe(self):
        """
        A function to return a string representation of the object for debugging and/or development
        :return:
        """
        return "CenterX :" + str(self.centerX) + ", centerY: " + str(self.centerY)

    def load_layer(self, layer_name, file_path, do_not_load=False, allow_out_of_scope=False):
        if do_not_load:
            return
        else:
            self.layers[layer_name] = {'gdf': gpd.read_file(file_path).to_crs(self.default_crs),
                                       'show': True,
                                       "file_path": file_path
                                       }
            self.color_layer(layer_name)
            if allow_out_of_scope:
                return
            else:
                self.geoprocess_layer(layer_name, clip_by=self.scope)
                return

    def filter_layer(self, layer_name, attribute, isin=None):
        filtered_gdf = self.layers[layer_name]["gdf"]

        if isin is not None:
            filtered_gdf = filtered_gdf[filtered_gdf[attribute].isin(isin)]

        self.layers[layer_name]["gdf"] = filtered_gdf
        return

    def geoprocess_layer(self, layer_name, xoff=None, yoff=None, clip_by=None):
        scope_gdf = gpd.GeoDataFrame({
            'name': ['scope'],
            'geometry': [self.scope]
        },
            crs=self.default_crs)
        layer_gdf = self.layers[layer_name]['gdf']

        if clip_by is not None:
            self.layers[layer_name]['gdf'] = gpd.clip(layer_gdf, scope_gdf)

        if xoff is not None:
            for idx, row in layer_gdf.iterrows():
                layer_gdf.at[idx, "geometry"] = affinity.translate(row["geometry"], xoff=xoff, yoff=yoff)

        return

    def create_deck_map(self, save_to_desk=False, show_in_notebook=True):
        pdk_layers = []
        for layer_name in self.layers:
            if self.layers[layer_name]["show"]:
                pdk_layer = pdk.Layer(
                    'GeoJsonLayer',
                    self.layers[layer_name]["gdf"],
                    opacity=1,
                    stroked=True,
                    filled=True,
                    extruded=False if "extrude" not in self.layers[layer_name] else self.layers[layer_name]["extrude"],
                    get_elevation=0 if "extrude_attribute" not in self.layers[layer_name] else self.layers[layer_name][
                        "extrude_attribute"],
                    wireframe=True,
                    get_width=0.1,
                    # get_radius= 5,
                    # get_line_color='[bs, 255, bs]',
                    get_line_color="color",
                    pickable=True
                )
                pdk_layers.append(pdk_layer)

        initial_view_state = pdk.ViewState(
            latitude=self.centerY,
            longitude=self.centerX,
            zoom=16,
            max_zoom=20,
            pitch=45,
            bearing=0
        )

        r = pdk.Deck(
            layers=pdk_layers,
            initial_view_state=initial_view_state,
            # map_provider=None
        )

        if save_to_desk:
            r.to_html(
                self.default_map_name + ".html",
                css_background_color="cornflowerblue"
            )
        if show_in_notebook:
            return r
        return

    def color_layer(self, layer_name, color_by_attribute=None, color_method="single_color", color=None):
        """
        A method to set geometry color

        :param layer_name: string, layer name
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
        if layer_name in self.default_colors.keys() and color_by_attribute is None and color is None:
            # set default colors first. all default layers call without specifying "color_by_attribute"
            # default layer creation always calls self.color_layer(layer_name) without any other parameters
            color = self.default_colors[layer_name]
            color_method = "single_color"
            if type(color) is dict:
                # the default color is categorical..
                color_by_attribute = color["__attribute_name__"]
                color_method = "categorical"
        elif color_by_attribute is None and color is None:
            # if "color_by_attribute" is not given, and its not a default layer, assuming color_method == "single_color"
            #if no color is given, assign random color, else, color=color
            color = [random.random() * 255, random.random() * 255, random.random() * 255]
            color_method = "single_color"
        elif color is None:
            # color by attribute ia given, but no color is given..
            if color_method == "single_color":
                # if color by attribute is given, and color method is single color, this is redundant but just in case:
                color = [random.random() * 255, random.random() * 255, random.random() * 255]
            if color_method == "categorical" :
                color = {}
                for distinct_value in  self.layers[layer_name]["gdf"][color_by_attribute].unique():
                    color[distinct_value] = [random.random() * 255, random.random() * 255, random.random() * 255]

        #create color column
        if color_method == "single_color":
            color_column = [color] * len(self.layers[layer_name]["gdf"])
        elif color_method == "categorical":
            color_column = []
            for value in self.layers[layer_name]["gdf"][color_by_attribute]:
                if value in color.keys():
                    color_column.append(color[value])
                else:
                    color_column.append(color["__other__"])
        elif color_method == "gradient":
            cbc = self.layers[layer_name]["gdf"][color_by_attribute]  # color by column
            nc = 255 * (cbc - cbc.min()) / (cbc.max() - cbc.min())  # normalized column
            color_column = [[255 - v, 0 + v, 0] for v in nc]            # convert normalized values to color spectrom.
            # TODO: insert color map options here..
        elif color_method == 'quantile':
            scaled_percentile_rank = 255 * self.layers[layer_name]["gdf"][color_by_attribute].rank(pct=True)
            color_column = [[255 - v, 0 + v, 0] for v in scaled_percentile_rank]  # convert normalized values to color spectrom.

        self.layers[layer_name]["gdf"]["color"] = color_column
        return

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

        gdf = gpd.GeoDataFrame(d, crs=self.default_crs)
        self.layers[layer_name] = {'gdf': gdf, 'show': visibility}
        self.color_layer(layer_name)
        return

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

    def generate_radial_streets(self, radius, nrays, nrows):
        rays = []

        main_ray = geo.LineString([
            (self.centerX, self.centerY),
            (self.centerX, self.centerY + radius)
        ])
        rays.append(main_ray)
        for i in range(nrays - 1):
            rays.append(
                affinity.rotate(main_ray,
                                (i + 1) / nrays * 360,
                                origin=(self.centerX, self.centerY)
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

    def create_street_nodes_edges(self, temporary_copy=False):
        ## The goal of this function is to ensure topological integrity of the network, provide lists of nodes and edges needed to produce a NetworkX Graph
        ## TODO: Thi function could be faster when avoiding the use of node_gdf = node_gdf.append(). construct a dictionary first, then when finished, make the node_gdf.

        #street_geometry = geo.MultiLineString(
#            list(self.layers['streets']['gdf']['geometry'])
#        )
        street_gdf = self.layers["streets"]['gdf']


        node_dict = {
            "id": [],
            "geometry": [],
            "source_layer": [],
            "source_id": [],
            "type": [],
            "weight": [],
            "nearest_street_id": [],
            "nearest_street_distance": [],
        }

        node_gdf = gpd.GeoDataFrame(node_dict, crs="EPSG:4326").astype({'id': 'int32'})
        node_gdf.set_index("id")
        edge_dict = {
            "id": [],
            "start": [],
            "end": [],
            "length": [],
            "type": [],
            "geometry": [],
            "parent_street_id": [],
        }
        edge_gdf = gpd.GeoDataFrame(edge_dict, crs="EPSG:4326").astype({'id': 'int32'})
        edge_gdf.set_index("id")

        node_id = 0
        edge_id = 0
        for idx, street_row in street_gdf.iterrows():
            street = street_row["geometry"]
            node_snapping_tolerance = 0.0005
            ## Add ends to node list, get IDs, make sure to avoid duplicates.
            start_point_index = None
            end_point_index = None
            if len(street.coords) != 2:  ## bad segment, consider segmenting to individual segments for accuracy..
                start_point = geo.Point(street.coords[0])
                end_point = geo.Point(street.coords[-1])
                street = geo.LineString([start_point, end_point])
            else:  ## Good segment with two ends
                start_point = geo.Point(street.coords[0])
                end_point = geo.Point(street.coords[1])

            for idx, row in node_gdf.iterrows():
                if row["geometry"].almost_equals(start_point, decimal=6):
                    start_point_index = idx
                if row["geometry"].almost_equals(end_point, decimal=6):
                    end_point_index = idx
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
                     "nearest_street_distance": 0,
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
                     "nearest_street_distance": 0,
                     }
                    , ignore_index=True)

            ## Need to add a segment
            edge_gdf = edge_gdf.append(
                {
                    "id": edge_id,
                    "start": start_point_index,
                    "end": end_point_index,
                    "length": street.length,
                    "type": "street",
                    "geometry": street,
                    "parent_street_id": street_row["id"],
                }
                , ignore_index=True)
            edge_id += 1

        if temporary_copy:
            return node_gdf, edge_gdf
        else:
            self.layers['network_nodes'] = {'gdf': node_gdf, 'show': True}
            self.color_layer('network_nodes')
            self.layers['network_edges'] = {'gdf': edge_gdf, 'show': True}
            self.color_layer('network_edges')
            return

    def insert_nodes(self, label, layer_name=None, filter=None, attachment_method="project",
                     representative_point="nearest_point", node_weight_attribute=None,
                     projection_edge_cost_attribute=None, source_identifier="id", temporary_copy=False,
                     node_gdf=None, edge_gdf=None, source_gdf=None):
        # attachment_method= "project" | "snap"
        # representative_point="nearest_point" | "centroid"
        # labels = "origin" | "destination" | "observer"

        # TODO: This function should preserve style. style settings should be saved intl the layer, and reapplied after any
        #  layer insert/delete/update operation.
        # TODO : for now, focus on a given layer as input, deal with a filter later..

        if not temporary_copy:
            print("permanent copy")
            node_gdf = self.layers['network_nodes']['gdf'].drop(self.STYLE_COLUMNS, axis=1, errors='ignore')
            edge_gdf = self.layers['network_edges']['gdf'].drop(self.STYLE_COLUMNS, axis=1, errors='ignore')
            source_gdf = self.layers[layer_name]["gdf"]
        # TODO: keep track of a layer's identifier in the Zonal object layers...
          # self.layers[layer_name]["identifier"]

        # counters to keep track of what a new node should have as an id. Increment prior to each use.
        new_node_id = int(node_gdf.iloc[-1]["id"])
        new_edge_id = int(edge_gdf.iloc[-1]["id"])

        # dictionaries are much more effecient for insertion than pandas df/gdfs.
        edge_dict = edge_gdf.to_dict()
        node_dict = node_gdf.to_dict()

        for source_id, row in source_gdf.iterrows():
            print ("inserting node " + str(source_id))
            # TODO: handle variations of geometries and representative point options here.
            source_representative_point = row["geometry"].centroid

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

            if attachment_method == "project":
                # need to store representative point (labeled as origin)
                # + a projection line (labeled as "projection_line"
                # + a projection point (point_on_the_street) labeled as "projection_point

                new_node_id += 1
                representative_point_id = new_node_id
                node_dict["id"][representative_point_id] = representative_point_id
                node_dict["geometry"][representative_point_id] = source_representative_point
                node_dict["source_layer"][representative_point_id] = layer_name
                node_dict["source_id"][representative_point_id] = row[source_identifier]
                node_dict["type"][representative_point_id] = label
                node_dict["weight"][representative_point_id] = 0 if node_weight_attribute is None else row[
                    node_weight_attribute]

                new_node_id += 1
                project_point_id = new_node_id
                node_dict["id"][project_point_id] = project_point_id
                node_dict["geometry"][project_point_id] = point_on_the_street
                node_dict["source_layer"][project_point_id] = layer_name
                node_dict["source_id"][project_point_id] = row[source_identifier]
                node_dict["type"][project_point_id] = "project_point"
                node_dict["weight"][project_point_id] = 0

                new_edge_id += 1
                projection_segment_id = new_edge_id
                projection_segment = geo.LineString((source_representative_point, point_on_the_street))

                edge_dict["id"][projection_segment_id] = projection_segment_id
                edge_dict["start"][projection_segment_id] = representative_point_id
                edge_dict["end"][projection_segment_id] = project_point_id
                edge_dict["length"][projection_segment_id] = 0 if projection_edge_cost_attribute is None else row[
                    projection_edge_cost_attribute]
                # projection_segment.length
                edge_dict["type"][projection_segment_id] = "project_line"
                edge_dict["geometry"][projection_segment_id] = projection_segment


            elif attachment_method == "snap":
                # need to only store point_on_the_street (labeled as origin)
                new_node_id += 1
                project_point_id = new_node_id
                node_dict["id"][project_point_id] = project_point_id
                node_dict["geometry"][project_point_id] = point_on_the_street
                node_dict["source_layer"][project_point_id] = layer_name
                node_dict["source_id"][project_point_id] = row[source_identifier]
                node_dict["type"][project_point_id] = label
                node_dict["weight"][project_point_id] = 0 if node_weight_attribute is None else row[
                    node_weight_attribute]
            else:
                print("method is incorrect or not implemented.")

            ## insert segmented street
            start_segment = geo.LineString((old_street_segment.coords[0], point_on_the_street))
            end_segment = geo.LineString((point_on_the_street, old_street_segment.coords[1]))

            new_edge_id += 1
            edge_dict["id"][new_edge_id] = new_edge_id
            edge_dict["start"][new_edge_id] = edge_dict["start"][nearest_street_id]
            edge_dict["end"][new_edge_id] = project_point_id
            edge_dict["length"][new_edge_id] = start_segment.length
            edge_dict["type"][new_edge_id] = "street"
            edge_dict["geometry"][new_edge_id] = start_segment
            edge_dict["parent_street_id"][new_edge_id] = edge_dict["parent_street_id"][nearest_street_id]

            new_edge_id += 1
            edge_dict["id"][new_edge_id] = new_edge_id
            edge_dict["start"][new_edge_id] = project_point_id
            edge_dict["end"][new_edge_id] = edge_dict["end"][nearest_street_id]
            edge_dict["length"][new_edge_id] = end_segment.length
            edge_dict["type"][new_edge_id] = "street"
            edge_dict["geometry"][new_edge_id] = end_segment
            edge_dict["parent_street_id"][new_edge_id] = edge_dict["parent_street_id"][nearest_street_id]

            ## remove old street
            [edge_dict[k].pop(nearest_street_id) for k in edge_dict.keys()]

        node_gdf = gpd.GeoDataFrame(node_dict, crs="EPSG:4326")
        node_gdf.set_index("id")

        edge_gdf = gpd.GeoDataFrame(edge_dict, crs="EPSG:4326")
        edge_gdf.set_index("id")

        if temporary_copy:
            return node_gdf, edge_gdf
        else:
            self.layers['network_nodes']['gdf'] = node_gdf
            self.color_layer('network_nodes')
            self.layers['network_edges']['gdf'] = edge_gdf
            self.color_layer('network_edges')

            # TODO: re-apply style with new points.
            return node_gdf, edge_gdf

    def create_graph(self, temporary_copy=False, node_gdf=None, edge_gdf=None):
        if not temporary_copy:
            node_gdf = self.layers['network_nodes']['gdf']
            edge_gdf = self.layers['network_edges']['gdf']
        G = nx.Graph()
        for idx, edge in edge_gdf.iterrows():
            G.add_edge(int(edge["start"]), int(edge["end"]), weight=edge["length"], type=edge["type"], id=edge["id"], )

        for idx, node in node_gdf.iterrows():
            G.nodes[int(node["id"])]['type'] = node["type"]

        if temporary_copy:
            return G
        else:
            self.G = G
            return

    def closest_destination(self, beta=0.003):
        node_gdf = self.layers['network_nodes']['gdf']
        origins = node_gdf[node_gdf["type"] == "origin"]
        destinations = node_gdf[node_gdf["type"] == "destination"]
        distance, path = nx.multi_source_dijkstra(
            self.G,
            sources=list(destinations["id"]),
            weight='weight'
        )
        for idx, origin in origins.iterrows():
            node_gdf.at[origin['id'], 'closest_destination'] = path[origin['id']][0]
            node_gdf.at[origin['id'], 'closest_destination_distance'] = distance[origin['id']]
            node_gdf.at[origin['id'], 'closest_destination_gravity'] = 1 / pow(math.e, (beta * distance[origin['id']]))

        self.layers['network_nodes']['gdf'] = node_gdf
        return

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
                    #if path_weight(self.G, path, weight="weight") > search_radius * detour_ratio:
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

    def create_building_envelop(self, source_layer="parcels", attribute="zone"):
        self.layers["buildings"] = {'gdf': self.layers[source_layer]["gdf"].copy(),
                                    'show': True,
                                    "extrude": True,
                                    "extrude_attribute": attribute
                                    }
        self.color_layer("buildings")
        return

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

    # TODO: finish and test THESE FUNCTIONS
    def node_degree(self):
        ## calculate degrees
        degrees = dict(self.G.degree)
        for i in degrees:
            self.node_gdf.loc[i, "degree"] = degrees[i]

    def redundant_paths():
        pass
        return

    # Static function ??
    def time_function(self):
        start = time.time()
        ## call input function
        print("street node edge creation: " + str(time.time() - start))

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


class Project:
    containers = []
    datagraph = None
    center_lat = 0
    center_long = 0
    name = ""

    def __init__(self, atribute_dict, container_list):
        pass

    def project_page(self):
        return html

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
