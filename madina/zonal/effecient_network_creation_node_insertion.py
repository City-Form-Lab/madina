import geopandas as gpd
import madina as md
import shapely.geometry as geo
import pandas as pd
import time
import numpy as np
import plotly.express as px
from pygeos.lib import get_x, get_y, get_point

import numba as nb


# Class Function
def create_nodes_edges(self, source_layer="streets", prepare_geometry=False, weight_attribute=None, tolerance=0.0, tag_edges=False, discard_redundant_edges=True):
    start = time.time()

    geometry_gdf = self.layers[source_layer]["gdf"]
    if prepare_geometry:
        geometry_gdf = _prepare_geometry(geometry_gdf)

    print(f"\t{(time.time()-start)*1000:6.2f}ms\t geometry prepared")
    start = time.time()

    # node_gdf, edge_gdf = _effecient_network_nodes_edges(
    node_gdf, edge_gdf = _node_edge_builder(
        geometry_gdf,
        weight_attribute=weight_attribute,
        tolerance=tolerance
    )

    print(f"\t{(time.time()-start)*1000:6.2f}ms\t nodes and edges constructed")
    start = time.time()

    if discard_redundant_edges:
        edge_gdf = _discard_redundant_edges(edge_gdf)

    print(f"\t{(time.time()-start)*1000:6.2f}ms\t redundant edges discaded")
    start = time.time()

    if tag_edges:
        edge_gdf = _tag_edges(edge_gdf, tolerance=tolerance)

    print(f"\t{(time.time()-start)*1000:6.2f}ms\t edges tagged")
    start = time.time()

    self.layers['network_nodes'] = {'gdf': node_gdf, 'show': True}
    self.layers['network_edges'] = {'gdf': edge_gdf, 'show': True}
    self.color_layer('network_nodes')
    self.color_layer('network_edges')

    print(f"\t{(time.time()-start)*1000:6.2f}ms\t object updated and colors applied")
    return node_gdf, edge_gdf

# Static Function
def _prepare_geometry(geometry_gdf):
    geometry_gdf = geometry_gdf.copy(deep=True)

    # making sure if geometry contains polygons, they are corrected by using the polygon exterior as a line.
    polygon_idxs = geometry_gdf[geometry_gdf["geometry"].geom_type ==
                                "Polygon"].index
    geometry_gdf.loc[polygon_idxs,
                     "geometry"] = geometry_gdf.loc[polygon_idxs, "geometry"].exterior

    # if geometry is multilineString, convert to lineString
    if (geometry_gdf["geometry"].geom_type ==  'MultiLineString').all()\
            and (np.array(list(map(len, geometry_gdf["geometry"].values))) == 1).all():
        geometry_gdf["geometry"] = geometry_gdf["geometry"].apply(lambda x: list(x)[0])

    # deleting any Z coordinate if they exist
    from shapely.ops import transform

    def _to_2d(x, y, z):
        return tuple(filter(None, [x, y]))
    if geometry_gdf.has_z.any():
        geometry_gdf["geometry"] = geometry_gdf["geometry"].apply(
            lambda s: transform(_to_2d, s))
    return geometry_gdf

# handles creating a network node gdf and edge gdf given geometry and a disconnection tolerance.
def _effecient_tolerance_network_nodes_edges(geometry_gdf, weight_attribute=None, tolerance=1.0):
    geometry_gdf["weight"] = geometry_gdf["geometry"].length if weight_attribute is None else geometry_gdf[weight_attribute].apply(
        lambda x: max(1, x))
    geometry_gdf = geometry_gdf.sort_values("weight", ascending=False)
    geometry_gdf.index

    point_geometries = geometry_gdf["geometry"].boundary.explode(
        index_parts=False).reset_index(drop=True)
    point_types = ["start", "end"] * geometry_gdf.shape[0]
    point_segment_ids = np.repeat(geometry_gdf.index, 2)

    point_count = geometry_gdf.shape[0] * 2
    geometry_collector = np.empty(point_count, dtype=object)
    degree_collector = np.empty(point_count, dtype="int64")

    edge_count = geometry_gdf.shape[0]

    edge_collector = {
        "start": np.zeros(edge_count, dtype="int64") - 1,
        "end": np.zeros(edge_count, dtype="int64") - 1
    }

    matching = point_geometries.sindex.query_bulk(
        point_geometries.buffer(tolerance))

    # grouping points..
    unique_points, intersection_count = np.unique(
        matching[0], return_counts=True)
    point_intersection = {}
    current_index = 0
    for point, count in zip(unique_points, intersection_count):
        point_intersection[point] = set(
            matching[1][current_index:current_index+count]) - {point}
        current_index = current_index+count

    new_node_id = -1
    current_point_idx = -1
    for edge_idx in geometry_gdf.index:
        for position in ["start", "end"]:
            current_point_idx += 1
            if edge_collector[position][edge_idx] != -1:
                continue
            current_intersecting_points = point_intersection[current_point_idx]

            degree = 1
            for intersecting_point in current_intersecting_points:
                intersecting_point_segment_idx = point_segment_ids[intersecting_point]
                intersecting_point_type = point_types[intersecting_point]

                if edge_collector[intersecting_point_type][intersecting_point_segment_idx] != -1:
                    continue
                degree += 1
                edge_collector[intersecting_point_type][intersecting_point_segment_idx] = new_node_id
            new_node_id += 1
            edge_collector[position][edge_idx] = new_node_id
            degree_collector[new_node_id] = degree
            geometry_collector[new_node_id] = point_geometries[current_point_idx]

    node_gdf = gpd.GeoDataFrame(
        {
            "id":           pd.Series(range(new_node_id), dtype="int64"),
            "source_layer": pd.Series(["streets"] * new_node_id, dtype="category"),
            "source_id":    pd.Series([0] * new_node_id, dtype="int64"),
            "type":         pd.Series(["street_node"] * new_node_id, dtype="category"),
            "weight":       pd.Series([0.0] * new_node_id, dtype="float64"),
            "nearest_street_id": pd.Series([0] * new_node_id, dtype="int64"),
            "nearest_street_node_distance": pd.Series([None] * new_node_id, dtype="object"),
            "degree":       pd.Series(degree_collector[:new_node_id], dtype="int64"),
            "geometry":     gpd.GeoSeries(geometry_collector[:new_node_id], crs=geometry_gdf.crs)

        }
    ).set_index("id")

    edge_gdf = gpd.GeoDataFrame(
        {
            "id":       pd.Series(range(edge_count), dtype="int64"),
            "geometry": geometry_gdf["geometry"],
            "length":   geometry_gdf["geometry"].length,
            "weight":   geometry_gdf["weight"],
            "type":     pd.Series(["street"] * edge_count, dtype="category"),
            "parent_street_id": pd.Series(list(geometry_gdf.index), dtype="int64"),
            "start":    pd.Series(edge_collector["start"], dtype="int64"),
            "end":      pd.Series(edge_collector["end"], dtype="int64"),
        }, crs=geometry_gdf.crs
    ).set_index("id")

    return node_gdf, edge_gdf

# Static Function
def _discard_redundant_edges(edge_gdf):
    se = np.vstack([
        edge_gdf["start"].values,
        edge_gdf["end"].values
    ]).transpose()

    uniques = np.unique(se, return_counts=True,
                        return_index=True, return_inverse=True, axis=0)
    sorted_unique = uniques[0]
    unique_indices = uniques[1]
    inverser = uniques[2]
    unique_frequency = uniques[3]
    return edge_gdf.loc[unique_indices]

# Static FUnction, needs improving?
# Strip down needed values and vectorize accordingly, Never use the .apply() function on a dataframe
# inserting data back into a dataframe should use fastpath and should be given a correct index
def _tag_edges(edge_gdf, tolerance=1.0):
    edge_gdf["tag"] = ""
    edge_gdf['tag'] = np.select(
        condlist=[
            edge_gdf["weight"] < tolerance,
            (edge_gdf["weight"] >= tolerance) & (
                edge_gdf["weight"] < 3.0 * tolerance),
            edge_gdf["start"] == edge_gdf["end"],
            edge_gdf["geometry"].geom_type != "LineString"
        ],
        choicelist=[
            "shorter than tolerance",
            "within 3x tolerance",
            "cycle_edge",
            "bad_geometry_type"
        ],
        default=edge_gdf['tag']
    )
    duplicates = edge_gdf.apply(
        lambda x: {x["start"], x["end"]}, axis=1).duplicated(keep=False)
    edge_gdf.loc[duplicates, "tag"] = 'redundant_edge'
    return edge_gdf

# Static function, wrapper to the numerical and spatial static network node-edge constructors
def _node_edge_builder(geometry_gdf, weight_attribute=None, tolerance=0.0):
    if tolerance == 0:
        # use vectorized implementaytion
        point_xy = GeoPandaExtractor(geometry_gdf.geometry.values.data)
        node_indexer, node_points, node_dgree, edge_start_node, edge_end_node = _vectorized_node_edge_builder(
            point_xy)
        # use spatial index implementation
        # could this gain effeciency by being sorted?
        # TODO: complete the implementation of this case by copyinfg the for-loop thT UTILIz MATCHING RESULTS..
        point_geometries = geometry_gdf["geometry"].boundary.explode(
            index_parts=False).reset_index(drop=True)
        matching = point_geometries.sindex.query_bulk(
            point_geometries.buffer(tolerance))

        # construct node and edge gdfs
        edge_count = geometry_gdf.shape[0]
        index = pd.Index(np.arange(edge_count), name="id")
        length = pd.Series(
            geometry_gdf["geometry"].length.values, fastpath=True, index=index)
        edge_gdf = gpd.GeoDataFrame(
            {
                "length":   length,
                "weight":   length if weight_attribute is None else pd.Series(geometry_gdf[weight_attribute].apply(lambda x: max(1, x)), fastpath=True, index=index),
                "type":     pd.Series(np.repeat(np.array(["street"], dtype=object), repeats=edge_count), fastpath=True, index=index, dtype="category"),
                "parent_street_id": pd.Series(geometry_gdf.index.values, fastpath=True, index=index),
                "start":    pd.Series(edge_start_node, fastpath=True, index=index),
                "end":      pd.Series(edge_end_node, fastpath=True, index=index),
            },
            index=index,
            geometry=geometry_gdf["geometry"]
        )

        node_count = node_points.shape[0]
        index = pd.Index(np.arange(node_count), name="id")
        node_gdf = gpd.GeoDataFrame(
            {
                "source_layer": pd.Series(np.repeat(np.array(["streets"], dtype=object), repeats=node_count), fastpath=True, index=index, dtype="category"),
                "source_id":    pd.Series(np.repeat(np.array([0], dtype=np.int32), repeats=node_count), fastpath=True, index=index, dtype=np.int32),
                "type":         pd.Series(np.repeat(np.array(["street_node"], dtype=object), repeats=node_count), fastpath=True, index=index, dtype="category"),
                "weight":       pd.Series(np.repeat(np.array([0.0], dtype=np.float32), repeats=node_count), fastpath=True, index=index, dtype=np.float32),
                "nearest_street_id": pd.Series(np.repeat(np.array([0], dtype=np.int32), repeats=node_count), fastpath=True, index=index, dtype=np.int32),
                "nearest_street_node_distance": pd.Series(np.repeat(np.array([{}], dtype=object), repeats=node_count), fastpath=True, index=index),
                "degree":       pd.Series(node_dgree, fastpath=True, index=index),
                # "connected_edges": connected_edges
            },
            index=index,
            # crs=geometry_gdf.crs,
            geometry=gpd.points_from_xy(
                x=point_xy[0][node_indexer], y=point_xy[1][node_indexer], crs=geometry_gdf.crs)
        )

    elif tolerance > 0:
        node_gdf, edge_gdf = _effecient_tolerance_network_nodes_edges(
            geometry_gdf=geometry_gdf,
            weight_attribute=weight_attribute,
            tolerance=tolerance
        )
    else:
        raise ValueError(f"tolerance must either be zero or a positive number, {tolerance} was given.")

    return node_gdf, edge_gdf # Return Adjacency List here...


# Static function for numeric network node-edge construction
# TODO: eliminate functions not compatible with nb.jit
def _vectorized_node_edge_builder(point_xy):
    # the point_xy input is a 2 by n matrix. where the start point and end point of an edge are in alternating order..
    # unique parameter is expected to come from a spatial index, on points that are sorted.

    # input setup
    point_count = point_xy.shape[1]
    edge_count = int(point_count/2)

    point_types = np.repeat(
        np.array([0.0, 1.0], dtype=np.int32), repeats=edge_count)
    edge_ids = np.arange(edge_count, dtype=np.int32)
    point_segment_ids = np.array(
        [edge_ids, edge_ids]).reshape([point_count, 1])

    # sorting points by x then by y. This is to make it effecient to link the similarly sorted unique nodes with each point occurance
    # sort by x then by y ## np.lexsort is not supported with njit
    sorter = np.lexsort((point_xy[1, :], point_xy[0, :]))
    sorted_point_xy = point_xy[:, sorter]
    point_xy_T = np.transpose(sorted_point_xy)

    # Resorting inputs:
    sorted_point_types = point_types[sorter]
    sorted_point_segment_ids = point_segment_ids[sorter]

    # finding unique points (which are from now on, refereed to as nodes.)
    # indices # np.unique is only supported in njit if no arguments are given.
    unique_nodes = np.unique(
        point_xy_T, axis=0, return_index=True, return_counts=True)

    # node data
    node_points = unique_nodes[0]
    node_first_occurance = unique_nodes[1]
    node_indexer = sorter[node_first_occurance]

    node_dgree = np.array(unique_nodes[2], dtype=np.int32)
    # connected_edges = np.empty(node_count, dtype=np.array)
    # edge data

    edge_start_node = np.zeros(edge_count, dtype=np.int32)
    edge_end_node = np.zeros(edge_count, dtype=np.int32)

    # this is to make sure if the last point occurs for the first time, we'll have a proper
    if node_first_occurance[-1] != (point_count-1):
        np.append(node_first_occurance, point_count-1)

    range_start = 0
    i = 0
    node_index = 0
    for range_end in node_first_occurance[1:]:
        # connected_edges[node_index] = sorted_point_segment_ids[range_start:range_end]
        for point in point_xy_T[range_start:range_end, :]:
            edge_index = sorted_point_segment_ids[i]
            if sorted_point_types[i] == 0:
                edge_start_node[edge_index] = node_index
            else:
                edge_end_node[edge_index] = node_index
            i += 1
        range_start = range_end
        node_index += 1

    return node_indexer, node_points, node_dgree, edge_start_node, edge_end_node

# static method to extract xy point data of a linestring to a numpy matrix.
def GeoPandaExtractor(poly_line_data):

    points = [f(point) for point in [get_point(poly_line_data, 0),
                                     get_point(poly_line_data, -1)] for f in [get_x, get_y]]
    return np.hstack(
        (
            np.array([points[0], points[1]]),
            np.array([points[2], points[3]])
        )
    )


# clss function for node insertion into network. uses spatial index

def effecient_node_insertion(self, layer_name, label="origin", weight_attribute=None):
    n_node_gdf = self.layers["network_nodes"]["gdf"]
    n_edge_gdf = self.layers["network_edges"]["gdf"]
    source_gdf = self.layers[layer_name]["gdf"]
    inserted_node_gdf = _effecient_node_insertion(n_node_gdf, n_edge_gdf, source_gdf, layer_name=layer_name, label=label, weight_attribute=weight_attribute)
    self.layers["network_nodes"]["gdf"] = pd.concat([n_node_gdf, inserted_node_gdf])
    return inserted_node_gdf

# helper function for node insertion, return a node_gdf containing inserted nodes only. 
def _effecient_node_insertion(n_node_gdf, n_edge_gdf, source_gdf, layer_name, label="origin", weight_attribute=None):
    # Assigning nodes to edges using a spatial index
    # TODO: VERIFY IF source_gdf["geometry"] contains POINTS. IF POLYGONS, USE THEIR CENTRPIOD.
    match = n_edge_gdf["geometry"].sindex.nearest(source_gdf["geometry"], return_all=False)

    # Finding out the distance along the edge for each node, and creating a point on the edge for visuals
    node_representative_points = source_gdf["geometry"].centroid.values
    closest_edge_geometries = n_edge_gdf["geometry"].values[match[1]]
    distance_along_closest_edge = closest_edge_geometries.project(node_representative_points)
    point_on_nearest_edge = closest_edge_geometries.interpolate(distance_along_closest_edge)





    # These are essential to construct an o_adjacency and a d_adjacency lists..
    closest_edge_weights = n_edge_gdf["weight"].values[match[1]]
    closest_edge_lengths = n_edge_gdf["length"].values[match[1]]
    closest_edge_starts = n_edge_gdf["start"].values[match[1]]
    closest_edge_ends = n_edge_gdf["end"].values[match[1]]



    weight_to_start = closest_edge_weights * (distance_along_closest_edge/closest_edge_lengths)
    weight_to_end = closest_edge_weights - weight_to_start


    # Nodes attributes
    node_count = source_gdf.shape[0]
    start_index = n_node_gdf.index[-1] + 1
    node_ids = np.arange(start=start_index, stop=start_index + node_count, dtype=np.int32)
    node_source_ids = source_gdf.index.values
    closest_edge_ids = n_edge_gdf.index.values[match[1]]

    # creating a dataframe for new nodes
    index = pd.Index(node_ids, name="id")
    node_weight = np.ones(node_count, dtype=np.float64) if weight_attribute is None else source_gdf[weight_attribute].values
    node_gdf = gpd.GeoDataFrame(
        {
            "source_layer":     pd.Series(np.repeat(np.array([layer_name], dtype=object), repeats=node_count), fastpath=True, index=index, dtype="category"),
            "source_id":        pd.Series(node_source_ids, fastpath=True, index=index, dtype=np.int32),
            "type":             pd.Series(np.repeat(np.array([label], dtype=object), repeats=node_count), fastpath=True, index=index, dtype="category"),
            "weight":           pd.Series(node_weight, fastpath=True, index=index, dtype=np.float32),
            "nearest_edge_id":  pd.Series(closest_edge_ids, fastpath=True, index=index, dtype=np.int32),
            "edge_start_node":  pd.Series(closest_edge_starts, fastpath=True, index=index, dtype=np.int32),
            "weight_to_start":  pd.Series(weight_to_start , fastpath=True, index=index, dtype=np.float32),
            "edge_end_node":    pd.Series(closest_edge_ends , fastpath=True, index=index, dtype=np.int32),
            "weight_to_end":    pd.Series(weight_to_end , fastpath=True, index=index, dtype=np.float32),
            "degree":           pd.Series(np.zeros(node_count, dtype=np.int32), fastpath=True, index=index),
        },
        index=index,
        crs=source_gdf.crs,
        geometry=pd.Series(point_on_nearest_edge, fastpath=True, index=index)
    )
    return node_gdf



### This is testing on the old (May 2023) zonal object..
if __name__ == 'main':
    start = time.time()
    analysis_area = md.Zonal()

    print(f"{(time.time()-start)*1000:6.2f}ms\t object created")
    start = time.time()

    analysis_area.load_layer(
        layer_name='streets',
        file_path="./Cities/Beirut/Data/Network_V08.geojson"
    )

    print(f"{(time.time()-start)*1000:6.2f}ms\t street data loaded")
    start = time.time()

    node_gdf, edge_gdf = create_nodes_edges(
        analysis_area,
        source_layer="streets",
        prepare_geometry=True,
        weight_attribute=None,
        tolerance=0,
        tag_edges=False,
        discard_redundant_edges=True
    )

    print(f"{(time.time()-start)*1000:6.2f}ms\t nodes and edges created created")
    start = time.time()

    analysis_area.load_layer(
        layer_name='buildings',
        file_path="./Cities/Beirut/Data/Buildings.geojson"
    )



    print(f"{(time.time()-start)*1000:6.2f}ms\t buildings loaded")
    start = time.time()

    effecient_node_insertion(
        analysis_area,
        layer_name='buildings',
        label='origin',
        weight_attribute="O_numJobsGFA"
    )

    print(f"{(time.time()-start)*1000:6.2f}ms\t origins inserted")
    start = time.time()

    effecient_node_insertion(
        analysis_area,
        layer_name='buildings',
        label='destination',
        weight_attribute="O_numJobsGFA"
    )

    print(f"{(time.time()-start)*1000:6.2f}ms\t destination inserted")
    print("done")