from geopandas import GeoDataFrame
import numpy as np
import geopandas as gpd
import pandas as pd
import shapely.geometry as geo
from shapely.ops import split, snap

# from pygeos.lib import get_x, get_y, get_point

def node_edge_builder(geometry_gdf, weight_attribute=None, tolerance=0.0, source_layer: str =None):
    if tolerance == 0.0:
        # use vectorized implementaytion
        point_xy = GeoPandaExtractor(np.array([geom for geom in geometry_gdf.geometry.values]))
        node_indexer, node_points, node_dgree, edge_start_node, edge_end_node = vectorized_node_edge_builder(point_xy)
        # use spatial index implementation
        # could this gain effeciency by being sorted?
        # TODO: complete the implementation of this case by copyinfg the for-loop thT UTILIz MATCHING RESULTS..
        point_geometries = geometry_gdf["geometry"].boundary.explode(index_parts=False).reset_index(drop=True)
        #matching = point_geometries.sindex.query(
            #point_geometries.buffer(tolerance))

        # construct node and edge gdfs
        edge_count = geometry_gdf.shape[0]
        index = pd.Index(np.arange(edge_count, dtype=int), dtype=int, name="id")
        length = pd.Series(
            geometry_gdf["geometry"].length.values, fastpath=True, index=index)
        edge_gdf = gpd.GeoDataFrame(
            {
                "length": length,
                "weight": length if weight_attribute is None else geometry_gdf.apply(
                    lambda x: max(x[weight_attribute], 0.01) if x[weight_attribute] != 0 else x["geometry"].length,
                    axis=1),
                "type": pd.Series(np.repeat(np.array(["street"], dtype=object), repeats=edge_count), fastpath=True,
                                  index=index, dtype="category"),
                "parent_street_id": pd.Series(geometry_gdf.index.values,dtype=int,  fastpath=True, index=index),
                "start": pd.Series(edge_start_node, dtype=np.int32, fastpath=True, index=index),
                "end": pd.Series(edge_end_node, dtype=np.int32, fastpath=True, index=index),
            },
            index=index,
            geometry=geometry_gdf["geometry"]
        )

        node_count = node_points.shape[0]
        index = pd.Index(np.arange(node_count), name="id")
        node_gdf = gpd.GeoDataFrame(
            {
                "source_layer": pd.Series(np.repeat(np.array([source_layer], dtype=object), repeats=node_count),
                                          fastpath=True, index=index, dtype="category"),
                "source_id": pd.Series(np.repeat(np.array([0], dtype=np.int32), repeats=node_count), fastpath=True,
                                       index=index, dtype=np.int32),
                "type": pd.Series(np.repeat(np.array(["street_node"], dtype=object), repeats=node_count), fastpath=True,
                                  index=index, dtype="category"),
                "weight": pd.Series(np.repeat(np.array([0.0], dtype=np.float32), repeats=node_count), fastpath=True,
                                    index=index, dtype=np.float32),
                # "nearest_street_id": pd.Series(np.repeat(np.array([0], dtype=np.int32), repeats=node_count), fastpath=True, index=index, dtype=np.int32),
                # "nearest_street_node_distance": pd.Series(np.repeat(np.array([{}], dtype=object), repeats=node_count), fastpath=True, index=index),
                "degree": pd.Series(node_dgree, fastpath=True, index=index),
                # "connected_edges": connected_edges
            },
            index=index,
            # crs=geometry_gdf.crs,
            geometry=gpd.points_from_xy(
                x=point_xy[0][node_indexer], y=point_xy[1][node_indexer], crs=geometry_gdf.crs)
        )

    elif tolerance > 0:
        # node_gdf, edge_gdf = _effecient_tolerance_network_nodes_edges(
        node_gdf, edge_gdf = tolerance_network_nodes_edges(
            geometry_gdf=geometry_gdf,
            weight_attribute=weight_attribute,
            tolerance=tolerance, 
            source_layer=source_layer,
        )
    else:
        raise ValueError(f"tolerance must either be zero or a positive number, {tolerance} was given.")

    return node_gdf, edge_gdf  # Return Adjacency List here...


def vectorized_node_edge_builder(point_xy):
    # the point_xy input is a 2 by n matrix. where the start point and end point of an edge are in alternating order..
    # unique parameter is expected to come from a spatial index, on points that are sorted.

    # input setup
    point_count = point_xy.shape[1]
    edge_count = int(point_count / 2)

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
    if node_first_occurance[-1] != (point_count - 1):
        np.append(node_first_occurance, point_count - 1)

    range_start = 0
    i = 0
    node_index = 0
    for range_end in (list(node_first_occurance[1:]) + [node_first_occurance[-1] + 1]):
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


def GeoPandaExtractor(poly_line_data):
    '''code that works with pygeos
    points = [f(point) for point in [get_point(poly_line_data, 0),
                                     get_point(poly_line_data, -1)] for f in [get_x, get_y]]
    return np.hstack(
        (
            np.array([points[0], points[1]]),
            np.array([points[2], points[3]])
        )
    )
    '''
    return np.hstack(
        (
            np.array([[line.coords[0][0] for line in poly_line_data], [line.coords[0][1] for line in poly_line_data]]),
            np.array([[line.coords[-1][0] for line in poly_line_data], [line.coords[-1][1] for line in poly_line_data]])
        )
    )


def tolerance_network_nodes_edges(geometry_gdf, weight_attribute=None, tolerance=1.0, source_layer=None):
    # geometry_gdf["weight"] = geometry_gdf["geometry"].length if weight_attribute is None else geometry_gdf[weight_attribute].apply(
    #    lambda x: max(1, x))

    geometry_gdf["weight"] = geometry_gdf["geometry"].length if weight_attribute is None else geometry_gdf.apply(
        lambda x: max(x[weight_attribute], 0.01) if x[weight_attribute] != 0 else x["geometry"].length, axis=1)

    geometry_gdf = geometry_gdf.sort_values("weight", ascending=False)

    # geometry_gdf = geometry_gdf[geometry_gdf["weight"] >= tolerance].reset_index(drop=True)
    point_geometries = pd.concat(
        [
            geometry_gdf["geometry"].apply(lambda x: geo.Point(x.coords[0])),
            geometry_gdf["geometry"].apply(lambda x: geo.Point(x.coords[-1]))
        ]
    ).reset_index(drop=True)

    point_types = ["start"] * geometry_gdf.shape[0] + ["end"] * geometry_gdf.shape[0]
    point_segment_ids = list(geometry_gdf.index) + list(geometry_gdf.index)

    point_count = geometry_gdf.shape[0] * 2
    geometry_collector = np.empty(point_count, dtype=object)
    degree_collector = np.empty(point_count, dtype="int64")

    edge_count = geometry_gdf.shape[0]

    edge_collector = {
        "start": np.zeros(edge_count, dtype="int64") - 1,
        "end": np.zeros(edge_count, dtype="int64") - 1
    }

    matching = point_geometries.sindex.query(
        point_geometries.buffer(tolerance), predicate="intersects")
    # grouping points..
    unique_points, intersection_count = np.unique(
        matching[0], return_counts=True)

    point_intersection = {}
    current_index = 0
    for point, count in zip(unique_points, intersection_count):
        point_intersection[point] = set(
            matching[1][current_index:current_index + count]) - {point}
        current_index = current_index + count

    new_node_id = -1
    current_point_idx = -1
    for position in ["start", "end"]:
        for edge_idx in geometry_gdf.index:
            current_point_idx += 1
            if edge_collector[position][edge_idx] != -1:
                continue
            current_intersecting_points = point_intersection[current_point_idx]
            degree = 1
            new_node_id += 1
            edge_collector[position][edge_idx] = new_node_id
            for intersecting_point in current_intersecting_points:
                intersecting_point_segment_idx = point_segment_ids[intersecting_point]
                intersecting_point_type = point_types[intersecting_point]
                if edge_collector[intersecting_point_type][
                    intersecting_point_segment_idx] != -1:  ## already filled by another edge..
                    continue
                degree += 1
                edge_collector[intersecting_point_type][intersecting_point_segment_idx] = new_node_id

            degree_collector[new_node_id] = degree
            geometry_collector[new_node_id] = point_geometries[current_point_idx]

    node_gdf = gpd.GeoDataFrame(
        {
            "id": pd.Series(range(new_node_id + 1), dtype="int64"),
            "source_layer": pd.Series([source_layer] * (new_node_id + 1), dtype="category"),
            "source_id": pd.Series([0] * (new_node_id + 1), dtype="int64"),
            "type": pd.Series(["street_node"] * (new_node_id + 1), dtype="category"),
            "weight": pd.Series([0.0] * (new_node_id + 1), dtype="float64"),
            # "nearest_street_id": pd.Series([0] * (new_node_id+1), dtype="int64"),
            # "nearest_street_node_distance": pd.Series([None] * (new_node_id+1), dtype="object"),
            "degree": pd.Series(degree_collector[:new_node_id + 1], dtype="int64"),
            "geometry": gpd.GeoSeries(geometry_collector[:new_node_id + 1], crs=geometry_gdf.crs)
        }
    ).set_index("id")

    geometries = []
    snapped = []
    snapping_distance = []

    # making sure geometry is snapped as well. Theooritically, this could be done in the previous loop, an could be done slightly more eddecoenctly by not accessing node_gdf, not using append,..
    # geometry_gdf["geometry"].sort_index()
    for segment_geometry, segment_start_node, segment_end_node in zip(geometry_gdf["geometry"].sort_index(),
                                                                      edge_collector["start"], edge_collector["end"]):
        start = node_gdf.at[segment_start_node, 'geometry'].coords[0]
        end = node_gdf.at[segment_end_node, 'geometry'].coords[0]
        middle = segment_geometry.coords[1:-1]

        new_geometry = geo.LineString([start] + middle + [end])
        difference = abs(segment_geometry.length - new_geometry.length)
        geometries.append(new_geometry)
        if difference > 0:
            snapped.append(True)
            snapping_distance.append(difference)
        else:
            snapped.append(False)
            snapping_distance.append(0)

    edge_gdf = gpd.GeoDataFrame(
        {
            "id": pd.Series(range(edge_count), dtype="int64"),
            "geometry": gpd.GeoSeries(geometries, crs=geometry_gdf.crs),
            "length": geometry_gdf["geometry"].length,
            "weight": geometry_gdf["weight"],
            "type": pd.Series(["street"] * edge_count, dtype="category"),
            "parent_street_id": pd.Series(list(geometry_gdf.sort_index().index), dtype="int64"),
            "start": pd.Series(edge_collector["start"], dtype="int64"),
            "end": pd.Series(edge_collector["end"], dtype="int64"),
            "snapped": pd.Series(snapped, dtype=bool),
            "snapping_distance": pd.Series(snapping_distance, dtype="float64")
        }
    ).set_index("id")
    return node_gdf, edge_gdf


def _discard_redundant_edges(edge_gdf: GeoDataFrame):
    ## finding edges that share the same start and end, including those who start and end at the same node
    edge_stack = pd.DataFrame({
        'edge_id': np.concatenate([edge_gdf.index.values, edge_gdf.index.values]),
        'start': np.concatenate([edge_gdf['start'].values, edge_gdf['end'].values]),
        'end': np.concatenate([edge_gdf['end'].values, edge_gdf['start'].values]),
    })
    redundnt_edge_ids = np.unique(
        edge_stack[edge_stack.duplicated(subset=['start', 'end'], keep=False)]['edge_id'].values)

    # keeping the shortest of all available 'redundant' edges
    keep = set({})
    redundnt_edge_queue = set(redundnt_edge_ids)

    # TODO: update degrees of end nodes of all dropped edges..
    while redundnt_edge_queue:
        edge_idx = redundnt_edge_queue.pop()
        this_edge_start = edge_gdf.at[edge_idx, 'start']
        this_edge_end = edge_gdf.at[edge_idx, 'end']
        if this_edge_start == this_edge_end:
            continue  # this excludes nodes that start and end at the same node from being kept
        this_edg_duplicates = edge_gdf[
            ((edge_gdf['start'] == this_edge_start) & (edge_gdf['end'] == this_edge_end))
            | ((edge_gdf['end'] == this_edge_start) & (edge_gdf['start'] == this_edge_end))
            ]

        keep.add(this_edg_duplicates["weight"].idxmin())
        redundnt_edge_queue = redundnt_edge_queue - set(this_edg_duplicates.index.values)

    remove_edges = set(redundnt_edge_ids) - keep
    unique_edges = list(set(edge_gdf.index.values) - remove_edges)
    return edge_gdf.loc[unique_edges]


def _split_redundant_edges(node_gdf: GeoDataFrame, edge_gdf: GeoDataFrame):
    ## finding edges that share the same start and end, including those who start and end at the same node
    edge_stack = pd.DataFrame(
        {
            'edge_id': np.concatenate([edge_gdf.index.values, edge_gdf.index.values]),
            'start': np.concatenate([edge_gdf['start'].values, edge_gdf['end'].values]),
            'end': np.concatenate([edge_gdf['end'].values, edge_gdf['start'].values]),
        }
    )

    redundnt_edge_ids = np.unique(
        edge_stack[edge_stack.duplicated(subset=['start', 'end'], keep=False)]['edge_id'].values)
    redundnt_edge_queue = set(redundnt_edge_ids)

    new_node_id = node_gdf.index.max()
    node_ids = []
    node_geometries = []

    edge_geometries = []
    edge_weights = []
    edge_parent_street_ids = []
    edge_starts = []
    edge_ends = []

    remove_edges = []

    # TODO: update degrees of end nodes of all dropped edges..
    while redundnt_edge_queue:
        edge_idx = redundnt_edge_queue.pop()

        this_edge_start = edge_gdf.at[edge_idx, 'start']
        this_edge_end = edge_gdf.at[edge_idx, 'end']

        # finding parallel edges to the current..
        this_edg_duplicates = edge_gdf[
            ((edge_gdf['start'] == this_edge_start) & (edge_gdf['end'] == this_edge_end))
            | ((edge_gdf['end'] == this_edge_start) & (edge_gdf['start'] == this_edge_end))
            ]

        # looped edge, no need to process.
        if len(this_edg_duplicates.index.values) == 1:
            continue

        # sort by weight, and skip the first edge (.index[1:]), the shortest edge kept intact, while the rest are split.
        for edge_idx in this_edg_duplicates.sort_values('weight').index[1:]:
            line = edge_gdf.at[edge_idx, 'geometry']
            split_point = line.interpolate(0.5, normalized=True)
            splited_lines = list(split(snap(line, split_point, 0.1), split_point).geoms)

            # construct central nodes
            new_node_id += 1
            node_ids.append(new_node_id)
            node_geometries.append(split_point)

            # construct two edges
            edge_geometries.append(splited_lines[0])
            edge_geometries.append(splited_lines[1])

            edge_weights.append(edge_gdf.at[edge_idx, 'weight'] / 2.0)
            edge_weights.append(edge_gdf.at[edge_idx, 'weight'] / 2.0)

            edge_parent_street_ids.append(edge_gdf.at[edge_idx, 'parent_street_id'])
            edge_parent_street_ids.append(edge_gdf.at[edge_idx, 'parent_street_id'])

            edge_starts.append(edge_gdf.at[edge_idx, 'start'])
            edge_ends.append(new_node_id)
            edge_starts.append(new_node_id)
            edge_ends.append(edge_gdf.at[edge_idx, 'end'])

            remove_edges.append(edge_idx)

        # remove already processed edges
        redundnt_edge_queue = redundnt_edge_queue - set(this_edg_duplicates.index.values)

    # construct new node and edge gdfs..
    node_index = pd.Index(node_ids, name="id")
    node_count = len(node_ids)
    node_geometry_series = gpd.GeoSeries(
        data=node_geometries,
        index=node_index,
        crs=node_gdf.crs
    )
    new_node_gdf = gpd.GeoDataFrame(
        {
            "source_layer": pd.Series(np.repeat(np.array(["streets"], dtype=object), repeats=node_count), fastpath=True,
                                      index=node_index, dtype="category"),
            "source_id": pd.Series(np.repeat(np.array([0], dtype=np.int32), repeats=node_count), fastpath=True,
                                   index=node_index, dtype=np.int32),
            "type": pd.Series(np.repeat(np.array(["street_node"], dtype=object), repeats=node_count), fastpath=True,
                              index=node_index, dtype="category"),
            "weight": pd.Series(np.repeat(np.array([0.0], dtype=np.float32), repeats=node_count), fastpath=True,
                                index=node_index, dtype=np.float32),
            "degree": pd.Series(np.repeat(np.array([2], dtype=np.int32), repeats=node_count), fastpath=True,
                                index=node_index),
        },
        index=node_index,
        geometry=node_geometry_series
    )

    edge_count = len(edge_geometries)
    new_edge_id = edge_gdf.index.max() + 1
    edge_index = pd.Index(np.arange(start=new_edge_id, stop=new_edge_id + edge_count), name="id")
    # print (edge_index)
    edge_geometry_series = gpd.GeoSeries(
        data=edge_geometries,
        index=edge_index,
        crs=edge_gdf.crs
    )
    length = pd.Series(edge_geometry_series.length.values, fastpath=True, index=edge_index)
    new_edge_gdf = gpd.GeoDataFrame(
        {
            "length": length,
            "weight": pd.Series(np.array(edge_weights), fastpath=True, index=edge_index),
            "type": pd.Series(np.repeat(np.array(["street"], dtype=object), repeats=edge_count), fastpath=True,
                              index=edge_index, dtype="category"),
            "parent_street_id": pd.Series(np.array(edge_parent_street_ids, dtype=int), fastpath=True, index=edge_index),
            "start": pd.Series(np.array(edge_starts, dtype=np.int32), fastpath=True, index=edge_index),
            "end": pd.Series(np.array(edge_ends, dtype=np.int32), fastpath=True, index=edge_index),
        },
        index=edge_index,
        geometry=edge_geometry_series
    )
    # print (f"{edge_gdf.shape = }\t{edge_gdf.drop(index=remove_edges).shape = }\t{len(remove_edges) = }")
    # print (f"{pd.concat([edge_gdf.drop(index=remove_edges), new_edge_gdf]).shape = }")

    return pd.concat([node_gdf, new_node_gdf]), pd.concat([edge_gdf.drop(index=remove_edges), new_edge_gdf])


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


def efficient_node_insertion(n_node_gdf: GeoDataFrame, n_edge_gdf: GeoDataFrame, source_gdf: GeoDataFrame,
                             layer_name: str, label: str = "origin", weight_attribute: str = None):
    # Assigning nodes to edges using a spatial index
    # TODO: CHECK IF THESE AR EPOINTS, IF POLYGONS, USE THEIR CENTRPIOD
    match = n_edge_gdf["geometry"].sindex.nearest(source_gdf["geometry"], return_all=False)

    # FInding out the distance along the edge for each node, and creating a point on the edge for visuals
    node_representative_points = source_gdf["geometry"].centroid.values
    closest_edge_geometries = n_edge_gdf["geometry"].values[match[1]]
    distance_along_closest_edge = closest_edge_geometries.project(node_representative_points)
    point_on_nearest_edge = closest_edge_geometries.interpolate(distance_along_closest_edge)

    # These are essential to construct an o_adjacency and a d_adjacency lists..
    closest_edge_weights = n_edge_gdf["weight"].values[match[1]]
    closest_edge_lengths = n_edge_gdf["length"].values[match[1]]
    closest_edge_starts = n_edge_gdf["start"].values[match[1]]
    closest_edge_ends = n_edge_gdf["end"].values[match[1]]

    weight_to_start = closest_edge_weights * (distance_along_closest_edge / closest_edge_lengths)
    weight_to_end = closest_edge_weights - weight_to_start

    # Nodes attributes
    node_count = source_gdf.shape[0]
    start_index = n_node_gdf.index[-1] + 1
    node_ids = np.arange(start=start_index, stop=start_index + node_count, dtype=np.int32)
    node_source_ids = source_gdf.index.values
    closest_edge_ids = n_edge_gdf.index.values[match[1]]

    # creating a dataframe for new nodes
    index = pd.Index(node_ids, name="id")
    node_weight = np.ones(node_count, dtype=np.float64) if weight_attribute is None else source_gdf[
        weight_attribute].values
    node_gdf = gpd.GeoDataFrame(
        {
            "source_layer": pd.Series(np.repeat(np.array([layer_name], dtype=object), repeats=node_count),
                                      fastpath=True, index=index, dtype="category"),
            "source_id": pd.Series(node_source_ids, fastpath=True, index=index, dtype=np.int32),
            "type": pd.Series(np.repeat(np.array([label], dtype=object), repeats=node_count), fastpath=True,
                              index=index, dtype="category"),
            "weight": pd.Series(node_weight, fastpath=True, index=index, dtype=np.float32),
            "nearest_edge_id": pd.Series(closest_edge_ids, fastpath=True, index=index, dtype=np.int32),
            "edge_start_node": pd.Series(closest_edge_starts, fastpath=True, index=index, dtype=np.int32),
            "weight_to_start": pd.Series(weight_to_start, fastpath=True, index=index, dtype=np.float32),
            "edge_end_node": pd.Series(closest_edge_ends, fastpath=True, index=index, dtype=np.int32),
            "weight_to_end": pd.Series(weight_to_end, fastpath=True, index=index, dtype=np.float32),
            "degree": pd.Series(np.zeros(node_count, dtype=np.int32), fastpath=True, index=index),
        },
        index=index,
        crs=source_gdf.crs,
        geometry=pd.Series(point_on_nearest_edge, fastpath=True, index=index)
    )
    return node_gdf
