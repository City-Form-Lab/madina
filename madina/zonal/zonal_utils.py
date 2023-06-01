from shapely import geometry as geo
from geopandas import GeoDataFrame


def flatten_multi_edge_segments(source_gdf, flatten_multiLineStrings=True, flatten_LineStrings=False):
    ls_dict = {"source_index": [], "geometry": []}

    if flatten_multiLineStrings:
        for source_idx in source_gdf.index:
            geometry = source_gdf.at[source_idx, "geometry"]
            geom_type = geometry.geom_type
            if geom_type == 'LineString':
                ls_dict["source_index"].append(source_idx)
                ls_dict["geometry"].append(geometry)
            elif geom_type == 'MultiLineString':
                for ls in list(geometry):
                    ls_dict["source_index"].append(source_idx)
                    ls_dict["geometry"].append(ls)
            else:
                raise TypeError(f"geometries could only be of types 'LineString' and 'MultiLineString', {geom_type} is "
                                f"not valid")

    good_lines = {"source_index": [], "geometry": []}

    for geometry_seq, geometry in enumerate(ls_dict["geometry"]):
        number_of_nodes = len(geometry.coords)
        if (number_of_nodes != 2) and flatten_LineStrings:
            # bad segment, segmenting to individual segments
            for i in range(number_of_nodes - 1):
                # create new segment
                start_point = geo.Point(geometry.coords[i])
                end_point = geo.Point(geometry.coords[i + 1])
                line_segment = geo.LineString([start_point, end_point])
                good_lines["source_index"].append(ls_dict["source_index"][geometry_seq])
                good_lines["geometry"].append(line_segment)
        else:
            # good segment, add as is
            good_lines["source_index"].append(ls_dict["source_index"][geometry_seq])
            good_lines["geometry"].append(geometry)
    flat = GeoDataFrame(good_lines, crs=source_gdf["geometry"].crs)
    flat["length"] = flat["geometry"].length
    return flat


def load_nodes_edges_from_gdf(
        node_dict=None,
        edge_dict=None,
        gdf=None,
        node_snapping_tolerance=0,
        return_dict=False,
        discard_redundant_edges=False,
        weight_attribute=None,
):
    if gdf is not None:
        if weight_attribute is not None:
            sorted_gdf = gdf.sort_values(weight_attribute, ascending=False)
        else:
            sorted_gdf = gdf.sort_values("length", ascending=False)
        street_geometry = list(sorted_gdf["geometry"])

    # find proper index for new nodes and edges
    if len(node_dict["id"]) == 0:
        new_node_id = 0
    else:
        new_node_id = node_dict["id"][-1] + 1

    if len(edge_dict["id"]) == 0:
        new_edge_id = 0
    else:
        new_edge_id = edge_dict["id"][-1] + 1

    counter = 0
    zero_length_edges = 0
    redundant_edges = 0
    list_len = len(street_geometry)
    for street_iloc, street in enumerate(street_geometry):
        counter += 1
        if counter % 100 == 0:
            print(f'{counter = }, progress = {counter / list_len * 100:5.2f}')
        # get this segment's nodes, if segment is more than one segment, get only beginning node and end node
        # , treat as a single sigment
        start_point_index = None
        end_point_index = None

        start_point_geometry = None
        end_point_geometry = None

        start_point = geo.Point(street.coords[0])
        end_point = geo.Point(street.coords[-1])

        smallest_distance_to_start = float('inf')
        for node_seq, node_point in enumerate(node_dict["geometry"]):
            distance = node_point.distance(start_point)
            if smallest_distance_to_start > distance:
                smallest_distance_to_start = distance
                if smallest_distance_to_start <= node_snapping_tolerance:
                    start_point_index = node_dict["id"][node_seq]
                    start_point_geometry = node_dict["geometry"][node_seq]
                if smallest_distance_to_start == 0:
                    break

        smallest_distance_to_end = float('inf')
        for node_seq, node_point in enumerate(node_dict["geometry"]):
            distance = node_point.distance(end_point)
            if smallest_distance_to_end > distance:
                smallest_distance_to_end = distance
                if smallest_distance_to_end <= node_snapping_tolerance:
                    end_point_index = node_dict["id"][node_seq]
                    end_point_geometry = node_dict["geometry"][node_seq]
                if smallest_distance_to_end == 0:
                    break

        if discard_redundant_edges:
            if (start_point_index is not None) and (end_point_index is not None):
                # Check and discard if this starts and ends at the same node
                if start_point_index == end_point_index:
                    zero_length_edges += 1
                    continue
                # CHeck and discard if a link already exist between start and end.
                found_redundant = False
                for starts, ends in zip(edge_dict["start"], edge_dict["end"]):
                    if (
                            (starts == start_point_index) and (ends == end_point_index)
                    ) or (
                            (ends == start_point_index) and (starts == end_point_index)
                    ):
                        redundant_edges += 1
                        found_redundant = True
                        break
                if found_redundant:
                    continue

        if start_point_index is not None:
            node_dict["degree"][start_point_index] += 1
        else:
            start_point_index = new_node_id
            new_node_id += 1
            node_dict["id"].append(start_point_index)
            node_dict["geometry"].append(start_point)
            node_dict["source_layer"].append("streets")
            node_dict["source_id"].append(None)
            node_dict["type"].append("street_node")
            node_dict["weight"].append(0)
            node_dict["degree"].append(1)
            node_dict["nearest_street_id"].append(None)
            node_dict["nearest_street_node_distance"].append(None)

        if end_point_index is not None:
            node_dict["degree"][end_point_index] += 1
        else:
            end_point_index = new_node_id
            new_node_id += 1

            node_dict["id"].append(end_point_index)
            node_dict["geometry"].append(end_point)
            node_dict["source_layer"].append("streets")
            node_dict["source_id"].append(None)
            node_dict["type"].append("street_node")
            node_dict["weight"].append(0)
            node_dict["degree"].append(1)
            node_dict["nearest_street_id"].append(None)
            node_dict["nearest_street_node_distance"].append(None)

        # construct a proper geometry that have the start and end nodes.
        if start_point_geometry:
            use_start = start_point_geometry
        else:
            use_start = start_point

        if end_point_geometry:
            use_end = end_point_geometry
        else:
            use_end = end_point
        new_street = geo.linestring.LineString([
            use_start,
            use_end
        ])

        # Need to add a segment
        edge_dict["id"].append(new_edge_id)
        edge_dict["start"].append(start_point_index)
        edge_dict["end"].append(end_point_index)
        edge_dict["length"].append(street.length)
        if weight_attribute is None:
            edge_dict["weight"].append(street.length)
        else:
            edge_dict["weight"].append(sorted_gdf.iloc[street_iloc][weight_attribute])
        edge_dict["type"].append("street")
        edge_dict["geometry"].append(street)
        edge_dict["parent_street_id"].append(sorted_gdf.iloc[street_iloc].name)
        new_edge_id += 1
    if return_dict:
        return node_dict, edge_dict
    else:
        node_gdf = GeoDataFrame(node_dict, crs=gdf["geometry"].crs).set_index("id")
        edge_gdf = GeoDataFrame(edge_dict, crs=gdf["geometry"].crs).set_index("id")
        # print(f"Node/Edge table constructed. {zero_length_edges = }, {redundant_edges = }")
    if discard_redundant_edges:
        print(f"redundant edge report: {zero_length_edges = }, {redundant_edges = }")
    return node_gdf, edge_gdf
