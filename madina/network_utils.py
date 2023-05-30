import random


def cut(edge, distance):
    """
    Divides an edge `edge` into two at a distance `distance` from its starting point.
    Previously a local function scoped to `insert_nodes_v2`
    """
    raise NotImplementedError


def color_gdf(gdf, by_attribute, method, color_scheme):
    """
    Sets the color of the nodes in a `gdf`, visualizing the attribute/column name to be
    colored by using `by_attribute`.

    Available schemes are: `single`, `categorical` and `gradient`.

     `color_scheme`: if color method is single color, expects one color. if categorical,
        expects nothing and would give automatic assignment, or a dict {"val': [0,0,0]}. if color_method is gradient,
        expects nothing for a default color map, or a color map name

    Returns:
        None

    """

    if not method:
        if by_attribute:
            method = "single"
        else:
            method = "categorical"

    # if "color_by_attribute" is not given, and its not a default layer, assuming color_method == "single_color"
    if by_attribute is None and color_scheme is None:
        color_scheme = [random.random() * 255, random.random() * 255, random.random() * 255]
        method = "single"
    elif not color_scheme:
        if method == "single":
            color_scheme = [random.random() * 255, random.random() * 255, random.random() * 255]
        elif method == "categorical":
            color_scheme = {"__other__": [255, 255, 255]}
            for distinct_value in gdf[by_attribute].unique():
                color_scheme[distinct_value] = [random.random() * 255, random.random() * 255, random.random() * 255]


empty_network_template = {
    'node': {
        "id": [],
        "geometry": [],
        "source_layer": [],
        "source_id": [],
        "type": [],
        "weight": [],
        "degree": [],
        "nearest_street_id": [],
        "nearest_street_node_distance": [],
    },
    'edge': {
        "id": [],
        "start": [],
        "end": [],
        "length": [],
        "weight": [],
        "type": [],
        "geometry": [],
        "parent_street_id": []
    }
}

DEFAULT_COLORS = {
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
