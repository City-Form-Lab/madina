def cut(edge, distance):
    """
    Divides an edge `edge` into two at a distance `distance` from its starting point.
    Previously a local function scoped to `insert_nodes_v2`
    """
    raise NotImplementedError


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
