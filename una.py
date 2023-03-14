from madina import Zonal


class UNA:
    """
    A set of network tools including betweenness, reach and gravity analysis.
    """
    DEFAULT_DETOUR_RATIO = 1.15
    DEFAULT_SEARCH_RADIUS = 800
    IS_TURN_PENALTY = False

    def __init__(self):
        pass

    @staticmethod
    def parallel_betweenness(zonal: Zonal,
                             search_radius,
                             detour,
                             decay,
                             decay_method,
                             beta,
                             path_detour_penalty
                             ):
        """
        Finds the betweenness
        """
        raise NotImplementedError

    @staticmethod
    def best_path_generator(
            zonal: Zonal,
            origin_index,
            search_radius=DEFAULT_SEARCH_RADIUS,
            detour_ratio=DEFAULT_DETOUR_RATIO,
            turn_penalty=IS_TURN_PENALTY
    ):
        """
        TODO: Fill out function spec
        """

        raise NotImplementedError

    @staticmethod
    def visualize_graph(zonal: Zonal):
        """
        Creates an HTML map of the `zonal` object.
        """
        raise NotImplementedError

    @staticmethod
    def get_od_subgraph(zonal: Zonal, origin_node, distance):
        """
        Creates a subgraph of the city network `zonal` from all nodes <= `distance` from the node at `origin_node`

        """
        raise NotImplementedError

    def _get_nodes_at_distance(self, zonal: Zonal, origin_node, distance, method='geometric'):
        """
        Gets all nodes at distance `distance` from the origin `origin_node` using the method `method`.
        Approximation of functionality from ln 496-592 of b_f.py
        """
        # TODO: Copy over remainder of distance methods. What does 'trail_blazer' mean?
        method_dict = {
            'geometric': self._get_nodes_at_geometric_distance,
            'network': self._get_nodes_at_network_distance,
            'trail_blazer': self._get_nodes_at_bf_distance

        }
        return method_dict[method](zonal, origin_node, distance)

    def _get_nodes_at_geometric_distance(self, zonal: Zonal, origin_node, distance):
        raise NotImplementedError

    def _get_nodes_at_network_distance(self, zonal: Zonal, origin_node, distance):
        raise NotImplementedError

    def _get_nodes_at_bf_distance(self, zonal: Zonal, origin_node, distance):
        raise NotImplementedError

    def _scan_for_intersections(self, zonal: Zonal, node_snapping_tolerance=1):
        '''
        TODO: Fill out function spec
        '''
        raise NotImplementedError
