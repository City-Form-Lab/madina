from madina.zonal import Zonal


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
    def parallel_betweenness(zonal: Zonal, search_radius, detour, decay, decay_method, beta, path_detour_penalty):
        """
        TODO: fill function spec
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
    def accessibility(zonal: Zonal, reach=False, gravity=False, closest_facility=False, alpha=1,
                                        beta=None, search_radius=None, weight=None):
        """
        Modifies the input zonal with accessibility metrics such as `reach`, `gravity`, and `closest_facility` analysis.
        Equivalent of 'una_accessibility' function in madina.py

        Returns:
            A modified Zonal object

        Raises:
            ValueError if `gravity` is True but beta is None/unspecified.

        """
        raise NotImplementedError

    @staticmethod
    def service_area(zonal: Zonal, origin_ids=None, search_radius=None):
        """
        Calculates the service area of the origins in `origin_ids`.

        Returns:
            a tuple of destinations accessible from the origins, edges traversed, and a pandas GeoDataFrame of their scope
        """
        raise NotImplementedError
