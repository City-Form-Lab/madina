from geopandas import GeoDataFrame
# street nodes, origin nodes, destination_nodes are the different types of nodes


class Network:
    """
    A road network with nodes `nodes` and edges `edges` weighted with factor `weight_attribute`,
    projected onto the `projected_crs` coordinate system.

    Internal class, meant to represent a network within the Zonal object.
    """

    def __init__(self, nodes: GeoDataFrame, edges: GeoDataFrame, projected_crs: str, weight_attribute=None):
        self.nodes = nodes
        self.edges = edges
        self.crs = projected_crs
        self.weight_attribute = weight_attribute
        self.G = None  # G represents a networkx graph. To be deprecated
        return

    def insert_node(self, label: str, layer_name: str, weight: int):
        """
        Adds a node with label `label` to the layer `layer_name` with weight `weight`.
        Maps to insert_nodes_v2 in madina.py
        """
        raise NotImplementedError

    def set_node_value(self, idx, label, new_value):
        """
        Sets the node at (`idx`, `label`) value in the network to `new_value`.
        """
        self.nodes.at[idx, label] = new_value
        return

    def create_graph(self, light_graph: bool, dense_graph: bool, d_graph: bool):
        """
        TODO: Fill out function spec
        """
        raise NotImplementedError
