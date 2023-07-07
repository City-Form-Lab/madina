import time

import numpy as np
import networkx as nx
from geopandas import GeoDataFrame
from shapely import wkt
from shapely import geometry as geo

from madina.zonal.layer import *
# street nodes, origin nodes, destination_nodes are the different types of nodes


class Network:
    """
    A road network with nodes `nodes` and edges `edges` weighted with factor `weight_attribute`,
    projected onto the `projected_crs` coordinate system.

    Internal class, meant to represent a network within the Zonal object.
    """

    def __init__(self, nodes: GeoDataFrame, edges: GeoDataFrame, projected_crs: str, turn_threshold_degree: float, turn_penalty_amount: float, weight_attribute=None,):

        if nodes.empty or edges.empty:
            pass # throw Error here

        self.nodes = nodes
        self.edges = edges
        self.crs = projected_crs
        self.turn_threshold_degree = turn_threshold_degree
        self.turn_penalty_amount = turn_penalty_amount
        self.weight_attribute = weight_attribute
        self.light_graph = None
        self.d_graph = None
        self.od_graph = None
        return
    
    def set_node_value(self, idx, label, new_value):
        """
        Sets the node at (`idx`, `label`) value in the network to `new_value`.
        """
        self.nodes.at[idx, label] = new_value
        return

    def create_graph(self, light_graph=False, d_graph=True, od_graph=False):
        """
        Creates the corresponding graphs in the Network object based on the current nodes and edges
        `light_graph` - contains only network nodes and edges
        `d_graph` - contains all destination nodes
        `od_graph` - contains all origin and destination nodes

        Args:
            light_graph: if true, create `self.light_graph`
            d_graph: if true, create `self.d_graph`
            od_graph: if true, create `self.od_graph`

        Returns:
            none
        """

        if light_graph:
            street_node_gdf = self.nodes[self.nodes["type"] == "street_node"]
            ## no need to filter edges, because in light graph, they haven't been inseted
            self.light_graph = nx.Graph()
            for idx in self.edges.index:
                self.light_graph.add_edge(
                    int(self.edges.at[idx, "start"]),
                    int(self.edges.at[idx, "end"]),
                    weight=max(self.edges.at[idx, "weight"], 0),
                    type=self.edges.at[idx, "type"],
                    id=idx
                )

            for idx in street_node_gdf.index:
                self.light_graph.nodes[int(idx)]['type'] = street_node_gdf.at[idx, "type"]

        if d_graph:
            d_list = list(self.nodes[self.nodes["type"] == "destination"].index)
            graph = self.light_graph.copy() # TODO: Consider deep copies?
            self.update_light_graph(graph, add_nodes=d_list)
            self.d_graph = graph

        if od_graph:
            od_list = list(self.nodes[self.nodes["type"].isin(["origin", "destination"])].index)
            graph = self.light_graph.copy()
            self.update_light_graph(graph, add_nodes=od_list)
            self.od_graph = graph

        return

    def visualize_graph(self):
        """
        Creates an HTML map of the `zonal` object.
        """
        raise NotImplementedError

    def get_od_subgraph(self, origin_idx, distance):
        """
        Creates a subgraph of the city network `self` from all nodes <= `distance` from the node at `origin_idx`

        Returns:
            A smaller network graph (?)

        """
        raise NotImplementedError

    def turn_o_scope(self, origin_idx, search_radius, detour_ratio, turn_penalty=True,
                     origin_graph=None, return_paths=True):
        """
        Runs a modified Dijkstra algorithm from the `origin_idx`, with a `turn_penalty` for
        making a turn along the path. Bounds the search by `search_radius`

        Returns:
            A tuple of destination indices, origin scope, and paths from origin to destination

        """
        raise NotImplementedError
    
    def update_light_graph(self, graph: nx.Graph, add_nodes: list = [], remove_nodes: list = []):
        """
        Updates the given graph object by adding nodes to and removing nodes from it.

        Args:
            graph: The given networkx Graph object to be edited
            add_nodes: a list of nodes to be added
            remove_nodes: a list of nodes to be removed

        Returns:
            none
        """
        
        if "added_nodes" not in graph.graph:
            graph.graph["added_nodes"] = []

        # Add nodes
        if len(add_nodes) > 0:

            existing_nodes = graph.graph["added_nodes"]
            for node_idx in add_nodes:
                if node_idx in existing_nodes:
                    print(f'node ({node_idx}) is already added...{add_nodes = }\t{existing_nodes = }')
                    print(graph)
                    add_nodes.remove(node_idx)
                else:
                    graph.graph["added_nodes"].append(node_idx)

            edge_nodes = {}
            for key, value in self.nodes.loc[graph.graph["added_nodes"]].groupby("nearest_edge_id"):
                edge_nodes[int(key)] = list(value.index)
                
            for edge_id in edge_nodes:
                neighbors = edge_nodes[edge_id]
                insert_neighbors = set(add_nodes).intersection(set(neighbors))
                existing_neighbors = set(neighbors) - insert_neighbors

                if len(insert_neighbors) == 0:
                    continue
                    #self.nodes.at[node_idx, ""]
                if len(neighbors) == 1:
                    node_idx = neighbors[0]
                    graph.add_edge(
                        int(self.nodes.at[node_idx, "edge_end_node"]),
                        int(node_idx),
                        weight=max(self.nodes.at[node_idx, "weight_to_end"], 0),
                        id=edge_id
                    )
                    graph.add_edge(
                        int(node_idx),
                        int(self.nodes.at[node_idx, "edge_start_node"]),
                        weight=max(self.nodes.at[node_idx, 'weight_to_start'], 0),
                        id=edge_id
                    )
                    graph.remove_edge(self.nodes.at[node_idx, "edge_end_node"], int(self.nodes.at[node_idx, "edge_start_node"]))
                    
                else:
                    # start a chain addition of neighbors, starting from the 'left',
                    # so, need to sort based on distance from left
                    segment_weight = self.edges.at[edge_id, "weight"]

                    chain_start = self.nodes.at[neighbors[0], "edge_end_node"]
                    chain_end = self.nodes.at[neighbors[0], "edge_start_node"]

                    chain_distances = [self.nodes.at[node, "weight_to_end"] for node in neighbors]

                    if len(existing_neighbors) == 0:  # if there are no existing neighbors, remove the original edge
                        graph.remove_edge(int(chain_start), int(chain_end))
                    else:  # if there are existing neighbors, remove them. This would also remove their associated edges
                        for node in existing_neighbors:
                            graph.remove_node(node)

                    chain_nodes = [chain_start] + neighbors + [chain_end]
                    chain_distances = [0] + chain_distances + [segment_weight]

                    chain_nodes = np.array(chain_nodes)
                    chain_distances = np.array(chain_distances)
                    sorting_index = np.argsort(chain_distances)

                    # np arrays allows calling them by a list index
                    chain_nodes = chain_nodes[sorting_index]
                    chain_distances = chain_distances[sorting_index]
                    # print(f"{chain_nodes = }\t{chain_distances}")
                    accumilated_weight = 0
                    for seq in range(len(chain_nodes) - 1):
                        graph.add_edge(
                            int(chain_nodes[seq]),
                            int(chain_nodes[seq + 1]),
                            # TODO: change this to either defaults to distance or a specified column for a weight..
                            weight=max(chain_distances[seq + 1] - chain_distances[seq], 0),
                            ##avoiding small negative numbers due to numerical error when two nodes are superimposed.
                            id=edge_id
                        )
                        accumilated_weight += (chain_distances[seq + 1] - chain_distances[seq])
        
        # Removing nodes
        for node_idx in remove_nodes:
            node_idx = int(node_idx)
            if node_idx not in graph.nodes:
                print(f"attempting to remove node {node_idx} that's not in graph {str(graph)}")
                continue

            if len(graph.adj[node_idx]) != 2:
                print(f"attempting to remove a node {node_idx = } that's not degree 2, adjacent to: {graph.adj[node_idx]}")
                continue

            neighbors = list(graph.adj[node_idx])

            start = int(neighbors[0])
            end = int(neighbors[1])
            weight = graph.adj[node_idx][start]["weight"] \
                    + graph.adj[node_idx][end]["weight"]

            original_edge_id = self.nodes.at[node_idx, "nearest_edge_id"]

            # remove node after we got the attributes we needed..
            graph.remove_node(node_idx)
            graph.graph["added_nodes"].remove(node_idx)
            graph.add_edge(
                start,
                end,
                weight=weight,
                id=original_edge_id
            )
            
        return graph

    def _get_nodes_at_distance(self, origin_node, distance, method='geometric'):
        """
        Gets all nodes at distance `distance` from the origin `origin_idx` using the method `method`.
        Approximation of functionality from ln 496-592 of b_f.py

        Returns:
            The distance first-class function to be used to find nodes
        """
        # TODO: Copy over remainder of distance methods. What does 'trail_blazer' mean?
        method_dict = {
            'geometric': self._get_nodes_at_geometric_distance,
            'network': self._get_nodes_at_network_distance,
            'trail_blazer': self._get_nodes_at_bf_distance

        }
        return method_dict[method](origin_node, distance)

    def _get_nodes_at_geometric_distance(self, origin_node, distance):
        raise NotImplementedError

    def _get_nodes_at_network_distance(self, origin_node, distance):
        raise NotImplementedError

    def _get_nodes_at_bf_distance(self, origin_node, distance):
        raise NotImplementedError

    def scan_for_intersections(self, node_snapping_tolerance=1):
        raise NotImplementedError

    def fuse_degree_2_nodes(self, tolerance_angle):
        raise NotImplementedError

    def network_to_layer(self):
        return Layer('network_nodes', self.nodes, True, '', ''), Layer('network_edges', self.edges, True, '', '')


