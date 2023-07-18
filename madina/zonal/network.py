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
    A network data structure composed of weighted nodes and edges that can be used to carry out
    network operations like betweenness and elastic weight calculation

    Parameters
    ----------
    nodes: GeoDataFrame
        The nodes in the network
    edges: GeoDataFrame
        The edges in the network
    projected_crs: str
        The CRS that the nodes and edges are in
    turn_threshold_degree: float, defaults to 45
        The threshold of a turn to be considered penalizable
    turn_penalty_amount: float, defaults to 30
        The penalty added to a turn
    weight_attribute: 
        The attribute name of the layer that was used as the edge weight in the created 
        network. If None, the geometric length of the edges was used
    """

    def __init__(self, 
                 nodes: GeoDataFrame, 
                 edges: GeoDataFrame, 
                 projected_crs: str, 
                 turn_threshold_degree: float = 45, 
                 turn_penalty_amount: float = 30, 
                 weight_attribute: str = None
                 ):

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
    
    def set_node_value(self, idx: int, label: str, new_value):
        """
        Sets the node at (`idx`, `label`) value in the network to `new_value`.

        Parameters
        ----------
        idx: int
            The index of the node to set value
        label: str
            The label of the given node to set to the new value
        new_value: any
            The new value to set to, raises exception if the type does not match with old label
        """
        self.nodes.at[idx, label] = new_value
        return

    def create_graph(self, light_graph=False, d_graph=True, od_graph=False):
        """
        Creates the corresponding graphs in the Network object based on the current nodes and edges
        `light_graph` - contains only network nodes and edges
        `d_graph` - contains all destination nodes
        `od_graph` - contains all origin and destination nodes

        Parameters
        ----------
        light_graph: bool, defaults to False
            if true, create `self.light_graph`
        d_graph: bool, defaults to True
            if true, create `self.d_graph`
        od_graph: bool, defaults to False
            if true, create `self.od_graph`
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

        Warnings
        --------
        Not Implemented Yet
        """
        raise NotImplementedError

    def update_light_graph(self, graph: nx.Graph, add_nodes: list = [], remove_nodes: list = []):
        """
        Updates the given graph object by adding nodes to and removing nodes from it.

        Parameters
        ----------
        graph: networkx.Graph
            The given networkx Graph object to be edited
        add_nodes: list, defaults to an empty list
            a list of nodes to be added
        remove_nodes: list, defaults to an empty list
            a list of nodes to be removed
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
