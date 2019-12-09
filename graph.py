from typing import List, Tuple, Dict, Any

import numpy as np
import pandas as pd
import networkx


class Graph(object):

    def __init__(
            self,
            nodes: List[int],
            edges: List[Tuple[int, int]] = None,
            weighted_edges: List[Tuple[int, int, float]] = None,
    ):
        if weighted_edges:
            self.graph = networkx.DiGraph()
        else:
            self.graph = networkx.Graph()
        self.graph.add_nodes_from(nodes)
        if weighted_edges:
            self.graph.add_weighted_edges_from(edges)
        else:
            self.graph.add_edges_from(edges)

        self.diameter = None

    def set_diameter(self):
        connected_components = sorted(networkx.connected_components(self.graph), key=lambda x: len(x))
        sub_graph = self.graph.subgraph(connected_components[-1])
        sample_nodes = np.random.choice(list(connected_components[-1]), 100)
        lengths = []
        for node in sample_nodes:
            lengths.append(np.max(list(networkx.shortest_path_length(sub_graph, node).values())))
        self.diameter = np.mean(lengths) + 2 * np.std(lengths)

    def set_names(self, node_names: Dict[int, str]):
        node_names_dict = {}
        for node, name in node_names.items():
            node_names_dict[node] = {'name': name}
        networkx.set_node_attributes(self.graph, node_names_dict)

    @staticmethod
    def from_dict_of_lists(input_dict: Dict[int, List[int]]):
        return Graph.from_dict_of_dicts({k: {item: None for item in l} for k, l in input_dict.items()})

    @staticmethod
    def from_dict_of_dicts(input_dict: Dict[int, Dict[int, Any]]):
        nodes = list(input_dict.keys())
        edges = []
        for from_node, sims_dict in input_dict.items():
            to_nodes = list(sims_dict.keys())
            nodes += to_nodes
            for to_node in to_nodes:
                edges.append((from_node, to_node))
        nodes = list(set(nodes))
        return Graph(nodes, edges=edges)

    @staticmethod
    def from_df(df: pd.DataFrame, from_col='C1', to_col='C2', weight_col='C3', directed=False) -> 'Graph':
        nodes = list(set(df[from_col].unique().tolist() + df[to_col].unique().tolist()))
        edges = []
        for row in df.itertuples():
            if directed:
                edges.append((getattr(row, from_col), getattr(row, to_col), getattr(row, weight_col)))
            else:
                edges.append((getattr(row, from_col), getattr(row, to_col)))

        if directed:
            return Graph(nodes, weighted_edges=edges)
        else:
            return Graph(nodes, edges=edges)

    def shortest_path(self, from_id, to_id, no_path_value=None):
        try:
            if self.graph.has_node(from_id) and self.graph.has_node(to_id):
                if self.graph.is_directed():
                    return networkx.shortest_path_length(self.graph, from_id, to_id, weight='weight')
                else:
                    return networkx.shortest_path_length(self.graph, from_id, to_id)
        except networkx.NetworkXNoPath:
            pass
        return no_path_value
