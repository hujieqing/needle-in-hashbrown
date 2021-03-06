import torch
import networkx as nx
import numpy as np
import multiprocessing as mp
import random
import mmh3
import graph

from typing import Iterable, Dict, List
from scipy.stats import kendalltau


# approximate
def get_edge_mask_link_negative_approximate(mask_link_positive, num_nodes, num_negtive_edges):
    links_temp = np.zeros((num_nodes, num_nodes)) + np.identity(num_nodes)
    mask_link_positive = duplicate_edges(mask_link_positive)
    links_temp[mask_link_positive[0],mask_link_positive[1]] = 1
    # add random noise
    links_temp += np.random.rand(num_nodes,num_nodes)
    prob = num_negtive_edges / (num_nodes*num_nodes-mask_link_positive.shape[1])
    mask_link_negative = np.stack(np.nonzero(links_temp<prob))
    return mask_link_negative


# exact version, slower
def get_edge_mask_link_negative(mask_link_positive, num_nodes, num_negtive_edges):
    mask_link_positive_set = []
    for i in range(mask_link_positive.shape[1]):
        mask_link_positive_set.append(tuple(mask_link_positive[:,i]))
    mask_link_positive_set = set(mask_link_positive_set)

    mask_link_negative = np.zeros((2,num_negtive_edges), dtype=mask_link_positive.dtype)
    for i in range(num_negtive_edges):
        while True:
            mask_temp = tuple(np.random.choice(num_nodes,size=(2,),replace=False))
            if mask_temp not in mask_link_positive_set:
                mask_link_negative[:,i] = mask_temp
                break

    return mask_link_negative


def resample_edge_mask_link_negative(data):
    data.mask_link_negative_train = get_edge_mask_link_negative(data.mask_link_positive_train, num_nodes=data.num_nodes,
                                                                num_negtive_edges=data.mask_link_positive_train.shape[1]
                                                                )
    data.mask_link_negative_val = get_edge_mask_link_negative(data.mask_link_positive, num_nodes=data.num_nodes,
                                                              num_negtive_edges=data.mask_link_positive_val.shape[1])
    data.mask_link_negative_test = get_edge_mask_link_negative(data.mask_link_positive, num_nodes=data.num_nodes,
                                                               num_negtive_edges=data.mask_link_positive_test.shape[1])


def deduplicate_edges(edges):
    edges_new = np.zeros((2, edges.shape[1]//2), dtype=int)
    # add none self edge
    j = 0
    skip_node = {}
    # node already put into result
    for i in range(edges.shape[1]):
        if edges[0, i] < edges[1, i]:
            edges_new[:, j] = edges[:, i]
            j += 1
        elif edges[0, i] == edges[1, i] and edges[0, i] not in skip_node:
            edges_new[:, j] = edges[:, i]
            skip_node.add(edges[0, i])
            j += 1

    return edges_new


def duplicate_edges(edges):
    return np.concatenate((edges, edges[::-1, :]), axis=-1)


# each node at least remain in the new graph
def split_edges(edges, remove_ratio, connected=False):
    e = edges.shape[1]
    edges = edges[:, np.random.permutation(e)]
    if connected:
        unique, counts = np.unique(edges, return_counts=True)
        node_count = dict(zip(unique, counts))

        index_train = []
        index_val = []
        for i in range(e):
            node1 = edges[0,i]
            node2 = edges[1,i]
            if node_count[node1]>1 and node_count[node2] > 1:
                # if degree>1
                index_val.append(i)
                node_count[node1] -= 1
                node_count[node2] -= 1
                if len(index_val) == int(e * remove_ratio):
                    break
            else:
                index_train.append(i)
        index_train = index_train + list(range(i + 1, e))
        index_test = index_val[:len(index_val)//2]
        index_val = index_val[len(index_val)//2:]

        edges_train = edges[:, index_train]
        edges_val = edges[:, index_val]
        edges_test = edges[:, index_test]
    else:
        split1 = int((1-remove_ratio)*e)
        split2 = int((1-remove_ratio/2)*e)
        edges_train = edges[:,:split1]
        edges_val = edges[:,split1:split2]
        edges_test = edges[:,split2:]

    return edges_train, edges_val, edges_test


def edge_to_set(edges):
    edge_set = []
    for i in range(edges.shape[1]):
        edge_set.append(tuple(edges[:, i]))
    edge_set = set(edge_set)
    return edge_set


def get_link_mask(data, remove_ratio=0.2, resplit=True, infer_link_positive=True):
    if resplit:
        if infer_link_positive:
            data.mask_link_positive = deduplicate_edges(data.edge_index.numpy())
        data.mask_link_positive_train, data.mask_link_positive_val, data.mask_link_positive_test = \
            split_edges(data.mask_link_positive, remove_ratio)
    resample_edge_mask_link_negative(data)


def add_nx_graph(data):
    G = nx.Graph()
    edge_numpy = data.edge_index.numpy()
    edge_list = []
    for i in range(data.num_edges):
        edge_list.append(tuple(edge_numpy[:, i]))
    G.add_edges_from(edge_list)
    data.G = G


def single_source_shortest_path_length_range(graph, node_range, cutoff):
    dists_dict = {}
    for node in node_range:
        dists_dict[node] = nx.single_source_shortest_path_length(graph, node, cutoff)
    return dists_dict


def merge_dicts(dicts):
    result = {}
    for dictionary in dicts:
        result.update(dictionary)
    return result


def all_pairs_shortest_path_length_parallel(graph, cutoff=None, num_workers=4):
    nodes = list(graph.nodes)
    random.shuffle(nodes)
    if len(nodes) < 50:
        num_workers = int(num_workers/4)
    elif len(nodes) < 400:
        num_workers = int(num_workers/2)

    pool = mp.Pool(processes=num_workers)
    results = [pool.apply_async(single_source_shortest_path_length_range,
                                args=(graph, nodes[int(len(nodes)/num_workers*i):int(len(nodes)/num_workers*(i+1))],
                                      cutoff)) for i in range(num_workers)]
    output = [p.get() for p in results]
    dists_dict = merge_dicts(output)
    pool.close()
    pool.join()
    return dists_dict


def precompute_dist_data(edge_index, num_nodes, approximate=0):
        '''
        Here dist is 1/real_dist, higher actually means closer, 0 means disconnected
        :return:
        '''
        graph = nx.Graph()
        edge_list = edge_index.transpose(1, 0).tolist()
        graph.add_edges_from(edge_list)

        n = num_nodes
        dists_array = np.zeros((n, n))
        # dists_dict = nx.all_pairs_shortest_path_length(graph,cutoff=approximate if approximate>0 else None)
        # dists_dict = {c[0]: c[1] for c in dists_dict}
        dists_dict = all_pairs_shortest_path_length_parallel(graph, cutoff=approximate if approximate > 0 else None)
        for i, node_i in enumerate(graph.nodes()):
            shortest_dist = dists_dict[node_i]
            for j, node_j in enumerate(graph.nodes()):
                dist = shortest_dist.get(node_j, -1)
                if dist != -1:
                    # dists_array[i, j] = 1 / (dist + 1)
                    dists_array[node_i, node_j] = 1 / (dist + 1)
        return dists_array


def get_random_anchorset(n, c=0.5):
    m = int(np.log2(n))
    copy = int(c*m)
    anchorset_id = []
    for i in range(m):
        anchor_size = int(n/np.exp2(i + 1))
        for j in range(copy):
            anchorset_id.append(np.random.choice(n,size=anchor_size,replace=False))
    return anchorset_id


def get_dist_max(anchorset_id, dist, device):
    dist_max = torch.zeros((dist.shape[0], len(anchorset_id))).to(device)
    dist_argmax = torch.zeros((dist.shape[0], len(anchorset_id))).long().to(device)
    for i in range(len(anchorset_id)):
        temp_id = anchorset_id[i]
        dist_temp = dist[:, temp_id]
        dist_max_temp, dist_argmax_temp = torch.max(dist_temp, dim=-1)
        dist_max[:,i] = dist_max_temp
        dist_argmax[:,i] = dist_argmax_temp
    return dist_max, dist_argmax


def preselect_anchor(data, layer_num=1, anchor_num=32, anchor_size_num=4, device='cpu'):

    data.anchor_size_num = anchor_size_num
    data.anchor_set = []
    anchor_num_per_size = anchor_num//anchor_size_num
    for i in range(anchor_size_num):
        anchor_size = 2**(i+1)-1
        anchors = np.random.choice(data.num_nodes, size=(layer_num, anchor_num_per_size, anchor_size), replace=True)
        data.anchor_set.append(anchors)
    data.anchor_set_indicator = np.zeros((layer_num, anchor_num, data.num_nodes), dtype=int)

    anchorset_id = get_random_anchorset(data.num_nodes, c=1)
    data.dists_max, data.dists_argmax = get_dist_max(anchorset_id, data.dists, device)


def int_to_hash_vector(input, n):
    """
    This function converts an input integer to a vector of dimension n.
    Each element in the vector is normalized from -1 to 1.
    """
    max32 = pow(2, 31)-1
    result = []
    last = input
    for i in range(n):
        a = mmh3.hash(str(last))
        result.append(a / max32)
        last = a
    return result


def avg_metric_at_k(relevance_scores: Dict[int, Iterable[float]], k: int, metric, method=0):
    dcgs = []
    for entity_id, relevance_scores_for_entity in relevance_scores.items():
        dcgs.append(metric(relevance_scores_for_entity, k, method))

    return np.mean(dcgs)


def avg_dcg_at_k(relevance_scores: Dict[int, Iterable[float]], k: int, method=0):
    return avg_metric_at_k(relevance_scores, k, dcg_at_k, method)


def avg_ndcg_at_k(relevance_scores: Dict[int, Iterable[float]], k: int, method=0):
    return avg_metric_at_k(relevance_scores, k, ndcg_at_k, method)


def dcg_at_k(relevance_scores: Iterable[float], k: int, method=0):

    """Score is discounted cumulative gain (dcg)
    Relevance is positive real values.  Can use binary
    as the previous methods.

    Args:
        relevance_scores: Relevance scores (list or numpy) in rank order
            (first element is the first item)
        k: Number of results to consider
        method: If 0 then weights are [1.0, 1.0, 0.6309, 0.5, 0.4307, ...]
                If 1 then weights are [1.0, 0.6309, 0.5, 0.4307, ...]
    Returns:
        Discounted cumulative gain
    """
    relevance_scores = np.asfarray(relevance_scores)[:k]
    if relevance_scores.size:
        if method == 0:
            return relevance_scores[0] + np.sum(relevance_scores[1:] / np.log2(np.arange(2, relevance_scores.size + 1)))
        elif method == 1:
            return np.sum(relevance_scores / np.log2(np.arange(2, relevance_scores.size + 2)))
        else:
            raise ValueError('method must be 0 or 1.')
    return 0.


def ndcg_at_k(relevance_scores: Iterable[float], k: int, method=0):

    """Score is normalized discounted cumulative gain (ndcg)
    Relevance is positive real values.  Can use binary
    as the previous methods.
    Example from
    http://www.stanford.edu/class/cs276/handouts/EvaluationNew-handout-6-per.pdf

    Args:
        relevance_scores: Relevance scores (list or numpy) in rank order
            (first element is the first item)
        k: Number of results to consider
        method: If 0 then weights are [1.0, 1.0, 0.6309, 0.5, 0.4307, ...]
                If 1 then weights are [1.0, 0.6309, 0.5, 0.4307, ...]
    Returns:
        Normalized discounted cumulative gain
    """
    dcg_max = dcg_at_k(sorted(relevance_scores, reverse=True), k, method)
    if not dcg_max:
        return 0.
    return dcg_at_k(relevance_scores, k, method) / dcg_max


def avg_kendall_tau(dict_to_consider: Dict[int, List[int]], baseline_dict: Dict[int, List[int]]):

    """Calculates Kendall’s tau, a correlation measure for ordinal data.
    Kendall’s tau is a measure of the correspondence between two rankings.
    Values close to 1 indicate strong agreement, values close to -1 indicate strong disagreement.
    This is the tau-b version of Kendall’s tau which accounts for ties."""

    avg_kt_val = 0.0
    count = 0

    entity_ids = set(list(dict_to_consider.keys())) | set(list(baseline_dict.keys()))

    for entity_id in entity_ids:
        consider_ranks = dict_to_consider.get(entity_id)
        ground_truth_ranks = baseline_dict.get(entity_id)
        if consider_ranks is not None and ground_truth_ranks is not None:
            count += 1
            k = min(len(consider_ranks), len(ground_truth_ranks))
            avg_kt_val += kendalltau(consider_ranks[:k], ground_truth_ranks[:k])

    if count > 0:
        avg_kt_val = avg_kt_val / count

    return avg_kt_val


def precision_for_entity(true_list: List, candidate_list: List) -> float:
    true_set = set(true_list)
    candidate_set = set(candidate_list)
    return len(candidate_set & true_set) / len(candidate_set)


def recall_for_entity(true_list: List, candidate_list: List) -> float:
    true_set = set(true_list)
    candidate_set = set(candidate_list)
    return len(candidate_set & true_set) / len(true_set)


def graph_shortest_paths(sims: Dict[int, List[int]], other_graph: graph.Graph):
    paths = []
    for from_node, to_nodes in sims.items():
        from_node_lengths = []
        for to_node in to_nodes:
            distance = other_graph.shortest_path(from_node, to_node, no_path_value=None)
            from_node_lengths.append(distance)
        paths.append(from_node_lengths)
    return paths


def average_path_metrics(sims: Dict[int, List[int]], candidate_graph: graph.Graph, weights: Dict[int, float]):
    paths = graph_shortest_paths(sims, candidate_graph)
    weights = np.array([weights[show] if show in weights else 0 for show, _ in sims.items()])
    weights /= weights.sum()
    mean_distance_for_connected = []
    frac_connected = []
    for paths_for_entity in paths:
        connected_distances = list(filter(lambda x: x is not None, paths_for_entity))
        frac_connected.append(len(connected_distances) / len(paths_for_entity))
        if connected_distances:
            mean_distance_for_connected.append(np.mean(connected_distances))
        else:
            mean_distance_for_connected.append(-1)

    mean_distance_for_connected = np.array(mean_distance_for_connected)
    connected_mask = mean_distance_for_connected >= 0
    mean_distance_for_connected = np.sum(
        mean_distance_for_connected[connected_mask] * weights[connected_mask]
    ) / np.sum(weights[connected_mask])
    frac_connected = np.array(frac_connected)
    frac_connected = np.sum(frac_connected * weights)

    return mean_distance_for_connected, frac_connected


def extract_edge_distances(distances, edges, alpha):
    edge_distances = []
    for row in range(edges.shape[0]):
        src_node = edges[row][0]
        dest_node = edges[row][1]
        edge_distances.append(pow(distances[src_node, dest_node], alpha))
    return np.asarray(edge_distances)
