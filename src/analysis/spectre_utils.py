###############################################################################
#
# Adapted from https://github.com/lrjconan/GRAN/ which in turn is adapted from https://github.com/JiaxuanYou/graph-generation
#
###############################################################################

import os
import torch.nn as nn
import numpy as np
import networkx as nx
import subprocess as sp
import concurrent.futures
try:
    import graph_tool.all as gt
except:
    print("Couldn't import graphtool, spectre utils won't work")
import secrets
from string import ascii_uppercase, digits
from datetime import datetime
from src.analysis.dist_helper import compute_mmd, gaussian_emd, gaussian
from torch_geometric.utils import to_networkx
import wandb

PRINT_TIME = False
__all__ = []

def degree_worker(G):
    return np.array(nx.degree_histogram(G))


def degree_stats(graph_ref_list, graph_pred_list):
    ''' Compute the distance between the degree distributions of two unordered sets of graphs.
        Args:
            graph_ref_list, graph_target_list: two lists of networkx graphs to be evaluated
        '''
    sample_ref = []
    sample_pred = []
    # in case an empty graph is generated
    graph_pred_list_remove_empty = [
        G for G in graph_pred_list if not G.number_of_nodes() == 0
    ]

    prev = datetime.now()
    with concurrent.futures.ThreadPoolExecutor() as executor:
        for deg_hist in executor.map(degree_worker, graph_ref_list):
            sample_ref.append(deg_hist)
    with concurrent.futures.ThreadPoolExecutor() as executor:
        for deg_hist in executor.map(degree_worker, graph_pred_list_remove_empty):
            sample_pred.append(deg_hist)

    # EMD option uses the same computation as GraphRNN, the alternative is MMD as computed by GRAN
    # mmd_dist = compute_mmd(sample_ref, sample_pred, kernel=emd)
    mmd_dist = compute_mmd(sample_ref, sample_pred, kernel=gaussian_emd)

    elapsed = datetime.now() - prev
    if PRINT_TIME:
        print('Time computing degree mmd: ', elapsed)
    return mmd_dist


###############################################################################

def clustering_worker(param):
    G, bins = param
    clustering_coeffs_list = list(nx.clustering(G).values())
    hist, _ = np.histogram(
        clustering_coeffs_list, bins=bins, range=(0.0, 1.0), density=False)
    return hist


def clustering_stats(graph_ref_list,
                     graph_pred_list,
                     bins=100):
    sample_ref = []
    sample_pred = []
    graph_pred_list_remove_empty = [
        G for G in graph_pred_list if not G.number_of_nodes() == 0
    ]

    prev = datetime.now()
    with concurrent.futures.ThreadPoolExecutor() as executor:
        for clustering_hist in executor.map(clustering_worker,
                                            [(G, bins) for G in graph_ref_list]):
            sample_ref.append(clustering_hist)
    with concurrent.futures.ThreadPoolExecutor() as executor:
        for clustering_hist in executor.map(
                clustering_worker, [(G, bins) for G in graph_pred_list_remove_empty]):
            sample_pred.append(clustering_hist)

    # EMD option uses the same computation as GraphRNN, the alternative is MMD as computed by GRAN
    # mmd_dist = compute_mmd(sample_ref, sample_pred, kernel=emd, sigma=1.0 / 10)
    mmd_dist = compute_mmd(sample_ref, sample_pred, kernel=gaussian_emd, sigma=1.0 / 10, distance_scaling=bins)

    elapsed = datetime.now() - prev
    if PRINT_TIME:
        print('Time computing clustering mmd: ', elapsed)
    return mmd_dist


# maps motif/orbit name string to its corresponding list of indices from orca output
COUNT_START_STR = 'orbit counts:'


def edge_list_reindexed(G):
    idx = 0
    id2idx = dict()
    for u in G.nodes():
        id2idx[str(u)] = idx
        idx += 1

    edges = []
    for (u, v) in G.edges():
        edges.append((id2idx[str(u)], id2idx[str(v)]))
    return edges


def orca(graph):
    tmp_fname = f'orca/tmp_{"".join(secrets.choice(ascii_uppercase + digits) for i in range(8))}.txt'
    tmp_fname = os.path.join(os.path.dirname(os.path.realpath(__file__)), tmp_fname)
    f = open(tmp_fname, 'w')
    f.write(
        str(graph.number_of_nodes()) + ' ' + str(graph.number_of_edges()) + '\n')
    for (u, v) in edge_list_reindexed(graph):
        f.write(str(u) + ' ' + str(v) + '\n')
    f.close()
    output = sp.check_output(
        [str(os.path.join(os.path.dirname(os.path.realpath(__file__)), 'orca/orca')), 'node', '4', tmp_fname, 'std'])
    output = output.decode('utf8').strip()
    idx = output.find(COUNT_START_STR) + len(COUNT_START_STR) + 2
    output = output[idx:]
    node_orbit_counts = np.array([
        list(map(int,
                 node_cnts.strip().split(' ')))
        for node_cnts in output.strip('\n').split('\n')
    ])

    try:
        os.remove(tmp_fname)
    except OSError:
        pass

    return node_orbit_counts


def orbit_stats_all(graph_ref_list, graph_pred_list):
    total_counts_ref = []
    total_counts_pred = []

    graph_pred_list_remove_empty = [
        G for G in graph_pred_list if not G.number_of_nodes() == 0
    ]

    for G in graph_ref_list:
        orbit_counts = orca(G)
        orbit_counts_graph = np.sum(orbit_counts, axis=0) / G.number_of_nodes()
        total_counts_ref.append(orbit_counts_graph)

    for G in graph_pred_list:
        orbit_counts = orca(G)
        orbit_counts_graph = np.sum(orbit_counts, axis=0) / G.number_of_nodes()
        total_counts_pred.append(orbit_counts_graph)

    total_counts_ref = np.array(total_counts_ref)
    total_counts_pred = np.array(total_counts_pred)

    # mmd_dist = compute_mmd(total_counts_ref, total_counts_pred, kernel=emd, sigma=30.0)
    # EMD option uses the same computation as GraphRNN, the alternative is MMD as computed by GRAN
    mmd_dist = compute_mmd(total_counts_ref, total_counts_pred, kernel=gaussian, is_hist=False, sigma=30.0)
    return mmd_dist


class Comm20SamplingMetrics(nn.Module):
    def __init__(self, test_loader):
        super().__init__()

        self.test_graphs = self.loader_to_nx(test_loader)

    def loader_to_nx(self, loader):
        networkx_graphs = []
        for batch in loader:
            # TODO: this does not run with current loader
            data_list = batch.to_data_list()
            for data in data_list:
                networkx_graphs.append(to_networkx(data, node_attrs=None, edge_attrs=None, to_undirected=True,
                                                   remove_self_loops=True))
        return networkx_graphs

    def forward(self, generated_graphs: list, name, current_epoch, val_counter, save_graphs=True, test=False):
        print(f"Computing sampling metrics between {len(generated_graphs)} generated graphs and {len(self.test_graphs)}"
              f" test graphs -- emd computation: True")
        networkx_graphs = []
        adjacency_matrices = []
        print("Building networkx graphs...")
        for graph in generated_graphs:
            node_types, edge_types = graph
            A = edge_types.bool().cpu().numpy()
            adjacency_matrices.append(A)

            nx_graph = nx.from_numpy_array(A)
            networkx_graphs.append(nx_graph)

        print("Saving all adjacency matrices")
        np.savez('generated_adjs.npz', *adjacency_matrices)

        # degree metric
        print("Computing degree stats..")
        degree = degree_stats(self.test_graphs, networkx_graphs, is_parallel=True)
        wandb.run.summary['degree'] = degree

        to_log = {}

        # 'clustering' metric
        print("Computing clustering stats...")
        clustering = clustering_stats(self.test_graphs, networkx_graphs, bins=100, is_parallel=True)
        to_log['clustering'] = clustering
        wandb.run.summary['clustering'] = clustering

        # orbit metric
        print("Computing orbit stats...")
        orbit = orbit_stats_all(self.test_graphs, networkx_graphs)
        to_log['orbit'] = orbit
        wandb.run.summary['orbit'] = orbit

        print("Sampling statistics", to_log)
        wandb.log(to_log, commit=False)

    def reset(self):
        pass
