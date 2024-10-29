# -*- coding: utf-8 -*-
import torch
import torch.nn.functional as F
from torch.utils.data.dataloader import default_collate
import torch_geometric.utils as utils
from torch_geometric.data import Data
import numpy as np
import os
from tqdm import tqdm
import nxmetis
import networkx as nx
from torch_geometric.utils.convert import to_networkx
import pygsp
from .coarsening_utils import coarsen
from .spectral_clustering import spectral_clustering

def trans_to_adj(graph):
    graph.remove_edges_from(nx.selfloop_edges(graph))
    nodes = range(len(graph.nodes))
    return np.array(nx.to_numpy_array(graph, nodelist=nodes))

def my_inc(self, key, value, *args, **kwargs):
    if key == 'subgraph_edge_index':
        return self.num_subgraph_nodes 
    if key == 'subgraph_node_idx':
        return self.num_nodes 
    if key == 'subgraph_indicator':
        return self.num_nodes 
    elif 'index' in key:
        return self.num_nodes
    else:
        return 0

import pickle
import matplotlib.pyplot as plt
from communities.algorithms import louvain_method,girvan_newman
from communities.visualization import draw_communities
class GraphDataset(object):
    def __init__(self, dataset, data_path, degree=False, k_hop=2, se="gnn", use_subgraph_edge_attr=False,
                 cache_path=None, return_complete_index=True, train='train', hdse = 0):
        self.dataset = dataset
        self.n_features = dataset[0].x.shape[-1]
        self.degree = degree
        self.hdse = hdse
        self.compute_degree()
        file_path = data_path+'/'+train+'_'+str(hdse)+'.pkl'
        print(file_path)
        print(os.path.exists(file_path))
        if(os.path.exists(file_path)):
            with open(file_path, 'rb') as f:
                self.dist_list, self.SPD_list = pickle.load(f)
        else:
            self.compute_dist()
            with open(file_path, 'wb') as f:
                pickle.dump((self.dist_list, self.SPD_list), f, pickle.HIGHEST_PROTOCOL)
        self.abs_pe_list = None
        self.return_complete_index = return_complete_index
        self.k_hop = k_hop
        self.se = se
        self.use_subgraph_edge_attr = use_subgraph_edge_attr
        self.cache_path = cache_path
        if self.se == 'khopgnn':
            Data.__inc__ = my_inc
            self.extract_subgraphs()
    
    def compute_dist(self):
        self.dist_list = []
        self.SPD_list = []
        idx = -1
        for data in tqdm(self.dataset):
            idx += 1
            N = data.x.shape[0]
            Maxnode = 300
            # flag = 0
            if(data.x.shape[0]>=Maxnode):
                print(data)
                raise Exception("Maxnode exceed")

            G = to_networkx(data, to_undirected=True)
            # deg = 0
            # for i in range(G.number_of_nodes()):
            #     if(G.degree(i)==0):
            #         deg+=1
            # if(deg > 5):
            #     flag = 1
            Maxdis = 10
            Maxspd = 30
            SPD = [[Maxspd for j in range(Maxnode)] for i in range(Maxnode)]
            dist = [[Maxdis for j in range(Maxnode)] for i in range(Maxnode)]
            G_all = G
            if(self.hdse==0):
                _, communities = nxmetis.partition(G, nparts=5)
                # print(communities)
            #Spectral clustering
            elif(self.hdse==1):
                # print(G.nodes(),G.edges())
                adj_matrix = nx.to_numpy_array(G_all)
                communities = spectral_clustering(adj_matrix, 5)
                # print(communities)
            #Spectral-cut
            elif(self.hdse==2):
                graph = pygsp.graphs.Graph(nx.adjacency_matrix(G).todense())
                C, Gc, _, _ = coarsen(graph, K=10, r=0.9, method='algebraic_JC')
                dense_matrix = C.toarray()
                row_indices, col_indices = dense_matrix.nonzero()
                max_cluster = np.max(row_indices) + 1
                cluster_list = [[] for _ in range(max_cluster)]
                for i in range(len(row_indices)):
                    cluster = col_indices[i]
                    point = row_indices[i]
                    cluster_list[point].append(cluster)
                communities = cluster_list
            #Girvan-Newman
            elif(self.hdse==3):
                adj_matrix = nx.to_numpy_array(G_all)
                communities, _ = girvan_newman(adj_matrix)
            #Louvains Method
            elif(self.hdse==4):
                adj_matrix = nx.to_numpy_array(G_all)
                communities, _ = louvain_method(adj_matrix)
            # communities, _ = louvain_method(adj_matrix)
            M = nx.quotient_graph(G_all, communities, relabel=True)
            # if(len(communities)>10):
            #     print(len(communities))
            #     raise Exception("communities exceed")
            dict_graph = {}
            for i in range(len(communities)):
                for j in communities[i]:
                    dict_graph[j] = i

            length = dict(nx.all_pairs_shortest_path_length(G_all))
            # print(length)
            for i in range(N):
                for j in range(N):
                    if(j in length[i]):
                        SPD[i][j] = length[i][j]
                        if(SPD[i][j]>=Maxspd):
                            SPD[i][j] = Maxspd

            G = M
            length = dict(nx.all_pairs_shortest_path_length(G))
            for i in range(N):
                for j in range(N):
                    if(dict_graph[j] in length[dict_graph[i]]):
                        dist[i][j] = length[dict_graph[i]][dict_graph[j]]
                        if(dist[i][j]>=Maxdis):
                            dist[i][j] = Maxdis
            dist = torch.tensor(dist).long()
            SPD = torch.tensor(SPD).long()
            complete_edge_index_dist = dist[:N,:N]
            complete_edge_index_dist = complete_edge_index_dist.reshape(-1)
            complete_edge_index_SPD = SPD[:N,:N]
            complete_edge_index_SPD = complete_edge_index_SPD.reshape(-1)
            self.SPD_list.append(complete_edge_index_SPD)
            self.dist_list.append(complete_edge_index_dist)

    def compute_degree(self):
        if not self.degree:
            self.degree_list = None
            return
        self.degree_list = []
        for g in self.dataset:
            deg = 1. / torch.sqrt(1. + utils.degree(g.edge_index[0], g.num_nodes))
            self.degree_list.append(deg)

    def extract_subgraphs(self):
        print("Extracting {}-hop subgraphs...".format(self.k_hop))
        # indicate which node in a graph it is; for each graph, the
        # indices will range from (0, num_nodes). PyTorch will then
        # increment this according to the batch size
        self.subgraph_node_index = []

        # Each graph will become a block diagonal adjacency matrix of
        # all the k-hop subgraphs centered around each node. The edge
        # indices get augumented within a given graph to make this
        # happen (and later are augmented for proper batching)
        self.subgraph_edge_index = []

        # This identifies which indices correspond to which subgraph
        # (i.e. which node in a graph)
        self.subgraph_indicator_index = []

        # This gets the edge attributes for the new indices
        if self.use_subgraph_edge_attr:
            self.subgraph_edge_attr = []

        for i in range(len(self.dataset)):
            if self.cache_path is not None:
                filepath = "{}_{}.pt".format(self.cache_path, i)
                if os.path.exists(filepath):
                    continue
            graph = self.dataset[i]
            node_indices = []
            edge_indices = []
            edge_attributes = []
            indicators = []
            edge_index_start = 0

            for node_idx in range(graph.num_nodes):
                sub_nodes, sub_edge_index, _, edge_mask = utils.k_hop_subgraph(
                    node_idx, 
                    self.k_hop, 
                    graph.edge_index,
                    relabel_nodes=True, 
                    num_nodes=graph.num_nodes
                    )
                node_indices.append(sub_nodes)
                edge_indices.append(sub_edge_index + edge_index_start)
                indicators.append(torch.zeros(sub_nodes.shape[0]).fill_(node_idx))
                if self.use_subgraph_edge_attr and graph.edge_attr is not None:
                    edge_attributes.append(graph.edge_attr[edge_mask]) # CHECK THIS DIDN"T BREAK ANYTHING
                edge_index_start += len(sub_nodes)

            if self.cache_path is not None:
                if self.use_subgraph_edge_attr and graph.edge_attr is not None:
                    subgraph_edge_attr = torch.cat(edge_attributes)
                else:
                    subgraph_edge_attr = None
                torch.save({
                    'subgraph_node_index': torch.cat(node_indices),
                    'subgraph_edge_index': torch.cat(edge_indices, dim=1),
                    'subgraph_indicator_index': torch.cat(indicators).type(torch.LongTensor),
                    'subgraph_edge_attr': subgraph_edge_attr
                }, filepath)
            else:
                self.subgraph_node_index.append(torch.cat(node_indices))
                self.subgraph_edge_index.append(torch.cat(edge_indices, dim=1))
                self.subgraph_indicator_index.append(torch.cat(indicators))
                if self.use_subgraph_edge_attr and graph.edge_attr is not None:
                    self.subgraph_edge_attr.append(torch.cat(edge_attributes))
        print("Done!")

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        data = self.dataset[index]
        dist = self.dist_list[index]
        SPD = self.SPD_list[index]
        if self.n_features == 1:
            data.x = data.x.squeeze(-1)
        if not isinstance(data.y, list):
            data.y = data.y.view(data.y.shape[0], -1)
        n = data.num_nodes
        s = torch.arange(n)
        if self.return_complete_index:
            data.complete_edge_index = torch.vstack((s.repeat_interleave(n), s.repeat(n)))
        # print(s.repeat(n))
        # print(s.repeat_interleave(n))
        data.complete_edge_dist = dist
        data.complete_edge_SPD = SPD
        data.degree = None
        if self.degree:
            data.degree = self.degree_list[index]
        data.abs_pe = None
        if self.abs_pe_list is not None and len(self.abs_pe_list) == len(self.dataset):
            data.abs_pe = self.abs_pe_list[index]
         
        # add subgraphs and relevant meta data
        if self.se == "khopgnn":
            if self.cache_path is not None:
                cache_file = torch.load("{}_{}.pt".format(self.cache_path, index))
                data.subgraph_edge_index = cache_file['subgraph_edge_index']
                data.num_subgraph_nodes = len(cache_file['subgraph_node_index'])
                data.subgraph_node_idx = cache_file['subgraph_node_index']
                data.subgraph_edge_attr = cache_file['subgraph_edge_attr']
                data.subgraph_indicator = cache_file['subgraph_indicator_index']
                return data
            data.subgraph_edge_index = self.subgraph_edge_index[index]
            data.num_subgraph_nodes = len(self.subgraph_node_index[index])
            data.subgraph_node_idx = self.subgraph_node_index[index]
            if self.use_subgraph_edge_attr and data.edge_attr is not None:
                data.subgraph_edge_attr = self.subgraph_edge_attr[index]
            data.subgraph_indicator = self.subgraph_indicator_index[index].type(torch.LongTensor)
        else:
            data.num_subgraph_nodes = None
            data.subgraph_node_idx = None
            data.subgraph_edge_index = None
            data.subgraph_indicator = None

        return data
