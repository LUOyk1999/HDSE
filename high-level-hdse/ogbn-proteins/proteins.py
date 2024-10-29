import argparse

import torch
import torch.nn.functional as F
import os
import torch_geometric.transforms as T
from torch_geometric.nn import GCNConv, SAGEConv

from ogb.nodeproppred import PygNodePropPredDataset, Evaluator

from logger import Logger
import numpy as np
from model import Transformer
import nxmetis
import networkx as nx
import numpy as np
from torch_geometric.utils import to_undirected
import pickle


def train(model, data, train_idx, optimizer):
    model.train()
    criterion = torch.nn.BCEWithLogitsLoss()
    optimizer.zero_grad()
    out = model.global_forward(data.x, data.adj_t, data.distance_matrix, data.nodes_to_community_tensor)[train_idx]
    loss = criterion(out, data.y[train_idx].to(torch.float))
    loss.backward()
    optimizer.step()

    return loss.item()


@torch.no_grad()
def test(model, data, split_idx, evaluator):
    model.eval()

    out = model.global_forward(data.x, data.adj_t, data.distance_matrix, data.nodes_to_community_tensor)
    train_acc = evaluator.eval({
        'y_true': data.y[split_idx['train']],
        'y_pred': out[split_idx['train']],
    })['rocauc']
    valid_acc = evaluator.eval({
        'y_true': data.y[split_idx['valid']],
        'y_pred': out[split_idx['valid']],
    })['rocauc']
    test_acc = evaluator.eval({
        'y_true': data.y[split_idx['test']],
        'y_pred': out[split_idx['test']],
    })['rocauc']

    return train_acc, valid_acc, test_acc


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', type=int, default=0)
    parser.add_argument('--eval_steps', type=int, default=5)
    parser.add_argument('--log_steps', type=int, default=1)
    parser.add_argument('--use_sage', action='store_true')
    parser.add_argument('--gnum_layers', type=int, default=3)
    parser.add_argument('--ghidden_channels', type=int, default=256)
    parser.add_argument('--gdropout', type=float, default=0.5)
    parser.add_argument('--lr', type=float, default=0.0005)
    parser.add_argument('--epochs', type=int, default=1000)
    parser.add_argument('--runs', type=int, default=2)

    parser.add_argument('--num_layers', type=int, default=1)
    parser.add_argument('--hidden_channels', type=int, default=128)
    parser.add_argument('--dropout', type=float, default=0.5)
    
    parser.add_argument('--num_heads', type=int, default=1) 
    parser.add_argument('--attn_dropout', type=float, default=0.5)
    parser.add_argument('--ff_dropout', type=float, default=0.5)
    parser.add_argument('--num_centroids', type=int, default=1024) 
    parser.add_argument('--no_bn', action='store_true')
    parser.add_argument('--norm_type', type=str, default='batch_norm')
    args = parser.parse_args()
    print(args)
    args.global_dim = args.hidden_channels

    device = f'cuda:{args.device}' if torch.cuda.is_available() else 'cpu'
    device = torch.device(device)


    dataset = PygNodePropPredDataset(
        name='ogbn-proteins', transform=T.ToSparseTensor(attr='edge_attr'))
    data = dataset[0]
    
    # Move edge features to node features.
    data.x = data.adj_t.mean(dim=1)
    data.adj_t.set_value_(None)
    
    print(data.num_features,data.x.shape)
    
    split_idx = dataset.get_idx_split()
    train_idx = split_idx['train'].to(device)

    print(f'metis begins: ./distance_{args.num_centroids}_proteins.pkl')
    if os.path.exists(f'./distance_{args.num_centroids}_proteins.pkl'):
        with open(f'./distance_{args.num_centroids}_proteins.pkl', 'rb') as f:
            data.distance_matrix, data.nodes_to_community_tensor = pickle.load(f)
    else:
        dataset_ = PygNodePropPredDataset(name='ogbn-proteins')
        data_ = dataset_[0]
       
        n = data_.num_nodes
        nx_G = nx.Graph()
        node_list = list(range(n)) 
        nx_G.add_nodes_from(node_list)
        edge_list = list(zip(data_.edge_index[0].numpy(), data_.edge_index[1].numpy()))
        nx_G.add_edges_from(edge_list)
        print(nx_G.number_of_nodes())
        a_values = [args.num_centroids]
        nodes_to_community_tensors = {}
        for a in a_values:
            _, communities = nxmetis.partition(nx_G, nparts=a)
            print(len(communities))
            nodes_to_community_tensor = torch.zeros(n, dtype=torch.long)
            # print(nodes_to_community_tensor)
            for i, community in enumerate(communities):
                for node in community:
                    nodes_to_community_tensor[node] = i
            nodes_to_community_tensors[a] = nodes_to_community_tensor     
        # print(nodes_to_community_tensors[args.num_centroids],nodes_to_community_tensors[args.num_centroids].shape)
        super_G = nx.Graph()
        # print(nodes_to_community_tensor)
        community_to_nodes = {}
        for node, community in enumerate(nodes_to_community_tensor):
            if community.item() not in community_to_nodes:
                # print(community.item())
                community_to_nodes[community.item()] = [node]
            else:
                community_to_nodes[community.item()].append(node)

        super_G.add_nodes_from(list(range(args.num_centroids)))
        print(super_G.number_of_nodes())
        for u, v in nx_G.edges():
            community_u = nodes_to_community_tensor[u].item()
            community_v = nodes_to_community_tensor[v].item()
            if community_u != community_v:
                super_G.add_edge(community_u, community_v)

        print(super_G.number_of_nodes(),super_G.number_of_edges())

        super_G_distances = dict(nx.all_pairs_shortest_path_length(super_G))
        # print(super_G_distances)

        distance_matrix = np.zeros((nx_G.number_of_nodes(), super_G.number_of_nodes()))+30
        for node in nx_G.nodes():
            super_node = nodes_to_community_tensor[node].item()
            for super_node_target in super_G.nodes():
                if(super_node_target in super_G_distances[super_node]):
                    distance_matrix[node, super_node_target] = super_G_distances[super_node][super_node_target]

        data.distance_matrix = torch.tensor(distance_matrix)
        data.nodes_to_community_tensor = nodes_to_community_tensor
        
        with open(f'./distance_{args.num_centroids}_proteins.pkl', 'wb') as f:
            pickle.dump((data.distance_matrix,data.nodes_to_community_tensor), f, pickle.HIGHEST_PROTOCOL)
    print(f'metis ends: ./distance_{args.num_centroids}_proteins.pkl')
    data = data.to(device)
    model = Transformer(
        num_nodes=data.num_nodes,
        in_channels=data.num_features,
        hidden_channels=args.hidden_channels, 
        out_channels=112,
        global_dim=args.global_dim,
        num_layers=args.num_layers,
        heads=args.num_heads,
        ff_dropout=args.ff_dropout,
        attn_dropout=args.attn_dropout,
        num_centroids=args.num_centroids,
        no_bn=args.no_bn,
        norm_type=args.norm_type,
        gnum_layers=args.gnum_layers,
        ghidden_channels=args.ghidden_channels,
        gdropout=args.gdropout,
        gcn=0
    ).to(device)

    evaluator = Evaluator(name='ogbn-proteins')
    logger = Logger(args.runs, args)

    for run in range(args.runs):
        model.reset_parameters()
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
        for epoch in range(1, 1 + args.epochs):
            loss = train(model, data, train_idx, optimizer)
            if epoch % args.eval_steps == 0:
                result = test(model, data, split_idx, evaluator)
                logger.add_result(run, result)

                if epoch % args.log_steps == 0:
                    train_acc, valid_acc, test_acc = result
                    print(f'Run: {run + 1:02d}, '
                        f'Epoch: {epoch:02d}, '
                        f'Loss: {loss:.4f}, '
                        f'Train: {100 * train_acc:.2f}%, '
                        f'Valid: {100 * valid_acc:.2f}% '
                        f'Test: {100 * test_acc:.2f}%')

        logger.print_statistics(run)
    logger.print_statistics()


if __name__ == "__main__":
    main()