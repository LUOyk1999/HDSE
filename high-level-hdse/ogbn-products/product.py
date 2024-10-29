import os.path as osp
import time
import argparse
import torch
import torch.nn.functional as F
from ogb.nodeproppred import Evaluator, PygNodePropPredDataset
from torch.nn import LayerNorm, Linear
from tqdm import tqdm
import numpy as np
import torch_geometric.transforms as T
from torch_geometric.loader import RandomNodeLoader
from torch_geometric.nn import SAGEConv
from torch_geometric.utils import index_to_mask

import nxmetis
import networkx as nx
import numpy as np
from torch_geometric.utils import to_undirected
import pickle
import os

from einops import rearrange, repeat, reduce
import math

import torch.nn as nn

class MLP(nn.Module):
    def __init__(self, embedding_dim, num_embeddings):
        super(MLP, self).__init__()
        self.fc = nn.Linear(1, 1)

    def forward(self, x):
        # Pass through embedding
        x = x.unsqueeze(-1)
        x = self.fc(x)
        x = x.squeeze(-1)
        return x
    def reset_parameters(self):
        self.fc.reset_parameters()
        
        
class GNNBlock(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.norm = LayerNorm(in_channels, elementwise_affine=True)
        self.conv = SAGEConv(in_channels, out_channels)

    def reset_parameters(self):
        self.norm.reset_parameters()
        self.conv.reset_parameters()

    def forward(self, x, edge_index, dropout_mask=None):
        x = self.norm(x).relu()
        x = F.dropout(x, p=0.5, training=self.training)
        return self.conv(x, edge_index)


class GOAT_HDSE(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers,
                 dropout, num_centroids=512, heads=4):
        super().__init__()
        
        self.fc_dis= MLP(hidden_channels, 31)
        self.hidden_channels = hidden_channels
        self.heads = heads
        self.num_centroids = num_centroids

        self.attn_fn = F.softmax

        self.lin_proj_g = Linear(in_channels, hidden_channels)
        self.lin_key_g = Linear(in_channels, hidden_channels)
        self.lin_query_g = Linear(hidden_channels, hidden_channels)
        self.lin_value_g = Linear(in_channels, hidden_channels)
        
        
        self.dropout = dropout
        self.vqs = torch.nn.ModuleList()
        self.lin1 = Linear(in_channels, hidden_channels)
        # self.lin2 = Linear(hidden_channels, out_channels)
        self.lin2 = Linear(hidden_channels*2, out_channels)
        self.norm = LayerNorm(hidden_channels, elementwise_affine=True)

        self.convs = torch.nn.ModuleList()
        for _ in range(num_layers):
            conv = GNNBlock(
                hidden_channels,
                hidden_channels,
            )
            self.convs.append(conv)
           

    def reset_parameters(self):
        self.lin1.reset_parameters()
        self.lin2.reset_parameters()
        self.norm.reset_parameters()
        for conv in self.convs:
            conv.reset_parameters()
        self.lin_proj_g.reset_parameters()
        self.lin_key_g.reset_parameters()
        self.lin_query_g.reset_parameters()
        self.lin_value_g.reset_parameters()
        self.fc_dis.reset_parameters()
    

    def global_forward(self, x, distance_matrix, nodes_to_community_tensor):
        
        distance_matrix = self.fc_dis(distance_matrix.float())
        d, h = self.hidden_channels // self.heads, self.heads
        scale = 1.0 / math.sqrt(d)
        # print(x.shape)

        q_x = self.lin_proj_g(x)

        P = torch.nn.functional.one_hot(nodes_to_community_tensor, num_classes=self.num_centroids).float()
        # print(P.shape)
        community_sizes = P.sum(dim=0).view(-1, 1) 
        community_sizes = community_sizes.clamp(min=1) 
        community_sums = P.T @ x
        # print(community_sums, community_sums.shape)
        community_avg = community_sums / community_sizes
        # print(community_sizes, community_sizes.shape)
        # print(community_avg, community_avg.shape, 'done')
        q = self.lin_query_g(q_x)
        k = self.lin_key_g(community_avg)
        v = self.lin_value_g(community_avg)
        # print(q.shape,k.shape)
        q, k, v = map(lambda t: rearrange(t, 'n (h d) -> h n d', h=h), (q, k, v))
        dots = torch.einsum('h i d, h j d -> h i j', q, k) * scale
        c, c_count = nodes_to_community_tensor.squeeze().to(torch.short).unique(return_counts=True)
        # print(c.shape,c_count.shape)
        centroid_count = torch.zeros(self.num_centroids, dtype=torch.long).to(x.device)
        centroid_count[c.to(torch.long)] = c_count
        # print(centroid_count.shape,centroid_count.view(1,1,-1).shape)
        dots += torch.log(centroid_count.view(1,1,-1))
        dots += distance_matrix.view(1,distance_matrix.shape[0],distance_matrix.shape[1])
        # print(distance_matrix.view(1,distance_matrix.shape[0],distance_matrix.shape[1]).shape)
        attn = self.attn_fn(dots, dim = -1)
        attn = F.dropout(attn, p=self.dropout, training=self.training)

        out = torch.einsum('h i j, h j d -> h i d', attn, v)
        out = rearrange(out, 'h n d -> n (h d)')

        return out
    
    def forward(self, x, edge_index, distance_matrix, nodes_to_community_tensor):
        x_ = x
        x = self.lin1(x)

        for conv in self.convs:
            x = conv(x, edge_index)
            
        x = self.norm(x).relu()
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = torch.cat([x,self.global_forward(x_, distance_matrix, nodes_to_community_tensor)],dim=-1)
        # x = x + self.global_forward(x_, distance_matrix, nodes_to_community_tensor)
        return self.lin2(x)

parser = argparse.ArgumentParser()
parser.add_argument('--device', type=int, default=2)
parser.add_argument('--num_centroids', type=int, default=512)
args = parser.parse_args()
print(args)

device = f'cuda:{args.device}' if torch.cuda.is_available() else 'cpu'
device = torch.device(device)
transform = T.Compose([T.ToDevice(device), T.ToSparseTensor()])
transform_cpu = T.Compose([T.ToDevice(torch.device("cpu")), T.ToSparseTensor()])
root = osp.join(osp.dirname(osp.realpath(__file__)), 'dataset')
dataset = PygNodePropPredDataset('ogbn-products', root,
                                 transform=T.AddSelfLoops())
evaluator = Evaluator(name='ogbn-products')

data = dataset[0]

print(f'metis begins: ./distance_{args.num_centroids}_products.pkl')
if os.path.exists(f'./distance_{args.num_centroids}_products.pkl'):
    with open(f'./distance_{args.num_centroids}_products.pkl', 'rb') as f:
        data.distance_matrix, data.nodes_to_community_tensor = pickle.load(f)
        data.distance_matrix = torch.tensor(data.distance_matrix, dtype=torch.int8)
else:
    dataset_ = PygNodePropPredDataset(name='ogbn-products', root=root)
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
    
    with open(f'./distance_{args.num_centroids}_products.pkl', 'wb') as f:
        pickle.dump((data.distance_matrix,data.nodes_to_community_tensor), f, pickle.HIGHEST_PROTOCOL)
print(f'metis ends: ./distance_{args.num_centroids}_products.pkl')

split_idx = dataset.get_idx_split()
for split in ['train', 'valid', 'test']:
    data[f'{split}_mask'] = index_to_mask(split_idx[split], data.y.shape[0])

train_loader = RandomNodeLoader(data, num_parts=10, shuffle=True,
                                num_workers=5)

test_loader = RandomNodeLoader(data, num_parts=1, num_workers=5)

model = GOAT_HDSE(
    in_channels=dataset.num_features,
    hidden_channels=256,
    out_channels=dataset.num_classes,
    num_layers=5,
    dropout=0.5, num_centroids=args.num_centroids
).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.003)


def train(epoch):
    model.train()

    pbar = tqdm(total=len(train_loader))
    pbar.set_description(f'Training epoch: {epoch:03d}')

    total_loss = total_examples = 0
    for data in train_loader:
        optimizer.zero_grad()

        data = transform(data)
        out= model(data.x, data.adj_t, data.distance_matrix, data.nodes_to_community_tensor)
        loss = F.cross_entropy(out[data.train_mask], data.y[data.train_mask].view(-1))
        (loss).backward()
        optimizer.step()

        total_loss += float(loss) * int(data.train_mask.sum())
        total_examples += int(data.train_mask.sum())
        pbar.update(1)

    pbar.close()

    return total_loss / total_examples


@torch.no_grad()
def test(epoch):

    model.to("cpu")
    model.eval()
    y_true = {"train": [], "valid": [], "test": []}
    y_pred = {"train": [], "valid": [], "test": []}

    pbar = tqdm(total=len(test_loader))
    pbar.set_description(f'Evaluating epoch: {epoch:03d}')

    for data in test_loader:

        data = transform_cpu(data)
        out= model(data.x, data.adj_t, data.distance_matrix, data.nodes_to_community_tensor)
        out = out.argmax(dim=-1, keepdim=True)
        for split in ['train', 'valid', 'test']:
            mask = data[f'{split}_mask']
            y_true[split].append(data.y[mask].cpu())
            y_pred[split].append(out[mask].cpu())

        pbar.update(1)

    pbar.close()

    train_acc = evaluator.eval({
        'y_true': torch.cat(y_true['train'], dim=0),
        'y_pred': torch.cat(y_pred['train'], dim=0),
    })['acc']

    valid_acc = evaluator.eval({
        'y_true': torch.cat(y_true['valid'], dim=0),
        'y_pred': torch.cat(y_pred['valid'], dim=0),
    })['acc']

    test_acc = evaluator.eval({
        'y_true': torch.cat(y_true['test'], dim=0),
        'y_pred': torch.cat(y_pred['test'], dim=0),
    })['acc']
    model.to(device)

    return train_acc, valid_acc, test_acc


times = []
best_val = 0.0
final_train = 0.0
final_test = 0.0
for epoch in range(1, 1001):
    start = time.time()
    loss = train(epoch)
    if epoch % 50 == 0:
        train_acc, val_acc, test_acc = test(epoch)
        if val_acc > best_val:
            best_val = val_acc
            final_train = train_acc
            final_test = test_acc
        print(f'Loss: {loss:.4f}, Train: {train_acc:.4f}, Val: {val_acc:.4f}, '
            f'Test: {test_acc:.4f}')
    times.append(time.time() - start)

print(f'Final Train: {final_train:.4f}, Best Val: {best_val:.4f}, '
      f'Final Test: {final_test:.4f}')
print(f"Median time per epoch: {torch.tensor(times).median():.4f}s")