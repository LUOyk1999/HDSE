from genericpath import exists
import math
from typing import Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch_sparse import SparseTensor

from torch_geometric.nn.conv import MessagePassing
from torch_geometric.nn.dense.linear import Linear
from torch_geometric.typing import Adj, OptTensor, PairTensor
from torch_geometric.utils import softmax

from einops import rearrange, repeat, reduce
from einops.layers.torch import Rearrange

from torch_geometric.nn import GCNConv, SAGEConv


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

class SAGE(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers,
                 dropout):
        super(SAGE, self).__init__()

        self.convs = torch.nn.ModuleList()
        self.convs.append(SAGEConv(in_channels, hidden_channels))
        self.bns = torch.nn.ModuleList()
        self.bns.append(torch.nn.BatchNorm1d(hidden_channels))
        for _ in range(num_layers - 2):
            self.convs.append(SAGEConv(hidden_channels, hidden_channels))
            self.bns.append(torch.nn.BatchNorm1d(hidden_channels))
        self.convs.append(SAGEConv(hidden_channels, out_channels))

        self.dropout = dropout

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()
        for bn in self.bns:
            bn.reset_parameters()

    def forward(self, x, adj_t):
        for i, conv in enumerate(self.convs[:-1]):
            x = conv(x, adj_t)
            x = self.bns[i](x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.convs[-1](x, adj_t)
        return x


class GCN(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers,
                 dropout):
        super(GCN, self).__init__()
        self.convs = torch.nn.ModuleList()
        self.convs.append(GCNConv(in_channels, hidden_channels, cached=True))
        self.bns = torch.nn.ModuleList()
        self.bns.append(torch.nn.BatchNorm1d(hidden_channels))
        for _ in range(num_layers - 2):
            self.convs.append(
                GCNConv(hidden_channels, hidden_channels, cached=True))
            self.bns.append(torch.nn.BatchNorm1d(hidden_channels))
        self.convs.append(GCNConv(hidden_channels, out_channels, cached=True))

        self.dropout = dropout

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()
        for bn in self.bns:
            bn.reset_parameters()

    def forward(self, x, adj_t):
        for i, conv in enumerate(self.convs[0:-1]):
            x = conv(x, adj_t)
            x = self.bns[i](x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.convs[-1](x, adj_t)
        return x


class TransformerConv(MessagePassing):

    _alpha: OptTensor

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        global_dim: int,
        num_nodes: int,
        heads: int = 1,
        dropout: float = 0.,
        num_centroids: Optional[int] = None,
        **kwargs,
    ):
        kwargs.setdefault('aggr', 'add')
        super(TransformerConv, self).__init__(node_dim=0, **kwargs)
        self.fc_dis= MLP(out_channels, 31)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.heads = heads
        self.dropout = dropout

        self.num_centroids = num_centroids

        self.attn_fn = F.softmax

        self.lin_proj_g = Linear(in_channels, global_dim)
        self.lin_key_g = Linear(global_dim, out_channels)
        self.lin_query_g = Linear(global_dim, out_channels)
        self.lin_value_g = Linear(global_dim, out_channels)

        self.reset_parameters()

    def reset_parameters(self):
        self.lin_proj_g.reset_parameters()
        self.lin_key_g.reset_parameters()
        self.lin_query_g.reset_parameters()
        self.lin_value_g.reset_parameters()
        self.fc_dis.reset_parameters()


    def global_forward(self, x, distance_matrix, nodes_to_community_tensor):
        
        distance_matrix = self.fc_dis(distance_matrix.float())
        d, h = self.out_channels // self.heads, self.heads
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


class Transformer(torch.nn.Module):
    def __init__(self, num_nodes, in_channels, hidden_channels, out_channels, global_dim, num_layers, heads, ff_dropout, attn_dropout, num_centroids, no_bn, norm_type, gnum_layers,
        ghidden_channels,
        gdropout, gcn):
        super(Transformer, self).__init__()
        self.num_centroids = num_centroids
        if gcn==1:
            self.gnn = GCN(in_channels, ghidden_channels,
                        out_channels, gnum_layers,
                        gdropout)
        else:
            self.gnn = SAGE(in_channels, ghidden_channels,
                        out_channels, gnum_layers,
                        gdropout)

        if norm_type == 'batch_norm' :
            norm_func = nn.BatchNorm1d
        elif norm_type == 'layer_norm' :
            norm_func = nn.LayerNorm

        if no_bn :
            self.fc_in = nn.Sequential(
                nn.Linear(in_channels, hidden_channels),
                nn.ReLU(),
                nn.Dropout(ff_dropout),
                nn.Linear(hidden_channels, hidden_channels)
            )            
        else :
            self.fc_in = nn.Sequential(
                nn.Linear(in_channels, hidden_channels),
                norm_func(hidden_channels),
                nn.ReLU(),
                nn.Dropout(ff_dropout),
                nn.Linear(hidden_channels, hidden_channels)
            )

        self.convs = torch.nn.ModuleList()
        self.ffs = torch.nn.ModuleList()

        for _ in range(num_layers):
            self.convs.append(
                TransformerConv(
                    in_channels=hidden_channels,
                    out_channels=hidden_channels,
                    global_dim=global_dim,
                    num_nodes=num_nodes,
                    heads=heads,
                    dropout=attn_dropout, 
                    num_centroids=num_centroids
                )
            )

            if no_bn :
                self.ffs.append(
                    nn.Sequential(
                        nn.Linear(hidden_channels, hidden_channels),
                        nn.ReLU(),
                        nn.Dropout(ff_dropout),
                        nn.Linear(hidden_channels, hidden_channels),
                        nn.ReLU(),
                        nn.Dropout(ff_dropout),
                    )
                )
            else :
                self.ffs.append(
                    nn.Sequential(
                        nn.Linear(hidden_channels, hidden_channels),
                        norm_func(hidden_channels),
                        nn.ReLU(),
                        nn.Dropout(ff_dropout),
                        nn.Linear(hidden_channels, hidden_channels),
                        norm_func(hidden_channels),
                        nn.ReLU(),
                        nn.Dropout(ff_dropout),
                    )
                )

        self.fc_out = torch.nn.Linear(hidden_channels, out_channels)

    def reset_parameters(self):
        for module in self.fc_in:
            if hasattr(module, 'reset_parameters'):
                module.reset_parameters()
        self.gnn.reset_parameters()
        for conv in self.convs:
            conv.reset_parameters()
        for ff in self.ffs:
            for module in ff:
                if hasattr(module, 'reset_parameters'):
                    module.reset_parameters()
        self.fc_out.reset_parameters()

    def global_forward(self, x, adj_t, distance_matrix, nodes_to_community_tensor):
        # print(x.shape)
        x_ = self.fc_in(x)
        for i, conv in enumerate(self.convs):
            x_ = conv.global_forward(x_, distance_matrix, nodes_to_community_tensor)
            x_ = self.ffs[i](x_)
        x_ = self.fc_out(x_)
        x = x_*0.5 + self.gnn(x, adj_t)*0.5
        return x
