import torch
from torch import nn
import torch_geometric.graphgym.register as register
from torch_geometric.graphgym.config import cfg
from torch_geometric.graphgym.models.gnn import GNNPreMP
from torch_geometric.graphgym.models.layer import (new_layer_config,
                                                   BatchNorm1dNode)
from torch_geometric.graphgym.register import register_network

from graphgps.layer.gps_layer import GPSLayer
from torch_geometric.utils import to_dense_batch, to_dense_adj

class FeatureEncoder(torch.nn.Module):
    """
    Encoding node and edge features

    Args:
        dim_in (int): Input feature dimension
    """
    def __init__(self, dim_in):
        super(FeatureEncoder, self).__init__()
        self.dim_in = dim_in
        if cfg.dataset.node_encoder:
            # Encode integer node features via nn.Embeddings
            NodeEncoder = register.node_encoder_dict[
                cfg.dataset.node_encoder_name]
            self.node_encoder = NodeEncoder(cfg.gnn.dim_inner)
            if cfg.dataset.node_encoder_bn:
                self.node_encoder_bn = BatchNorm1dNode(
                    new_layer_config(cfg.gnn.dim_inner, -1, -1, has_act=False,
                                     has_bias=False, cfg=cfg))
            # Update dim_in to reflect the new dimension fo the node features
            self.dim_in = cfg.gnn.dim_inner
        if cfg.dataset.edge_encoder:
            # Hard-set edge dim for PNA.
            cfg.gnn.dim_edge = 16 if 'PNA' in cfg.gt.layer_type else cfg.gnn.dim_inner
            # Encode integer edge features via nn.Embeddings
            EdgeEncoder = register.edge_encoder_dict[
                cfg.dataset.edge_encoder_name]
            self.edge_encoder = EdgeEncoder(cfg.gnn.dim_edge)
            if cfg.dataset.edge_encoder_bn:
                self.edge_encoder_bn = BatchNorm1dNode(
                    new_layer_config(cfg.gnn.dim_edge, -1, -1, has_act=False,
                                     has_bias=False, cfg=cfg))

    def forward(self, batch):
        # print(batch)
        for module in self.children():
            batch = module(batch)
        return batch

class CateFeatureEmbedding(nn.Module):
    def __init__(self, num_uniq_values, embed_dim, dropout=0.0):
        '''
        '''
        super().__init__()
        if len(num_uniq_values)==1:
            self.linear = nn.Linear(embed_dim, embed_dim)
        else:
            self.linear = nn.Linear(embed_dim*2, embed_dim)
        num_uniq_values = torch.LongTensor(num_uniq_values)
        csum = torch.cumsum(num_uniq_values, dim=0)
        num_emb = csum[-1]
        num_uniq_values = torch.LongTensor(num_uniq_values).reshape(1, 1, -1)
        self.register_buffer('num_uniq_values', num_uniq_values)
        
        starts = torch.cat(
            (torch.LongTensor([0]), csum[:-1])).reshape(1, -1)
        self.register_buffer('starts', starts)
        
        self.embeddings = nn.Embedding(
            num_emb, embed_dim)
        
        self.dropout_proba = dropout
        
        self.layer_norm_output = nn.LayerNorm(embed_dim)
        pass

    def forward(self, x):
        # x = x + 1
        if torch.any(x < 0):
            raise RuntimeError(str(x))
        
        if torch.any(torch.ge(x, self.num_uniq_values)):
            print(torch.max(x[:,:,:,0]))
            print(torch.max(x[:,:,:,1]))
            raise RuntimeError(str(x))
            pass
        
        x = x + self.starts
        
        if self.training:
            # x[torch.rand(size=x.shape, device=x.device) < self.dropout_proba] = 0
            pass
        # print(self.embeddings(x).shape)
        # emb = self.embeddings(x).sum(dim=-2)
        emb = self.embeddings(x)
        emb = torch.reshape(emb,(emb.shape[0],emb.shape[1],emb.shape[2],-1))
        emb = self.linear(emb)
        return emb
    pass

@register_network('GPSModel')
class GPSModel(torch.nn.Module):
    """Multi-scale graph x-former.
    """

    def __init__(self, dim_in, dim_out):
        super().__init__()
        self.encoder = FeatureEncoder(dim_in)
        dim_in = self.encoder.dim_in

        if cfg.gnn.layers_pre_mp > 0:
            self.pre_mp = GNNPreMP(
                dim_in, cfg.gnn.dim_inner, cfg.gnn.layers_pre_mp)
            dim_in = cfg.gnn.dim_inner

        assert cfg.gt.dim_hidden == cfg.gnn.dim_inner == dim_in, \
            "The inner and hidden dims must match."

        try:
            local_gnn_type, global_model_type = cfg.gt.layer_type.split('+')
        except:
            raise ValueError(f"Unexpected layer type: {cfg.gt.layer_type}")
        layers = []
        if (cfg.dataset.name=='ogbg-molhiv'):
            self.catelist = [11, 31]
        elif(cfg.dataset.name=='ogbg-molpcba'):
            self.catelist = [11, 31]
        # elif(cfg.dataset.name=='CIFAR10'):
        #     self.catelist = [16]
        elif(cfg.dataset.name=='peptides-functional'):
            self.catelist = [31]
        else:
            self.catelist = [11, 31]
        for _ in range(cfg.gt.layers):
            layers.append(GPSLayer(
                dataset=cfg.dataset.name,
                dim_h=cfg.gt.dim_hidden,
                local_gnn_type=local_gnn_type,
                global_model_type=global_model_type,
                num_heads=cfg.gt.n_heads,
                pna_degrees=cfg.gt.pna_degrees,
                equivstable_pe=cfg.posenc_EquivStableLapPE.enable,
                dropout=cfg.gt.dropout,
                attn_dropout=cfg.gt.attn_dropout,
                layer_norm=cfg.gt.layer_norm,
                batch_norm=cfg.gt.batch_norm,
                bigbird_cfg=cfg.gt.bigbird,
            ))
        self.layers = torch.nn.Sequential(*layers)
        self.layer_structure_embed_cate = CateFeatureEmbedding(self.catelist, cfg.gt.dim_hidden, dropout=0.1)
        if (cfg.dataset.name=='ogbg-molpcba' or cfg.dataset.name=='peptides-structural' or cfg.dataset.name=='peptides-functional'):
            self.layer_structure_embed_cate = CateFeatureEmbedding(self.catelist, 16, dropout=0.1)
        # self.layer_structure_embed_cate = CateFeatureEmbedding(self.catelist, cfg.gt.dim_hidden, dropout=0.1)
        GNNHead = register.head_dict[cfg.gnn.head]
        self.post_mp = GNNHead(dim_in=cfg.gnn.dim_inner, dim_out=dim_out)

    def forward(self, batch):
        batch = self.encoder(batch)
        # print(batch)
        dist = to_dense_adj(batch.complete_edge_index, batch.batch, edge_attr = batch.complete_edge_dist)
        SPD = to_dense_adj(batch.complete_edge_index, batch.batch, edge_attr = batch.complete_edge_SPD)
        if(len(self.catelist)==1):
            dist = torch.stack([SPD], axis=3)
            # print(dist.shape)
        else:
            dist = torch.stack([dist,SPD], axis=3)

        batch.dist = self.layer_structure_embed_cate(dist)
        for module in self.layers:
            batch = module(batch)
        batch = self.post_mp(batch)
        return batch

