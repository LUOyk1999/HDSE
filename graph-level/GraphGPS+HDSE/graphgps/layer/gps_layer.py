import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric.nn as pygnn
from performer_pytorch import SelfAttention
from torch_geometric.data import Batch
import torch_geometric.utils as utils
from torch_geometric.nn import Linear as Linear_pyg
from torch_geometric.utils import to_dense_batch, degree
from torch_geometric.data import Data
from graphgps.layer.gatedgcn_layer import GatedGCNLayer
from graphgps.layer.gine_conv_layer import GINEConvESLapPE
from graphgps.layer.bigbird_layer import SingleBigBirdLayer
import matplotlib.pyplot as plt
from torch import _VF
from typing import Callable, List, Optional, Tuple
import math
Tensor = torch.Tensor
# # Activation functions
def dropout(input: Tensor, p: float = 0.5, training: bool = True, inplace: bool = False) -> Tensor:
    if p < 0.0 or p > 1.0:
        raise ValueError("dropout probability has to be between 0 and 1, " "but got {}".format(p))
    return _VF.dropout_(input, p, training) if inplace else _VF.dropout(input, p, training)


def softmax(input: Tensor, dim: Optional[int] = None, _stacklevel: int = 3, dtype: Optional[int] = None) -> Tensor:
    if dtype is None:
        ret = input.softmax(dim)
    else:
        ret = input.softmax(dim, dtype=dtype)
    return ret

def unbatch_edge_index(edge_index: Tensor, batch: Tensor) -> List[Tensor]:
    r"""Splits the :obj:`edge_index` according to a :obj:`batch` vector.

    Args:
        edge_index (Tensor): The edge_index tensor. Must be ordered.
        batch (LongTensor): The batch vector
            :math:`\mathbf{b} \in {\{ 0, \ldots, B-1\}}^N`, which assigns each
            node to a specific example. Must be ordered.

    :rtype: :class:`List[Tensor]`

    Example:

        >>> edge_index = torch.tensor([[0, 1, 1, 2, 2, 3, 4, 5, 5, 6],
        ...                            [1, 0, 2, 1, 3, 2, 5, 4, 6, 5]])
        >>> batch = torch.tensor([0, 0, 0, 0, 1, 1, 1])
        >>> unbatch_edge_index(edge_index, batch)
        (tensor([[0, 1, 1, 2, 2, 3],
                [1, 0, 2, 1, 3, 2]]),
        tensor([[0, 1, 1, 2],
                [1, 0, 2, 1]]))
    """
    deg = degree(batch, dtype=torch.int64)
    ptr = torch.cat([deg.new_zeros(1), deg.cumsum(dim=0)[:-1]], dim=0)

    edge_batch = batch[edge_index[0]]
    edge_index = edge_index - ptr[edge_batch]
    sizes = degree(edge_batch, dtype=torch.int64).cpu().tolist()
    return edge_index.split(sizes, dim=1)


def bias_scaled_dot_product_attention(
    q: Tensor,
    k: Tensor,
    v: Tensor,
    attn_mask: Optional[Tensor] = None,
    dropout_p: float = 0.0, dist = None
) -> Tuple[Tensor, Tensor]:
    B, Nt, E = q.shape
    q = q / math.sqrt(E)
    # (B, Nt, E) x (B, E, Ns) -> (B, Nt, Ns)
    attn = torch.bmm(q, k.transpose(-2, -1))
    attn_ = softmax(attn, dim=-1)
    # print("ok!!!!!!!!!")
    structure_query = dist
    tgt_len = structure_query.shape[1]
    structure_query = structure_query.contiguous().transpose(2, 3)
    structure_query = structure_query.contiguous().transpose(1, 2)
    structure_query = structure_query.contiguous().view(attn.shape[0], tgt_len, tgt_len)
    attn = attn + structure_query
    
    attn_ = softmax(structure_query, dim=-1)
    if attn_mask is not None:
        attn += attn_mask
    attn = softmax(attn, dim=-1)
    if dropout_p > 0.0:
        attn = dropout(attn, p=dropout_p)
    # (B, Nt, Ns) x (B, Ns, E) -> (B, Nt, E)
    output = torch.bmm(attn, v)
    return output, attn, attn_

def multi_head_attention_forward(
    query: Tensor,
    key: Tensor,
    value: Tensor,
    embed_dim_to_check: int,
    num_heads: int,
    in_proj_weight: Tensor,
    in_proj_bias: Optional[Tensor],
    bias_k: Optional[Tensor],
    bias_v: Optional[Tensor],
    add_zero_attn: bool,
    dropout_p: float,
    out_proj_weight: Tensor,
    out_proj_bias: Optional[Tensor],
    training: bool = True,
    key_padding_mask: Optional[Tensor] = None,
    need_weights: bool = True,
    attn_mask: Optional[Tensor] = None,
    use_separate_proj_weight: bool = False,
    q_proj_weight: Optional[Tensor] = None,
    k_proj_weight: Optional[Tensor] = None,
    v_proj_weight: Optional[Tensor] = None,
    static_k: Optional[Tensor] = None,
    static_v: Optional[Tensor] = None,
    dist = None
) -> Tuple[Tensor, Optional[Tensor]]:
    r"""
    Args:
        query, key, value: map a query and a set of key-value pairs to an output.
            See "Attention Is All You Need" for more details.
        embed_dim_to_check: total dimension of the model.
        num_heads: parallel attention heads.
        in_proj_weight, in_proj_bias: input projection weight and bias.
        bias_k, bias_v: bias of the key and value sequences to be added at dim=0.
        add_zero_attn: add a new batch of zeros to the key and
                       value sequences at dim=1.
        dropout_p: probability of an element to be zeroed.
        out_proj_weight, out_proj_bias: the output projection weight and bias.
        training: apply dropout if is ``True``.
        key_padding_mask: if provided, specified padding elements in the key will
            be ignored by the attention. This is an binary mask. When the value is True,
            the corresponding value on the attention layer will be filled with -inf.
        need_weights: output attn_output_weights.
        attn_mask: 2D or 3D mask that prevents attention to certain positions. A 2D mask will be broadcasted for all
            the batches while a 3D mask allows to specify a different mask for the entries of each batch.
        use_separate_proj_weight: the function accept the proj. weights for query, key,
            and value in different forms. If false, in_proj_weight will be used, which is
            a combination of q_proj_weight, k_proj_weight, v_proj_weight.
        q_proj_weight, k_proj_weight, v_proj_weight, in_proj_bias: input projection weight and bias.
        static_k, static_v: static key and value used for attention operators.


    Shape:
        Inputs:
        - query: :math:`(L, N, E)` where L is the target sequence length, N is the batch size, E is
          the embedding dimension.
        - key: :math:`(S, N, E)`, where S is the source sequence length, N is the batch size, E is
          the embedding dimension.
        - value: :math:`(S, N, E)` where S is the source sequence length, N is the batch size, E is
          the embedding dimension.
        - key_padding_mask: :math:`(N, S)` where N is the batch size, S is the source sequence length.
          If a ByteTensor is provided, the non-zero positions will be ignored while the zero positions
          will be unchanged. If a BoolTensor is provided, the positions with the
          value of ``True`` will be ignored while the position with the value of ``False`` will be unchanged.
        - attn_mask: 2D mask :math:`(L, S)` where L is the target sequence length, S is the source sequence length.
          3D mask :math:`(N*num_heads, L, S)` where N is the batch size, L is the target sequence length,
          S is the source sequence length. attn_mask ensures that position i is allowed to attend the unmasked
          positions. If a ByteTensor is provided, the non-zero positions are not allowed to attend
          while the zero positions will be unchanged. If a BoolTensor is provided, positions with ``True``
          are not allowed to attend while ``False`` values will be unchanged. If a FloatTensor
          is provided, it will be added to the attention weight.
        - static_k: :math:`(N*num_heads, S, E/num_heads)`, where S is the source sequence length,
          N is the batch size, E is the embedding dimension. E/num_heads is the head dimension.
        - static_v: :math:`(N*num_heads, S, E/num_heads)`, where S is the source sequence length,
          N is the batch size, E is the embedding dimension. E/num_heads is the head dimension.

        Outputs:
        - attn_output: :math:`(L, N, E)` where L is the target sequence length, N is the batch size,
          E is the embedding dimension.
        - attn_output_weights: :math:`(N, L, S)` where N is the batch size,
          L is the target sequence length, S is the source sequence length.
    """
    tens_ops = (query, key, value, in_proj_weight, in_proj_bias, bias_k, bias_v, out_proj_weight, out_proj_bias)

    # set up shape vars
    tgt_len, bsz, embed_dim = query.shape
    src_len, _, _ = key.shape
    assert embed_dim == embed_dim_to_check, \
        f"was expecting embedding dimension of {embed_dim_to_check}, but got {embed_dim}"
    if isinstance(embed_dim, torch.Tensor):
        # embed_dim can be a tensor when JIT tracing
        head_dim = embed_dim.div(num_heads, rounding_mode='trunc')
    else:
        head_dim = embed_dim // num_heads
    assert head_dim * num_heads == embed_dim, f"embed_dim {embed_dim} not divisible by num_heads {num_heads}"
    if use_separate_proj_weight:
        # allow MHA to have different embedding dimensions when separate projection weights are used
        assert key.shape[:2] == value.shape[:2], \
            f"key's sequence and batch dims {key.shape[:2]} do not match value's {value.shape[:2]}"
    else:
        assert key.shape == value.shape, f"key shape {key.shape} does not match value shape {value.shape}"

    #
    # compute in-projection
    #
    if not use_separate_proj_weight:
        q, k, v = F._in_projection_packed(query, key, value, in_proj_weight, in_proj_bias)
    else:
        assert q_proj_weight is not None, "use_separate_proj_weight is True but q_proj_weight is None"
        assert k_proj_weight is not None, "use_separate_proj_weight is True but k_proj_weight is None"
        assert v_proj_weight is not None, "use_separate_proj_weight is True but v_proj_weight is None"
        if in_proj_bias is None:
            b_q = b_k = b_v = None
        else:
            b_q, b_k, b_v = in_proj_bias.chunk(3)
        q, k, v = F._in_projection(query, key, value, q_proj_weight, k_proj_weight, v_proj_weight, b_q, b_k, b_v)
    
    # prep attention mask
    if attn_mask is not None:
        if attn_mask.dtype == torch.uint8:
            attn_mask = attn_mask.to(torch.bool)
        else:
            assert attn_mask.is_floating_point() or attn_mask.dtype == torch.bool, \
                f"Only float, byte, and bool types are supported for attn_mask, not {attn_mask.dtype}"
        # ensure attn_mask's dim is 3
        if attn_mask.dim() == 2:
            correct_2d_size = (tgt_len, src_len)
            if attn_mask.shape != correct_2d_size:
                raise RuntimeError(f"The shape of the 2D attn_mask is {attn_mask.shape}, but should be {correct_2d_size}.")
            attn_mask = attn_mask.unsqueeze(0)
        elif attn_mask.dim() == 3:
            correct_3d_size = (bsz * num_heads, tgt_len, src_len)
            if attn_mask.shape != correct_3d_size:
                raise RuntimeError(f"The shape of the 3D attn_mask is {attn_mask.shape}, but should be {correct_3d_size}.")
        else:
            raise RuntimeError(f"attn_mask's dimension {attn_mask.dim()} is not supported")

    # prep key padding mask
    if key_padding_mask is not None and key_padding_mask.dtype == torch.uint8:
        key_padding_mask = key_padding_mask.to(torch.bool)

    # add bias along batch dimension (currently second)
    if bias_k is not None and bias_v is not None:
        assert static_k is None, "bias cannot be added to static key."
        assert static_v is None, "bias cannot be added to static value."
        k = torch.cat([k, bias_k.repeat(1, bsz, 1)])
        v = torch.cat([v, bias_v.repeat(1, bsz, 1)])
        if attn_mask is not None:
            attn_mask = F.pad(attn_mask, (0, 1))
        if key_padding_mask is not None:
            key_padding_mask = F.pad(key_padding_mask, (0, 1))
    else:
        assert bias_k is None
        assert bias_v is None

    #
    # reshape q, k, v for multihead attention and make em batch first
    #
    q = q.contiguous().view(tgt_len, bsz * num_heads, head_dim).transpose(0, 1)
    
    if static_k is None:
        k = k.contiguous().view(k.shape[0], bsz * num_heads, head_dim).transpose(0, 1)
    else:
        # TODO finish disentangling control flow so we don't do in-projections when statics are passed
        assert static_k.size(0) == bsz * num_heads, \
            f"expecting static_k.size(0) of {bsz * num_heads}, but got {static_k.size(0)}"
        assert static_k.size(2) == head_dim, \
            f"expecting static_k.size(2) of {head_dim}, but got {static_k.size(2)}"
        k = static_k
    if static_v is None:
        v = v.contiguous().view(v.shape[0], bsz * num_heads, head_dim).transpose(0, 1)
    else:
        # TODO finish disentangling control flow so we don't do in-projections when statics are passed
        assert static_v.size(0) == bsz * num_heads, \
            f"expecting static_v.size(0) of {bsz * num_heads}, but got {static_v.size(0)}"
        assert static_v.size(2) == head_dim, \
            f"expecting static_v.size(2) of {head_dim}, but got {static_v.size(2)}"
        v = static_v

    # add zero attention along batch dimension (now first)
    if add_zero_attn:
        zero_attn_shape = (bsz * num_heads, 1, head_dim)
        k = torch.cat([k, torch.zeros(zero_attn_shape, dtype=k.dtype, device=k.device)], dim=1)
        v = torch.cat([v, torch.zeros(zero_attn_shape, dtype=v.dtype, device=v.device)], dim=1)
        if attn_mask is not None:
            attn_mask = F.pad(attn_mask, (0, 1))
        if key_padding_mask is not None:
            key_padding_mask = F.pad(key_padding_mask, (0, 1))

    # update source sequence length after adjustments
    src_len = k.size(1)

    # merge key padding and attention masks
    if key_padding_mask is not None:
        assert key_padding_mask.shape == (bsz, src_len), \
            f"expecting key_padding_mask shape of {(bsz, src_len)}, but got {key_padding_mask.shape}"
        key_padding_mask = key_padding_mask.view(bsz, 1, 1, src_len).   \
            expand(-1, num_heads, -1, -1).reshape(bsz * num_heads, 1, src_len)
        if attn_mask is None:
            attn_mask = key_padding_mask
        elif attn_mask.dtype == torch.bool:
            attn_mask = attn_mask.logical_or(key_padding_mask)
        else:
            attn_mask = attn_mask.masked_fill(key_padding_mask, float("-inf"))

    # dist = dist.reshape(bsz * num_heads, tgt_len, src_len)

    # convert mask to float
    if attn_mask is not None and attn_mask.dtype == torch.bool:
        new_attn_mask = torch.zeros_like(attn_mask, dtype=torch.float)
        new_attn_mask.masked_fill_(attn_mask, float("-inf"))
        attn_mask = new_attn_mask

    # adjust dropout probability
    if not training:
        dropout_p = 0.0

    #
    # (deep breath) calculate attention and out projection
    #
    attn_output, attn_output_weights, attn_output_weights_ = bias_scaled_dot_product_attention(q, k, v, attn_mask, dropout_p, dist)
    attn_output = attn_output.transpose(0, 1).contiguous().view(tgt_len, bsz, embed_dim)
    attn_output = F.linear(attn_output, out_proj_weight, out_proj_bias)

    if need_weights:
        # average attention weights over heads
        attn_output_weights = attn_output_weights.view(bsz, num_heads, tgt_len, src_len)
        attn_output_weights_ = attn_output_weights_.view(bsz, num_heads, tgt_len, src_len)
        return attn_output, (attn_output_weights.sum(dim=1) / num_heads, attn_output_weights_.sum(dim=1) / num_heads)
    else:
        return attn_output, None

from communities.algorithms import louvain_method,girvan_newman
from collections import defaultdict
from networkx.drawing.nx_pydot import graphviz_layout
import networkx as nx
import os
def draw_graph_with_attn(
    graph,
    outdir,
    filename,
    nodecolor=["tag", "attn"],
    dpi=300
):
    adj_matrix = nx.to_numpy_array(graph)
    partitions, _ = girvan_newman(adj_matrix)
    # print(partitions)
    color_map = []
    for node in graph.nodes:
        for i, cluster in enumerate(partitions):
            if node in cluster:
                color_map.append(i)
    if len(graph.edges) == 0:
        return
    import matplotlib.pyplot as plt
    plt.switch_backend("agg")
    fig = plt.figure(figsize=(4*len(nodecolor), 4), dpi=dpi)

    node_colors = defaultdict(list)

    titles = {
        'tag': 'Molecular graph',
        'attn1': 'Transformer attn',
        'attn2': 'HDSE attn'
    }


    for i in graph.nodes():
        for key in nodecolor:
            node_colors[key].append(graph.nodes[i][key])

    node_colors['tag'] = color_map
    node_colors['tag'][0] = 100
    vmax = {}
    cmap = {}
    for key in nodecolor:
        vmax[key] = 19
        cmap[key] = 'tab20'
        if 'attn' in key:
            vmax[key] = max(node_colors[key])
            cmap[key] = 'Reds'

    pos_layout = nx.spring_layout(graph, weight=None)
    # kamada_kawai_layout
    for i, key in enumerate(nodecolor):
        ax = fig.add_subplot(1, len(nodecolor), i+1)
        ax.set_title(titles[key], fontweight='bold')
        nx.draw(
            graph,
            pos=pos_layout,
            with_labels=False,
            font_size=4,
            node_color=node_colors[key],
            vmin=0,
            vmax=vmax[key],
            cmap=cmap[key],
            width=1.3,
            node_size=20,
            alpha=1.0,
        )
        if 'attn' in key:
            from mpl_toolkits.axes_grid1 import make_axes_locatable
            divider = make_axes_locatable(ax)
            cax = divider.append_axes('right', size='5%', pad=0.05)
            sm = plt.cm.ScalarMappable(cmap=cmap[key], norm=plt.Normalize(vmin=0, vmax=vmax[key]))
            sm._A = []
            plt.colorbar(sm, cax=cax)

    fig.axes[0].xaxis.set_visible(False)
    try:
        fig.canvas.draw()
    except:
        return

    save_path = os.path.join(outdir, filename)
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    print(save_path)
    plt.savefig(save_path)
    
class biasMultiheadAttention(nn.MultiheadAttention):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
    def forward(self, query: Tensor, key: Tensor, value: Tensor, key_padding_mask: Optional[Tensor] = None,
                need_weights: bool = True, attn_mask: Optional[Tensor] = None, dist=None) -> Tuple[Tensor, Optional[Tensor]]:
        if self.batch_first:
            query, key, value = [x.transpose(1, 0) for x in (query, key, value)]
        
        if not self._qkv_same_embed_dim:
            attn_output, attn_output_weights = multi_head_attention_forward(
                query, key, value, self.embed_dim, self.num_heads,
                self.in_proj_weight, self.in_proj_bias,
                self.bias_k, self.bias_v, self.add_zero_attn,
                self.dropout, self.out_proj.weight, self.out_proj.bias,
                training=self.training,
                key_padding_mask=key_padding_mask, need_weights=need_weights,
                attn_mask=attn_mask, use_separate_proj_weight=True,
                q_proj_weight=self.q_proj_weight, k_proj_weight=self.k_proj_weight,
                v_proj_weight=self.v_proj_weight, dist=dist)
        else:
            attn_output, attn_output_weights = multi_head_attention_forward(
                query, key, value, self.embed_dim, self.num_heads,
                self.in_proj_weight, self.in_proj_bias,
                self.bias_k, self.bias_v, self.add_zero_attn,
                self.dropout, self.out_proj.weight, self.out_proj.bias,
                training=self.training,
                key_padding_mask=key_padding_mask, need_weights=need_weights,
                attn_mask=attn_mask, dist=dist)
        if self.batch_first:
            return attn_output.transpose(1, 0), attn_output_weights
        else:
            return attn_output, attn_output_weights
class GPSLayer(nn.Module):
    """Local MPNN + full graph attention x-former layer.
    """

    def __init__(self, dataset, dim_h,
                 local_gnn_type, global_model_type, num_heads,
                 pna_degrees=None, equivstable_pe=False, dropout=0.0,
                 attn_dropout=0.0, layer_norm=False, batch_norm=True,
                 bigbird_cfg=None):
        super().__init__()

        self.dim_h = dim_h
        self.num_heads = num_heads
        self.attn_dropout = attn_dropout
        self.layer_norm = layer_norm
        self.batch_norm = batch_norm
        self.equivstable_pe = equivstable_pe

        # Local message-passing model.
        if local_gnn_type == 'None':
            self.local_model = None
        elif local_gnn_type == 'GENConv':
            self.local_model = pygnn.GENConv(dim_h, dim_h)
        elif local_gnn_type == 'GINE':
            gin_nn = nn.Sequential(Linear_pyg(dim_h, dim_h),
                                   nn.ReLU(),
                                   Linear_pyg(dim_h, dim_h))
            if self.equivstable_pe:  # Use specialised GINE layer for EquivStableLapPE.
                self.local_model = GINEConvESLapPE(gin_nn)
            else:
                self.local_model = pygnn.GINEConv(gin_nn)
        elif local_gnn_type == 'GAT':
            self.local_model = pygnn.GATConv(in_channels=dim_h,
                                             out_channels=dim_h // num_heads,
                                             heads=num_heads,
                                             edge_dim=dim_h)
        elif local_gnn_type == 'PNA':
            # Defaults from the paper.
            # aggregators = ['mean', 'min', 'max', 'std']
            # scalers = ['identity', 'amplification', 'attenuation']
            aggregators = ['mean', 'max', 'sum']
            scalers = ['identity']
            deg = torch.from_numpy(np.array(pna_degrees))
            self.local_model = pygnn.PNAConv(dim_h, dim_h,
                                             aggregators=aggregators,
                                             scalers=scalers,
                                             deg=deg,
                                             edge_dim=16, # dim_h,
                                             towers=1,
                                             pre_layers=1,
                                             post_layers=1,
                                             divide_input=False)
        elif local_gnn_type == 'CustomGatedGCN':
            self.local_model = GatedGCNLayer(dim_h, dim_h,
                                             dropout=dropout,
                                             residual=True,
                                             equivstable_pe=equivstable_pe)
        else:
            raise ValueError(f"Unsupported local GNN model: {local_gnn_type}")
        self.local_gnn_type = local_gnn_type

        # torch.nn.functional._scaled_dot_product_attention = bias_scaled_dot_product_attention
        # Global attention transformer-style model.
        if global_model_type == 'None':
            self.self_attn = None
        elif global_model_type == 'Transformer':
            self.self_attn = biasMultiheadAttention(
                dim_h, num_heads, dropout=self.attn_dropout, batch_first=True)
            # self.global_model = torch.nn.TransformerEncoderLayer(
            #     d_model=dim_h, nhead=num_heads,
            #     dim_feedforward=2048, dropout=0.1, activation=F.relu,
            #     layer_norm_eps=1e-5, batch_first=True)
        elif global_model_type == 'Performer':
            self.self_attn = SelfAttention(
                dim=dim_h, heads=num_heads,
                dropout=self.attn_dropout, causal=False)
        elif global_model_type == "BigBird":
            bigbird_cfg.dim_hidden = dim_h
            bigbird_cfg.n_heads = num_heads
            bigbird_cfg.dropout = dropout
            self.self_attn = SingleBigBirdLayer(bigbird_cfg)
        else:
            raise ValueError(f"Unsupported global x-former model: "
                             f"{global_model_type}")
        self.global_model_type = global_model_type

        if self.layer_norm and self.batch_norm:
            raise ValueError("Cannot apply two types of normalization together")

        # Normalization for MPNN and Self-Attention representations.
        if self.layer_norm:
            # self.norm1_local = pygnn.norm.LayerNorm(dim_h)
            # self.norm1_attn = pygnn.norm.LayerNorm(dim_h)
            self.norm1_local = pygnn.norm.GraphNorm(dim_h)
            self.norm1_attn = pygnn.norm.GraphNorm(dim_h)
            # self.norm1_local = pygnn.norm.InstanceNorm(dim_h)
            # self.norm1_attn = pygnn.norm.InstanceNorm(dim_h)
        if self.batch_norm:
            self.norm1_local = nn.BatchNorm1d(dim_h)
            self.norm1_attn = nn.BatchNorm1d(dim_h)
        self.dropout_local = nn.Dropout(dropout)
        self.dropout_attn = nn.Dropout(dropout)

        # Feed Forward block.
        self.activation = F.relu
        self.ff_linear1 = nn.Linear(dim_h, dim_h * 2)
        self.ff_linear2 = nn.Linear(dim_h * 2, dim_h)
        if self.layer_norm:
            # self.norm2 = pygnn.norm.LayerNorm(dim_h)
            self.norm2 = pygnn.norm.GraphNorm(dim_h)
            # self.norm2 = pygnn.norm.InstanceNorm(dim_h)
        if self.batch_norm:
            self.norm2 = nn.BatchNorm1d(dim_h)
        self.ff_dropout1 = nn.Dropout(dropout)
        self.ff_dropout2 = nn.Dropout(dropout)
        if (dataset=='ogbg-molpcba' or dataset=='peptides-structural' or dataset=='peptides-functional'):
            self.structure_map_query = nn.Sequential(
                nn.Linear(16, num_heads))
            # self.structure_map_key = nn.Sequential(
            #     nn.Linear(16, num_heads))
        else:
            self.structure_map_query = nn.Sequential(
                nn.Linear(dim_h, num_heads))
            self.structure_map_key = nn.Sequential(
                nn.Linear(dim_h, num_heads))
        # self.cnn = nn.Sequential(nn.Conv2d(1, num_heads, kernel_size=1), nn.ReLU(), nn.Conv2d(num_heads, num_heads, kernel_size=1))

    def forward(self, batch):
        h = batch.x
        h_in1 = h  # for first residual connection

        h_out_list = []
        # Local MPNN with edge attributes.
        if self.local_model is not None:
            self.local_model: pygnn.conv.MessagePassing  # Typing hint.
            if self.local_gnn_type == 'CustomGatedGCN':
                es_data = None
                if self.equivstable_pe:
                    es_data = batch.pe_EquivStableLapPE
                local_out = self.local_model(Batch(batch=batch,
                                                   x=h,
                                                   edge_index=batch.edge_index,
                                                   edge_attr=batch.edge_attr,
                                                   pe_EquivStableLapPE=es_data))
                # GatedGCN does residual connection and dropout internally.
                h_local = local_out.x
                batch.edge_attr = local_out.edge_attr
            else:
                if self.equivstable_pe:
                    h_local = self.local_model(h, batch.edge_index, batch.edge_attr,
                                               batch.pe_EquivStableLapPE)
                else:
                    h_local = self.local_model(h, batch.edge_index, batch.edge_attr)
                h_local = self.dropout_local(h_local)
                h_local = h_in1 + h_local  # Residual connection.

            if self.layer_norm:
                h_local = self.norm1_local(h_local, batch.batch)
            if self.batch_norm:
                h_local = self.norm1_local(h_local)
            h_out_list.append(h_local)

        # Multi-head attention.
        if self.self_attn is not None:
            h_dense, mask = to_dense_batch(h, batch.batch)
            # dist = 1.0 / dist
            # dist_conv = (self.structure_map_query(batch.dist),self.structure_map_key(batch.dist))
            dist_conv = self.structure_map_query(batch.dist)
            # dist_conv = batch.dist

            if self.global_model_type == 'Transformer':
                h_attn_ = self._sa_block(h_dense, None, ~mask, dist_conv)
                h_attn = h_attn_[0][mask]
                
                # if ((self.training)):
                #     print(batch, batch.ptr.shape[0], self.training)
                #     for index in range(batch.ptr.shape[0]-1):
                #         h_attn_1 = h_attn_[0][index][mask[index]]
                #         attn_1 = h_attn_[1][0][index][mask[index]][:,mask[index]]
                #         attn_2 = h_attn_[1][1][index][mask[index]][:,mask[index]]
                #         plt.clf()
                #         plt.imshow(attn_1.cpu().detach().numpy(), cmap='hot_r', interpolation='nearest')
                #         plt.colorbar()
                #         plt.savefig('attn_1_{}.png'.format(index))
                #         plt.clf()
                #         plt.imshow(attn_2.cpu().detach().numpy(), cmap='hot_r', interpolation='nearest')
                #         plt.colorbar()
                #         plt.savefig('attn_2_{}.png'.format(index))
                #         plt.clf()
                #         batch_egde_1 = unbatch_edge_index(batch.edge_index, batch.batch)[index]
                #         graph = Data(x=h_attn_1, edge_index=batch_egde_1)
                #         graph.tag = graph.x.argmax(dim=-1)
                #         graph.attn1 = attn_1[0]
                #         graph.attn2 = attn_2[0]
                #         graph = utils.to_networkx(graph, node_attrs=['tag', 'attn1', 'attn2'], to_undirected=True)
                #         draw_graph_with_attn(
                #             graph, '.',
                #             'graph{}.png'.format(index),
                #             nodecolor=['tag', 'attn1', 'attn2']
                #         )
            elif self.global_model_type == 'Performer':
                h_attn = self.self_attn(h_dense, mask=mask)[mask]
            elif self.global_model_type == 'BigBird':
                h_attn = self.self_attn(h_dense, attention_mask=mask)
            else:
                raise RuntimeError(f"Unexpected {self.global_model_type}")

            h_attn = self.dropout_attn(h_attn)
            h_attn = h_in1 + h_attn  # Residual connection.
            if self.layer_norm:
                h_attn = self.norm1_attn(h_attn, batch.batch)
            if self.batch_norm:
                h_attn = self.norm1_attn(h_attn)
            h_out_list.append(h_attn)

        # Combine local and global outputs.
        # h = torch.cat(h_out_list, dim=-1)
        h = sum(h_out_list)

        # Feed Forward block.
        h = h + self._ff_block(h)
        if self.layer_norm:
            h = self.norm2(h, batch.batch)
        if self.batch_norm:
            h = self.norm2(h)

        batch.x = h
        return batch

    def _sa_block(self, x, attn_mask, key_padding_mask, dist_conv):
        """Self-attention block.
        """
        x, _ = self.self_attn(x, x, x,
                           attn_mask=attn_mask,
                           key_padding_mask=key_padding_mask,
                           need_weights=True, dist=dist_conv)
        
        return x, _

    def _ff_block(self, x):
        """Feed Forward block.
        """
        x = self.ff_dropout1(self.activation(self.ff_linear1(x)))
        return self.ff_dropout2(self.ff_linear2(x))

    def extra_repr(self):
        s = f'summary: dim_h={self.dim_h}, ' \
            f'local_gnn_type={self.local_gnn_type}, ' \
            f'global_model_type={self.global_model_type}, ' \
            f'heads={self.num_heads}'
        return s
