# -*- coding: utf-8 -*-
import os
import copy
import argparse
import numpy as np
import pandas as pd
from collections import defaultdict
from sklearn.metrics import confusion_matrix

from sklearn.metrics import roc_auc_score, average_precision_score
import torch
from torch import nn, optim
import torch.nn.functional as F
from torch_geometric.data import DataLoader
from torch_geometric import datasets
import torch_geometric.utils as utils
from sat.models import GraphTransformer
from sat.data import GraphDataset
from sat.utils import count_parameters
from sat.position_encoding import POSENCODINGS
from sat.gnn_layers import GNN_TYPES
from timeit import default_timer as timer
# from sat.metric import accuracy_SBM

def load_args():
    parser = argparse.ArgumentParser(
        description='Structure-Aware Transformer on cifar',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--seed', type=int, default=0,
                        help='random seed')
    parser.add_argument('--dataset', type=str, default="CIFAR10",
                        help='name of dataset')
    parser.add_argument('--num-heads', type=int, default=4, help="number of heads")
    parser.add_argument('--num-layers', type=int, default=6, help="number of layers")
    parser.add_argument('--dim-hidden', type=int, default=64, help="hidden dimension of Transformer")
    parser.add_argument('--dropout', type=float, default=0.2, help="dropout")
    parser.add_argument('--epochs', type=int, default=100,
                        help='number of epochs')
    parser.add_argument('--lr', type=float, default=0.001,
                        help='initial learning rate')
    parser.add_argument('--weight-decay', type=float, default=1e-5, help='weight decay')
    parser.add_argument('--batch-size', type=int, default=16,
                        help='batch size')
    parser.add_argument('--abs-pe', type=str, default=None, choices=POSENCODINGS.keys(),
                        help='which absolute PE to use?')
    parser.add_argument('--abs-pe-dim', type=int, default=20, help='dimension for absolute PE')
    parser.add_argument('--outdir', type=str, default='',
                        help='output path')
    parser.add_argument('--warmup', type=int, default=5000, help="number of iterations for warmup")
    parser.add_argument('--layer-norm', action='store_true', help='use layer norm instead of batch norm')
    parser.add_argument('--use-edge-attr', action='store_true', help='use edge features')
    parser.add_argument('--edge-dim', type=int, default=32, help='edge features hidden dim')
    parser.add_argument('--gnn-type', type=str, default='graphsage',
                        choices=GNN_TYPES,
                        help="GNN structure extractor type")
    parser.add_argument('--k-hop', type=int, default=2, 
        help="Number of hops to use when extracting subgraphs around each node")
    parser.add_argument('--hdse', type=int, default=0)
    parser.add_argument('--global-pool', type=str, default='mean', choices=['mean', 'cls', 'add'],
                        help='global pooling method')
    parser.add_argument('--se', type=str, default="gnn", 
            help='Extractor type: khopgnn, or gnn')
    parser.add_argument('--sat', type=int, default=0)

    args = parser.parse_args()
    args.use_cuda = torch.cuda.is_available()
    args.batch_norm = not args.layer_norm

    args.save_logs = False
    if args.outdir != '':
        args.save_logs = True
        outdir = args.outdir
        if not os.path.exists(outdir):
            try:
                os.makedirs(outdir)
            except Exception:
                pass
        outdir = outdir + '/{}'.format(args.dataset)
        if not os.path.exists(outdir):
            try:
                os.makedirs(outdir)
            except Exception:
                pass
        outdir = outdir + '/seed{}'.format(args.seed)
        if not os.path.exists(outdir):
            try:
                os.makedirs(outdir)
            except Exception:
                pass
        if args.use_edge_attr:
            outdir = outdir + '/edge_attr'
            if not os.path.exists(outdir):
                try:
                    os.makedirs(outdir)
                except Exception:
                    pass
        pedir = 'None' if args.abs_pe is None else '{}_{}'.format(args.abs_pe, args.abs_pe_dim)
        outdir = outdir + '/{}'.format(pedir)
        if not os.path.exists(outdir):
            try:
                os.makedirs(outdir)
            except Exception:
                pass
        bn = 'BN' if args.batch_norm else 'LN'
        if args.se == "khopgnn":
            outdir = outdir + '/{}_{}_{}_{}_{}_{}_{}_{}_{}_{}'.format(
                args.se, args.gnn_type, args.k_hop, args.dropout, args.lr, args.weight_decay,
                args.num_layers, args.num_heads, args.dim_hidden, bn,
            )

        else:
            outdir = outdir + '/{}_{}_{}_{}_{}_{}_{}_{}_{}'.format(
                args.gnn_type, args.k_hop, args.dropout, args.lr, args.weight_decay,
                args.num_layers, args.num_heads, args.dim_hidden, bn,
            )
        if not os.path.exists(outdir):
            try:
                os.makedirs(outdir)
            except Exception:
                pass
        args.outdir = outdir
    return args


def train_epoch(model, loader, criterion, optimizer, lr_scheduler, epoch, use_cuda=False):
    model.train()

    running_loss = 0.0

    tic = timer()
    for i, data in enumerate(loader):
        #print(data)
        size = len(data.y)
        if args.warmup is not None:
            iteration = epoch * len(loader) + i
            for param_group in optimizer.param_groups:
                param_group["lr"] = lr_scheduler(iteration)
        if args.abs_pe == 'lap':
            # sign flip as in Bresson et al. for laplacian PE
            sign_flip = torch.rand(data.abs_pe.shape[-1])
            sign_flip[sign_flip >= 0.5] = 1.0
            sign_flip[sign_flip < 0.5] = -1.0
            data.abs_pe = data.abs_pe * sign_flip.unsqueeze(0)

        if use_cuda:
            data = data.cuda()
        if(args.dataset=='MNIST'):
            data.x = data.x.unsqueeze(-1)
        data.edge_attr = data.edge_attr.unsqueeze(-1)
        # print(data.x.shape,data.edge_attr.shape)
        optimizer.zero_grad()
        output = model(data)
        # print(output.shape, data.y.shape)
        loss = criterion(output, data.y.squeeze())
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item() * size

    toc = timer()
    n_sample = len(loader.dataset)
    epoch_loss = running_loss / n_sample
    print('Train loss: {:.4f} time: {:.2f}s'.format(
          epoch_loss, toc - tic))
    return epoch_loss

def accuracy_SBM(scores, targets):
    targets = targets.cpu().numpy()
    scores = scores.argmax(dim=-1).cpu().numpy()
    return torch.from_numpy(confusion_matrix(targets, scores, labels=np.array([range(10)])).astype('float32'))

def eval_epoch(model, loader, criterion, use_cuda=False, split='Val'):
    model.eval()

    running_loss = 0.0
    n_sample = 0
    cm = torch.zeros((args.num_class, args.num_class))

    tic = timer()
    with torch.no_grad():
        for i, data in enumerate(loader):
            size = len(data.y)
            if use_cuda:
                data = data.cuda()
            if(args.dataset=='MNIST'):
                data.x = data.x.unsqueeze(-1)
            data.edge_attr = data.edge_attr.unsqueeze(-1)
            output = model(data)
            loss = criterion(output, data.y.squeeze())
            # print(output.shape)
            cm += accuracy_SBM(output, data.y.squeeze())

            running_loss += loss.item() * size
            n_sample += size
    toc = timer()

    epoch_loss = running_loss / n_sample
    epoch_acc = torch.mean(
        cm.diag() / cm.sum(1)).item()
    print('{} loss: {:.4f} ACC: {:.4f} time: {:.2f}s'.format(
          split, epoch_loss, epoch_acc, toc - tic))
    return epoch_acc, epoch_loss


def main():
    global args
    args = load_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    print(args)
    data_path = '../datasets'

    num_class = 10
    num_edge_features = 1
    args.num_class = num_class
    train_dset = GraphDataset(datasets.GNNBenchmarkDataset(data_path,name=args.dataset,
        split='train'), degree=True, k_hop=args.k_hop, se=args.se,
        use_subgraph_edge_attr=args.use_edge_attr, data_path=data_path+'/'+args.dataset, train='train', hdse=args.hdse)

    train_loader = DataLoader(train_dset, batch_size=args.batch_size,
            shuffle=True, num_workers=2)

    print(train_dset[0])

    val_dset = GraphDataset(datasets.GNNBenchmarkDataset(data_path,name=args.dataset,
        split='val'), degree=True, k_hop=args.k_hop, se=args.se,
        use_subgraph_edge_attr=args.use_edge_attr, data_path=data_path+'/'+args.dataset, train='val', hdse=args.hdse)
    val_loader = DataLoader(val_dset, batch_size=args.batch_size, shuffle=False, num_workers=2)
    # train_dset = val_dset
    # train_loader = val_loader
    
    abs_pe_encoder = None
    if args.abs_pe and args.abs_pe_dim > 0:
        abs_pe_method = POSENCODINGS[args.abs_pe]
        abs_pe_encoder = abs_pe_method(args.abs_pe_dim, normalization='sym')
        if abs_pe_encoder is not None:
            abs_pe_encoder.apply_to(train_dset)
            abs_pe_encoder.apply_to(val_dset)

    deg = torch.cat([
        utils.degree(data.edge_index[1], num_nodes=data.num_nodes) for
        data in train_dset])

    if(args.dataset!='MNIST'):
        input_size = train_dset[0].x.shape[1]
    else:
        input_size = 1
    model = GraphTransformer(in_size=input_size,
                             num_class=num_class,
                             d_model=args.dim_hidden,
                             sat=args.sat,
                             dim_feedforward=2*args.dim_hidden,
                             dropout=args.dropout,
                             num_heads=args.num_heads,
                             num_layers=args.num_layers,
                             batch_norm=args.batch_norm,
                             abs_pe=args.abs_pe,
                             abs_pe_dim=args.abs_pe_dim,
                             gnn_type=args.gnn_type,
                             use_edge_attr=args.use_edge_attr,
                             num_edge_features=num_edge_features,
                             edge_dim=args.edge_dim,
                             k_hop=args.k_hop,
                             se=args.se,
                             in_embed=False,
                             edge_embed=False,
                             deg=deg,
                             global_pool=args.global_pool) 

    if args.use_cuda:
        model.cuda()
    print("Total number of parameters: {}".format(count_parameters(model)))

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    if args.warmup is None:
        lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min',
                                                            factor=0.5,
                                                            patience=15,
                                                            min_lr=1e-05,
                                                            verbose=False)
    else:
        lr_steps = (args.lr - 1e-6) / args.warmup
        decay_factor = args.lr * args.warmup ** .5
        def lr_scheduler(s):
            if s < args.warmup:
                lr = 1e-6 + s * lr_steps
            else:
                lr = decay_factor * s ** -.5
            return lr

    test_dset = GraphDataset(datasets.GNNBenchmarkDataset(data_path,name=args.dataset,
        split='test'), degree=True, k_hop=args.k_hop, se=args.se,
        use_subgraph_edge_attr=args.use_edge_attr, data_path=data_path+'/'+args.dataset, train='test', hdse=args.hdse)

    test_loader = DataLoader(test_dset, batch_size=args.batch_size, shuffle=False, num_workers=2)
    
    #FIXME
    if abs_pe_encoder is not None:
        abs_pe_encoder.apply_to(test_dset)

    print("Training...")
    best_val_loss = float('inf')
    best_val_score = 0
    best_test_score = 0
    best_model = None
    best_epoch = 0
    logs = defaultdict(list)
    start_time = timer()
    for epoch in range(args.epochs):
        print("Epoch {}/{}, LR {:.6f}".format(epoch + 1, args.epochs, optimizer.param_groups[0]['lr']))
        train_loss = train_epoch(model, train_loader, criterion, optimizer, lr_scheduler, epoch, args.use_cuda)
        val_score, val_loss = eval_epoch(model, val_loader, criterion, args.use_cuda, split='Val')
        test_score, test_loss = eval_epoch(model, test_loader, criterion, args.use_cuda, split='Test')

        if args.warmup is None:
            lr_scheduler.step(val_loss)

        logs['train_loss'].append(train_loss)
        logs['val_score'].append(val_score)
        logs['test_score'].append(test_score)
        if val_score > best_val_score:
            best_val_score = val_score
            best_test_score = test_score
            best_val_loss = val_loss
            best_epoch = epoch
            best_weights = copy.deepcopy(model.state_dict())
        print(best_val_score, best_test_score)

    total_time = timer() - start_time
    print("best epoch: {} best val loss: {:.4f}".format(best_epoch, best_val_loss))
    model.load_state_dict(best_weights)

    print()
    print("Testing...")
    test_score, test_loss = eval_epoch(model, test_loader, criterion, args.use_cuda, split='Test')

    print("test ACC {:.4f}".format(test_score))




if __name__ == "__main__":
    main()
