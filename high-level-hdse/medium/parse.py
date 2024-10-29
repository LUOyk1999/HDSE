from model import Transformer


def parse_method(method, args, c, d, device):
    if method == 'hd':
        model = Transformer(
        in_channels=d,
        hidden_channels=args.hidden_channels, 
        out_channels=c,
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
        no_gnn=args.no_gnn
        ).to(device)
    else:
        raise ValueError(f'Invalid method {method}')
    return model


def parser_add_main_args(parser):
    # dataset and evaluation
    parser.add_argument('--data_dir', type=str, default='../data/')
    parser.add_argument('--dataset', type=str, default='cora')
    parser.add_argument('--device', type=int, default=7,
                        help='which gpu to use if any (default: 0)')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--cpu', action='store_true')
    parser.add_argument('--epochs', type=int, default=500)
    parser.add_argument('--runs', type=int, default=10,
                        help='number of distinct runs')
    parser.add_argument('--train_prop', type=float, default=.5,
                        help='training label proportion')
    parser.add_argument('--valid_prop', type=float, default=.25,
                        help='validation label proportion')
    parser.add_argument('--protocol', type=str, default='semi',
                        help='protocol for cora datasets, semi or supervised')
    parser.add_argument('--rand_split', action='store_true',
                        help='use random splits')
    parser.add_argument('--rand_split_class', action='store_true',
                        help='use random splits with a fixed number of labeled nodes for each class')
    parser.add_argument('--label_num_per_class', type=int, default=20,
                        help='labeled nodes per class(randomly selected)')
    parser.add_argument('--valid_num', type=int, default=500,
                        help='Total number of validation')
    parser.add_argument('--test_num', type=int, default=500,
                        help='Total number of test')
    
    # model
    parser.add_argument('--method', type=str, default='hd')
    parser.add_argument('--hidden_channels', type=int, default=64)
    parser.add_argument('--num_layers', type=int, default=2,
                        help='number of layers for deep methods')
    parser.add_argument('--num_heads', type=int, default=1,
                        help='number of heads for attention')
    parser.add_argument('--no_gnn', action='store_true')

    parser.add_argument('--attn_dropout', type=float, default=0.5)
    parser.add_argument('--ff_dropout', type=float, default=0.5)
    parser.add_argument('--num_centroids', type=int, default=64) 
    parser.add_argument('--no_bn', action='store_true')
    parser.add_argument('--norm_type', type=str, default='batch_norm')

    parser.add_argument('--gnum_layers', type=int, default=2)
    parser.add_argument('--ghidden_channels', type=int, default=64)
    parser.add_argument('--gdropout', type=float, default=0.5)

    # training
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--weight_decay', type=float, default=5e-3)
    parser.add_argument('--dropout', type=float, default=0.5)

    # display and utility
    parser.add_argument('--display_step', type=int,
                        default=50, help='how often to print')

    parser.add_argument('--no_feat_norm', action='store_true',
                        help='Not use feature normalization.')


def parser_add_default_args(args):
    args.global_dim=args.hidden_channels