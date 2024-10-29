python -u train_zinc.py --abs-pe rw --se khopgnn --gnn-type pna2 --dropout 0.2 --k-hop 3 --seed 0 --use-edge-attr --hdse 0 --sat 1
python -u train_zinc.py --abs-pe rw --se khopgnn --gnn-type pna2 --dropout 0.2 --k-hop 3 --seed 1 --use-edge-attr --hdse 0 --sat 1

python train_SBMs.py --dataset CLUSTER --weight-class --abs-pe rw --abs-pe-dim 3 --se gnn --gnn-type pna2 --dropout 0.4 --k-hop 3 --num-layers 16 --dim-hidden 48 --lr 0.0003 --seed 0 --sat 1
python train_SBMs.py --dataset CLUSTER --weight-class --abs-pe rw --abs-pe-dim 3 --se gnn --gnn-type pna2 --dropout 0.4 --k-hop 3 --num-layers 16 --dim-hidden 48 --lr 0.0003 --seed 1 --sat 1
python train_SBMs.py --dataset CLUSTER --weight-class --abs-pe rw --abs-pe-dim 3 --se gnn --gnn-type pna2 --dropout 0.4 --k-hop 3 --num-layers 16 --dim-hidden 48 --lr 0.0003 --seed 2 --sat 1
python train_SBMs.py --dataset CLUSTER --weight-class --abs-pe rw --abs-pe-dim 3 --se gnn --gnn-type pna2 --dropout 0.4 --k-hop 3 --num-layers 16 --dim-hidden 48 --lr 0.0003 --seed 3 --sat 1
python train_SBMs.py --dataset CLUSTER --weight-class --abs-pe rw --abs-pe-dim 3 --se gnn --gnn-type pna2 --dropout 0.4 --k-hop 3 --num-layers 16 --dim-hidden 48 --lr 0.0003 --seed 4 --sat 1

python -u train_SBMs.py --dataset PATTERN --weight-class --abs-pe rw --abs-pe-dim 7 --se gnn --gnn-type pna3 --dropout 0.3 --k-hop 3 --num-layers 6 --lr 0.0003 --seed 0 --sat 1
python -u train_SBMs.py --dataset PATTERN --weight-class --abs-pe rw --abs-pe-dim 7 --se gnn --gnn-type pna3 --dropout 0.3 --k-hop 3 --num-layers 6 --lr 0.0003 --seed 1 --sat 1
python -u train_SBMs.py --dataset PATTERN --weight-class --abs-pe rw --abs-pe-dim 7 --se gnn --gnn-type pna3 --dropout 0.3 --k-hop 3 --num-layers 6 --lr 0.0003 --seed 0 --sat 1  --hdse 1
python -u train_SBMs.py --dataset PATTERN --weight-class --abs-pe rw --abs-pe-dim 7 --se gnn --gnn-type pna3 --dropout 0.3 --k-hop 3 --num-layers 6 --lr 0.0003 --seed 1 --sat 1  --hdse 1
