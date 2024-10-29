python -u train_zinc.py --abs-pe rw --dropout 0.2 --seed 0 --use-edge-attr --hdse 3
python -u train_zinc.py --abs-pe rw --dropout 0.2 --seed 1 --use-edge-attr --hdse 3
python -u train_zinc.py --abs-pe rw --dropout 0.2 --seed 2 --use-edge-attr --hdse 3

python -u train_cifar.py --abs-pe lap --abs-pe-dim 8 --dropout 0.2 --num-heads 8 --seed 0 --use-edge-attr
python -u train_cifar.py --abs-pe lap --abs-pe-dim 8 --dropout 0.2 --num-heads 8  --seed 1 --use-edge-attr
python -u train_cifar.py --abs-pe lap --abs-pe-dim 8 --dropout 0.2 --num-heads 8  --seed 2 --use-edge-attr

python -u train_cifar.py --dataset MNIST --abs-pe lap --abs-pe-dim 8 --dropout 0.2 --num-heads 8 --seed 0 --use-edge-attr
python -u train_cifar.py --dataset MNIST --abs-pe lap --abs-pe-dim 8 --dropout 0.2 --num-heads 8  --seed 1 --use-edge-attr
python -u train_cifar.py --dataset MNIST --abs-pe lap --abs-pe-dim 8 --dropout 0.2 --num-heads 8  --seed 2 --use-edge-attr

python train_SBMs.py --dataset CLUSTER --weight-class --abs-pe rw --abs-pe-dim 3 --dropout 0.4 --num-layers 16 --dim-hidden 48 --lr 0.0001 --seed 0 --epochs 300
python train_SBMs.py --dataset CLUSTER --weight-class --abs-pe rw --abs-pe-dim 3 --dropout 0.4 --num-layers 16 --dim-hidden 48 --lr 0.0001 --seed 1 --epochs 300
python train_SBMs.py --dataset CLUSTER --weight-class --abs-pe rw --abs-pe-dim 3 --dropout 0.4 --num-layers 16 --dim-hidden 48 --lr 0.0001 --seed 2 --epochs 300
python train_SBMs.py --dataset CLUSTER --weight-class --abs-pe rw --abs-pe-dim 3 --dropout 0.4 --num-layers 16 --dim-hidden 48 --lr 0.0001 --seed 3 --epochs 300

python -u train_SBMs.py --dataset PATTERN --weight-class --abs-pe rw --abs-pe-dim 7 --dropout 0.2 --num-layers 6 --lr 0.0003 --seed 0
python -u train_SBMs.py --dataset PATTERN --weight-class --abs-pe rw --abs-pe-dim 7 --dropout 0.2 --num-layers 6 --lr 0.0003 --seed 1
python -u train_SBMs.py --dataset PATTERN --weight-class --abs-pe rw --abs-pe-dim 7 --dropout 0.2 --num-layers 6 --lr 0.0003 --seed 2
python -u train_SBMs.py --dataset PATTERN --weight-class --abs-pe rw --abs-pe-dim 7 --dropout 0.2 --num-layers 6 --lr 0.0003 --seed 3