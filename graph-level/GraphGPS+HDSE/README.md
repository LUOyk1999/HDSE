# HDSE

### Python environment setup with Conda

Tested with Python 3.9, PyTorch 1.10.2, and PyTorch Geometric 2.0.4.
```bash
conda create -n graphgps python=3.9
conda activate graphgps

pip install rdkit
pip install torchmetrics
pip install performer-pytorch
pip install ogb
pip install tensorboardX
pip install wandb
pip install communities
pip install networkx-metis
conda clean --all
```

### Benchmarking GPS+HDSE on 11 datasets

Alternatively, you can run them in terminal following the example below. Configs for all 7 datasets are in `configs/GPS/`.

```bash
conda activate graphgps
python main.py --cfg configs/GPS/zinc-GPS+RWSE_3.yaml  --repeat 4  seed 0  wandb.use False
python main.py --cfg configs/GPS/cifar10-GPS-ESLapPE4.yaml --repeat 4  seed 0  wandb.use False 
python main.py --cfg configs/GPS/cluster-GPS-ESLapPE4.yaml --repeat 4  seed 0  wandb.use False 
python main.py --cfg configs/GPS/mnist-GPS-ESLapPE10.yaml --repeat 4  seed 0  wandb.use False 
python main.py --cfg configs/GPS/pattern-GPS-ESLapPE2.yaml --repeat 4  seed 0  wandb.use False 
python main.py --cfg configs/GPS/peptides-func-GPS3.yaml --repeat 4  seed 0  wandb.use False 
python main.py --cfg configs/GPS/peptides-struct-GPS10.yaml --repeat 4  seed 0  wandb.use False 
```
