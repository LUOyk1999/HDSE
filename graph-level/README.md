# HDSE on graph-level tasks

## Python environment setup with Conda

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

## Overview

* `./GraphGPS+HDSE` Experiment code of GraphGPS+HDSE.

* `./GT+HDSE` Experiment code of GT+HDSE and SAT+HDSE.

