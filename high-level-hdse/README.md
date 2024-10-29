# High-level HDSE

## Python environment setup with Conda

Tested with Python 3.7, PyTorch 1.12.1, and PyTorch Geometric 2.3.1, dgl 1.0.2.
```bash
pip install pandas
pip install scikit_learn
pip install numpy
pip install scipy
pip install einops
pip install ogb
pip install googledrivedownloader
pip install networkx
pip install networkx-metis
```

## Dataset

One can download the datasets (Planetoid, Actor/Film) from the google drive link below:

https://drive.google.com/drive/folders/1rr3kewCBUvIuVxA6MJ90wzQuF-NnCRtf?usp=drive_link (provided by Qitian Wu and Wentao Zhao and Chenxiao Yang and Hengrui Zhang and Fan Nie and Haitian Jiang and Yatao Bian and Junchi Yan, Simplifying and empowering transformers for large-graph representations. In Thirty-seventh Conference on Neural Information Processing Systems, 2023b.)

Download the geom-gcn folder, place it in `./data/` and unzip it.
Download the Planetoid folder, place it in `./data/` and unzip it. Rename `Cora` to `cora`. Delete all `processed` folders.

For Chameleon and Squirrel, we use the [new splits](https://github.com/yandex-research/heterophilous-graphs/tree/main/data) that filter out the overlapped nodes. (provided by Qitian Wu and Wentao Zhao and Chenxiao Yang and Hengrui Zhang and Fan Nie and Haitian Jiang and Yatao Bian and Junchi Yan, Simplifying and empowering transformers for large-graph representations. In Thirty-seventh Conference on Neural Information Processing Systems, 2023b.)

Download `chameleon_filtered.npz`, put it into `./data/geom-gcn/chameleon/`.
Download `squirrel_filtered.npz`, put it into `./data/geom-gcn/squirrel/`.

For the OGB datasets, they will be downloaded automatically when running the code.

## Run the codes

Please refer to the bash script `run.sh` in each folder for running the training and evaluation pipeline.

