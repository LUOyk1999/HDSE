# Base code of ogbn-100M sourced from https://github.com/qitianwu/SGFormer (SGFormer: Simplified Graph Transformers)
# Modifications have been applied to adapt to our experiment

# metis 
python -u metis_graph.py --num_parts 1024 --output data_1024

# pretrain
python -u nb-sample.py --dataset ogbn-papers100M --method ours --lr 0.001 --num_layers 3 \
    --hidden_channels 256 --dropout 0.2 --weight_decay 1e-5 --use_residual --use_weight --use_bn --use_init --use_act \
    --ours_layers 1 --ours_dropout 0.5 --ours_use_residual --ours_use_weight --ours_use_bn \
    --use_graph \
    --batch_size 1000  --seed 123 --runs 1 --epochs 23 --display_step 5 --device 6 --save_model --data_dir ./dataset/ --num_centroids 1024

# finetune
python -u nb-sample.py --dataset ogbn-papers100M --data_dir ./dataset/ --method ours --lr 0.0001 --num_layers 3 \
    --hidden_channels 256 --dropout 0.2 --weight_decay 1e-5 --use_residual --use_weight --use_bn --use_init --use_act \
    --ours_layers 1 --ours_dropout 0.5 --ours_use_residual --ours_use_weight --ours_use_bn \
    --use_graph \
    --batch_size 1000  --seed 123 --runs 5 --epochs 10 --display_step 1 --device 0 --save_model \
    --use_pretrained --model_dir models/ogbn-papers100M_ours_23_1_1024.pt --num_centroids 1024
