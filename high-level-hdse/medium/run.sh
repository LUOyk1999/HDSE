
python main-hd.py --method hd --dataset cora --lr 0.01 --num_layers 1 \
            --ghidden_channels 128 --hidden_channels 128 --weight_decay 5e-4 --dropout 0.5 \
            --rand_split_class --valid_num 500 --test_num 1000 --no_feat_norm \
            --seed 123 --num_heads 4 --num_centroids 32 

python main-hd.py --method hd --dataset citeseer --lr 0.01 --num_layers 1 \
          --hidden_channels 128 --weight_decay 0.01 --dropout 0.5 \
          --rand_split_class --valid_num 500 --test_num 1000 --no_feat_norm \
          --seed 123 --num_heads 2 --num_centroids 200 --ghidden_channels 128 

python main-hd.py --method hd --dataset pubmed --lr 0.01 --num_layers 1 \
              --ghidden_channels 128 --hidden_channels 128 --weight_decay 5e-4 --dropout 0.5 \
              --rand_split_class --valid_num 500 --test_num 1000 --no_feat_norm \
              --seed 123 --num_heads 1 --num_centroids 64 

python main-hd.py --method hd --dataset film --lr 0.01 --num_layers 1 \
    --hidden_channels 128 --weight_decay 0.0005 --epoch 1000 --dropout 0.5 --num_heads 2 --num_centroids 200 --ghidden_channels 128 --no_gnn

python main-hd.py --method hd  --dataset squirrel --lr 0.01 --num_layers 3 \
      --hidden_channels 128 --weight_decay 5e-4 --dropout 0.5 --num_heads 1 --num_centroids 128 --ghidden_channels 128 

python main-hd.py --method hd --dataset chameleon --lr 0.01 --num_layers 3 \
            --hidden_channels 128 --weight_decay 0.001 --dropout 0.5 --num_heads 1 --num_centroids 32 --ghidden_channels 128  

