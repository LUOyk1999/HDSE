python arxiv_node2vec.py
python arxiv_ERM_ns.py --dataset arxiv-year --lr 1e-3 --batch_size 1024 --test_batch_size 256 --hidden_dim 128 --test_freq 5 --num_workers 4 --conv_type full --num_heads 4 --num_centroids 2048 --hetero_train_prop 0.5
