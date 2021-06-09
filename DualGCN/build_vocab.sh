#!/bin/bash
# build vocab for different datasets
python ./DualGCN/prepare_vocab.py --data_dir DualGCN/dataset/Restaurants_stanza --vocab_dir DualGCN/dataset/Restaurants_stanza
python ./DualGCN/prepare_vocab.py --data_dir DualGCN/dataset/Laptops_stanza --vocab_dir DualGCN/dataset/Laptops_stanza
python ./DualGCN/prepare_vocab.py --data_dir DualGCN/dataset/Tweets_stanza --vocab_dir DualGCN/dataset/Tweets_stanza
