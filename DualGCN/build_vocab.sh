#!/bin/bash
# build vocab for different datasets

python ./DualGCN/prepare_vocab.py --data_dir DualGCN/dataset/Restaurants_corenlp --vocab_dir DualGCN/dataset/Restaurants_corenlp
python ./DualGCN/prepare_vocab.py --data_dir DualGCN/dataset/Laptops_corenlp --vocab_dir DualGCN/dataset/Laptops_corenlp
python ./DualGCN/prepare_vocab.py --data_dir DualGCN/dataset/Tweets_corenlp --vocab_dir DualGCN/dataset/Tweets_corenlp

python ./DualGCN/prepare_vocab.py --data_dir DualGCN/dataset/Restaurants_allennlp --vocab_dir DualGCN/dataset/Restaurants_allennlp
python ./DualGCN/prepare_vocab.py --data_dir DualGCN/dataset/Laptops_allennlp --vocab_dir DualGCN/dataset/Laptops_allennlp
python ./DualGCN/prepare_vocab.py --data_dir DualGCN/dataset/Tweets_allennlp --vocab_dir DualGCN/dataset/Tweets_allennlp

python ./DualGCN/prepare_vocab.py --data_dir DualGCN/dataset/Restaurants_stanza --vocab_dir DualGCN/dataset/Restaurants_stanza
python ./DualGCN/prepare_vocab.py --data_dir DualGCN/dataset/Laptops_stanza --vocab_dir DualGCN/dataset/Laptops_stanza
python ./DualGCN/prepare_vocab.py --data_dir DualGCN/dataset/Tweets_stanza --vocab_dir DualGCN/dataset/Tweets_stanza
