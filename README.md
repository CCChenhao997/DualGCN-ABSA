# DualGCN

Code and datasets of our paper "Dual Graph Convolutional Networks for Aspect-based Sentiment Analysis" accepted by ACL 2021.



## Requirements

- torch==1.4.0
- scikit-learn==0.23.2
- transformers==3.2.0
- cython==0.29.13
- nltk==3.5

To install requirements, run `pip install -r requirements.txt`.

## Preparation

1. Download and unzip GloVe vectors(`glove.840B.300d.zip`) from [https://nlp.stanford.edu/projects/glove/](https://nlp.stanford.edu/projects/glove/) and put it into  `DualGCN/glove` directory.

2. Prepare vocabulary with:

   `sh DualGCN/build_vocab.sh`

3. Download the best model [best_parser.pt](LAL-Parser/best_model/readme.md) of [LAL-Parser](https://github.com/KhalilMrini/LAL-Parser).

## Training

To train the DualGCN model, run:

`sh DualGCN/run.sh`

## Credits

The code and datasets in this repository are based on [ABSA-PyTorch](https://github.com/songyouwei/ABSA-PyTorch) and [CDT_ABSA](https://github.com/Guangzidetiaoyue/CDT_ABSA).

