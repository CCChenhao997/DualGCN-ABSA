__author__ = 'max'

import pickle
import numpy as np
import gzip

import utils_io


def load_embedding_dict(embedding, embedding_path, normalize_digits= False):
    """
    load word embeddings from file
    :param embedding:
    :param embedding_path:
    :return: embedding dict, embedding dimention, caseless
    """
    print("loading embedding: %s from %s" % (embedding, embedding_path))
    if embedding == 'glove':
        # loading GloVe
        embedd_dim = -1
        embedd_dict = dict()
        with gzip.open(embedding_path, 'r') as file:
            for line in file:
                line = line.strip()
                line = line.decode('utf-8')
                if len(line) == 0:
                    continue

                tokens = line.split()
                if embedd_dim < 0:
                    embedd_dim = len(tokens) - 1
                else:
                    assert (embedd_dim + 1 == len(tokens))
                embedd = np.empty([1, embedd_dim], dtype=np.float32)
                embedd[:] = tokens[1:]
                word = utils_io.DIGIT_RE.sub(b"0", tokens[0]) if normalize_digits else tokens[0]
                embedd_dict[word] = embedd
        return embedd_dict, embedd_dim
    elif embedding == 'sskip':
        embedd_dim = -1
        embedd_dict = dict()
        with gzip.open(embedding_path, 'r') as file:
            # skip the first line
            file.readline()
            for line in file:
                line = line.strip()
                try:
                    line = line.decode('utf-8')
                    if len(line) == 0:
                        continue

                    tokens = line.split()
                    if len(tokens) < embedd_dim:
                        continue

                    if embedd_dim < 0:
                        embedd_dim = len(tokens) - 1

                    embedd = np.empty([1, embedd_dim], dtype=np.float32)
                    start = len(tokens) - embedd_dim
                    word = ' '.join(tokens[0:start])
                    embedd[:] = tokens[start:]
                    word = utils_io.DIGIT_RE.sub(b"0", word) if normalize_digits else word
                    embedd_dict[word] = embedd
                except UnicodeDecodeError:
                    continue
        return embedd_dict, embedd_dim
    else:
        raise ValueError("embedding should choose from [glove, sskip]")
