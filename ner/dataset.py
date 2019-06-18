import os

import numpy as np
from gensim.models import KeyedVectors
from gensim.models.wrappers import FastText

from ner.vocab import create_vocab


def load_embeddings(embeddings_path):
    # Creating vocab from train and test file
    word2index = create_vocab()
    model = load_embedding_file(embeddings_path)
    vectors = np.zeros((len(word2index), model.vector_size))
    i = 0
    for index, word in enumerate(word2index):
        try:
            vectors[index, :] = model[word]
        except KeyError:  # word not in embedding file
            vectors[index, :] = np.random.rand(model.vector_size)
            i += 1
        except AttributeError:
            vectors[index, :] = np.random.rand(model.vector_size)
            i += 1
    return vectors, word2index


def load_embedding_file(embeddings_path):
    model = None
    if os.path.isfile(embeddings_path + '.model'):
        model = KeyedVectors.load(embeddings_path + ".model")
    if os.path.isfile(embeddings_path + '.vec'):
        model = FastText.load_word2vec_format(embeddings_path + '.vec')
    if model is None:
        raise Exception("No vaild path to embeddings")
    return model

