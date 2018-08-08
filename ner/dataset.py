import os


import os

from gensim.models import KeyedVectors
from gensim.models.wrappers import FastText
import numpy as np

def load_word_vectors(embeddings_path):
    model = None
    if os.path.isfile(embeddings_path + '.model'):
        model = KeyedVectors.load(embeddings_path + ".model")
    if os.path.isfile(embeddings_path + '.vec'):
        model = FastText.load_word2vec_format(embeddings_path + '.vec')
    if model is None:
        raise Exception("No vaild path to embeddings")
    index2word = model.index2word
    word2index = {}
    vectors = np.zeros((len(index2word)+2, model.vector_size))
    for index, word in enumerate(index2word):
        word2index[word] = index
        vectors[index, :] = model[word]

    index2word.append("UNK")
    word2index["UNK"] = len(index2word)-1
    vectors[word2index["UNK"] , :] = np.random.rand(model.vector_size)
    index2word.append("PAD")
    word2index["PAD"] = len(index2word)-1
    vectors[word2index["PAD"], :] = np.zeros(model.vector_size)
    return vectors, word2index


