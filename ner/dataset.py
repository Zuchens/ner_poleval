import os

import os

from gensim.models import KeyedVectors
from gensim.models.wrappers import FastText
import numpy as np

#SPECIAL_CHARS = ["unk", ".",",","-","?","!"]
SPECIAL_CHARS = ["UNKNOWN"]


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
    vectors = np.zeros((len(index2word) + len(SPECIAL_CHARS) + 1, model.vector_size))
    for index, word in enumerate(index2word):
        word2index[word] = index + 1
        vectors[index + 1, :] = model[word]

    for idx, word in enumerate(SPECIAL_CHARS):
        assert word not in index2word
        index2word.append(word)
        word2index[word] = index + idx + 2
        vectors[word2index[word], :] = np.random.rand(model.vector_size)

    index2word.append("pad")
    word2index["pad"] = 0
    vectors[0, :] = np.zeros(model.vector_size)
    return vectors, word2index
