from keras.layers import GRU, LSTM
from sklearn.model_selection import ParameterGrid

from ner.config import parameters
from ner.dataset import load_word_vectors
from ner.train import train_and_eval


def grid_search():
    grid_parameters = \
        {   'padding' : [100],
            "lowercase": [False,True],
            'rnn': [GRU, LSTM],
            'output_dim_rnn':  [300],
            'activation_rnn': ['relu', 'tanh'],
            'dropout': [0.5],
            'trainable_embeddings': [False,True],
            'optimizer': ['adam', 'adagrad'],
            'iob': ['io','iob'],
        }
    grid = ParameterGrid(grid_parameters)

    vectors, word2index = load_word_vectors(parameters["emb_file"])

    idx = 0
    for param in list(grid):
        with open('../output/'+str(idx)+'_grid_results.csv', 'w+') as f:
            values = train_and_eval(vectors, word2index, param)
            f.write('{}\t{}'.format(param, values))
            idx+=1


if __name__ == "__main__":
    grid_search()