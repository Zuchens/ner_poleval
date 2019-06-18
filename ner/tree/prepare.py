from copy import deepcopy, copy
from random import shuffle

import torch
from sklearn.model_selection import train_test_split
from torch.nn import Embedding
from ner.preprocess import preprocess_training_data
from ner.tree import config
from ner.tree.config import tree_config
from ner.tree.data.dataset import SSTDataset
from ner.tree.model.training import train


def prepare_embeddings(vectors, word2index):
    embedding_model = Embedding(len(word2index), config.tree_config["input_dim"])

    if config.tree_config["cuda"]:
        embedding_model = embedding_model.cuda()

    torch_vectors = torch.tensor(vectors)

    if config.tree_config["cuda"]:
        torch_vectors = torch_vectors.cuda()
    # plug these into embedding matrix inside model
    embedding_model.state_dict()['weight'].copy_(torch_vectors)
    return  embedding_model


def trees_mockup(input, output):
    ans = []
    for x,y in zip(input,output):
        data = {}
        parents = list(range(len(x)))
        shuffle(parents)
        data['parents'] = parents
        data['labels'] =  deepcopy(y)
        data['tokens'] = deepcopy(x)
        data['relations'] = ['comp'] * len(x)
        ans.append(data)
    return ans


def train_tree(vectors, word2index, model_params):

    target, sentences, label2idx, features = preprocess_training_data(word2index, model_params)
    trees = trees_mockup(sentences,target)
    #TODO add features later
    embedding_model = prepare_embeddings(vectors, word2index)
    train_dataset = SSTDataset(word2index, len(label2idx))
    tree_config['num_classes'] = len(label2idx)
    dev_dataset = SSTDataset(word2index, len(label2idx))
    train_dataset, dev_dataset = split_dataset_random(target, sentences, trees, train_dataset,dev_dataset)
    max_dev_epoch, max_dev_acc, max_model_filename = train(train_dataset, dev_dataset, embedding_model, config.tree_config)


def split_dataset_random(target, sentences, trees, train_dataset,dev_dataset, test_size=0.1 ):
    X_train, X_dev, y_train, y_dev = train_test_split([{'trees':x, 'sentence':y} for x, y in zip(trees, sentences)], target, test_size=test_size, random_state=0)

    train_dataset.create_trees( [x['trees'] for x in X_train])
    dev_dataset.create_trees([x['trees'] for x in X_dev])
    train_dataset.sentences, dev_dataset.sentences = [x['sentence'] for x in X_train], [x['sentence'] for x in X_dev]

    return train_dataset, dev_dataset