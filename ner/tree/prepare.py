from copy import deepcopy

import torch
from sklearn.model_selection import train_test_split
from torch.nn import Embedding
from ner.tree import config
from ner.tree.config import tree_config
from ner.tree.data.dataset import SSTDataset
from ner.tree.model.sentiment_trainer import SentimentTrainer
from ner.tree.model.training import train


def prepare_embeddings(vectors, word2index):
    embedding_model = Embedding(len(word2index), config.tree_config["input_dim"])

    if config.tree_config["cuda"]:
        embedding_model = embedding_model.cuda()

    torch_vectors = torch.tensor(vectors)

    if config.tree_config["cuda"]:
        torch_vectors = torch_vectors.cuda()
    embedding_model.state_dict()['weight'].copy_(torch_vectors)
    return embedding_model


def trees_mockup(input, relations, parents, output):
    ans = []
    for x, y, par, rel in zip(input, output, parents, relations[0]):
        data = {}
        data['parents'] = [int(x)  if int(x) > 0 else 0 for x in par]
        data['labels'] = deepcopy(y)
        data['tokens'] = deepcopy(x)
        data['relations'] = deepcopy(rel)
        ans.append(data)
    return ans


def test_tree(model, categories, input, label2idx, features, dependencies, dependencyLabels, search_parameters):
    trees = trees_mockup(input, dependencyLabels, dependencies, categories)
    # TODO add features later
    test_dataset = SSTDataset(num_classes= len(label2idx))
    test_dataset.create_trees(trees)
    test_dataset.sentences = input
    trainer = SentimentTrainer(config.tree_config, model, criterion=torch.nn.NLLLoss(), optimizer=None)
    loss, accuracies, outputs, output_trees = trainer.test(test_dataset)
    test_acc = torch.mean(accuracies)
    print("Test acc  "+ str(test_acc))
    # max_dev_epoch, max_dev_acc, model = model.predict(test_dataset)

def train_tree(categories, input, label2idx, features, dependencies, dependencyLabels, search_parameters, vectors,
               vocabulary):
    trees = trees_mockup(input, dependencyLabels, dependencies, categories)
    # TODO add features later
    embedding_model = prepare_embeddings(vectors, vocabulary)
    train_dataset = SSTDataset(num_classes= len(label2idx))
    tree_config['num_classes'] = len(label2idx)
    dev_dataset = SSTDataset(len(label2idx))
    train_dataset, dev_dataset = split_dataset_random(categories, input, trees, train_dataset, dev_dataset)
    max_dev_epoch, max_dev_acc, model = train(train_dataset, dev_dataset, embedding_model, config.tree_config)
    return max_dev_epoch, max_dev_acc, model



def split_dataset_random(target, sentences, trees, train_dataset, dev_dataset, test_size=0.1):
    X_train, X_dev, y_train, y_dev = train_test_split([{'trees': x, 'sentence': y} for x, y in zip(trees, sentences)],
                                                      target, test_size=test_size, random_state=0)

    train_dataset.create_trees([x['trees'] for x in X_train])
    dev_dataset.create_trees([x['trees'] for x in X_dev])
    train_dataset.sentences, dev_dataset.sentences = [x['sentence'] for x in X_train], [x['sentence'] for x in X_dev]

    return train_dataset, dev_dataset
