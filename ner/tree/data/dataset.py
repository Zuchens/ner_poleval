import os
from copy import deepcopy

import torch
import torch.utils.data as data
from tqdm import tqdm

from  ner.tree.data import constants
from ner.tree.model.tree import Tree


class SSTDataset(data.Dataset):
    """
    A wrapper class for dataset in the format of Stanford Sentiment Treebank 
    (SST) (https://nlp.stanford.edu/sentiment/)
    """

    def __init__(self, vocab=None, num_classes=None, ):
        super(SSTDataset, self).__init__()
        # self.vocab = vocab
        self.num_classes = num_classes
        self.sentences = []
        self.labels = []

    def set_labels(self):
        self.labels = []
        for i in range(0, len(self.trees)):
            self.labels.append(self.trees[i].gold_label)
        self.labels = torch.Tensor(self.labels)

    @classmethod
    def create_dataset_from_user_input(cls, sentence_path, parents_path,
                                       vocab=None, num_classes=None):
        dataset = cls()
        dataset.vocab = vocab
        dataset.num_classes = num_classes
        parents_file = open(parents_path, 'r', encoding='utf-8')
        tokens_file = open(sentence_path, 'r', encoding='utf-8')
        dataset.trees = [
            dataset.read_tree(parents, 0, tokens, tokens)
            for parents, tokens in zip(parents_file.readlines(),
                                       tokens_file.readlines())
        ]
        dataset.sentences = dataset.read_sentences(sentence_path)
        dataset.labels = torch.Tensor(len(dataset.sentences))
        return dataset

    def __len__(self):
        return len(self.trees)

    def __getitem__(self, index):
        tree = deepcopy(self.trees[index])
        sent = deepcopy(self.sentences[index])
        label = deepcopy(self.labels[index])
        return tree, sent, label

    def create_trees(self, data):
            self.trees = self.read_trees(
                parents_data=[x['parents'] for x in data],
                labels_data=[x['labels'] for x in data],
                tokens_data=[x['tokens'] for x in data],
                relations_data=[x['relations'] for x in data],
            )

            self.set_labels()

    def read_sentences(self, filename):
        with open(filename, 'r', encoding='utf-8') as f:
            sentences = [self.read_sentence(line)
                         for line in tqdm(f.readlines())]
        return sentences

    def read_sentence(self, line):
        # indices = self.vocab.convert_to_idx(line.split(), constants.UNK_WORD)
        return torch.LongTensor(line)

    def read_trees(self, parents_data, labels_data, tokens_data, relations_data):
        if labels_data:
            iterator = zip(parents_data, labels_data,
                           tokens_data, relations_data)
            trees = [self.read_tree(parents, labels, tokens, relations)
                     for parents, labels, tokens, relations in tqdm(iterator)]
        else:
            iterator = zip(parents_data, tokens_data, relations_data)
            trees = [self.read_tree(parents, None, tokens, relations)
                     for parents, tokens, relations in tqdm(iterator)]

        return trees

    def parse_label(self, label):
        return int(label) + 1

    def read_tree(self, parents, labels, words, relations):
        trees = dict()
        root = None

        for i in range(1, len(parents) + 1):
            if i not in trees.keys():
                idx = i
                prev = None
                while True:
                    parent = parents[idx - 1]
                    tree = Tree()
                    if prev:
                        tree.add_child(prev)
                    trees[idx] = tree
                    tree.idx = idx
                    if labels:
                        tree.gold_label = labels[idx - 1]
                    else:
                        tree.gold_label = None
                    tree.word = words[idx - 1]
                    tree.relation = relations[idx - 1]
                    if parent in trees.keys():
                        trees[parent].add_child(tree)
                        break
                    elif parent == 0:
                        root = tree
                        break
                    else:
                        prev = tree
                        idx = parent
        # helper for visualization
        root._viz_all_children = trees
        root._viz_sentence = words
        root._viz_relations = relations
        root._viz_labels = labels
        return root

