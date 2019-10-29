import numpy as np
from keras_preprocessing.sequence import pad_sequences


class Sentence:
    def __init__(self, doc_text):
        self.targets = []
        self.offsets = []
        self.dependency_labels = []
        self.dependencies = []
        self.doc_text = doc_text
        self.words = []
        self.words_idx = []
        self.dependency_labels_idx = []
        self.features = []

    def __repr__(self):
        return " ".join(self.words)

    def __str__(self):
        return " ".join(self.words)

    def get_padded_words_idx(self, padding, padding_label):
        zeros = [padding_label for _ in range(padding)]
        for idx in range(padding):
            if idx < len(self.words_idx):
                zeros[idx] = self.words_idx[idx]
        return zeros

    def get_padded_dependency_labels_idx(self, padding):
        zeros = [0 for _ in range(padding)]
        for idx in range(padding):
            if idx < len(self.dependency_labels_idx):
                zeros[idx] = self.dependency_labels_idx[idx]
        return zeros

    def get_padded_features(self, padding):
        zeros = [[0 for _ in range(len(self.features[0]))] for _ in range(padding)]
        for idx in range(padding):
            if idx < len(self.features):
                zeros[idx] = self.features[idx]
        return zeros

    def get_padded_dependencies(self, padding):
        zeros = [0 for _ in range(padding)]
        for idx in range(padding):
            if idx < len(self.dependencies):
                zeros[idx] = int(self.dependencies[idx])
        return zeros

    def get_padded_target(self, padding):
        zeros = [0 for _ in range(padding)]
        for idx in range(padding):
            if idx < len(self.targets):
                zeros[idx] = self.targets[idx]
        return zeros