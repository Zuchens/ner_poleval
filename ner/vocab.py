import json

from ner.config import config_parameters
from ner.treebank_span import TreebankSpanTokenizer


def add_words_to_vocabulary(vocabulary, i, name):
    with open(config_parameters[name]) as f:
        unprocessed_data = json.load(f)["texts"]
    for sentence in unprocessed_data:
        for word in sentence['tokens']:
            if word.lower() not in vocabulary:
                vocabulary[word.lower()] = i
                i = i + 1
    return i



def create_vocab():
    vocabulary = {'PAD': 0, 'UNKNOWN': 1}
    i = 2
    i = add_words_to_vocabulary(vocabulary, i, "train_dataset_path")
    add_words_to_vocabulary(vocabulary, i, "test_dataset_path")
    return vocabulary