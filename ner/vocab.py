import json

from ner.config import parameters
from ner.treebank_span import TreebankSpanTokenizer


def add_train_words_to_vocabulary(vocabulary, i):
    with open(parameters["train_dataset_path"]) as f:
        unprocessed_data = json.load(f)["texts"]
    for sentence in unprocessed_data:
        for word in sentence['tokens']:
            if word.lower() not in vocabulary:
                vocabulary[word.lower()] = i
                i = i + 1
    return i


def add_test_words_to_vocabulary(vocabulary, i):
    with open(parameters["test_dataset_path"]) as f:
        test_data = json.load(f)
    tokenizer = TreebankSpanTokenizer()
    for doc in test_data:
        doc['answers'] = ""
        sentence = tokenizer.tokenize(doc["text"])
        for word in sentence:
            if word.lower() not in vocabulary:
                vocabulary[word.lower()] = i
                i = i + 1


def create_vocab():
    vocabulary = {'PAD': 0, 'UNKNOWN': 1}
    i = 2
    i = add_train_words_to_vocabulary(vocabulary, i)
    add_test_words_to_vocabulary(vocabulary, i)
    return vocabulary
