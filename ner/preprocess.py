import json

import nltk
from keras_preprocessing.sequence import pad_sequences

from ner.config import parameters
from ner.treebank_span import TreebankSpanTokenizer
from ner.utils import split_by_sentence_train


# TODO implement cutting sentence size
def convert_entities(token_categories, model_parameters):
    categories = get_categories(token_categories)

    label2idx = {}
    label2idx_iterator = 1

    categories_idx = []
    from collections import defaultdict

    label2count = defaultdict(int)
    for i, iob_sentence in enumerate(categories):
        idx_iob = []
        for sentence_categories in iob_sentence:
            sentence_categories = sorted(sentence_categories)
            data = "-".join(sentence_categories)
            if data not in label2idx:
                label2idx[data] = label2idx_iterator
                label2idx_iterator += 1

            label2count[data] += 1
            idx_iob.append(label2idx[data])
        categories_idx.append(idx_iob)

    print(json.dumps(label2count, indent=2))
    return label2idx, categories_idx


def get_categories(token_categories):
    categories = []
    for sentence_entities in token_categories:
        sentence_categories = []
        for idx, token_categories in enumerate(sentence_entities):
            if token_categories:
                for label in token_categories:
                    if label.get("subtype"):
                        ne = label["type"] + "_" + label["subtype"]
                    else:
                        ne = label["type"]
                    for i in range(len(label["offsets"])):
                        if i == 0:
                            next_token = "B_" + ne
                        else:
                            next_token = "I_" + ne
                        if len(sentence_categories) >= (idx + i + 1):
                            sentence_categories[idx + i].add(next_token)
                        else:
                            sentence_categories.append(set([next_token]))
            else:
                category = set("O")
                sentence_categories.append(category)
        categories.append(sentence_categories)
    return categories


import re


def create_features(tokens):
    features = []
    for doc in tokens:
        sent_features = []
        for word in doc:
            starts_uppercase = 2 if word[0].isupper() else 1
            has_dot = 2 if "." in word else 1
            has_num = 2 if re.search('\d+', word) else 1
            word_features = [starts_uppercase, has_dot, has_num]
            sent_features.append(word_features)
        features.append(sent_features)
    return features


def preprocess_training_data(word2index, model_parameters):
    with open(parameters["train_dataset_path"]) as f:
        unprocessed_data = json.load(f)["texts"]
    # unprocessed_data_sentences = split_by_sentence_train(unprocessed_data)
    # TODO add words from train to vocab
    words = [[word.lower() if model_parameters["lowercase"] else word for word in doc["tokens"]] for doc in
             unprocessed_data]
    embedding_indices = [[word2index.get(word, word2index["UNKNOWN"]) for word in doc] for doc in words]
    features = create_features(words)
    model_parameters["padding"] = max([len(doc) for doc in words])
    # model_parameters["padding"] = max([len(doc["tokens"]) for doc in unprocessed_data])

    entities = [doc["entities"] for doc in unprocessed_data]
    label2idx, idx_iobs = convert_entities(entities, model_parameters)
    return idx_iobs, embedding_indices, label2idx, features
