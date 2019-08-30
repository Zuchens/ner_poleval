import json
import keras
import nltk
from keras_preprocessing.sequence import pad_sequences

from ner.config import parameters
from ner.treebank_span import TreebankSpanTokenizer
from ner.utils import split_by_sentence_train


# TODO implement cutting sentence size
def convert_entities(token_categories, model_parameters, labels):
    categories = get_categories(token_categories)

    label2idx ={} if not labels else labels
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
    for a, sentence_entities in enumerate(token_categories):
        sentence_categories = [set() for x in range(len(sentence_entities))]
        for idx, token_categories in enumerate(sentence_entities):
            if token_categories:
                for label in token_categories:
                    if label:
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
                if not sentence_categories[idx]:
                    sentence_categories[idx].add("O")

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


def preprocess_training_data(word2index, model_parameters, name, label2idx=None, depLabel=None):
    with open(parameters[name]) as f:
        #TODO remove cutting
        unprocessed_data = json.load(f)["texts"][:100]
    unprocessed_data = [x for x in unprocessed_data if x["tokens"]]
    words = []
    dependencies = []
    dependencyLabels = []
    dependencyLabel2idx = {} if not depLabel else depLabel
    dependencyLabelIterator = 1
    for doc in unprocessed_data:
        word_idx = 0

        for sentDep, sentDepLabel in zip(doc["dependencies"], doc["dependencyLabels"]):
            xwords = []
            xdependencies = []
            xdependencyLabels = []
            for dep, label in zip(sentDep, sentDepLabel):
                if label not in dependencyLabel2idx.keys():
                    dependencyLabel2idx[label] = dependencyLabelIterator
                    dependencyLabelIterator += 1
                if dep == "None":
                    dep = -2
                xdependencies.append(dep)
                xdependencyLabels.append(label)
                if model_parameters["lowercase"]:
                    word = doc["tokens"][word_idx].lower()
                else:
                    word = doc["tokens"][word_idx]
                word_idx += 1
                xwords.append(word)
            dependencies.append(xdependencies)
            dependencyLabels.append(xdependencyLabels)
            words.append(xwords)
            assert len(doc["entities"]) == len(doc["tokens"])

    dependencyLabels = [[dependencyLabel2idx[label] for label in doc] for doc in dependencyLabels]
    embedding_indices = [[word2index.get(word, word2index["UNKNOWN"]) for word in doc] for doc in words]
    features = create_features(words)
    if label2idx == None:
        model_parameters["padding"] = max([len(doc) for doc in words])

    entities = [doc["entities"] for doc in unprocessed_data]

    label2idx, idx_iobs = convert_entities(entities, model_parameters, label2idx)
    new_idx_iobs = []
    i_iob = 0
    idx = 0
    for w in words:
        xnew_idx_iobs = []
        for _ in w:
            val = idx_iobs[i_iob][idx]
            xnew_idx_iobs.append(val)
            idx += 1
        new_idx_iobs.append(xnew_idx_iobs)
        if idx == len(idx_iobs[i_iob]):
            i_iob += 1
            idx = 0
    assert [len(w) for w in words] == [len(i) for i in new_idx_iobs]
    return new_idx_iobs, embedding_indices, label2idx, features, dependencies, (dependencyLabels, dependencyLabel2idx)
