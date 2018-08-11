import json

from keras_preprocessing.sequence import pad_sequences

from ner.config import parameters
from ner.utils import split_by_sentence_train


def convert_entities(entities, tokens,model_parameters):
    iobs = []
    for sentence_entities in entities:
        # TODO implement cutting sentence size
        iob = [set("P", )] * model_parameters["padding"]
        #TODO quickfix
        for idx, entity in enumerate(sentence_entities[:model_parameters["padding"]]):
            val = set(iob[idx])
            val.add("O")
            iob[idx] = val

            if len(entity) != 0:
                for label in entity:
                    if label.get("subtype"):
                        ne = label["type"] + "_" + label["subtype"]
                    else:
                        ne = label["type"]

                    for i in range(len(label["offsets"])):
                        # TODO quickfix
                        if idx + i >=model_parameters["padding"]:
                            break
                        val = set(iob[idx + i])
                        if(model_parameters["iob"]=="iob"):
                            if (i == 0):
                                val.add("B_" + ne)
                            else:
                                val.add("I_" + ne)
                        else:
                            val.add(ne)
                        iob[idx + i] = val
        iobs.append(iob)
    # for doc, iob in zip(tokens, iobs):
    #     for idx, word in enumerate(doc):
    #
    #         if iob[idx] != set(("P", "O")):
    #             print(word + " " + str(iob[idx]))

    label2idx = {}
    label2idx_iterator = 0

    idx_iobs = []

    for iob_sentence in iobs:
        idx_iob = []
        for iob in iob_sentence:
            iob = sorted(iob)
            data = "-".join(iob)
            if data not in label2idx:
                label2idx[data] = label2idx_iterator
                label2idx_iterator += 1
            idx_iob.append(label2idx[data])
        idx_iobs.append(idx_iob)
    return label2idx, idx_iobs

import re

def create_features(tokens):
    features = []
    for doc in tokens:
        sent_features = []
        for word in doc:
            starts_uppercase = 2 if word[0].isupper() else 1
            has_dot = 2 if "." in word else 1
            has_num = 2 if re.search('\d+', word) else 1
            word_features = [starts_uppercase,has_dot, has_num]
            sent_features.append(word_features)
        features.append(sent_features)
    return features


def preprocess_training_data(word2index, model_parameters):
    with open(parameters["train_dataset_path"]) as f:
        unprocessed_data = json.load(f)["texts"]
        unprocessed_data = unprocessed_data
    unprocessed_data_sentences = split_by_sentence_train(unprocessed_data)
    # TODO add words from train to vocab
    tokens = [[word.lower() if model_parameters["lowercase"] else word for word in doc["tokens"]] for doc in
              unprocessed_data_sentences]
    input = [[word2index.get(word, word2index["UNK"]) for word in doc] for doc in tokens]
    features = create_features(tokens)
    model_parameters["padding"] = max([len(doc) for doc in tokens])
    entities = [doc["entities"] for doc in unprocessed_data_sentences]
    label2idx, idx_iobs = convert_entities(entities, tokens, model_parameters)
    input = pad_sequences(input, maxlen=model_parameters["padding"], padding="post", truncating="post")
    features = pad_sequences(features, maxlen=model_parameters["padding"], padding="post", truncating="post")

    return idx_iobs, input, label2idx, features


