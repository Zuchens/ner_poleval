from collections import defaultdict
import json

import keras
from keras_preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split

from ner.config import parameters
from ner.dataset import load_word_vectors
from sklearn.preprocessing import OneHotEncoder
import numpy as np
import nltk
from ner.model import create_model

import nltk

from nltk.tokenize.treebank import TreebankWordTokenizer

class TreebankSpanTokenizer(TreebankWordTokenizer):

    def __init__(self):
        self._word_tokenizer = TreebankWordTokenizer()

    def span_tokenize(self, text):
        ix = 0
        for word_token in self.tokenize(text):
            ix = text.find(word_token, ix)
            end = ix+len(word_token)
            yield ix, end
            ix = end

    def tokenize(self, text):
        return self._word_tokenizer.tokenize(text)


def convert_entities(entities, tokens):
    iobs = []
    for sentence_entities in entities:
        # TODO implement cutting sentence size
        iob = [set("P", )] * parameters["padding"]
        for idx, entity in enumerate(sentence_entities):
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
                        val = set(iob[idx + i])
                        if (i == 0):
                            val.add("B_" + ne)
                        else:
                            val.add("I_" + ne)
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
                label2idx_iterator += 1
                label2idx[data] = label2idx_iterator
            idx_iob.append(label2idx[data])
        idx_iobs.append(idx_iob)
    return label2idx, idx_iobs


def preprocess():
    with open(parameters["train_dataset_path"]) as f:
        unprocessed_data = json.load(f)["texts"]
        unprocessed_data = unprocessed_data
    if parameters["use_test_file"]:

        with open(parameters["test_dataset_path"]) as f:
            import nltk
            # nltk.download()
            test_data = json.load(f)
            tokenizer = TreebankSpanTokenizer()
            test_tokens = [tokenizer.tokenize(x["text"]) for x in test_data]
            test_spans = [list(tokenizer.span_tokenize(x["text"])) for x in test_data]
            a =1
    # TODO add words from train to vocab
    vectors, word2index = load_word_vectors(parameters["emb_file"])
    tokens = [[word.lower() if parameters["lowercase"] else word for word in doc["tokens"]] for doc in
              unprocessed_data]
    input = [[word2index.get(word, word2index["UNK"]) for word in doc] for doc in tokens]
    uppercase = [[2 if word[0].isupper() else 1 for word in doc] for doc in tokens]
    parameters["padding"] = max([len(doc) for doc in tokens])
    entities = [doc["entities"] for doc in unprocessed_data]
    label2idx, idx_iobs = convert_entities(entities, tokens)

    input = pad_sequences(input, maxlen=parameters["padding"], padding="post")
    uppercase = pad_sequences(uppercase, maxlen=parameters["padding"], padding="post")

    mlb_uppercase = OneHotEncoder()
    mlb_uppercase.fit([[0], [1], [2]])
    uppercase_feature = np.asarray([mlb_uppercase.transform([[i] for i in x]).toarray() for x in uppercase]).tolist()

    mlb_targets = OneHotEncoder()
    mlb_targets.fit([[x] for x in label2idx.values()])
    target = np.asarray([mlb_targets.transform([[i] for i in x]).toarray() for x in idx_iobs]).tolist()



    input_train,input_val,  uppercase_feature_train, uppercase_feature_val, target_train, target_val = train_test_split(
        input,
        uppercase_feature,
        target,
        test_size=parameters["validation_size"],
        shuffle=False)


    model = create_model(vectors, emb_features=vectors.shape[1],feature_size=3,  maxlen=parameters["padding"], output_size=len(target_train[0][0]))
    model.fit([np.asarray(input_train), np.asarray(uppercase_feature_train)], np.asarray(target_train), batch_size=16, nb_epoch=10,
              validation_split=0.1,
              verbose=0)

    loss, train_accuracy = model.evaluate([np.asarray(input_train), np.asarray(uppercase_feature_train)], np.asarray(target_train), verbose=0)
    print('Accuracy test: %f' % (train_accuracy * 100))

    loss, val_accuracy = model.evaluate([np.asarray(input_val), np.asarray(uppercase_feature_val)], np.asarray(target_val), verbose=0)
    print('Accuracy test: %f' % (val_accuracy * 100))

    model.predict([np.asarray(input_val), np.asarray(uppercase_feature_val)], np.asarray(target_val), verbose=0)

if __name__ == "__main__":
    preprocess()
