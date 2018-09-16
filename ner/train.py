import json

import numpy as np
import keras
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder

from ner.config import parameters, search_parameters
from ner.dataset import load_word_vectors, load_word_vectors_with_dictionary
from ner.model import create_model
from ner.preprocess import preprocess_training_data
from ner.test import predict_test
from ner.treebank_span import TreebankSpanTokenizer


def train_and_eval(vectors, word2index, model_params):

    idx_iobs, input, label2idx, features = preprocess_training_data(word2index, model_params)
    features = np.asarray(features)
    target = one_hot_encode(idx_iobs, label2idx)

    input_train, input_val, uppercase_feature_train, uppercase_feature_val, target_train, target_val = train_test_split(
        input,
        features,
        target,
        test_size=parameters["validation_size"],
        shuffle=True)

    model = create_model(vectors, emb_features=vectors.shape[1], feature_size=3, maxlen=model_params["padding"],
                         output_size=len(target_train[0][0]), model_parameters=model_params)

    idx2label = {v: k for k, v in label2idx.items()}

    model.fit([np.asarray(input_train), np.asarray(uppercase_feature_train)], np.asarray(target_train), batch_size=16,
              nb_epoch=parameters["epochs"],
              validation_split=0.1,
              verbose=2)

    loss, train_accuracy = model.evaluate([np.asarray(input_train), np.asarray(uppercase_feature_train)],
                                          np.asarray(target_train), verbose=2)
    values = 'Accuracy train: %f\n' % (train_accuracy * 100)

    values += test_validation(idx2label, input_val, model, target_val, uppercase_feature_val, word2index, model_params)

    test_data = predict_test(idx2label, model, word2index, model_params)
    print(values)

    print("Saved model to disk")

    return values, test_data


def test_validation(idx2label, input_val, model, target_val, uppercase_feature_val, word2index, model_params):
    index2word = dict((v, k) for k, v in word2index.items())
    loss, val_accuracy = model.evaluate([np.asarray(input_val), np.asarray(uppercase_feature_val)],
                                        np.asarray(target_val), verbose=2)
    values = 'Accuracy val: %f \n' % (val_accuracy * 100)
    val_predictions = model.predict([np.asarray(input_val), np.asarray(uppercase_feature_val)], verbose=2)
    all = 0
    true = 0
    for sent_idx in range(input_val.shape[0]):
        for token_idx in range(input_val.shape[1]):
            if token_idx >= model_params["padding"]:
                break
            if 0 != input_val[sent_idx][token_idx]:
                idx = np.argmax(val_predictions[sent_idx][token_idx])
                if idx2label[idx] != 'O-P':
                    if np.argmax(target_val[sent_idx][token_idx]) == idx:
                        true+=1
                    all+=1
                # print(str(input_val[sent_idx][token_idx]) + " " + index2word[] + " " +idx2label[idx])
    print('Confusion Matrix')
    t = np.argmax(target_val, axis=1).flatten()
    p = np.argmax(val_predictions, axis=1).flatten()

    np.savetxt('test.out', confusion_matrix(t, p), delimiter=',')
    acc = true/all if all != 0 else 0
    return values+ " Value on entitied "+  str(acc)

def one_hot_encode(idx_iobs, label2idx):
    mlb_targets = OneHotEncoder(sparse=False)
    mlb_targets.fit([[x] for x in label2idx.values()])
    target = [mlb_targets.transform([[i] for i in x]) for x in idx_iobs]
    return target


if __name__ == "__main__":

    with open(parameters["train_dataset_path"]) as f:
        unprocessed_data = json.load(f)["texts"]
    word2index = {'PAD': 0, "UNKNOWN":1}
    i = 1
    for sentence in unprocessed_data:
        for word in sentence:
            if word not in word2index:
                word2index[word] = i
                i = i + 1
    with open(parameters["test_dataset_path"]) as f:
        test_data = json.load(f)
    tokenizer = TreebankSpanTokenizer()
    for doc in test_data:
        doc['answers'] = ""
        sentence = tokenizer.tokenize(doc["text"])
        for word in sentence:
            if word not in word2index:
                word2index[word] = i
                i = i + 1

    vectors, word2index = load_word_vectors_with_dictionary(parameters["emb_file"], word2index)
    values, test_data = train_and_eval(vectors, word2index, search_parameters)
    import os
    if not os.path.exists("output"):
        os.mkdir("output")
    import datetime
    dt = datetime.datetime.now()
    with open('output/train_results-'+str(dt)+'.csv', 'w+') as f:
        f.write('{}\t{}'.format(search_parameters, values))
    with open('output/test_results-'+str(dt)+'.json', 'w+') as f:
        json.dump(test_data, f, indent=2)