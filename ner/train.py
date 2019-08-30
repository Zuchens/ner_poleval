import numpy as np
from keras_preprocessing.sequence import pad_sequences
from sklearn.feature_extraction import DictVectorizer
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder

from tqdm import tqdm
from ner.config import parameters
from ner.preprocess import preprocess_training_data
from ner.simple.model import create_model
from ner.test import predict_test

import tensorflow as tf
sess = tf.Session(config=tf.ConfigProto(log_device_placement=True))

def train_and_eval(categories, input, label2idx, features, dependencies, dependencyLabels, model_params, vectors,
                   vocabulary):
    input = pad_sequences(input, maxlen=model_params["padding"], padding="post", truncating="post")

    features = np.array(pad_sequences(features, maxlen=model_params["padding"], padding="post", truncating="post"))
    dependencies = np.reshape(
        np.array(pad_sequences(dependencies, maxlen=model_params["padding"], padding="post", truncating="post")),
        (features.shape[0], features.shape[1], 1))

    dependencyLabelsArray = np.array(
        pad_sequences(dependencyLabels[0], maxlen=model_params["padding"], padding="post", truncating="post"))
    dependencyLabelsArray = one_hot_encode(dependencyLabelsArray, dependencyLabels[1])

    features = np.concatenate((features, dependencies, dependencyLabelsArray), axis=2)

    categories = pad_sequences(categories, maxlen=model_params["padding"], padding="post", truncating="post")
    target = one_hot_encode(categories, label2idx)

    embeddings_train, embeddings_val, features_train, features_val, target_train, target_val = train_test_split(
        input,
        features,
        target,
        test_size=parameters["validation_size"],
        shuffle=True)
    idx2label = {v: k for k, v in label2idx.items()}

    model, train_accuracy = train(embeddings_train, model_params, target_train, features_train, vectors)
    print('Accuracy train: %f\n' % (train_accuracy * 100))
    values = 'Accuracy train: %f\n' % (train_accuracy * 100)


    values += test_validation(idx2label, embeddings_val, model, target_val, features_val, vocabulary, model_params)

    return values, model, idx2label


def train(input_train, model_params, target_train, uppercase_feature_train, vectors):
    model = create_model(vectors, emb_features=vectors.shape[1], feature_size=uppercase_feature_train.shape[2],
                         maxlen=model_params["padding"],
                         output_size=len(target_train[0][0]), model_parameters=model_params)
    model.fit([np.asarray(input_train), np.asarray(uppercase_feature_train)], np.asarray(target_train), batch_size=64,
              nb_epoch=parameters["epochs"],
              validation_split=0.1,
              verbose=2)
    import gc
    gc.collect()
    loss, train_accuracy = model.evaluate([np.asarray(input_train), np.asarray(uppercase_feature_train)],
                                          np.asarray(target_train), verbose=2)
    return model, train_accuracy


def test(model, categories, input, label2idx, features, dependencies, dependencyLabels, model_params):
    print("Prepare test")
    input = pad_sequences(input, maxlen=model_params["padding"], padding="post", truncating="post")

    features = np.array(pad_sequences(features, maxlen=model_params["padding"], padding="post", truncating="post"))
    dependencies = np.reshape(
        np.array(pad_sequences(dependencies, maxlen=model_params["padding"], padding="post", truncating="post")),
        (features.shape[0], features.shape[1], 1))

    dependencyLabelsArray = np.array(
        pad_sequences(dependencyLabels[0], maxlen=model_params["padding"], padding="post", truncating="post"))
    dependencyLabelsArray = one_hot_encode(dependencyLabelsArray, dependencyLabels[1])

    features = np.concatenate((features, dependencies, dependencyLabelsArray), axis=2)

    categories = pad_sequences(categories, maxlen=model_params["padding"], padding="post", truncating="post")
    target = one_hot_encode(categories, label2idx)
    predictions = model.predict([np.asarray(input), np.asarray(features)], batch_size=16, verbose=2)

    print("Test")
    all = 0
    true = 0
    for sent_idx in tqdm(range(input.shape[0])):

        for token_idx in range(input.shape[1]):
            if token_idx >= model_params["padding"]:
                break
            if 0 != input[sent_idx][token_idx]:
                idx = np.argmax(predictions[sent_idx][token_idx])
                if idx != 0 and idx != 1:
                    if np.argmax(target[sent_idx][token_idx]) == idx:
                        true += 1
                    all += 1
    print("Test evaluation")
    print(true / all)


def test_validation(idx2label, input_val, model, target_val, uppercase_feature_val, vocabulary, model_params):
    index2word = dict((v, k) for k, v in vocabulary.items())
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
                if idx != 0 and idx != 1:
                    if np.argmax(target_val[sent_idx][token_idx]) == idx:
                        true += 1
                    all += 1

                    # print(str(input_val[sent_idx][token_idx]) + " " + index2word[] + " " +idx2label[idx])
    print('Confusion Matrix')
    t = np.argmax(target_val, axis=1).flatten()
    p = np.argmax(val_predictions, axis=1).flatten()

    np.savetxt('test.out', confusion_matrix(t, p), delimiter=',')
    acc = true / all if all != 0 else 0
    print("Accuracy on the validation set " + str(acc))
    return values + " Value on entitied " + str(acc)

def one_hot_encode(categories, label2idx):
    one_hot_categories = OneHotEncoder(sparse=False)
    one_hot_categories.fit([[x] for x in label2idx.values()])
    target = [one_hot_categories.transform([[i] for i in x]) for x in categories]
    return target
