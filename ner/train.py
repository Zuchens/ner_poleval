import numpy as np
from collections import defaultdict
from keras_preprocessing.sequence import pad_sequences
from matplotlib.pyplot import clf
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder

from ner.common.phrase import Phrase
from ner.config import config_parameters
from ner.simple.model import create_model
import json
import tensorflow as tf

sess = tf.Session(config=tf.ConfigProto(log_device_placement=True))

def get_features(sentences, padding, dep2idx):
    features = np.array([sentence.get_padded_features(padding) for sentence in sentences])
    dependencies = np.array([sentence.get_padded_dependencies(padding) for sentence in sentences])
    dependencies = np.reshape(dependencies, (features.shape[0], features.shape[1], 1))
    dep_labels = []
    for sentence in sentences:
        dep_labels.append(one_hot_encode(sentence.get_padded_dependency_labels_idx(padding), dep2idx))
    dep_labels = np.array(dep_labels)
    A = np.concatenate((features, dependencies, dep_labels), axis=2)
    return A


def train_and_eval(parameters, label2idx, vocab, sentences, dep2idx):
    print(json.dumps(label2idx, indent=2))

    data_train, data_val = train_test_split(sentences, test_size=config_parameters["validation_size"], shuffle=True)
    features_train = get_features(data_train, parameters["padding"], dep2idx)
    targets_train = get_targets(label2idx, parameters, data_train)
    input_train = get_input(parameters, data_train, vocab)
    model, train_accuracy = train(input_train, parameters, targets_train, features_train, vocab.vectors)
    print('Accuracy train: %f\n' % (train_accuracy * 100))
    values = 'Accuracy train: %f\n' % (train_accuracy * 100)

    features_val = get_features(data_val, parameters["padding"], dep2idx)
    targets_val = get_targets(label2idx, parameters, data_val)
    input_val = get_input(parameters, data_val, vocab)

    idx2label = {v: k for k, v in label2idx.items()}
    test_validation(idx2label, input_val, model, targets_val, features_val, vocab.vocabulary, parameters, sentences)

    return values, model, idx2label


def get_input(parameters, sentences, vocab):
    return np.array(
        [sentence.get_padded_words_idx(parameters["padding"], vocab.vocabulary["PAD"]) for sentence in sentences])


def get_targets(label2idx, parameters, sentences):
    target = []
    for sentence in sentences:
        target.append(one_hot_encode(sentence.get_padded_target(parameters["padding"]), label2idx))
    return np.array(target)

def train(input_train, model_params, target_train, uppercase_feature_train, vectors):
    import numpy as np
    class_weight = {x: 25 for x in range(0, target_train.shape[2])}
    class_weight[0] = 1
    class_weight[1] = 2
    model = create_model(vectors, emb_features=vectors.shape[1], feature_size=uppercase_feature_train.shape[2],
                         maxlen=model_params["padding"],
                         output_size=len(target_train[0][0]), model_parameters=model_params, class_weights=class_weight)
    history = model.fit([input_train, uppercase_feature_train], target_train,
                        batch_size=128,
                        nb_epoch=config_parameters["epochs"],
                        validation_split=0.1,
                        verbose=2)
    import matplotlib.pyplot as plt
    print(history.history.keys())
    plt.plot(history.history['acc'])
    plt.plot(history.history['val_acc'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.savefig('accuracy.png')

    clf()
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.savefig('loss.png')

    # loss, train_accuracy = model.evaluate([np.asarray(input_train), np.asarray(uppercase_feature_train)],
    #                                       np.asarray(target_train), verbose=2)
    return model, 0


# def test(model, categories, input, label2idx, features, dependencies, dependencyLabels, model_params):
#     print("Prepare test")
#     input = pad_sequences(input, maxlen=model_params["padding"], padding="post", truncating="post")
#
#     features = np.array(pad_sequences(features, maxlen=model_params["padding"], padding="post", truncating="post"))
#     dependencies = np.reshape(
#         np.array(pad_sequences(dependencies, maxlen=model_params["padding"], padding="post", truncating="post")),
#         (features.shape[0], features.shape[1], 1))
#
#     dependencyLabelsArray = np.array(
#         pad_sequences(dependencyLabels[0], maxlen=model_params["padding"], padding="post", truncating="post"))
#     dependencyLabelsArray = one_hot_encode(dependencyLabelsArray, dependencyLabels[1])
#
#     features = np.concatenate((features, dependencies, dependencyLabelsArray), axis=2)
#
#     categories = pad_sequences(categories, maxlen=model_params["padding"], padding="post", truncating="post")
#     target = one_hot_encode(categories, label2idx)
#     predictions = model.predict([np.asarray(input), np.asarray(features)], batch_size=128, verbose=2)
#
#     print("Test")
#     all = 0.01
#     true = 0
#     for sent_idx in tqdm(range(input.shape[0])):
#
#         for token_idx in range(input.shape[1]):
#             if token_idx >= model_params["padding"]:
#                 break
#             if 0 != input[sent_idx][token_idx]:
#                 idx = np.argmax(predictions[sent_idx][token_idx])
#                 if idx != 0 and idx != 1:
#                     if np.argmax(target[sent_idx][token_idx]) == idx:
#                         true += 1
#
#                     all += 1
#     print("Test evaluation")
#     print(true / all)


def test_validation(idx2label, input_val, model, target_val, uppercase_feature_val, vocabulary, model_params, offsets):
    index2word = dict((v, k) for k, v in vocabulary.items())
    loss, val_accuracy = model.evaluate([input_val, uppercase_feature_val],target_val, verbose=2)
    print('Accuracy val (default): %f \n' % (val_accuracy * 100))
    val_predictions = model.predict([input_val, uppercase_feature_val], verbose=2)
    print("Test evaluation")
    # print(val_predictions)
    all = 0
    true = 0
    labels = ["persName", "persName_addName", "persName_surname", "persName_forename",
              "placeName_settlement", "placeName_country", "placeName_district", "placeName_bloc", "placeName_region",
              "placeName",
              "orgName", "date", "time", "geogName"]

    predictions = []
    targets = []

    for sent_idx in range(input_val.shape[0]):
        predictions_sent = dict()
        for i in labels:
            predictions_sent[i] = [0 for x in range(input_val.shape[1])]

        targets_sent = dict()
        for i in labels:
            targets_sent[i] = [0 for x in range(input_val.shape[1])]

        for token_idx in range(input_val.shape[1]):
            idx = np.argmax(val_predictions[sent_idx][token_idx])
            idx_target = np.argmax(target_val[sent_idx][token_idx])
            if idx_target > 1 or (idx_target == 1 and idx > 2):
                if idx_target == idx:
                    true += 1
                else:
                    print("{} : {}\t{}".format(idx2label[idx], idx2label[idx_target],
                                               index2word[input_val[sent_idx][token_idx]]))
                sentence_pred = idx2label[idx].split("-")
                for i in sentence_pred:
                    if (i.startswith("B")):
                        predictions_sent[i[2:]][token_idx] = 2
                    if (i.startswith("I")):
                        predictions_sent[i[2:]][token_idx] = 1

                sentence_target = idx2label[idx_target].split("-")
                for i in sentence_target:
                    if (i.startswith("B")):
                        targets_sent[i[2:]][token_idx] = 2
                    if (i.startswith("I")):
                        targets_sent[i[2:]][token_idx] = 1
                all += 1
        predictions.append(predictions_sent)
        targets.append(targets_sent)

    save_predictions(offsets, targets, "out/validation_target.json")
    save_predictions(offsets, predictions, "out/validation_prediction.json")
    print('Confusion Matrix')
    predictions = np.argmax(target_val, axis=1).flatten()
    p = np.argmax(val_predictions, axis=1).flatten()

    np.savetxt('test.out', confusion_matrix(predictions, p), delimiter=',')
    acc = true / all if all != 0 else 0
    print("Accuracy on the validation set " + str(acc))


def save_predictions(sentence, targets, name):
    phrase_per_sentence = defaultdict(list)
    for sentence_pred, sentence in zip(targets, sentence):
        text = sentence.doc_text
        offsets = sentence.offsets
        phrases_predictions = []
        for category in sentence_pred.keys():
            idx = 0
            while idx < len(sentence_pred[category]):
                if sentence_pred[category][idx] == 1:
                    phrase = Phrase(idx)
                    while sentence_pred[category][idx] == 1:
                        phrase.end_idx = idx
                        idx += 1
                    phrase.category = category
                    phrase.start = offsets[phrase.start_idx]
                    phrase.end = offsets[phrase.end_idx + 1] \
                        if phrase.end_idx + 1 < len(offsets) else offsets[phrase.end_idx]
                    if text[phrase.start: phrase.end].endswith(" "):
                        phrase.end -= 1
                    phrase.text = text[phrase.start: phrase.end]
                    phrases_predictions.append(str(phrase))
                idx += 1
        phrase_per_sentence[text].extend(phrases_predictions)
    phrase_per_sentence = [{"text": key, "answers": "\n".join(values)} for key, values in phrase_per_sentence.items()]
    with open(name, "w") as f:
        json.dump(phrase_per_sentence, f, indent=2, ensure_ascii=False)


def one_hot_encode(categories, label2idx):
    one_hot_categories = OneHotEncoder(sparse=False)
    one_hot_categories.fit([[x] for x in label2idx.values()])
    return one_hot_categories.transform([[i] for i in categories])
