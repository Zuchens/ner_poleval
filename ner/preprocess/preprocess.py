import json

from ner.config import config_parameters
from ner.preprocess.features import create_features
from ner.preprocess.outputs import convert_entities
from ner.common.sentence import Sentence


def preprocess_training_data(word2index, model_parameters, name, label2idx=None, depLabel=None):
    with open(config_parameters[name]) as f:
        unprocessed_docs = json.load(f)["texts"][:600]
    unprocessed_docs = [x for x in unprocessed_docs if x["tokens"]]
    with open("tested.json", "w") as f:
        json.dump(unprocessed_docs, f, indent=2)
    for doc in unprocessed_docs:
        doc["offsets"] = sorted([int(token) for token in list(doc["offsets2Entities"].keys())])
    dependency_label2idx = {} if not depLabel else depLabel
    dependency_label_iterator = 1

    sentences = []
    for doc in unprocessed_docs:
        word_idx = 0
        text = doc["text"]
        for sentence_dependencies, sentence_dependency_labels in zip(doc["dependencies"], doc["dependencyLabels"]):
            sentence = Sentence(text)
            sentences.append(sentence)
            set_dependencies(dependency_label2idx, dependency_label_iterator, sentence_dependencies, sentence_dependency_labels, sentence)
            set_words(doc, model_parameters, sentence_dependencies, sentence, word2index, word_idx)
            sentence.features = create_features(sentence.words)
            assert len(doc["entities"]) == len(doc["tokens"])

    if label2idx is None:
        model_parameters["padding"] = max([len(doc.words) for doc in sentences])
        # model_parameters["padding"] = 150

    label2idx = set_target(label2idx, model_parameters, sentences, unprocessed_docs)
    return label2idx, dependency_label2idx, sentences


def set_target(label2idx, model_parameters, sentences, unprocessed_docs):
    entities = [doc["entities"] for doc in unprocessed_docs]
    label2idx, idx_iobs = convert_entities(entities, model_parameters, label2idx)
    i_iob = 0
    idx = 0
    for sentence in sentences:
        xnew_idx_iobs = []
        for _ in sentence.words:
            val = idx_iobs[i_iob][idx]
            xnew_idx_iobs.append(val)
            idx += 1
        sentence.targets = xnew_idx_iobs
        if idx == len(idx_iobs[i_iob]):
            i_iob += 1
            idx = 0
    assert [len(w.words) for w in sentences] == [len(i.targets) for i in sentences]
    return label2idx


def set_words(doc, model_parameters, sent_dep, sentence, word2index, word_idx):
    for _ in sent_dep:
        if model_parameters["lowercase"]:
            word = doc["tokens"][word_idx].lower()
        else:
            word = doc["tokens"][word_idx]
        sentence.words.append(word)
        sentence.offsets.append(doc["offsets"][word_idx])
        word_idx += 1
    sentence.words_idx = [word2index.get(word, word2index["UNKNOWN"]) for word in sentence.words]


def set_dependencies(dependency_label2idx, dependency_label_iterator, sent_dep, sent_dep_label, sentence):
    for dep, label in zip(sent_dep, sent_dep_label):
        if label not in dependency_label2idx.keys():
            dependency_label2idx[label] = dependency_label_iterator
            dependency_label_iterator += 1
        if dep == "None":
            dep = -2
        sentence.dependencies.append(dep)
        sentence.dependency_labels.append(label)
    sentence.dependency_labels_idx = [dependency_label2idx[label] for label in sentence.dependency_labels]
