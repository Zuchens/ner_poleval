import json

from ner.config import parameters, search_parameters
from ner.dataset import load_embeddings
from ner.preprocess import preprocess_training_data
from ner.test import predict_test
from ner.train import train_and_eval
from ner.tree.prepare import train_tree


def write_output(values, test_data, search_parameters):
    import os
    if not os.path.exists("output"):
        os.mkdir("output")
    import datetime
    dt = datetime.datetime.now()
    with open('output/train_results-' + str(dt) + '.csv', 'w+') as f:
        f.write('{}\t{}'.format(search_parameters, values))
    with open('output/test_results-' + str(dt) + '.json', 'w+') as f:
        json.dump(test_data, f, indent=2)


if __name__ == "__main__":
    vectors, vocabulary = load_embeddings(parameters["emb_path"])
    categories, input, label2idx, features, dependencies, dependencyLabels = preprocess_training_data(vocabulary, search_parameters, "train_dataset_path")
    categories, input, label2idx, features, dependencies, dependencyLabels = preprocess_training_data(vocabulary,
                                                                                                      search_parameters,
                                                                                                      "test_dataset_path",label2idx,dependencyLabels[1]  )

    # train_tree(categories, input, label2idx, features, dependencies, dependencyLabels, search_parameters, vectors, vocabulary)
    if search_parameters["type"] == "simple":
        values, model, idx2label = train_and_eval(categories, input, label2idx, features, dependencies, dependencyLabels, search_parameters, vectors, vocabulary)
        test_data = predict_test(idx2label, model, vocabulary, search_parameters)
        write_output(values, "", search_parameters)
