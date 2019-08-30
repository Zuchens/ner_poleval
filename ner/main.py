import json

from ner.config import parameters, search_parameters
from ner.dataset import load_embeddings
from ner.preprocess import preprocess_training_data
from ner.test import predict_test
from ner.train import train_and_eval, test_validation, test
from ner.tree.prepare import train_tree, test_tree


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
    import gc
    gc.collect()
    categories, input, label2idx, features, dependencies, dependencyLabels = preprocess_training_data(vocabulary,
                                                                                                      search_parameters,
                                                                                                      "train_dataset_path")
    import gc

    gc.collect()
    categoriesTest, inputTest, _, featuresTest, dependenciesTest, dependencyLabelsTest = preprocess_training_data(
        vocabulary,
        search_parameters,
        "test_dataset_path", label2idx, dependencyLabels[1])
    if search_parameters["type"] == "simple":

        values, model, idx2label = train_and_eval(categories, input, label2idx, features, dependencies,
                                                  dependencyLabels, search_parameters, vectors, vocabulary)
        test(model, categoriesTest, inputTest, label2idx, featuresTest, dependenciesTest, dependencyLabelsTest, search_parameters)
        write_output(values, "", search_parameters)
    elif search_parameters["type"] == "tree":
        _, _, model = train_tree(categories, input, label2idx, features, dependencies, dependencyLabels,
                                 search_parameters, vectors, vocabulary)
        test_tree(model, categoriesTest, inputTest, label2idx, featuresTest, dependenciesTest, dependencyLabelsTest,
                  search_parameters)
