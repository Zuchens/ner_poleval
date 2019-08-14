import json

from ner.config import parameters, search_parameters
from ner.dataset import load_embeddings
from ner.preprocess import preprocess_training_data
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
    # with open("/home/p.zak2/PycharmProjects/ner_poleval/data/train/out2.json") as f:
    #     data = json.load(f)
    # with open("/home/p.zak2/PycharmProjects/ner_poleval/data/train/out-small-2.json", "w") as f:
    #     json.dump({"texts":data["texts"][:100]},f)
    vectors, vocabulary = load_embeddings(parameters["emb_path"])
    # train_tree(vectors, vocabulary, search_parameters)
    if search_parameters["type"] == "simple":
        categories, input, label2idx, features, dependencies, dependencyLabels = preprocess_training_data(vocabulary, search_parameters)
        values, test_data = train_and_eval(categories, input, label2idx, features, dependencies, dependencyLabels, search_parameters, vectors, vocabulary)
        write_output(values, test_data, search_parameters)
