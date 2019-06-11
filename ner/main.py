import json

from ner.config import parameters, search_parameters
from ner.dataset import load_embeddings
from ner.train import train_and_eval


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
    values, test_data = train_and_eval(vectors, vocabulary, search_parameters)
    write_output(values, test_data, search_parameters)
