import json

import nltk
import numpy as np
from keras_preprocessing.sequence import pad_sequences
from tqdm import tqdm
from ner.config import config_parameters
from ner.preprocess.preprocess import create_features
from ner.treebank_span import TreebankSpanTokenizer


def predict_test(label2idx, model, word2index, model_params):
    if config_parameters["use_test_file"]:
        with open(config_parameters["test_dataset_path"]) as f:
            test_data = json.load(f)

        # prepare_data(categories, dependencies, dependencyLabels, features, input, label2idx, model_params)
        tokenizer = TreebankSpanTokenizer()
        sent_tokenizer = nltk.PunktSentenceTokenizer()
        for doc in test_data["texts"]:
            doc["text"] = doc["text"].replace("\"","|")
            doc["text"] = doc["text"].replace("\n", " ")
            text = doc["text"]
            sentences = sent_tokenizer.tokenize(text)
            test_spans_sentences = list(sent_tokenizer.span_tokenize(doc["text"]))
            for sent, span in zip(sentences, test_spans_sentences):
                assert text[span[0]:span[1]]== sent
            # test_spans_sentences = list(sent_tokenizer.sentence_span_tokenize(doc["text"]))
            test_tokens = [tokenizer.tokenize(x) for x in sentences]
            test_spans = [list(tokenizer.span_tokenize(x)) for x in sentences]
            for t_sent, t_span, sent, span in zip(test_tokens, test_spans, sentences, test_spans_sentences):
                for t_token, t_token_span in zip(t_sent, t_span):
                    assert sent[t_token_span[0]:t_token_span[1]] == t_token
                    assert text[span[0]+t_token_span[0]:span[0]+t_token_span[1]] == t_token
            input_test = [[word2index.get(word, word2index["UNKNOWN"]) for word in x] for x in test_tokens]
            features = create_features(test_tokens)

            input_test = pad_sequences(input_test, maxlen=model_params["padding"], padding="post")
            test_uppercase = pad_sequences(features, maxlen=model_params["padding"], padding="post")


            predictions = model.predict([np.asarray(input_test), np.asarray(test_uppercase)], verbose=2)

            test_labels = []
            for sent_idx, sentence in tqdm(enumerate(test_tokens)):
                consecutive = []
                for token_idx, token in enumerate(sentence):
                    word_label = {}
                    # TODO quickfix
                    if token_idx >= model_params["padding"]:
                        break
                    if token_idx < len(predictions[sent_idx]):
                        idx = np.argmax(predictions[sent_idx][token_idx])
                        if idx == 0:
                            continue
                        labels = idx2label[idx].split("-")

                        for label in labels:
                            if label != 'O' and label != 'P':
                                # print(token + " " + idx2label[idx] + " " + str(test_spans[sent_idx][token_idx]))
                                start = test_spans_sentences[sent_idx][0]+test_spans[sent_idx][token_idx][0]
                                end = test_spans_sentences[sent_idx][0] +test_spans[sent_idx][token_idx][1]
                                word_label[label] = {"entity": token,
                                                     "span": (start,end)}
                                assert token == doc["text"][start:end]
                    consecutive.append(word_label)
                test_labels.append(consecutive)

            for sentence_idx, sentence, labels in zip(range(len(test_tokens)), test_tokens, test_labels):
                for idx in range(len(labels)):
                    for category, label in labels[idx].items():
                        label_to_write = {"start": label["span"][0], "stop": label["span"][1]}
                        label_to_write["category"] =category
                        i = 1
                        while (idx + i < len(labels) and category in labels[idx + i].keys()):
                            label_to_write["stop"] = labels[idx + i][category]["span"][1]
                            del labels[idx + i][category]
                            i += 1
                        label_to_write["text"] = doc["text"][label_to_write["start"]:label_to_write["stop"]]
                        doc['answers']+="{} {} {}\t{}\n".format(
                            label_to_write["category"],label_to_write["start"],label_to_write["stop"],label_to_write["text"]
                        )

        return test_data





