import nltk


def split_by_sentence_train(unprocessed_data):
    unprocessed_data_sentences = []
    for data in unprocessed_data:
        tokens = [word.replace(" ", "-") for word in data["tokens"]]
        text = " ".join(tokens)
        sent_text = nltk.sent_tokenize(text)
        idx = 0
        for sentence in sent_text:
            text_tokens = sentence.split(" ")
            sent = {
                "tokens": data["tokens"][idx:idx + len(text_tokens)],
                "entities": data["entities"][idx:idx + len(text_tokens)]
            }
            unprocessed_data_sentences.append(sent)
            idx = idx + len(text_tokens)
            assert len(text_tokens) == len(sent["tokens"])
        assert idx == len(data["tokens"])
    return unprocessed_data_sentences


