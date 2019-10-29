# TODO implement cutting sentence size
def convert_entities(token_categories, model_parameters, labels):
    categories = get_categories(token_categories)

    label2idx ={} if not labels else labels
    label2idx_iterator = len(label2idx) + 1
    label2idx["P"] = 0
    categories_idx = []
    from collections import defaultdict

    label2count = defaultdict(int)
    for i, iob_sentence in enumerate(categories):
        idx_iob = []
        for sentence_categories in iob_sentence:
            sentence_categories = sorted(set(sentence_categories))
            data = "-".join(sentence_categories)
            if data not in label2idx:
                label2idx[data] = label2idx_iterator
                label2idx_iterator += 1

            label2count[data] += 1
            idx_iob.append(label2idx[data])
        categories_idx.append(idx_iob)

    return label2idx, categories_idx


def get_categories(token_categories):
    categories = []
    for a, sentence_entities in enumerate(token_categories):
        sentence_categories = [set() for x in range(len(sentence_entities))]
        for idx, token_categories in enumerate(sentence_entities):
            if token_categories:
                for label in token_categories:
                    if label:
                        if label.get("subtype"):
                            ne = label["type"] + "_" + label["subtype"]
                        else:
                            ne = label["type"]
                        for i in range(len(label["offsets"])):
                            # if i == 0:
                            #     next_token = "B_" + ne
                            # else:
                            next_token = "I_" + ne
                            if len(sentence_categories) >= (idx + i + 1):
                                sentence_categories[idx + i].add(next_token)

            else:
                if not sentence_categories[idx]:
                    sentence_categories[idx].add("O")

        categories.append(sentence_categories)
    return categories

