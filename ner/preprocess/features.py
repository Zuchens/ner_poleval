import re


def create_features(tokens):
    features = []
    for word in tokens:
        starts_uppercase = 2 if word[0].isupper() else 1
        has_dot = 2 if "." in word else 1
        has_num = 2 if re.search('\d+', word) else 1
        word_features = [starts_uppercase, has_dot, has_num]
        features.append(word_features)
    return features
