from keras.layers import GRU

parameters = \
{
    "use_test_file" : True,
    #"emb_file" : "data/emb/waw/w2v_allwiki_nkjp300_50",
     "emb_file" : "/home/ubuntu/ner_poleval/data/emb/wiki.pl",
    # "emb_file" : "data/emb/fasttext/wiki.pl",
    # "train_dataset_path" : "data/train/out_small.json",
    #"train_dataset_path": "data/train/out_middle.json",
     "train_dataset_path": "data/train/out.json",
    # "train_dataset_path": "data/train/shuffled_out.json",
    "test_dataset_path": "data/test/poleval_test_ner_2018.json",

    "validation_size":0.1,
    'epochs': 15

}


search_parameters = \
{
    "padding": 100,
    "lowercase": False,
    'rnn': GRU,
    'output_dim_rnn': 200,
    'activation_rnn': 'relu',
    'dropout': 0.5,
    'trainable_embeddings': True,
    'optimizer': 'adam',
    'iob': 'io',
}