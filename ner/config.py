from keras.layers import GRU

parameters = \
{
    "use_test_file" : True,
    "emb_file" : "../data/emb/emb_50/w2v_allwiki_nkjp300_50",
    "train_dataset_path" : "../data/train/out.json",
    "test_dataset_path": "../data/test/poleval_test_ner_2018.json",
    "lowercase" : False,
    "padding" : 50,
    "validation_size":0.1,
    'rnn': GRU,
    'output_dim_rnn': 200,
    'activation_rnn': 'relu',
    'dropout': 0.5,
    'optimizer': 'adam',
    'trainable_embeddings': True

}


grid_search_parameters = \
{

}