from keras import Input, Model, optimizers
from keras.layers import Flatten, Bidirectional, Concatenate, Reshape, GRU, TimeDistributed, Masking
from keras.layers.core import Dense, Dropout
from keras.layers.embeddings import Embedding
from keras_contrib.layers import CRF
from keras_contrib.losses import crf_loss, crf_nll
from keras_contrib.metrics import crf_viterbi_accuracy, crf_accuracy, crf_marginal_accuracy


def create_model(embeddings, emb_features, feature_size, maxlen, output_size, model_parameters,class_weights):
    features = Input(shape=(maxlen, feature_size,), name='words')
    emb_input = Input(shape=(maxlen,), name='emb_input')
    embedding = Embedding(embeddings.shape[0], emb_features, input_length=maxlen, weights=[embeddings],
                          trainable=model_parameters['trainable_embeddings'])(
        emb_input)
    concat = Concatenate(axis=-1)([embedding, features])
    mask = Masking()(concat)  # insert this
    lstm = Bidirectional(
        model_parameters['rnn'](output_dim=model_parameters['output_dim_rnn'],
                                activation=model_parameters['activation_rnn'], return_sequences=True))(
        mask)
    dropout = Dropout(model_parameters['dropout'])(lstm)
    if model_parameters['is_crf']:
        part1 = TimeDistributed(Dense(600, activation="relu"))(dropout)
        # final_dense = Masking()(part1)  # insert this
        crf = CRF(output_size, sparse_target=True, learn_mode="marginal")(part1)
        model = Model(inputs=[emb_input, features], outputs=[crf])
        model.compile(optimizer=model_parameters['optimizer'], loss='binary_crossentropy', metrics=['acc'])

        # model.compile(optimizer=model_parameters['optimizer'], loss=crf_loss, metrics=[crf_viterbi_accuracy],class_weights=class_weights)
    else:
        part1 = TimeDistributed(Dense(600, activation="relu"))(dropout)
        final_dense = Masking()(part1)  # insert this
        out = Dense(output_size, activation='softmax')(final_dense)
        model = Model(inputs=[emb_input, features], outputs=[out])
        model.compile(optimizer=model_parameters['optimizer'], loss='binary_crossentropy', metrics=['acc'])
    model.summary()
    # from keras.utils import plot_model
    # plot_model(model, to_file='model.png', show_shapes=True)
    return model
