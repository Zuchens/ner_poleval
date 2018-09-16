from keras import Input, Model, optimizers
from keras.layers import Flatten, Bidirectional, Concatenate, Reshape, GRU
from keras.layers.core import Dense, Dropout
from keras.layers.embeddings import Embedding
from keras_contrib.layers import CRF

def create_model(embeddings, emb_features, feature_size, maxlen, output_size,model_parameters ):
    features = Input(shape=(maxlen,feature_size, ), name='words')
    # reshaped_features = Reshape((maxlen, 1,))(features)
    emb_input = Input(shape=(maxlen,), name='emb_input')
    embedding = Embedding(embeddings.shape[0], emb_features, input_length=maxlen, weights=[embeddings],
                          trainable=model_parameters['trainable_embeddings'])(
        emb_input)
    concat = Concatenate(axis=-1)([embedding, features])
    lstm = Bidirectional(
        model_parameters['rnn'](output_dim=model_parameters['output_dim_rnn'], activation=model_parameters['activation_rnn'], return_sequences=True))(
        concat)
    dropout = Dropout(model_parameters['dropout'])(lstm)
    # crf = CRF(output_size, sparse_target=True,learn_mode='marginal')(dropout)
    # model = Model(inputs=[emb_input, features], outputs=[crf])

    out = Dense(output_size, activation='sigmoid')(dropout)
    model = Model(inputs=[emb_input, features], outputs=[out])

    # model.compile(optimizer=model_parameters['optimizer'])

    model.compile(optimizer=model_parameters['optimizer'], loss='binary_crossentropy', metrics=['acc'])
    #plot(model)

    return model


def plot(model):
    from keras.utils import plot_model
    from IPython.display import SVG
    from keras.utils.vis_utils import model_to_dot

    plot_model(model, to_file='model.pdf', show_shapes=True)
    SVG(model_to_dot(model).create(prog='dot', format='svg'))