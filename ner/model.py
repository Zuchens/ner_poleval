from keras import Input, Model, optimizers
from keras.layers import Flatten, Bidirectional, Concatenate, Reshape, GRU
from keras.layers.core import Dense, Dropout
from keras.layers.embeddings import Embedding



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
    out = Dense(output_size, activation='sigmoid')(dropout)

    model = Model(inputs=[emb_input, features], outputs=[out])

    model.compile(optimizer=model_parameters['optimizer'], loss='binary_crossentropy', metrics=['acc'])
    from keras.utils import plot_model
    plot_model(model, to_file='model.png', show_shapes=True)
    model.save('../generated/model.h5')
    return model