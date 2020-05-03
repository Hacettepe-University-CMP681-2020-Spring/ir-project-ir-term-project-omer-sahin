import os
from datetime import datetime

from tensorflow_core.python.keras import Input, Sequential, Model, losses, optimizers
from tensorflow_core.python.keras.layers import Embedding, BatchNormalization, Concatenate, \
    Dense, Dropout, MaxPooling1D, Flatten

from tensorflow_core.python.keras.layers.convolutional import Conv1D

import numpy as np
from multiprocessing import Pool, cpu_count

from model.preprocess import Preprocessor
from model.util import search_and_evaluate


if __name__ == '__main__':
    preprocessor = Preprocessor()
    preprocessor.load_data(path='../../query_reformulation_dataset')

    query_objs, query_sequence, terms_sequence, candidate_terms = \
        preprocessor.get_query_and_candidate_terms(sequence_length=20)

    query_input = Input(shape=(query_sequence.shape[1],), name='query_input')
    terms_input = Input(shape=(terms_sequence.shape[1],), name='keyword_input')

    embedding_block = Sequential(layers=[
        Embedding(preprocessor.word_embedding.vocabulary_size, preprocessor.word_embedding.dimensions,
                  weights=[preprocessor.word_embedding.embedding_matrix],
                  trainable=True, mask_zero=False),
        BatchNormalization()
        # Bidirectional(LSTM(256, return_sequences=True))
    ])

    query_embedding = embedding_block(query_input)
    terms_embedding = embedding_block(terms_input)

    query_cnn = Conv1D(filters=64, kernel_size=3, strides=1)(query_embedding)
    query_cnn = MaxPooling1D(pool_size=2)(query_cnn)

    terms_cnn = Conv1D(filters=64, kernel_size=3, strides=1)(terms_embedding)
    terms_cnn = MaxPooling1D(pool_size=4)(terms_cnn)

    merged_cnn = Concatenate()([query_cnn, terms_cnn])

    merged_cnn = Conv1D(filters=256, kernel_size=3, strides=1)(merged_cnn)
    merged_cnn = MaxPooling1D(pool_size=3)(merged_cnn)

    merged_flat = Flatten()(merged_cnn)

    dense = BatchNormalization()(merged_flat)
    dense = Dense(256, activation='sigmoid')(dense)
    # dense = Dropout(0.4)(dense)
    out = Dense(terms_sequence.shape[1], activation='linear')(dense)

    model = Model(inputs=[query_input, terms_input], outputs=out)
    model.compile(optimizer='adam', loss=losses.mean_squared_error)

    epochs = 20
    batch_size = 4
    pool = Pool(batch_size)
    for e in range(epochs):
        print('Epochs: %3d/%d' % (e + 1, epochs))

        reward = np.zeros(shape=(len(query_objs)))
        for i in range(0, len(query_objs), batch_size):
            print('  [%4d-%-4d/%d]' % (i, i+batch_size, len(query_objs)))

            query = query_objs[i:i + batch_size]
            q_seq = query_sequence[i:i+batch_size]
            t_seq = terms_sequence[i:i+batch_size]
            terms = candidate_terms[i:i+batch_size]

            weights = model.predict(x=[q_seq, t_seq])

            batch_reward = pool.map(search_and_evaluate, zip(weights, terms, query))
            batch_reward = 0.8 * np.asarray(batch_reward) + 0.2 * reward[i:i+batch_size]

            model.train_on_batch(x=[q_seq, t_seq], y=weights, sample_weight=batch_reward)

            reward[i:i+batch_size] = batch_reward

        # Save model
        model_path = '../../saved_model/model_' + str(datetime.now().date()) + '_' + str(e)
        os.makedirs(model_path)
        model.save(filepath=model_path)

    pool.close()
    pool.join()
