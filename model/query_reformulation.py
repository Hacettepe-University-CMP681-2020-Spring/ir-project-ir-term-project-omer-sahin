import os
from datetime import datetime

import numpy as np
from multiprocessing import Pool

from tensorflow_core.python.keras import Input, Sequential, Model
from tensorflow_core.python.keras.layers import Embedding, BatchNormalization, Concatenate, \
    Dense, MaxPooling1D, Flatten, Conv1D

from tensorflow_core.python.keras import losses
from tensorflow_core.python.keras.models import load_model

from model.util import get_batch_data, evaluate_reward_precision, evaluate_precision_recall, recreate_query


class QueryReformulation:

    def __init__(self, model_path=None):
        self.model = None
        if model_path:
            self.model = load_model(model_path)
            self.model.summary()

    def build_cnn_model(self, query_dim, terms_dim, output_dim, word_embedding):
        query_input = Input(shape=(query_dim,), name='query_input')
        terms_input = Input(shape=(terms_dim,), name='terms_input')

        embedding_cnn_block = Sequential(layers=[
            Embedding(word_embedding.vocabulary_size, word_embedding.dimensions,
                      weights=[word_embedding.embedding_matrix],
                      trainable=True, mask_zero=False),
            BatchNormalization(),
            Conv1D(filters=64, kernel_size=3, strides=1),
            MaxPooling1D(pool_size=3)
        ])

        query_embedding_cnn = embedding_cnn_block(query_input)
        terms_embedding_cnn = embedding_cnn_block(terms_input)

        merged_cnn = Concatenate()([query_embedding_cnn, terms_embedding_cnn])

        merged_cnn = Conv1D(filters=256, kernel_size=3, strides=1)(merged_cnn)
        merged_cnn = MaxPooling1D(pool_size=3)(merged_cnn)

        merged_flat = Flatten()(merged_cnn)

        dense = BatchNormalization()(merged_flat)
        dense = Dense(256, activation='sigmoid')(dense)
        # dense = Dropout(0.4)(dense)
        out = Dense(output_dim, activation='linear')(dense)

        self.model = Model(inputs=[query_input, terms_input], outputs=out)
        self.model.compile(optimizer='adam', loss=losses.mean_squared_error)
        self.model.summary()

    def train_model(self, query_objs, query_sequence, terms_sequence, candidate_terms, epochs=20, batch_size=4):
        best_precision = 0
        pool = Pool(batch_size)
        for e in range(epochs):
            print('Epochs: %3d/%d' % (e + 1, epochs))

            reward = np.zeros(shape=(len(query_objs)))
            precision = np.zeros(shape=(len(query_objs)))
            for i, query, q_seq, t_seq, terms in get_batch_data(query_objs, query_sequence,
                                                                terms_sequence, candidate_terms,
                                                                batch_size):

                print('  [%4d-%-4d/%d]' % (i, i + batch_size, len(query_objs)))

                weights = self.model.predict(x=[q_seq, t_seq])

                batch_reward_precision = pool.map(evaluate_reward_precision, zip(weights, terms, query))
                batch_reward_precision = np.array(batch_reward_precision)

                batch_reward = 0.8 * np.asarray(batch_reward_precision[:, 0]) + 0.2 * reward[i:i + batch_size]

                self.model.train_on_batch(x=[q_seq, t_seq], y=weights, sample_weight=batch_reward)

                reward[i:i + batch_size] = batch_reward_precision[:, 0]
                precision[i:i + batch_size] = batch_reward_precision[:, 1]

            # Save model
            avg_precision = precision.mean()
            print('  Average precision %.5f on epoch %d, best precision %.5f' % (avg_precision, e+1, best_precision))
            if avg_precision > best_precision:
                best_precision = avg_precision
                model_path = '../../saved_model/reformulation_model_' + str(datetime.now().date())
                if not os.path.exists(model_path):
                    os.makedirs(model_path)
                self.model.save(filepath=model_path)

        pool.close()
        pool.join()

    def test_model(self, query_objs, query_sequence, terms_sequence, candidate_terms, batch_size=4):

        pool = Pool(batch_size)
        precision_recall = np.zeros(shape=(len(query_objs), 2))
        for i, query, q_seq, t_seq, terms in get_batch_data(query_objs, query_sequence,
                                                            terms_sequence, candidate_terms,
                                                            batch_size):

            print('[%4d-%-4d/%d]' % (i, i + batch_size, len(query_objs)))

            weights = self.model.predict(x=[q_seq, t_seq])

            batch_precision_recall = pool.map(evaluate_precision_recall, zip(weights, terms, query))

            precision_recall[i:i + batch_size] = np.array(batch_precision_recall)

        pool.close()
        pool.join()

        return precision_recall.mean(axis=0)

    def reformulate_query(self, query_sequence, terms_sequence, candidate_terms, threshold=0.5):
        weights = self.model.predict(x=[query_sequence, terms_sequence])
        reformulated_query = recreate_query(terms=candidate_terms, weights=weights, threshold=threshold)
        return reformulated_query
