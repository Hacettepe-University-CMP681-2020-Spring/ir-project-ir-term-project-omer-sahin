import os
import pickle

import numpy as np
import pandas as pd
from keras_preprocessing.text import Tokenizer

from model.embedding import WordEmbedding


def convert_to_data_frame(baseline):
    query_obj_list = baseline.get_query_list(min_precision=0.2, min_recall=0.01)
    query_list = list()
    keyword_list = list()
    keyword_line_list = list()
    article_list = list()
    base_precisions = np.zeros(shape=(len(query_obj_list),))
    for i, query in enumerate(query_obj_list):
        base_precisions[i] = query.base_precision
        query_list.append(query.query)
        article_list.append(query.article)
        keyword_list.append(query.keywords)
        keywords = ' '.join([keyword for _, keyword in query.keywords])
        keyword_line_list.append(keywords)

    data_frame = pd.DataFrame({'query': query_list, 'keyword': keyword_list, 'keyword_line': keyword_line_list,
                               'base_precision': base_precisions, 'article': article_list})

    return data_frame


class Preprocessor:

    def __init__(self, emb_path=None):
        self.tokenizer = Tokenizer()
        self.word_embedding = None
        if emb_path:
            self.word_embedding = WordEmbedding(embfile=emb_path)
        self.query_df = None

    def initialize(self, baseline):
        self.query_df = convert_to_data_frame(baseline)

        self.tokenizer.fit_on_texts(baseline.corpus)
        self.tokenizer.fit_on_texts(self.query_df['query'])
        self.word_embedding.create_embedding_matrix(self.tokenizer)

        self.query_df['query_sequence'] = self.tokenizer.texts_to_sequences(self.query_df['query'])
        self.query_df['keyword_sequence'] = self.tokenizer.texts_to_sequences(self.query_df['keyword_line'])

    def save_data(self, path):
        # Create directory if not exist
        if not os.path.exists(path):
            os.makedirs(path)

        # Serialize word embedding
        with open(path + '/embedding.pickle', 'wb') as handle:
            pickle.dump(self.word_embedding, handle, protocol=pickle.HIGHEST_PROTOCOL)

        # Serialize tokenizer
        with open(path + '/tokenizer.pickle', 'wb') as handle:
            pickle.dump(self.tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)

        # Serialize query data frame
        with open(path + '/query_df.pickle', 'wb') as handle:
            pickle.dump(self.query_df, handle, protocol=pickle.HIGHEST_PROTOCOL)

    def load_data(self, path):
        # Load word embedding
        with open(path + '/embedding.pickle', 'rb') as handle:
            self.word_embedding = pickle.load(handle)

        # Load tokenizer
        with open(path + '/tokenizer.pickle', 'rb') as handle:
            self.tokenizer = pickle.load(handle)

        # Load query data frame
        with open(path + '/query_df.pickle', 'rb') as handle:
            self.query_df = pickle.load(handle)





