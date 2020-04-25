import os
import pickle
import pandas as pd

from keras_preprocessing.sequence import pad_sequences
from keras_preprocessing.text import Tokenizer

from model.embedding import WordEmbedding


def extract_query_keyword_list(query_map):
    qid_list = list()
    query_list = list()
    keyword_list = list()
    for qid, query in query_map.items():
        qid_list.append(qid)
        query_list.append(query.query)
        keywords = ' '.join([keyword for _, keyword in query.keywords])
        keyword_list.append(keywords)

    return qid_list, query_list, keyword_list


class Preprocessor:

    def __init__(self, emb_path=None):
        self.tokenizer = Tokenizer()
        self.word_embedding = None
        if emb_path:
            self.word_embedding = WordEmbedding(embfile=emb_path)

        self.query_map = None
        self.query_df = None

    def initialize(self, query_manager):
        self.query_map = query_manager.query_list

        qid, query, keyword = extract_query_keyword_list(self.query_map)
        self.tokenizer.fit_on_texts(query_manager.corpus)
        self.tokenizer.fit_on_texts(query)
        self.word_embedding.create_embedding_matrix(self.tokenizer)

        self.query_df = pd.DataFrame({'qid': qid,
                                      'query_sequence': self.tokenizer.texts_to_sequences(query),
                                      'keyword_sequence': self.tokenizer.texts_to_sequences(keyword)})

    def get_inputs(self, sequence_length=50):
        qid = self.query_df['qid'].to_numpy()
        query_sequence = pad_sequences(self.query_df['query_sequence'], maxlen=sequence_length)
        keyword_sequence = pad_sequences(self.query_df['keyword_sequence'], maxlen=sequence_length)
        return qid, query_sequence, keyword_sequence

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

        # Serialize query map
        with open(path + '/query_map.pickle', 'wb') as handle:
            pickle.dump(self.query_map, handle, protocol=pickle.HIGHEST_PROTOCOL)

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

        # Load query map
        with open(path + '/query_map.pickle', 'rb') as handle:
            self.query_map = pickle.load(handle)




