import os
import pickle

from keras_preprocessing.sequence import pad_sequences
from keras_preprocessing.text import Tokenizer, text_to_word_sequence

from model.embedding import WordEmbedding


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

        query_list = [query.query for query in self.query_map.values()]
        self.tokenizer.fit_on_texts(query_list)
        self.tokenizer.fit_on_texts(query_manager.corpus)
        self.word_embedding.create_embedding_matrix(self.tokenizer)

    def get_query_and_candidate_terms(self, sequence_length=32):

        qid_list = list()
        query_texts = list()
        candidate_terms = list()

        for qid, query in self.query_map.items():
            qid_list.append(qid)
            query_texts.append(query.query)

            terms = '' * (sequence_length*2)
            terms[:sequence_length] = text_to_word_sequence(query.query)[:sequence_length]
            terms[sequence_length:] = query.keywords[:sequence_length]
            candidate_terms.append(terms)

        query_sequence = self.tokenizer.texts_to_sequences(query_texts)
        terms_sequence = self.tokenizer.texts_to_sequences(candidate_terms)

        query_sequence = pad_sequences(query_sequence, maxlen=sequence_length)
        terms_sequence = pad_sequences(terms_sequence, maxlen=sequence_length*2)

        return qid_list, query_sequence, terms_sequence, candidate_terms

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

        # Load query map
        with open(path + '/query_map.pickle', 'rb') as handle:
            self.query_map = pickle.load(handle)




