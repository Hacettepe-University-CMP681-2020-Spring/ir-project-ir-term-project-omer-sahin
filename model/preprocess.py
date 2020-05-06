import os
import pickle
import numpy as np
from nltk.corpus import stopwords
from keras_preprocessing.sequence import pad_sequences
from keras_preprocessing.text import Tokenizer, text_to_word_sequence

from model.embedding import WordEmbedding


class Preprocessor:

    def __init__(self):
        self.tokenizer = None
        self.word_embedding = None
        self.query_list = None
        self.query_df = None

    def initialize(self, query_manager, emb_path):
        self.tokenizer = Tokenizer()
        self.word_embedding = WordEmbedding(embfile=emb_path)

        self.query_list = query_manager.query_list

        query_list = [query.query for query in self.query_list]
        self.tokenizer.fit_on_texts(query_list)
        self.tokenizer.fit_on_texts(query_manager.corpus)
        self.word_embedding.create_embedding_matrix(self.tokenizer)

    def get_query_and_candidate_terms(self, sequence_length=20):

        stop_words = set(stopwords.words('english'))

        query_texts = list()
        candidate_terms = list()
        query_terms_texts = list()
        keyword_terms_texts = list()

        for query in self.query_list:
            query_texts.append(query.query)

            terms = [''] * (sequence_length*2)
            query_terms = text_to_word_sequence(query.query)
            terms[:sequence_length] = [term for term in query_terms if term not in stop_words][:sequence_length]
            terms[sequence_length:] = [keyword for _, keyword in query.keywords][:sequence_length]
            candidate_terms.append(terms)

            query_terms_texts.append(' '.join(terms[:sequence_length]))
            keyword_terms_texts.append(' '.join(terms[sequence_length:]))

        query_sequence = self.tokenizer.texts_to_sequences(query_texts)
        query_sequence = pad_sequences(query_sequence, maxlen=sequence_length*2)

        query_terms_sequence = self.tokenizer.texts_to_sequences(query_terms_texts)
        query_terms_sequence = pad_sequences(query_terms_sequence, maxlen=sequence_length, padding='post')
        keyword_terms_sequence = self.tokenizer.texts_to_sequences(keyword_terms_texts)
        keyword_terms_sequence = pad_sequences(keyword_terms_sequence, maxlen=sequence_length)

        terms_sequence = np.hstack([query_terms_sequence, keyword_terms_sequence])

        return self.query_list, query_sequence, terms_sequence, candidate_terms

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
        with open(path + '/query_list.pickle', 'wb') as handle:
            pickle.dump(self.query_list, handle, protocol=pickle.HIGHEST_PROTOCOL)

    def load_data(self, path):
        # Load word embedding
        with open(path + '/embedding.pickle', 'rb') as handle:
            self.word_embedding = pickle.load(handle)

        # Load tokenizer
        with open(path + '/tokenizer.pickle', 'rb') as handle:
            self.tokenizer = pickle.load(handle)

        # Load query map
        with open(path + '/query_list.pickle', 'rb') as handle:
            self.query_list = pickle.load(handle)
