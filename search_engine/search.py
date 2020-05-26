import os
import pickle

from search_engine.rank_bm25 import BM25Okapi

from nltk.tokenize import word_tokenize


def tokenize(corpus):
    if type(corpus) is not list:
        corpus = corpus.lower()
        return word_tokenize(corpus)

    tokenized_corpus = list()
    for text in corpus:
        text = text.lower()
        tokens = word_tokenize(text)
        tokenized_corpus.append(tokens)

    return tokenized_corpus


class SearchEngine:

    def __init__(self, corpus):
        self.corpus = corpus
        self.bm25 = BM25Okapi(corpus, tokenizer=tokenize)

    def get_top_documents(self, query, top_k=10):
        tokenized_query = tokenize(query)
        bm25_scores = self.bm25.get_scores(tokenized_query)
        indices = (-bm25_scores).argsort()
        top_documents = [self.corpus[i] for i in indices[:top_k]]
        return top_documents


def save_search_engine(search_engine, path):
    if not os.path.exists(path):
        os.makedirs(path)

    with open(path + '/search_engine.pickle', 'wb') as handle:
        pickle.dump(search_engine, handle, protocol=pickle.HIGHEST_PROTOCOL)


def load_search_engine(path):
    try:
        with open(path + '/search_engine.pickle', 'rb') as handle:
            search_engine = pickle.load(handle)
    except FileNotFoundError:
        search_engine = None
    return search_engine


# Search engine instance
search_engine_instance = load_search_engine('../../query_reformulation_dataset')
