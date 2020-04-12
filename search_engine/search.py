from search_engine.rank_bm25 import BM25Okapi

from nltk.tokenize import word_tokenize


def tokenize(corpus):
    if type(corpus) is not list:
        return word_tokenize(corpus)

    tokenized_corpus = list()
    for text in corpus:
        tokens = word_tokenize(text)
        tokenized_corpus.append(tokens)

    return tokenized_corpus


class SearchEngine:

    def __init__(self, corpus):
        self.corpus = corpus
        self.bm25 = BM25Okapi(corpus, tokenizer=tokenize)

    def get_top_documents(self, query, top_k=10):
        tokenized_query = tokenize(query.query)
        bm25_scores = self.bm25.get_scores(tokenized_query)
        indices = (-bm25_scores).argsort()
        top_documents = [self.corpus[i] for i in indices[:top_k]]
        return top_documents
