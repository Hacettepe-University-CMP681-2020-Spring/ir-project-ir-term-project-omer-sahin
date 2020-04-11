import os
import pickle

import numpy as np
from search_engine.search import SearchEngine


class Query:

    def __init__(self, query, article):
        self.query = query
        self.article = article
        self.keywords = None
        self.base_recall = None
        self.base_precision = None

    def extract_keywords(self, retrieved_documents, tfidf_vectorizer):
        vocabulary = tfidf_vectorizer.get_feature_names()
        text = ' '.join(retrieved_documents)
        features = tfidf_vectorizer.transform([text])
        vector = features[0].toarray()
        indices = (-vector).argsort()
        self.keywords = [vocabulary[i] for i in indices[0, :20]]

    def calc_base_precision_recall(self, retrieved_documents):
        self.base_precision, self.base_recall = self.calc_precision_recall(retrieved_documents)
        return self.base_precision, self.base_recall

    def calc_precision_recall(self, retrieved_documents):
        document_set = set(retrieved_documents)
        precision = self.article.precision_at(retrieved_documents=document_set)
        recall = self.article.recall_at(retrieved_documents=document_set)
        return precision, recall


class Baseline:

    def __init__(self, qta_indexer):
        self.qta_indexer = qta_indexer
        self.query_list = self.qta_indexer.get_query_list(n=20)
        self.corpus = self.qta_indexer.get_paragraph_list(n=20)
        self.search_engine = SearchEngine(corpus=self.corpus)

    def search_queries(self):
        precisions = np.zeros(shape=(len(self.query_list)))
        recalls = np.zeros(shape=(len(self.query_list)))
        for i, query in enumerate(self.query_list):
            print('Searching - [%3d/%d]...  ' % (i+1, len(self.query_list)), end='')
            top_documents = self.search_engine.get_top_documents(query=query, top_k=10)
            query.extract_keywords(retrieved_documents=top_documents,
                                   tfidf_vectorizer=self.qta_indexer.tfidf_vectorizer)
            precision, recall = query.calc_base_precision_recall(retrieved_documents=top_documents)
            print('P:%.5f, R:%.5f' % (precision, recall), end='')
            precisions[i] = precision
            recalls[i] = recall
            print('\t\tdone')
        print('Avg. Precision :', precisions.mean())
        print('Avg. Recall    :', recalls.mean())

    def save_queries(self, path):
        if not os.path.exists(path):
            os.makedirs(path)

        with open(path + '/queries.pickle', 'wb') as handle:
            pickle.dump(self.query_list, handle, protocol=pickle.HIGHEST_PROTOCOL)

    def load_queries(self, path):
        with open(path + '/queries.pickle', 'rb') as handle:
            self.query_list = pickle.load(handle)