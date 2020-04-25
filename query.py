import os
import pickle

from multiprocessing import Pool, cpu_count
from search_engine.search import SearchEngine


class Query:

    def __init__(self, query, article):
        self.query = query
        self.article = article
        self.keywords = None
        self.base_recall = None
        self.base_precision = None

    def extract_keywords(self, retrieved_documents, tfidf_vectorizer, keyword_number=20):
        vocabulary = tfidf_vectorizer.get_feature_names()
        text = ' '.join(retrieved_documents)
        features = tfidf_vectorizer.transform([text])
        vector = features[0].toarray()
        indices = (-vector).argsort()
        self.keywords = [(features[0, i], vocabulary[i]) for i in indices[0, :keyword_number]]

    def calc_base_precision_recall(self, retrieved_documents):
        self.base_precision, self.base_recall = self.calc_precision_recall(retrieved_documents)
        return self.base_precision, self.base_recall

    def calc_precision_recall(self, retrieved_documents):
        document_set = set(retrieved_documents)
        precision = self.article.precision_at(retrieved_documents=document_set)
        recall = self.article.recall_at(retrieved_documents=document_set)
        return precision, recall


class QueryManager:

    def __init__(self, qta_indexer, min_paragraph_number=20, top_document_number=10, keyword_number=20):
        self.qta_indexer = qta_indexer
        self.top_document_number = top_document_number
        self.keyword_number = keyword_number

        self.query_list = self.qta_indexer.get_query_list(min_paragraph=min_paragraph_number)
        self.corpus = self.qta_indexer.get_paragraph_list(min_paragraph=min_paragraph_number)
        self.search_engine = SearchEngine(corpus=self.corpus)
        self.query_number = len(self.query_list)

    def search_queries(self):
        pool = Pool(cpu_count())
        self.query_list = pool.map(self.search_base_query, enumerate(self.query_list))
        pool.close()
        pool.join()

    def search_base_query(self, query_tuple):
        index, query = query_tuple
        top_documents = self.search_engine.get_top_documents(query=query, top_k=self.top_document_number)
        query.extract_keywords(retrieved_documents=top_documents,
                               tfidf_vectorizer=self.qta_indexer.tfidf_vectorizer,
                               keyword_number=self.keyword_number)
        precision, recall = query.calc_base_precision_recall(retrieved_documents=top_documents)
        print('Searching [%5d/%d] - P:%.5f, R:%.5f  < %s >' %
              (index+1, self.query_number, precision, recall, query.query))
        return query

    def clear_query_list(self, min_precision=0, min_recall=0):
        if not isinstance(self.query_list, list):
            return
        query_map = dict()
        for query in self.query_list:
            if query.base_precision >= min_precision and query.base_recall >= min_recall:
                qid = hash(query.query)
                query_map[qid] = query
        self.query_list = query_map

    def save_queries(self, path):
        if not os.path.exists(path):
            os.makedirs(path)

        with open(path + '/queries.pickle', 'wb') as handle:
            pickle.dump(self.query_list, handle, protocol=pickle.HIGHEST_PROTOCOL)

    def load_queries(self, path):
        with open(path + '/queries.pickle', 'rb') as handle:
            self.query_list = pickle.load(handle)
