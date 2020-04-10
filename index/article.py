

class WikiArticle:

    def __init__(self, title):
        self.title = title
        self.query_list = set()
        self.paragraph_list = set()

    def add_queries(self, queries):
        for query in queries:
            self.query_list.add(query)

    def add_paragraph(self, paragraph):
        self.paragraph_list.add(paragraph)

    def recall_at(self, retrieved_documents):
        return len(self.paragraph_list.intersection(retrieved_documents)) / len(self.paragraph_list)

    def precision_at(self, retrieved_documents):
        return len(self.paragraph_list.intersection(retrieved_documents)) / len(retrieved_documents)


class Query:

    def __init__(self, query, article):
        self.query = query
        self.article = article
        self.retrieved_documents = None
        self.keywords = None
        self.base_recall = None
        self.base_precision = None

    def set_retrieved_documents(self, retrieved_documents, tfidf_vectorizer):
        self.retrieved_documents = retrieved_documents
        text = ' '.join(retrieved_documents)
        features = tfidf_vectorizer.transform([text])
        vector = features[0].toarray()
        indices = (-vector).argsort()
        self.keywords = [tfidf_vectorizer.get_feature_names() for i in indices[0, :20]]

    def calc_base_precision_recall(self, n=10):
        document_set = set(self.retrieved_documents[:n])
        self.base_precision = self.article.precision_at(retrieved_documents=document_set)
        self.base_recall = self.article.recall_at(retrieved_documents=document_set)
        return self.base_precision, self.base_recall
