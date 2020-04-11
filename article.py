import os
import pickle
import re
from urllib.parse import unquote

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from trec_car import read_data

from query import Query


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


class QTAIndexer:

    def __init__(self):
        self.title_query_map = dict()
        self.article_title_map = dict()
        self.paragraph_map = dict()
        self.article_list = dict()
        self.tfidf_vectorizer = TfidfVectorizer(stop_words='english')

    def create_title_query_map(self, query_path):
        query_df = pd.read_csv(query_path)
        for i, row in query_df.iterrows():
            try:
                title = row[' Answer']
                query = row[' Question']

                if 'href=' in query:
                    continue

                title = title.lower()
                title = title.strip('"')
                title = re.sub(pattern='^(a|an|the) ', repl='', string=title)

                # Skip all digit titles
                if title.isdigit():
                    continue

                if title not in self.title_query_map:
                    self.title_query_map[title.lower()] = list()

                self.title_query_map[title].append(query)
            except Exception:
                continue

    def create_article_title_map(self, qrel_path):
        with open(qrel_path, mode='r') as qrel_file:
            for line in qrel_file.readlines():
                token = line.split(' ')
                article_id = token[2]

                title = unquote(token[0])
                title = title[7:title.find('/')]
                title = title.lower()

                self.article_title_map[article_id] = title

    def create_paragraph_map(self, paragraph_path):
        for i, page in enumerate(read_data.iter_paragraphs(open(paragraph_path, 'rb'))):
            self.paragraph_map[page.para_id] = page.get_text()

    def index_documents(self, query_path, qrel_path, paragraph_path):
        print('Title-Query mapping...')
        self.create_title_query_map(query_path)
        print('Article-Title mapping...')
        self.create_article_title_map(qrel_path)
        print('Paragraph mapping...')
        self.create_paragraph_map(paragraph_path)
        print('Article-[Paragraphs+Queries] indexing...')
        for i, paragraph_id in enumerate(self.paragraph_map.keys()):
            try:
                title = self.article_title_map[paragraph_id]
                queries = self.title_query_map[title]

                if title not in self.article_list:
                    self.article_list[title] = WikiArticle(title=title)

                self.article_list[title].add_queries(queries=queries)

                self.article_list[title].add_paragraph(paragraph=self.paragraph_map[paragraph_id])

            except Exception:
                continue

        print('Number of title in Jeopardy   :', len(self.title_query_map))
        print('Number of article in TREC-CAR :', len(self.article_title_map))
        print('Number of matched article     :', len(self.article_list))

        # Free memory
        del self.title_query_map
        del self.article_title_map
        del self.paragraph_map

    def inverse_article_frequency(self):
        text_list = list()
        for title, article in self.article_list.items():
            text = ' '.join(article.paragraph_list)
            text_list.append(text)

        self.tfidf_vectorizer.fit(text_list)

    def get_paragraph_list(self, n=0):
        paragraph_list = list()
        for title, article in self.article_list.items():
            if len(article.paragraph_list) >= n:
                paragraph_list += article.paragraph_list

        return paragraph_list

    def get_query_list(self, n=10):
        query_list = list()
        for title, article in self.article_list.items():
            if len(article.paragraph_list) >= n:
                for query in article.query_list:
                    query_list.append(Query(query=query, article=article))

        return query_list

    def save_articles(self, path):
        if not os.path.exists(path):
            os.makedirs(path)

        with open(path + '/articles.pickle', 'wb') as handle:
            pickle.dump(self.article_list, handle, protocol=pickle.HIGHEST_PROTOCOL)

    def load_articles(self, path):
        with open(path + '/articles.pickle', 'rb') as handle:
            self.article_list = pickle.load(handle)
