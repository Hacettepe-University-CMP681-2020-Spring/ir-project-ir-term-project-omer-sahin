import re
import pandas as pd
from urllib.parse import unquote

from trec_car import read_data


class Document:

    def __init__(self, query, title):
        self.query = query
        self.title = title
        self.article_list = list()


class Indexer:

    def __init__(self):
        self.title_query_map = dict()
        self.article_title_map = dict()
        self.paragraph_map = dict()
        self.article_list = dict()
        self.document_list = dict()

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

        self.create_title_query_map(query_path)
        self.create_article_title_map(qrel_path)
        self.create_paragraph_map(paragraph_path)

        num_paragraph = len(self.paragraph_map)
        for i, article_id in enumerate(self.paragraph_map.keys()):
            print('[%6d/%d] : ' % (i+1, num_paragraph), end='')
            try:
                title = self.article_title_map[article_id]
                queries = self.title_query_map[title]

                if title not in self.article_list:
                    self.article_list[title] = list()

                self.article_list[title].append(self.paragraph_map[article_id])

                for query in queries:
                    qid = hash(query)
                    if qid not in self.document_list:
                        document = Document(title=title, query=query)
                        self.document_list[qid] = document

                    document = self.document_list[qid]
                    document.article_list = self.article_list[title]

                print('successful')

            except Exception as e:
                print(e)

    def iterate_documents(self):
        for document in self.document_list.values():
            yield document
