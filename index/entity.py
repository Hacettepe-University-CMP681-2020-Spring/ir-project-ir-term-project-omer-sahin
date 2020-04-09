

class WikiArticle:

    def __init__(self, title):
        self.title = title
        self.query_list = set()
        self.paragraph_list = list()

    def add_queries(self, queries):
        for query in queries:
            if query not in self.query_list:
                self.query_list.add(query)

    def add_paragraph(self, pid, text):
        self.paragraph_list.append((pid, text))
