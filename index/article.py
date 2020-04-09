

class WikiArticle:

    def __init__(self, title):
        self.title = title
        self.query_list = set()
        self.paragraph_list = dict()
        self.keywords = None

    def add_queries(self, queries):
        for query in queries:
            if query not in self.query_list:
                self.query_list.add(query)

    def add_paragraph(self, pid, text):
        self.paragraph_list[pid] = text

    def set_keywords(self, keywords):
        self.keywords = keywords
