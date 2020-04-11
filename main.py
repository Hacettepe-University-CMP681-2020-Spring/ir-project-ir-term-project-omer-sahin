from article import QTAIndexer
from query import Baseline

if __name__ == '__main__':
    # Paths
    query_path = '../query_reformulation_dataset/JEOPARDY_CSV.csv'
    qrel_path = '../query_reformulation_dataset/trec_car/base.train.cbor-article.qrels'
    paragraph_path = '../query_reformulation_dataset/trec_car/base.train.cbor-paragraphs.cbor'
    output_path = '../query_reformulation_dataset'

    qta_indexer = QTAIndexer()
    qta_indexer.index_documents(query_path=query_path, qrel_path=qrel_path, paragraph_path=paragraph_path)
    qta_indexer.save_articles(path=output_path)
    qta_indexer.load_articles(path=output_path)
    qta_indexer.inverse_article_frequency()

    num_article_gte_10 = 0
    num_query_gte_10 = 0
    for title, article in qta_indexer.article_list.items():
        num_query = len(article.query_list)
        num_paragraph = len(article.paragraph_list)

        if num_paragraph >= 10:
            num_article_gte_10 += 1
            num_query_gte_10 += num_query
            print('%-24s -> Queries: %3d, Paragraphs: %3d' % (article.title, num_query, num_paragraph))

    print('Number of article with 10+ paragraphs :', num_article_gte_10)
    print('Number of query with 10+ paragraphs   :', num_query_gte_10)

    baseline = Baseline(qta_indexer=qta_indexer)
    baseline.search_queries()
    baseline.save_queries(path=output_path)
