from article import QTAIndexer
from query import Baseline

if __name__ == '__main__':
    # Paths
    query_path = '../query_reformulation_dataset/JEOPARDY_CSV.csv'
    qrel_path = '../query_reformulation_dataset/trec_car/base.train.cbor-article.qrels'
    paragraph_path = '../query_reformulation_dataset/trec_car/base.train.cbor-paragraphs.cbor'
    output_path = '../query_reformulation_dataset'

    qta_indexer = QTAIndexer()
    # qta_indexer.index_documents(query_path=query_path, qrel_path=qrel_path, paragraph_path=paragraph_path)
    # qta_indexer.save_articles(path=output_path)
    qta_indexer.load_articles(path=output_path)
    qta_indexer.inverse_article_frequency()

    for title, article in qta_indexer.article_list.items():
        num_query = len(article.query_list)
        num_paragraph = len(article.paragraph_list)
        if num_paragraph >= 20:
            print('%-32s -> Queries: %3d, Paragraphs: %3d' % (article.title, num_query, num_paragraph))

    baseline = Baseline(qta_indexer=qta_indexer, min_paragraph_number=20, top_document_number=10, keyword_number=20)
    print('Corpus size :', len(baseline.corpus))
    baseline.search_queries()
    baseline.save_queries(path=output_path)
    baseline.load_queries(path=output_path)

    query_list = baseline.get_query_list()
    print('Usable query number:', len(query_list))
    for query in query_list:
        print('P:%.5f, R:%.5f  < %s >' % (query.base_precision, query.base_recall, query.query))
        print('\tKeywords:', query.keywords)

