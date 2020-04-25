import numpy as np
from article import QTAIndexer
from model.preprocess import Preprocessor
from query import QueryManager
from search_engine.search import save_search_engine

if __name__ == '__main__':
    # Paths
    query_path = '../query_reformulation_dataset/JEOPARDY_CSV.csv'
    qrel_path = '../query_reformulation_dataset/trec_car/base.train.cbor-article.qrels'
    paragraph_path = '../query_reformulation_dataset/trec_car/base.train.cbor-paragraphs.cbor'
    output_path = '../query_reformulation_dataset'

    embedding_file_path = '../../word_embedding/glove.840B.300d.txt'

    # Query-Title-Article Indexer ######################################################################################

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

    # Baseline Query Search ############################################################################################

    query_manager = QueryManager(qta_indexer=qta_indexer, min_paragraph_number=20,
                                 top_document_number=10, keyword_number=50)
    save_search_engine(search_engine=query_manager.search_engine, path=output_path)
    print('Corpus size :', len(query_manager.corpus))
    # query_manager.search_queries()
    # query_manager.save_queries(path=output_path)
    query_manager.load_queries(path=output_path)

    query_manager.clear_query_list(min_precision=0.2, min_recall=0.01)
    query_list = query_manager.query_list.items()
    precisions = np.zeros(shape=(len(query_list),))
    recalls = np.zeros(shape=(len(query_list),))
    for i, (qid, query) in enumerate(query_list):
        print('P:%.5f, R:%.5f  < %s >' % (query.base_precision, query.base_recall, query.query))
        print('\tKeywords:', query.keywords)
        precisions[i] = query.base_precision
        recalls[i] = query.base_recall

    print('Usable query number:', len(query_list))
    print('Avg. baseline precision :', precisions.mean())
    print('Avg. baseline recall    :', recalls.mean())

    # Preprocess #######################################################################################################

    # preprocessor = Preprocessor(emb_path=embedding_file_path)
    # preprocessor.initialize(query_manager=query_manager)
    # preprocessor.save_data(path=output_path)
    preprocessor = Preprocessor()
    preprocessor.load_data(path=output_path)
    print('--')

