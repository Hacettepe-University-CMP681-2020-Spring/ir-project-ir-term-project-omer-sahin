
import numpy as np
from index.query_title_article_indexer import Indexer
from rank_bm25 import BM25Okapi
from util import tokenize

if __name__ == '__main__':
    query_path = '../query_reformulation_dataset/JEOPARDY_CSV.csv'
    qrel_path = '../query_reformulation_dataset/trec_car/base.train.cbor-article.qrels'
    paragraph_path = '../query_reformulation_dataset/trec_car/base.train.cbor-paragraphs.cbor'

    output_path = '../query_reformulation_dataset/indexed'

    qta_indexer = Indexer()
    # qta_indexer.index_documents(query_path=query_path,
    #                             qrel_path=qrel_path,
    #                             paragraph_path=paragraph_path)
    # qta_indexer.save(path=output_path)

    qta_indexer.load(path=output_path)
    qta_indexer.inverse_article_frequency()

    corpus = qta_indexer.get_paragraph_list(n=20)
    tokenized_corpus = tokenize(corpus)
    bm25 = BM25Okapi(tokenized_corpus)

    queries = qta_indexer.get_query_list(n=20)
    for i, query in enumerate(queries):
        print('BM25 - [%3d/%d]...' % (i, len(queries)), end='')
        tokenized_query = tokenize(query.query)
        bm25_scores = bm25.get_scores(tokenized_query)
        indices = (-bm25_scores).argsort()
        top_documents = [corpus[i] for i in indices[:10]]
        query.set_retrieved_documents(retrieved_documents=top_documents, tfidf_vectorizer=qta_indexer.tfidf_vectorizer)
        print('\t\tdone')

    precisions = np.zeros(shape=(len(queries)))
    recalls = np.zeros(shape=(len(queries)))
    for i, query in enumerate(queries):
        print('Precision/Recall - [%3d/%d]...' % (i, len(queries)), end='')
        precision, recall = query.calc_base_precision_recall()
        print('P:%.5f, R:%.5f' % (precision, recall), end='')
        precisions[i] = precision
        recalls[i] = recall
        print('\t\tdone')

    print('Avg. Precision :', precisions.mean())
    print('Avg. Recall    :', recalls.mean())



