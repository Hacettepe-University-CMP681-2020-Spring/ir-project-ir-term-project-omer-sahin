
from index.query_title_article_indexer import Indexer

if __name__ == '__main__':
    query_path = '../query_reformulation_dataset/JEOPARDY_CSV.csv'
    qrel_path = '../query_reformulation_dataset/trec_car/base.train.cbor-article.qrels'
    paragraph_path = '../query_reformulation_dataset/trec_car/base.train.cbor-paragraphs.cbor'

    qta_indexer = Indexer()
    qta_indexer.index_documents(query_path=query_path,
                                qrel_path=qrel_path,
                                paragraph_path=paragraph_path)

    for doc in qta_indexer.iterate_documents():
        print(doc.title)
        print(doc.query)
        print('Number of article:', len(doc.article_list))
        print('-'*50)

    count = 0
    for title in qta_indexer.article_list.keys():
        for article in qta_indexer.article_list[title]:
            print(title, ':', article)
            count += 1

    print('Number of articles:', count)
