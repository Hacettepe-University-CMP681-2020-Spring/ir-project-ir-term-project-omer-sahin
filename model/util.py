import numpy as np
from search_engine.search import load_search_engine

search_engine = load_search_engine(path='../../query_reformulation_dataset')


def search_and_evaluate(inputs):
    weights, terms, query = inputs

    ref_query = recreate_query(terms=terms, weights=weights, threshold=0.5)

    top_docs = search_engine.get_top_documents(query=ref_query)
    precision, recall = query.calc_precision_recall(retrieved_documents=top_docs)

    reward = max((precision - query.base_precision) / query.base_precision, 0.0)

    print('    Base -> [P:%.5f, R:%.5f]  Reformulated -> [P:%.5f, R:%.5f]  Reward:%10.5f | %s -> %s'
          % (query.base_precision, query.base_recall, precision, recall, reward, query.query, ref_query))

    return reward


def recreate_query(terms, weights, threshold=0.5):
    try:
        index = np.argwhere(weights > threshold)[:, 0]
        query = ' '.join([terms[i] for i in index if terms[i] is not ''])
        return query.strip()
    except IndexError:
        return ''


# def precision_loss(search_engine, query_map, alpha=0.1):
#     def loss_function(y_true, y_pred):
#         query = query_map[y_true]
#         reformulated_query = recreate_query(terms=query.keywords, weights=y_pred)
#         top_documents = search_engine.get_top_documents(query=reformulated_query, top_k=10)
#         precision = query.article.precision_at(retrieved_documents=top_documents)
#         return alpha * (precision - y_true) ** 2
#
#     return loss_function

