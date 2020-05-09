import numpy as np
from search_engine.search import search_engine_instance as search_engine


def get_batch_data(query_objs, query_sequence, terms_sequence, candidate_terms, batch_size=4):
    for i in range(0, len(query_objs), batch_size):
        query = query_objs[i:i + batch_size]
        q_seq = query_sequence[i:i + batch_size]
        t_seq = terms_sequence[i:i + batch_size]
        terms = candidate_terms[i:i + batch_size]
        yield i, query, q_seq, t_seq, terms


def recreate_query(terms, weights, threshold=0.5):
    try:
        index = np.argwhere(weights > threshold)[:, 0]
        query = ' '.join([terms[i] for i in index if terms[i] is not ''])
        return query.strip()
    except IndexError:
        return ''


def evaluate_reward_precision(inputs):
    weights, terms, query = inputs

    ref_query = recreate_query(terms=terms, weights=weights, threshold=0.5)

    top_docs = search_engine.get_top_documents(query=ref_query)
    precision, recall = query.calc_precision_recall(retrieved_documents=top_docs)

    reward = max((precision - query.base_precision) / query.base_precision, 0.0)

    print('    Base -> [P:%.5f, R:%.5f]  Reformulated -> [P:%.5f, R:%.5f]  Reward:%10.5f | %s -> %s'
          % (query.base_precision, query.base_recall, precision, recall, reward, query.query, ref_query))

    return reward, precision


def evaluate_precision_recall(inputs):
    weights, terms, query = inputs

    ref_query = recreate_query(terms=terms, weights=weights, threshold=0.5)

    top_docs = search_engine.get_top_documents(query=ref_query)
    precision, recall = query.calc_precision_recall(retrieved_documents=top_docs)

    print('  Base -> [P:%.5f, R:%.5f]  Reformulated -> [P:%.5f, R:%.5f] | %s -> %s'
          % (query.base_precision, query.base_recall, precision, recall, query.query, ref_query))

    return precision, recall
