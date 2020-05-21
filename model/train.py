from sklearn.model_selection import train_test_split

from model.preprocess import Preprocessor
from model.query_reformulation import QueryReformulation

if __name__ == '__main__':
    preprocessor = Preprocessor()
    preprocessor.load_data(path='../../query_reformulation_dataset')

    query_objs, query_sequence, terms_sequence, candidate_terms = \
        preprocessor.get_query_and_candidate_terms(sequence_length=20)

    # sublist for testing
    size = 12
    query_objs = query_objs[:size]
    query_sequence = query_sequence[:size]
    terms_sequence = terms_sequence[:size]
    candidate_terms = candidate_terms[:size]

    trn_query_objs, tst_query_objs, \
    trn_query_sequence, tst_query_sequence, \
    trn_terms_sequence, tst_terms_sequence, \
    trn_candidate_terms, tst_candidate_terms = train_test_split(query_objs, query_sequence,
                                                                terms_sequence, candidate_terms,
                                                                test_size=0.3, random_state=42)
    q_reform = QueryReformulation(output_path='../../saved_model')
    q_reform.build_model(model_name='cnn',  # 'cnn', 'lstm', 'bilstm'
                         query_dim=query_sequence.shape[1],
                         terms_dim=terms_sequence.shape[1],
                         output_dim=terms_sequence.shape[1],
                         word_embedding=preprocessor.word_embedding)

    q_reform.train_model(query_objs=trn_query_objs,
                         query_sequence=trn_query_sequence, terms_sequence=trn_terms_sequence,
                         candidate_terms=trn_candidate_terms, epochs=20, batch_size=4)

    avg_precision, avg_recall = q_reform.test_model(query_objs=tst_query_objs,
                                                    query_sequence=tst_query_sequence,
                                                    terms_sequence=tst_terms_sequence,
                                                    candidate_terms=tst_candidate_terms, batch_size=4)

    base_precision = 0
    base_recall = 0
    for query in tst_query_objs:
        base_precision += query.base_precision
        base_recall += query.base_recall

    print('Base query precision/recall:')
    print('  Avg. Precision : ', base_precision / len(tst_query_objs))
    print('  Avg. Recall    : ', base_recall / len(tst_query_objs))

    print('Reformulated query precision/recall:')
    print('  Avg. Precision : ', avg_precision)
    print('  Avg. Recall    : ', avg_recall)
