import numpy as np
from multiprocessing import Pool, cpu_count

from sklearn.model_selection import train_test_split

from model.preprocess import Preprocessor
from model.query_reformulation import QueryReformulation
from search_engine.search import search_engine_instance as search_engine


class Evaluate:

    def __init__(self, search_engine, reformulation_model, query_list, sample_size):
        self.search_engine = search_engine
        self.reformulation_model = reformulation_model
        self.query_list = query_list
        self.k_list = [1, 5, 10, 20, 40]

        self.eval_base = np.zeros(shape=(len(self.k_list), 4, sample_size))
        self.eval_reform = np.zeros(shape=(len(self.k_list), 4, sample_size))

    def evaluate_queries(self):
        # pool = Pool(4)
        # pool.map(self.evaluate_query, enumerate(self.query_list))
        # pool.close()
        # pool.join()

        for query_tuple in enumerate(self.query_list):
            self.evaluate_query(query_tuple)

        for i, k in enumerate(self.k_list):
            print('                P@%d \t\t R@%d \t\t MAP@%d \t NDCG@%d' % (k, k, k, k))
            print('Baseline      : %.5f \t %.5f \t %.5f \t %.5f' % (
                self.eval_base[i, 0, :].mean(), self.eval_base[i, 1, :].mean(),
                self.eval_base[i, 2, :].mean(), self.eval_base[i, 3, :].mean()))
            print('Reformulated  : %.5f \t %.5f \t %.5f \t %.5f' % (
                self.eval_reform[i, 0, :].mean(), self.eval_reform[i, 1, :].mean(),
                self.eval_reform[i, 2, :].mean(), self.eval_reform[i, 3, :].mean()))
            print('-'*60)

    def evaluate_query(self, query_tuple):
        instance, (query, q_seq, t_seq, terms) = query_tuple

        base_documents = self.search_engine.get_top_documents(query=query.query, top_k=40)

        reformulated_query = self.reformulation_model.reformulate_query(query_sequence=q_seq,
                                                                        terms_sequence=t_seq,
                                                                        candidate_terms=terms)

        reformulated_documents = self.search_engine.get_top_documents(query=reformulated_query, top_k=40)

        for i, k in enumerate(self.k_list):
            base_precision, base_recall = query.calc_precision_recall(retrieved_documents=base_documents[:k])
            ref_precision, ref_recall = query.calc_precision_recall(retrieved_documents=reformulated_documents[:k])

            print('  Evaluate @%d  : %s -> %s \n'
                  '    Precision   : [%.6f -> %.6f]\n'
                  '    Recall      : [%.6f -> %.6f]\n'
                  '    MAP         : [----]\n'
                  '    NDCG        : [----]'
                  % (k, query.query, reformulated_query,
                     base_precision, base_recall,
                     ref_precision, ref_recall))

            self.eval_base[i, 0, instance] = base_precision
            self.eval_base[i, 1, instance] = base_recall
            self.eval_base[i, 2, instance] = 0  # base_map
            self.eval_base[i, 3, instance] = 0  # base_ndcg

            self.eval_reform[i, 0, instance] = ref_precision
            self.eval_reform[i, 1, instance] = ref_recall
            self.eval_reform[i, 2, instance] = 0  # ref_map
            self.eval_reform[i, 3, instance] = 0  # ref_ndcg


if __name__ == '__main__':
    # q_reform = QueryReformulation(model_path='../../saved_model/reformulation_model_2020-05-06')

    preprocessor = Preprocessor()
    preprocessor.load_data(path='../../query_reformulation_dataset')
    query_objs, query_sequence, terms_sequence, candidate_terms = \
        preprocessor.get_query_and_candidate_terms(sequence_length=20)

    size = 100
    query_objs = query_objs[:size]
    query_sequence = query_sequence[:size]
    terms_sequence = terms_sequence[:size]
    candidate_terms = candidate_terms[:size]

    _, query_objs, _, query_sequence, _, terms_sequence, _, candidate_terms = train_test_split(query_objs,
                                                                                               query_sequence,
                                                                                               terms_sequence,
                                                                                               candidate_terms,
                                                                                               test_size=0.3,
                                                                                               random_state=42)

    q_reform = QueryReformulation()
    q_reform.build_cnn_model(query_dim=query_sequence.shape[1],
                             terms_dim=terms_sequence.shape[1],
                             output_dim=terms_sequence.shape[1],
                             word_embedding=preprocessor.word_embedding)

    evaluate = Evaluate(search_engine=search_engine, reformulation_model=q_reform,
                        query_list=zip(query_objs, query_sequence, terms_sequence, candidate_terms),
                        sample_size=len(query_objs))

    evaluate.evaluate_queries()
