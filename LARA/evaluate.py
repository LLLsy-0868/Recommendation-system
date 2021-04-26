import numpy as np
import pandas as pd

def dcg_at_k(r, k):
    r = np.asfarray(r)[:k]
    if r.size:
        return np.sum(r/np.log2(np.arange(2, r.size+2)))
    else:
        return 0

def ndcg_at_k(item, k_similar_user, k):
    sum = 0.0
    ui_matrix = pd.read_csv('data/ui_matrix.csv', header=None)
    ui_matrix = np.array(ui_matrix[:])
    for test_i, test_user_list in zip(item, k_similar_user):
        r = []
        for user in test_user_list:
            r.append(ui_matrix[user][test_i])
        r_ideal = sorted(r, reverse=True)
        ideal_dcg = dcg_at_k(r_ideal, k)
        if ideal_dcg == 0:
            sum += 0
        else:
            sum += (dcg_at_k(r, k)/ideal_dcg)
    return sum/item.__len__()


def precision_at_k(item, k_similar_user, k):
    count = 0
    test_batch_size = item.__len__()
    ui_matrix = pd.read_csv('data/ui_matrix.csv', header=None)
    ui_matrix = np.array(ui_matrix[:])
    for test_i, test_user_list in zip(item, k_similar_user):
        for test_u in test_user_list:
            if ui_matrix[test_u, test_i] == 1:
                count += 1
    p_k = count / (test_batch_size * k)
    return p_k


def evaluate(item, rec_user):
    n_10 = ndcg_at_k(item, rec_user, 10)
    n_20 = ndcg_at_k(item, rec_user, 20)
    p_10 = precision_at_k(item, rec_user, 10)
    p_20 = precision_at_k(item, rec_user, 20)
    columns = [n_10, n_20,p_10,p_20]
    df = pd.DataFrame(columns=columns)
    df.to_csv('data/result.csv', line_terminator='\n', index=False, mode='a', encoding='utf8')
    return n_10, p_10, n_20, p_20
