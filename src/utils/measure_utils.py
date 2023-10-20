import time
from tqdm import tqdm
import numpy as np

from utils.logger import Logger


class ds:
    def __init__(self):
        self.output = []
        self.label = []


def partition_arg_topK(matrix, K, axis=0):
    """
    perform topK based on np.argpartition
    :param matrix: to be sorted
    :param K: select and sort the top K items
    :param axis: 0 or 1. dimension to be sorted.
    :return:
    """
    a_part = np.argpartition(matrix, K, axis=axis)
    if axis == 0:
        row_index = np.arange(matrix.shape[1 - axis])
        a_sec_argsort_K = np.argsort(matrix[a_part[0:K, :], row_index], axis=axis)
        return a_part[0:K, :][a_sec_argsort_K, row_index]
    else:
        column_index = np.arange(matrix.shape[1 - axis])[:, None]
        a_sec_argsort_K = np.argsort(matrix[column_index, a_part[:, 0:K]], axis=axis)
        return a_part[:, 0:K][column_index, a_sec_argsort_K]




class LabelMatchs(object):
    def __init__(self, label_match_matrix):
        self.label_match_matrix = label_match_matrix
        self.all_sims = np.sum(label_match_matrix, axis=1)


def calc_label_match_matrix(database_labels, query_labels):
    """
    :param database_labels:
    :param query_labels:
    :return: T * N matrix: N for database size and T for query size
    Notes
    -----
    There is more than one definition of sign in common use for complex
    numbers.  The definition used here is equivalent to :math:`x/\sqrt{x*x}`
    which is different from a common alternative, :math:`x/|x|`.
    Examples
    --------
        query_labels = np.array([[0,1,0], [1,1,0]])
            array([[0, 1, 0],
                   [1, 1, 0]])
        database_labels = np.array([[1,0,0], [1,1,0], [1,0,1], [0,0,1]])
            array([[1, 0, 0],
                   [1, 1, 0],
                   [1, 0, 1],
                   [0, 0, 1]])
        ret = np.dot(query_labels, database_labels.T) > 0
            array([[False,  True, False, False],
                   [ True,  True,  True, False]])
        """
    return LabelMatchs(np.dot(query_labels, database_labels.T) > 0)


################################################################

def mean_average_precision(database_hash, test_hash, database_labels, test_labels, option, sim=None,
                           ids=None):  # R = 1000
    # binary the hash code
    R = option.R
    T = option.T
    database_hash[database_hash < T] = -1
    database_hash[database_hash >= T] = 1
    test_hash[test_hash < T] = -1
    test_hash[test_hash >= T] = 1

    query_num = test_hash.shape[0]  # total number for testing
    if sim is None:
        sim = np.dot(database_hash, test_hash.T)
    if ids is None:
        ids = np.argsort(-sim, axis=0)
    del sim
    # data_dir = 'data/' + args.data_name
    # ids_10 = ids[:10, :]

    # np.save(data_dir + '/ids.npy', ids_10)
    APx = []
    Recall = []
    Pre = []
    iteration = tqdm(range(query_num), desc="CalMAP")
    for i in iteration:  # for i=0
        label = test_labels[i, :]  # the first test labels
        if np.sum(label) == 0:  # ignore images with meaningless label in nus wide
            continue
        label[label == 0] = -1
        idx = ids[:, i]
        imatch = np.sum(database_labels[idx[0:R], :] == label, axis=1) > 0
        relevant_num = np.sum(imatch)
        Lx = np.cumsum(imatch)
        Px = Lx.astype(float) / np.arange(1, R + 1, 1)  #
        if relevant_num != 0:
            APx.append(np.sum(Px * imatch) / relevant_num)
        if relevant_num == 0:  # even no relevant image, still need add in APx for calculating the mean
            APx.append(0)
        all_relevant = np.sum(database_labels == label, axis=1) > 0
        all_num = np.sum(all_relevant)
        r = relevant_num / np.float(all_num)
        Pre.append(relevant_num / R)
        Recall.append(r)
    return np.mean(np.array(APx)), np.mean(np.array(Recall)), np.mean(np.array(Pre))


def get_precision_recall_by_Hamming_Radius_optimized(database_output, database_labels, query_output, query_labels,
                                                     radius=2, label_matchs=None, coarse_sign=True, fine_sign=False):
    """
    :param database:
    :param query:
    :param radius:
    :param label_match_matrix: In this optimization, we suppose the test and database lists are fixed, so we only
    calculate the test-db label matching relation once and store it in a matrix with space complexity O(db_size * test_size).
    :return:
    """
    # query_output = query.output
    # database_output = database.output
    # query_labels = query.label
    # database_labels = database.label
    # prevent impact from other measure function
    query_labels[query_labels < 0] = 0
    database_labels[database_labels < 0] = 0
    bit_n = query_output.shape[1]  # i.e. K
    coarse_query_output = np.sign(query_output)
    coarse_database_output = np.sign(database_output)
    del query_output, database_output
    # fine_query_output = coarse_query_output if fine_sign else query_output
    # fine_database_output = coarse_database_output if fine_sign else database_output
    fine_query_output = coarse_query_output
    fine_database_output = coarse_database_output

    label_matrix_time = -1
    Logger.info("calculate match matrix")
    if label_matchs is None:
        tmp_time = time.time()
        label_matchs = calc_label_match_matrix(database_labels, query_labels)
        label_matrix_time = time.time() - tmp_time
        Logger.info("calc label matrix: time: {:.3f}\n".format(label_matrix_time))
    start_time = time.time()
    ips = np.dot(coarse_query_output, coarse_database_output.T)
    ips = (bit_n - ips) / 2
    ids = np.argsort(ips, 1)
    end_time = time.time()
    sort_time = end_time - start_time
    Logger.info("total query: {:d}, sorting time: {:.3f}\n".format(ips.shape[0], sort_time))
    all_nums = np.sum(ips <= radius, axis=1)
    del ips
    precX = []
    recX = []
    mAPX = []
    matchX = []
    allX = []
    iteration = tqdm(range(coarse_query_output.shape[0]), desc="CalMAP")
    for i in iteration:
        # if i % 100 == 0:
        #     tmp_time = time.time()
        #     # print("query map {:d}, time: {:.3f}".format(i, tmp_time - end_time))
        #     end_time = tmp_time
        all_num = all_nums[i]

        if all_num != 0:
            idx = ids[i, 0:all_num]
            if fine_sign:
                imatch = label_matchs.label_match_matrix[i, idx[:]]
            else:
                ips_continue = np.dot(fine_query_output[i, :], fine_database_output[idx, :].T)
                subset_idx = np.argsort(-ips_continue, axis=0)
                idx_continue = idx[subset_idx]
                imatch = label_matchs.label_match_matrix[i, idx_continue]

            match_num = int(np.sum(imatch))
            matchX.append(match_num)
            allX.append(all_num)
            precX.append(np.float(match_num) / all_num)
            all_sim_num = label_matchs.all_sims[i]
            recX.append(np.float(match_num) / (all_sim_num + 1e-6))

            Lx = np.cumsum(imatch)
            Px = Lx.astype(float) / np.arange(1, all_num + 1, 1)
            if match_num != 0:
                mAPX.append(np.sum(Px * imatch) / match_num)
            else:
                mAPX.append(0)

    # print("total query: {:d}, sorting time: {:.3f}".format(ips.shape[0], sort_time))
    # print("total time(no label matrix): {:.3f}".format(time.time() - start_time))
    if label_matrix_time > 0:
        pass
        # print("calc label matrix: time: {:.3f}".format(label_matrix_time))
    meanPrecX = 0 if len(precX) == 0 else np.mean(np.array(precX))
    meanRecX = 0 if len(recX) == 0 else np.mean(np.array(recX))
    meanMAPX = 0 if len(mAPX) == 0 else np.mean(np.array(mAPX))
    del fine_database_output, fine_query_output, label_matchs
    return meanPrecX, meanRecX, meanMAPX
