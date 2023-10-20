import torch
from pyclustering.cluster.center_initializer import kmeans_plusplus_initializer, random_center_initializer
from pyclustering.utils import distance_metric, type_metric
from sklearn.cluster import KMeans
import numpy as np

from hyptorch.pmath import _dist_matrix, _dist_matrix_pyclustering
from models.k_means import K_Means_hyper

"""
implementation of different hierarchical clustering methods
"""

"""
pyclustering
nltk
"""


# def hierarchical_clustering_K_Means(option, x, num_cluster):
#     """
#     input:
#     option: configurations
#     x: input data, numpy array
#     num_cluster:[100,50,20]
#     :return: clustering results
#     """
#     # number of levels
#     level = len(num_cluster)
#     data_queue = [x]
#     results = {'im2cluster': [], 'centroids': [], 'density': []}
#     for i in range(level):
#         # perform k means for every level
#         k_means = KMeans(n_clusters=num_cluster[i], max_iter=30, random_state=3407).fit(data_queue[len(data_queue) - 1])
#         results['centroids'].append(k_means.cluster_centers_)
#         results['density'].append(k_means.inertia_)
#         results['im2cluster'].append(k_means.labels_)
#         data_queue.append(k_means.cluster_centers_)
#     return results

def hierarchical_clustering_K_Means(option, x, num_cluster, hyper_c=0.):
    """
    input:
    option: configurations
    x: input data, numpy array
    num_cluster:[100,50,20]
    :return: clustering results
    """
    # number of levels
    level = len(num_cluster)
    data_queue = [x]
    results = {'im2cluster': [], 'centroids': [], 'density': []}
    for i in range(level):
        # perform k means for every level
        initial_centers = random_center_initializer(data_queue[len(data_queue) - 1], num_cluster[i]).initialize()
        if hyper_c == 0.:
            k_means = K_Means_hyper(data_queue[len(data_queue) - 1], initial_centers, ccore=False, itermax=30)
            k_means.process()
        else:
            hyper_metric = distance_metric(type_metric.USER_DEFINED,
                                           func=lambda x, y: _dist_matrix_pyclustering(x, y, hyper_c))
            k_means = K_Means_hyper(data_queue[len(data_queue) - 1], initial_centers, ccore=False, itermax=30,
                                    hyper_c=hyper_c, metric=hyper_metric)
            k_means.process()
        results['centroids'].append(np.array(k_means.get_centers()))
        results['density'].append(k_means.get_total_wce())
        results['im2cluster'].append(k_means.get_clusters())
        data_queue.append(k_means.get_centers())
    return results


def hierarchical_clustering_HC():
    pass


if __name__ == '__main__':
    # test_data = np.array([[1, 1], [2, 2], [2.5, 2.3], [5.5, 6.6], [5.5, 6.1], [5.4, 6.6], [100, 101],
    #                       [100, 102], [101, 103], [120, 121], [122, 121], [123, 122],
    #                       [1000, 1001], [1001, 1002], [1002, 1003], [1010, 1011], [1010, 1012], [1011, 1012],
    #                       [1200, 1201], [1201, 1202], [1202, 1203], [1210, 1211], [1210, 1212], [1211, 1212]])
    test_data = np.random.uniform(-1, 1, (5000, 16))
    result = hierarchical_clustering_K_Means(None, test_data, [50, 20, 10], hyper_c=0.01)
