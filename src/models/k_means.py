import copy
import numpy

import matplotlib.pyplot as plt
import matplotlib.animation as animation

import pyclustering.core.kmeans_wrapper as wrapper
import torch

from pyclustering.core.wrapper import ccore_library
from pyclustering.core.metric_wrapper import metric_wrapper

from pyclustering.cluster.encoder import type_encoding
from pyclustering.cluster import cluster_visualizer

from pyclustering.utils.metric import distance_metric, type_metric

from hyptorch.pmath import p2k, lorenz_factor, k2p


class K_Means_hyper():
    def __init__(self, data, initial_centers, tolerance=0.001, ccore=True, hyper_c=None, **kwargs):
        """!
        @brief Constructor of clustering algorithm K-Means.
        @details Center initializer can be used for creating initial centers, for example, K-Means++ method.

        @param[in] data (array_like): Input data that is presented as array of points (objects), each point should be represented by array_like data structure.
        @param[in] initial_centers (array_like): Initial coordinates of centers of clusters that are represented by array_like data structure: [center1, center2, ...].
        @param[in] tolerance (double): Stop condition: if maximum value of change of centers of clusters is less than tolerance then algorithm stops processing.
        @param[in] ccore (bool): Defines should be CCORE library (C++ pyclustering library) used instead of Python code or not.
        @param[in] **kwargs: Arbitrary keyword arguments (available arguments: 'observer', 'metric', 'itermax').

        <b>Keyword Args:</b><br>
            - observer (kmeans_observer): Observer of the algorithm to collect information about clustering process on each iteration.
            - metric (distance_metric): Metric that is used for distance calculation between two points (by default euclidean square distance).
            - itermax (uint): Maximum number of iterations that is used for clustering process (by default: 200).

        @see center_initializer

        """
        self.__pointer_data = numpy.array(data)
        self.__clusters = []
        self.__centers = numpy.array(initial_centers)
        self.__tolerance = tolerance
        self.__total_wce = 0.0

        self.__observer = kwargs.get('observer', None)
        self.__metric = copy.copy(kwargs.get('metric', distance_metric(type_metric.EUCLIDEAN_SQUARE)))
        self.__itermax = kwargs.get('itermax', 100)

        if self.__metric.get_type() != type_metric.USER_DEFINED:
            self.__metric.enable_numpy_usage()
        else:
            self.__metric.disable_numpy_usage()

        self.__ccore = ccore and self.__metric.get_type() != type_metric.USER_DEFINED
        if self.__ccore is True:
            self.__ccore = ccore_library.workable()

        self.__verify_arguments()
        self.hyper_c = hyper_c

    def process(self):
        """!
        @brief Performs cluster analysis in line with rules of K-Means algorithm.

        @return (kmeans) Returns itself (K-Means instance).

        @see get_clusters()
        @see get_centers()

        """

        if len(self.__pointer_data[0]) != len(self.__centers[0]):
            raise ValueError("Dimension of the input data and dimension of the initial cluster centers must be equal.")

        if self.__ccore is True:
            self.__process_by_ccore()
        else:
            self.__process_by_python()

        return self

    def __process_by_ccore(self):
        """!
        @brief Performs cluster analysis using CCORE (C/C++ part of pyclustering library).

        """
        ccore_metric = metric_wrapper.create_instance(self.__metric)

        results = wrapper.kmeans(self.__pointer_data, self.__centers, self.__tolerance, self.__itermax,
                                 (self.__observer is not None), ccore_metric.get_pointer())

        self.__clusters = results[0]
        self.__centers = results[1]

        if self.__observer is not None:
            self.__observer.set_evolution_clusters(results[2])
            self.__observer.set_evolution_centers(results[3])

        self.__total_wce = results[4][0]

    def __process_by_python(self):
        """!
        @brief Performs cluster analysis using python code.

        """

        maximum_change = float('inf')
        iteration = 0

        if self.__observer is not None:
            initial_clusters = self.__update_clusters()
            self.__observer.notify(initial_clusters, self.__centers.tolist())

        while maximum_change > self.__tolerance and iteration < self.__itermax:
            self.__clusters = self.__update_clusters()
            updated_centers = self.__update_centers()  # changes should be calculated before assignment

            if self.__observer is not None:
                self.__observer.notify(self.__clusters, updated_centers.tolist())

            maximum_change = self.__calculate_changes(updated_centers)

            self.__centers = updated_centers  # assign center after change calculation
            iteration += 1

        self.__calculate_total_wce()

    def predict(self, points):
        """!
        @brief Calculates the closest cluster to each point.

        @param[in] points (array_like): Points for which closest clusters are calculated.

        @return (list) List of closest clusters for each point. Each cluster is denoted by index. Return empty
                 collection if 'process()' method was not called.

        """

        nppoints = numpy.array(points)
        if len(self.__clusters) == 0:
            return []

        differences = numpy.zeros((len(nppoints), len(self.__centers)))
        for index_point in range(len(nppoints)):
            if self.__metric.get_type() != type_metric.USER_DEFINED:
                differences[index_point] = self.__metric(nppoints[index_point], self.__centers)
            else:
                differences[index_point] = [self.__metric(nppoints[index_point], center) for center in self.__centers]

        return numpy.argmin(differences, axis=1)

    def get_clusters(self):
        """!
        @brief Returns list of allocated clusters, each cluster contains indexes of objects in list of data.

        @see process()
        @see get_centers()

        """
        cluster_index = numpy.zeros(self.__pointer_data.shape[0])
        for i in range(len(self.__clusters)):
            cluster_index[self.__clusters[i]] = i
        return cluster_index

    def get_centers(self):
        """!
        @brief Returns list of centers of allocated clusters.

        @see process()
        @see get_clusters()

        """

        if isinstance(self.__centers, list):
            return self.__centers

        return self.__centers.tolist()

    def get_total_wce(self):
        """!
        @brief Returns sum of metric errors that depends on metric that was used for clustering (by default SSE - Sum of Squared Errors).
        @details Sum of metric errors is calculated using distance between point and its center:
                 \f[error=\sum_{i=0}^{N}distance(x_{i}-center(x_{i}))\f]

        @see process()
        @see get_clusters()

        """

        return self.__total_wce

    def get_cluster_encoding(self):
        """!
        @brief Returns clustering result representation type that indicate how clusters are encoded.

        @return (type_encoding) Clustering result representation.

        @see get_clusters()

        """

        return type_encoding.CLUSTER_INDEX_LIST_SEPARATION

    def __update_clusters(self):
        """!
        @brief Calculate distance (in line with specified metric) to each point from the each cluster. Nearest points
                are captured by according clusters and as a result clusters are updated.

        @return (list) Updated clusters as list of clusters. Each cluster contains indexes of objects from data.

        """

        clusters = [[] for _ in range(len(self.__centers))]
        # 这个过程应该是耗时的关键
        dataset_differences = self.__calculate_dataset_difference(len(clusters))

        optimum_indexes = numpy.argmin(dataset_differences, axis=0)
        for index_point in range(len(optimum_indexes)):
            index_cluster = optimum_indexes[index_point]
            clusters[index_cluster].append(index_point)

        clusters = [cluster for cluster in clusters if len(cluster) > 0]

        return clusters

    def __update_centers(self):
        """!
        @brief Calculate centers of clusters in line with contained objects.

        @return (numpy.array) Updated centers.

        """

        dimension = self.__pointer_data.shape[1]
        centers = numpy.zeros((len(self.__clusters), dimension))

        for index in range(len(self.__clusters)):
            cluster_points = self.__pointer_data[self.__clusters[index], :]
            if self.hyper_c is not None:
                # calculate prototypes in Poincare ball
                dim = 0
                cluster_points = torch.tensor(cluster_points)
                x = p2k(cluster_points, self.hyper_c)
                lamb = lorenz_factor(x, c=self.hyper_c, keepdim=True)
                mean = torch.sum(lamb * x, dim=dim, keepdim=True) / torch.sum(
                    lamb, dim=dim, keepdim=True
                )
                mean = k2p(mean, self.hyper_c)
                centers[index] = mean.squeeze(dim).numpy()
            else:
                centers[index] = cluster_points.mean(axis=0)
        return numpy.array(centers)

    def __calculate_total_wce(self):
        """!
        @brief Calculate total within cluster errors that is depend on metric that was chosen for K-Means algorithm.

        """

        dataset_differences = self.__calculate_dataset_difference(len(self.__clusters))

        self.__total_wce = 0.0
        for index_cluster in range(len(self.__clusters)):
            for index_point in self.__clusters[index_cluster]:
                self.__total_wce += dataset_differences[index_cluster][index_point]

    def __calculate_dataset_difference(self, amount_clusters):
        """!
        @brief Calculate distance from each point to each cluster center.
        # 这里没有进行矩阵运算，所以是慢的根源#########################################
        """
        dataset_differences = numpy.zeros((amount_clusters, len(self.__pointer_data)))
        for index_center in range(amount_clusters):
            if self.__metric.get_type() != type_metric.USER_DEFINED:
                dataset_differences[index_center] = self.__metric(self.__pointer_data, self.__centers[index_center])
            else:
                dataset_differences[index_center] = self.__metric(self.__pointer_data, self.__centers[index_center])

        return dataset_differences

    def __calculate_changes(self, updated_centers):
        """!
        @brief Calculates changes estimation between previous and current iteration using centers for that purpose.

        @param[in] updated_centers (array_like): New cluster centers.

        @return (float) Maximum changes between centers.

        """
        if len(self.__centers) != len(updated_centers):
            maximum_change = float('inf')

        else:
            if self.__metric.get_type() != type_metric.USER_DEFINED:
                changes = self.__metric(self.__centers, updated_centers)
            else:
                # changes = [self.__metric(center, updated_center) for center, updated_center in
                #            zip(self.__centers, updated_centers)]
                changes = self.__metric(self.__centers, updated_centers)

            maximum_change = numpy.max(changes)

        return maximum_change

    def __verify_arguments(self):
        """!
        @brief Verify input parameters for the algorithm and throw exception in case of incorrectness.

        """
        if len(self.__pointer_data) == 0:
            raise ValueError("Input data is empty (size: '%d')." % len(self.__pointer_data))

        if len(self.__centers) == 0:
            raise ValueError("Initial centers are empty (size: '%d')." % len(self.__pointer_data))

        if self.__tolerance < 0:
            raise ValueError("Tolerance (current value: '%d') should be greater or equal to 0." %
                             self.__tolerance)

        if self.__itermax < 0:
            raise ValueError("Maximum iterations (current value: '%d') should be greater or equal to 0." %
                             self.__tolerance)
