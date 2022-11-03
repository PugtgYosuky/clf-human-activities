import pandas as pd
from utils import normalize_data
import numpy as np
import matplotlib.pyplot as plt


class KMeans:
    """
    KMeans class
    Implements the Kmeans algorithm

    Methods:
        - predict
        - update_clusters
        - get_distances
        - calculate_error
        - get_outliers
        - get_labels_with_outliers
    """

    def __init__(self, k=3, max_iterations=150):
        """
        Constructor of the class
        @param k: number of clusters
        @param max_iterations: max number of iterations to update the clusters
        """
        self.labels = None
        self.distances = None
        self.inertia = None
        self.centroids = None
        self.data = None
        self.k = k
        self.max_iterations = max_iterations

    def predict(self, dataset):
        """
        Predicts the clusters of a given dataset
        Uses a for loop to update the clusters until they are unchanged or until it reaches max_iterations
        In each iteration, updates the clusters and the new centroids. Also calculates the distances of each point to the
        centroid.
        @param dataset: dataset to predict the cluster
        @return: predicted labels (one for each row in the dataset)
        """
        self.data = normalize_data(dataset, feature_range=(1, 10))
        self.centroids = self.data.sample(n=self.k, random_state=42, ignore_index=True)

        for i in range(self.max_iterations):
            if i % 10 == 0:
                print(f'Iteration: {i}')
            new_centroids = self.update_clusters()
            if self.centroids.equals(new_centroids):
                break
            else:
                self.centroids = new_centroids

        self.inertia = np.sum(self.get_distances())

        return self.labels

    def update_clusters(self):
        """
        Updates the clusters by calculating the distances of each point to every centroid, then each point is attributed
        to the closest centroid. Finally, updates the centroids coordinates based on the points coordinates.
        @return: the new centroids calculated
        """
        self.distances = self.centroids.apply(lambda x: np.sqrt(((self.data - x) ** 2).sum(axis=1)), axis=1).T
        self.labels = self.distances.idxmin(axis=1)
        new_centroids = self.data.groupby(self.labels).mean()
        return new_centroids

    def get_distances(self):
        """
        Gets the distance of each point to its centroid
        @return: pandas series with the distances
        """
        return self.distances.min(axis=1)

    def calculate_error(self):
        """
        Calculates the error. The error is calculated by the sum of all distances to their centroid
        @return: the error (float)
        """
        return np.sum(self.get_distances())

    def get_outliers(self, threshold):
        """
        Calculates outliers
        @param threshold: Maximum distance to the point be considered as non-outlier
        @return: a pandas series (of booleans - each value corresponds to a row) indicating if the value is an outlier (True)
        or not (False)
        """
        return self.get_distances() > threshold

    def get_labels_with_outliers(self, threshold):
        """
        After calculating all the outliers, attributes a new label called 'outlier' to the outliers.
        @param threshold: Maximum distance to the point be considered as non-outlier
        @return: list of labels (one for each row in the dataset)
        """
        labels = self.labels.copy().astype('str')
        outliers = self.get_outliers(threshold)
        labels[outliers] = 'outlier'
        return labels


def best_number_clusters(data, threshold=0.85, init=1, stop=10):
    """
    Calculates the best number of outliers that minimizes the total error. And plots the results
    @param data: dataset to use in the KMeans
    @param threshold: The difference between errors - Elbow Method. When the difference between errors is greater than
    the threshold, considers that value as the best number of clusters
    @param init: Minimum number of outliers
    @param stop: Maximum number of outliers
    @return: The best number of outliers
    """
    k_distances = []
    best_k = stop
    for k in range(init, stop, 1):
        kmeans = KMeans(k)
        values = kmeans.predict(data)
        inertia = kmeans.inertia  # lib_kmeans(n_clusters=k).fit(data).inertia_
        if k != init and inertia / k_distances[-1] > threshold and best_k == stop:
            best_k = k - 1
        k_distances += [inertia]

    plt.figure()
    x_values = [(i + init) for i in range(len(k_distances))]
    plt.plot(x_values, k_distances, 'r*')
    plt.xlabel('Number of clusters')
    plt.ylabel('Sum distances to centroids')
    plt.title('Relations between clusters and centroids')
    plt.show()
    return best_k  # best value
