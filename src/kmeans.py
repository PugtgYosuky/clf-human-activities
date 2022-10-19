import pandas as pd
from utils import normalize_data
import numpy as np
from sklearn.cluster import KMeans as lib_kmeans 
import matplotlib.pyplot as plt

class KMeans:

    def __init__(self, k=3, max_iterations=100):
        self.k = k
        self.max_iterations = max_iterations

    def predict(self, dataset):
        self.data = normalize_data(dataset, feature_range=(1, 10))
        self.centroids = self.data.sample(n=self.k, random_state=42, ignore_index=True)

        for i in range(self.max_iterations):
            print(f'Iteration: {i}')
            new_centroids = self.update_clusters()
            if self.centroids.equals(new_centroids):
                break
            else:
                self.centroids = new_centroids

        return self.labels

    def update_clusters(self):
        self.distances = self.centroids.apply(lambda x : np.sqrt(((self.data - x) **2).sum(axis=1)), axis=1).T
        self.labels = self.distances.idxmin(axis=1)
        new_centroids = self.data.groupby(self.labels).mean()
        return new_centroids

    def get_distances(self):
        return self.distances.min(axis=1)

def best_number_clusters(data, threshold=0.75, init=3, stop=20):
    k_distances = []
    best_k = stop
    for k in range(init, stop, 1):
        inertia = lib_kmeans(n_clusters=k).fit(data).inertia_
        if k != init and inertia / k_distances[-1] > threshold and best_k == stop:
            best_k = k - 1
        k_distances += [inertia]

    plt.figure()
    x_values = [ (i + init) for i in range(len(k_distances))]
    plt.plot(x_values, k_distances, 'r*')
    plt.xlabel('Number of clusters')
    plt.ylabel('Sum distances to centroids')
    plt.title('Relations between clusters and centroids')
    plt.show()
    return best_k # best value