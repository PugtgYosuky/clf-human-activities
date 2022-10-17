import pandas as pd
from utils import normalize_data
import numpy as np

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