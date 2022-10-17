import pandas as pd
import numpy as np
import random 

class ZScore:
    def __init__(self, data, k=3.5):
        self.data = data
        self.k = k
        self.calculate_zscore()
    def calculate_zscore(self):
        self.z = (self.data - self.data.mean()) / self.data.std()
    def outliers_indexes(self):
        self.calculate_zscore()
        self.indexes= self.z > self.k
        return self.indexes

    def calculate_density(self):
        outliers = self.outliers_indexes()
        density = outliers.sum() / self.data.count()
        print(f'Density: {density}')
        return density

    def add_outliers(self, percentage, z=1):
        outliers_density = self.calculate_density()
        if percentage > outliers_density:
            indexes_non_outliers = pd.Series(self.indexes.index[self.indexes == False])
            number_values = int((percentage - outliers_density) * self.data.count())
            selected_values_to_change = pd.Series(indexes_non_outliers).sample(number_values)
            print(len(self.data))
            print(len(selected_values_to_change))
            mean = self.data.mean()
            std = self.data.std()
            transform_data = self.data.copy()
            mean = self.data.mean()
            std = self.data.std()
            transform_data[selected_values_to_change] = transform_data[selected_values_to_change].apply(
                lambda x: (mean + random.choice([1, -1])*self.k*(std+np.random.uniform(0, z)))
                )
            return transform_data
