import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

def normalize_data(dataset, feature_range=(0, 1)):
    data = dataset.copy()
    data = minmax_scale(data, feature_range=feature_range, axis=0)
    return pd.DataFrame(data=data, columns=dataset.columns)

def scale_features(dataset):
    return (dataset-dataset.mean()) ( dataset.std())

def append_arrays(data_array):
    data = data_array[0]
    for i in range(1, len(data_array)):
        data = np.concatenate((data, data_array[i]), axis=0)
    return data