import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, minmax_scale


def normalize_data(dataset, feature_range=(0, 1)):
    """
    Normalizes each column between the given range
    @param dataset: dataset
    @param feature_range: range to normalize the values
    @return: the normalized dataset
    """
    data = dataset.copy()
    data = minmax_scale(data, feature_range=feature_range, axis=0)
    return pd.DataFrame(data=data, columns=dataset.columns)


def append_arrays(data_array):
    """
    Merges all the given arrays in just one
    @param data_array: a list of arrays to merge
    @return: dataset
    """
    data = data_array[0]
    for i in range(1, len(data_array)):
        data = np.concatenate((data, data_array[i]), axis=0)
    return data
