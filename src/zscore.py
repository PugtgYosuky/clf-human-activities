import pandas as pd
import numpy as np
import random 
import matplotlib.pyplot as plt

def calculate_zscore(data):
    mean = data.mean()
    std = data.std()
    z = (data - mean) / std
    return z

def calculate_outliers_indexes(data, k):
    z = calculate_zscore(data)
    outliers = (z > k) | (z < -k) # boolean pandas series
    return outliers

def calculate_zscore_density(data, outliers):
    density = outliers.sum() / data.count()
    print(f'Density - activity {activity}: {density * 100} %')
    return density # decimal value

def plot_zscore_outliers(data, k, variable):
    plt.figure()
    for activity in data['activity'].unique():
        activity_data = data[data['activity'] == activity][variable]
        outliers_indexes = calculate_outliers_indexes(activity_data, k)
        outliers = activity_data[outliers_indexes]
        x_data = np.ones_like(activity_data) * activity
        x_outliers = np.ones_like(outliers) * activity
        plt.plot(x_data, activity_data, 'b*')
        plt.plot(x_outliers, outliers, 'r*')
    plt.title(f'Z-Score outliers - variable {variable}, k={k}')
    plt.xticks([activity for activity in data['activity'].unique()])
    plt.xlabel('Activity')
    plt.show()