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

def calculate_zscore_density(data, outliers, activity):
    density = outliers.sum() / data.count()
    # print(f'Density - activity {activity}: {density * 100} %')
    return density # decimal value

def plot_zscore_outliers(data, variable, k_value=None):
    if k_value :
        ks = [k_value]
    else:
        ks = [3, 3.5, 4]
    i = 1
    f = plt.figure(figsize=(20,15))
    for k in ks:
        ax = f.add_subplot(1,3,i)
        for activity in data['activity'].unique():
            activity_data = data[data['activity'] == activity][variable]
            outliers_indexes = calculate_outliers_indexes(activity_data, k)
            density = calculate_zscore_density(activity_data, outliers_indexes, activity)
            print('Density: ', density * 100)
            outliers = activity_data[outliers_indexes]
            x_data = np.ones_like(activity_data) * activity
            x_outliers = np.ones_like(outliers) * activity
            ax.plot(x_data, activity_data, 'b*')
            ax.plot(x_outliers, outliers, 'r*')
        ax.set_title(f'Z-Score outliers - variable {variable}, k={k}')
        ax.set_xticks([activity for activity in data['activity'].unique()])
        ax.set_xlabel('Activity')
        
        i += 1
    plt.show()
