import pandas as pd
import numpy as np
import random
import matplotlib.pyplot as plt


def calculate_zscore(data):
    """
    Calculate th Z-Score of each column in the dataframe
    @param data:
    @return:
    """
    z = (data - data.mean()) / data.std()
    return z


def calculate_outliers_indexes(data, k):
    """
    Calculate the Z-Score outliers
    @param data: data to analyse
    @param k: max distance to the value to be considered non-outliers
    @return: array og booleans indicating the outliers in the data
    """
    z = calculate_zscore(data)
    outliers = (z > k) | (z < -k)  # boolean pandas series
    return outliers


def calculate_density(outliers):
    """
    Calculate the density
    @param outliers: array of booleans indication the outliers
    @return: the density between 0 and 1
    """
    return outliers.sum() / len(outliers)


def calculate_zscore_densities(data, activities_labels, k=3):
    """
    Calculates the density of outliers for each variable [vector modules] of each activity
    @param data: data no analyse
    @param activities_labels: list with the activities names
    @param k: max distance for the value to be considered non-outlier
    @return: a dataframe with the densities of outliers
    """
    activities = data['activity'].unique()
    arr = np.zeros((3, len(activities)))
    densities = pd.DataFrame(arr, columns=activities_labels,
                             index=['accelerometer_module', 'gyroscope_module', 'magnetometer_module'])
    for variable in ['accelerometer_module', 'gyroscope_module', 'magnetometer_module']:
        for activity in activities:
            activity_data = data[data['activity'] == activity][variable]
            outliers_indexes = calculate_outliers_indexes(activity_data, k)
            density = calculate_density(outliers_indexes)
            densities[activities_labels[activity - 1]][variable] = density * 100

    return densities


def plot_zscore_outliers(data, variable, k_value=None):
    """
    Plots the Z-Score outliers
    For each value of K, calculates the z-score outliers (for each activity) and plots the results
    (non-outliers in blue and outliers in red)
    @param data: data to analyse
    @param variable: variable to analyse
    @param k_value: maximum distance to be considered as non-outliers
    """
    if k_value:
        ks = [k_value]
    else:
        ks = [3, 3.5, 4]
    i = 1
    f = plt.figure(figsize=(20, 15))
    for k in ks:
        ax = f.add_subplot(1, 3, i)
        for activity in data['activity'].unique():
            activity_data = data[data['activity'] == activity][variable]
            outliers_indexes = calculate_outliers_indexes(activity_data, k)
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


def inject_outliers(quantity, data, outliers_indexes, k, z=1):
    """
    Injects outliers in the dataset
    @param quantity: quantity of outliers to add
    @param data: data where the outliers will be injected
    @param outliers_indexes: indexes of outliers
    @param k: constants
    @param z: constant
    @return: the data with the new outliers
    """
    non_outliers = pd.Series(outliers_indexes[outliers_indexes == False].index)
    indexes = non_outliers.sample(quantity, random_state=42)
    data[indexes] = data[indexes].apply(
        lambda x: (data.mean() + k * random.choice([1, -1]) * (data.std() + np.random.uniform(0, z))))
    return data


def add_outliers(percentage, k, data, step=1000):
    """
    Given a certain percentage of outliers, add new outliers into the data until it reaches the desired percentage
    @param percentage: percentage of outliers that the data should have
    @param k: constant
    @param data: data where the outliers will be injected
    @return: data with the new outliers and final density reached
    """
    data = data.copy()
    indexes = calculate_outliers_indexes(data, k)
    density = calculate_density(indexes)
    count = 0
    quantity = int((percentage - density) * len(data))
    print('Quantity:', quantity)
    # add outliers until it reaches the desired percentage of outliers or until it reaches the maximum number of
    # outliers to add
    while density < percentage and count < quantity + step:
        data = inject_outliers(step, data, indexes, k)
        indexes = calculate_outliers_indexes(data, k)
        density = calculate_density(indexes)
        print(density * 100, '%')
        count += step
    return data, density
