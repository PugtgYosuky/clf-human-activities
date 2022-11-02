import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def calculate_density_interquantile(data, density_col):
    first_quantile_accelerometer = data['accelerometer_module'].quantile(0.25)
    third_quantile_accelerometer = data['accelerometer_module'].quantile(0.75)
    first_quantile_gyroscope = data['gyroscope_module'].quantile(0.25)
    third_quantile_gyroscope = data['gyroscope_module'].quantile(0.75)
    first_quantile_magnetometer = data['magnetometer_module'].quantile(0.25)
    third_quantile_magnetometer = data['magnetometer_module'].quantile(0.75)

    accelerometer_range = third_quantile_accelerometer - first_quantile_accelerometer
    gyroscope_range = third_quantile_gyroscope - first_quantile_gyroscope
    magnetometer_range = third_quantile_magnetometer - first_quantile_magnetometer


    accelerometer_outliers = data[data['accelerometer_module'] < (first_quantile_accelerometer - 1.5*accelerometer_range)]['accelerometer_module'].count() + data[data['accelerometer_module'] > (third_quantile_accelerometer + 1.5*accelerometer_range)]['accelerometer_module'].count()
    gyroscope_outliers = data[data['gyroscope_module'] < (first_quantile_gyroscope - 1.5*gyroscope_range)]['gyroscope_module'].count() + data[data['gyroscope_module'] > (third_quantile_gyroscope + 1.5*gyroscope_range)]['gyroscope_module'].count()
    magnetometer_outliers = data[data['magnetometer_module'] < (first_quantile_magnetometer - 1.5*magnetometer_range)]['magnetometer_module'].count() + data[data['magnetometer_module'] > (third_quantile_magnetometer + 1.5* magnetometer_range)]['magnetometer_module'].count()

    data_size = data['magnetometer_module'].count()

    accelerometer_density = accelerometer_outliers / data_size
    gyroscope_density = gyroscope_outliers / data_size
    magnetometer_density = magnetometer_outliers / data_size

    density_col['accelerometer_module'] = accelerometer_density * 100
    density_col['gyroscope_module'] = gyroscope_density * 100
    density_col['magnetometer_module'] = magnetometer_density * 100

def calculate_density_by_activity(data, activities_labels):
    activities = data['activity'].unique()
    arr = np.zeros((3, len(activities)))
    densities = pd.DataFrame(arr, columns=activities_labels, index=['accelerometer_module', 'gyroscope_module', 'magnetometer_module'])
    for activity in activities:
        #activity_data = data[data['activity'] == activity]
        calculate_density_interquantile( data[data['activity'] == activity], densities[activities_labels[activity -1]])

    return densities

def plot_densities(densities):
    plt.figure()
    ax = densities.loc['accelerometer_module'].plot(kind='barh', xlabel='Percentage of outliers', ylabel='Activity')
    ax.set_title('Density of outliers of each activity - accelerometer sensor')
    plt.show()
    plt.figure()
    ax = densities.loc['gyroscope_module'].plot(kind='barh', xlabel='Percentage of outliers', ylabel='Activity')
    ax.set_title('Density of outliers of each activity - gyroscope sensor')
    plt.show()
    plt.figure()
    ax = densities.loc['magnetometer_module'].plot(kind='barh',xlabel='Percentage of outliers', ylabel='Activity')
    ax.set_title('Density of outliers of each activity - magnetometer sensor')
    plt.show()

def calculate_common_outliers(outliers_a, outliers_b):
    common = outliers_a & outliers_b
    return np.sum(common) / np.maximum(np.sum(outliers_a), np.sum(outliers_b))
