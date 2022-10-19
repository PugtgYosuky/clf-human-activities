import pandas as pd

def calculate_density(data, activity=None):
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
    print(f'{activity} Accelerometer density: {accelerometer_density * 100}') 
    print(f'{activity} Gyroscope density: {gyroscope_density * 100}') 
    print(f'{activity} Magnetometer density: {magnetometer_density * 100}') 


def calculate_density_by_activity(data, activities_labels):
    activities = data['activity'].unique()
    print(f'Activities: {len(activities)}')
    for activity in activities:
        activity_data = data[data['activity'] == activity]
        calculate_density(activity_data, activity=f'{activities_labels[activity]}-{activity}')
