import os
import pandas as pd
import numpy as np

from utils import *

def to_pandas_dataframe(dataset):
    columns = ['device_id', 
        'accelerometer_x', 'accelerometer_y', 'accelerometer_z',
        'gyroscope_x', 'gyroscope_y', 'gyroscope_z',
        'magnetometer_x', 'magnetometer_y', 'magnetometer_z',
        'timestamp', 'activity'
    ]

    data = pd.DataFrame(data=dataset, columns=columns)
    data['device_id'] = data['device_id'].astype('int64')
    data['activity'] = data['activity'].astype('int64')
    return data

def get_user_data(path, user_id, user_path=None, save=False):
    """
    reads user data from all files and appends it to just one

    """
    if user_path is None:
        user_path = f'part{user_id}'
        path = os.path.join(path, user_path)
    else:
        path = os.path.join(path, user_path)
    files_names = os.listdir(path)
    data_array = []
    for i in range(1, len(files_names)):
        if files_names[i].startswith('part'):
            data_path = os.path.join(path, files_names[i])
            aux = np.genfromtxt(data_path, delimiter=',')
            data_array += [aux]
            print(files_names[i], aux.shape)
    data = append_arrays(data_array)
    if save:
        data = to_pandas_dataframe(data)
        save_path = os.join(path, 'users', f'{user_path}.csv')
        data.to_csv(save_path, index=False)
    return data

def get_all_data(data_path):
    path = os.path.join(data_path, 'dataset.csv')
    if os.path.exists(path):
        return pd.read_csv(path)
    users_list = os.listdir(data_path)
    data_array = []
    for user_path in users_list:
        if user_path.startswith('part'):
            data_array += [get_user_data(path=data_path, user_id=user_path, user_path=user_path)]

    data = append_arrays(data_array)
    data = to_pandas_dataframe(data)
    save_path = os.path.join(data_path, 'dataset.csv')
    data.to_csv(save_path, index=False)
    return data

def get_device_data(data, device_id):
    return data[data['device_id'] == device_id]


def append_metrics(dataset):
    data = dataset.copy()
    data['accelerometer_module'] = np.sqrt(data['accelerometer_x']**2 + data['accelerometer_y']**2 + data['accelerometer_z']**2)
    data['gyroscope_module'] = np.sqrt(data['gyroscope_x']**2 + data['gyroscope_y']**2 + data['gyroscope_z']**2)
    data['magnetometer_module'] = np.sqrt(data['magnetometer_x']**2 + data['magnetometer_y']**2 + data['magnetometer_z']**2)
    return data
