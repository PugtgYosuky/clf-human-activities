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

def calculate_density(outliers):
    return outliers.sum() / len(outliers)

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
            density = calculate_density(outliers_indexes)
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

def inject_outliers(quantity, data, outliers_indexes, k,  z=1):
    non_outliers = pd.Series(outliers_indexes[outliers_indexes == False].index)
    indexes = non_outliers.sample(quantity, random_state=42)
    data[indexes] = data[indexes].apply(lambda x :  (data.mean() + k * random.choice([1, -1]) * (data.std() + np.random.uniform(0, z))))
    return data

def add_outliers(percentage, k, data):
    data = data.copy()
    indexes = calculate_outliers_indexes(data, k)
    density = calculate_density(indexes)
    step = 1000
    count = 0
    quantity = int((percentage - density) * len(data))
    print('Quantity:', quantity)
    while density < percentage and count < quantity + step:
        data = inject_outliers(step, data, indexes, k)
        indexes = calculate_outliers_indexes(data, k)
        density = calculate_density(indexes)
        print(density)
        count += step
    return data, density



