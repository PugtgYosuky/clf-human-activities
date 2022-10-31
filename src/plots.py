import matplotlib.pyplot as plt
import pandas as pd
import plotly.graph_objs as go
from plotly import tools
from plotly.subplots import make_subplots
import plotly.offline as py
import plotly.io as pio
from mpl_toolkits.mplot3d import Axes3D

def boxplot_features(data, var=0, title=None):
    """
    :param var: different activities; values:  0 = accelerometer_module, 1 = gyroscope_module, 2=magnetometer_module
    """
    variables = {0: 'accelerometer_module', 1: 'gyroscope_module', 2: 'magnetometer_module'} 
    plt.figure(figsize=(21, 10))
    data.boxplot(column=[variables[var]], by='activity')
    if title is not None:
        plt.title(f'{title} - {variables[var]}')
    else:
        plt.title(variables[var])
    plt.show()

def plot_points_and_outliers(data, outliers, title=None):
    #outliers = calculate_zscore(data, k).to_numpy()
    plt.figure()
    data.plot(style='b*')
    # data.plot(style='b*', markerfacecolor='r', markevery=outliers)
    plt.show() 

def plot_3d(data, labels, title):
    fig = plt.figure(figsize = (15,15))
    ax = fig.add_subplot(111, projection='3d')
    for label in labels.unique():
        ax.scatter(data[labels == label][data.columns[0]],
            data[labels == label][data.columns[1]],
            data[labels== label][data.columns[2]],
            label=f'cluster {label}'
            )
            
    ax.set_xlabel(data.columns[0])
    ax.set_ylabel(data.columns[1])
    ax.set_zlabel(data.columns[2])
    plt.title(title)
    ax.legend()
    plt.show()
