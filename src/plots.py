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

def plot_kmeans_clusters(data, labels, k, variable='variable'):
    # scatter = go.Scatter3d( x=data['accelerometer_x'], 
    #                         y=data['accelerometer_y'],
    #                         z=data['accelerometer_z'],
    #                         mode='markers',
    #                         marker=dict(color = labels, size= 2, 
    #                         line=dict(color= 'black',width = 10))
    # )
    # layout = go.Layout(scene={  'xaxis' : {'title': f'{variable}_x'},
    #                             'yaxis' : {'title' : f'{variable}_y'},
    #                             'zaxis' : {'title' : f'{variable}_z'}       
    # }, margin={'l': 0, 'r':0}, height = 800,width = 800)

    

    # fig = go.Figure(data=[scatter], layout=layout)
    # fig.show()
    
    
    fig = plt.figure(figsize = (15,15))
    ax = fig.add_subplot(111, projection='3d')
    for label in labels.unique():
        ax.scatter(data[labels == label][f'{variable}_x'],
            data[labels == label][f'{variable}_y'],
            data[labels== label][f'{variable}_z'],
            label=f'cluster {label}'
            )

    ax.set_xlabel(f'{variable}_x')
    ax.set_ylabel(f'{variable}_y')
    ax.set_zlabel(f'{variable}_z')
    plt.title(f'Clusters KMeans: {k} clusters - {variable} vector')
    ax.legend()
    plt.show()