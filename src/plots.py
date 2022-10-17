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
    plt.title(variables[var])
    plt.show()

def plot_points_and_outliers(data, outliers, title=None):
    #outliers = calculate_zscore(data, k).to_numpy()
    plt.figure()
    data.plot(style='b*', markerfacecolor='r', markevery=outliers)
    plt.show() 

def plot_kmeans_clusters(data, labels, variable='variable'):
    scatter = go.Scatter3d( x=data['accelerometer_module'],
                            y=data['gyroscope_module'],
                            z=data['magnetometer_module'],
                            mode='markers',
                            marker=dict(color = labels, size= 2, 
                            line=dict(color= 'black',width = 10))
    )
    layout = go.Layout(scene={  'xaxis' : {'title': f'{variable}_x'},
                                'yaxis' : {'title' : f'{variable}_y'},
                                'zaxis' : {'title' : f'{variable}_z'}       
    }, margin={'l': 0, 'r':0}, height = 800,width = 800)

    

    fig = go.Figure(data=[scatter], layout=layout)
    fig.show()
    
    """
        for cluster in range(clusters):
        ax.scatter(data[kmeans.labels_ == cluster]['accelerometer_module'],
            data[kmeans.labels_ == cluster]['gyroscope_module'],
            data[kmeans.labels_ == cluster]['magnetometer_module'],
            label=f'cluster {cluster}'
            )

        ax.set_xlabel('accelerometer_module')
        ax.set_ylabel('gyroscope_module')
        ax.set_zlabel('magnetometer_module')
        ax.legend()
        plt.show()
    """