import pandas as pd
import numpy as np
from scipy.stats import kstest, norm, kruskal
import matplotlib.pyplot as plt


def ks_test(data, variable, activities_labels, threshold=0.05):
    """
    Checks if the activities have a notmal / gaussian distribution using the Kolgomorov-Smirov test
    https://www.statology.org/plot-normal-distribution-python/
    @param data: dataframe to analyse
    @param variable: variable no analyse
    @param activities_labels: list with the name of all activities
    @param threshold: threshold to validate the p-value calculated with the test
    """
    plt.figure()
    plt.title(f'{variable} distribution')
    for activity in data['activity'].unique():
        # divide data by activities
        activity_data = data[data['activity'] == activity][variable]
        statistics, p_value = kstest(activity_data, norm.cdf)
        if p_value < threshold:
            print(
                f'Variable {variable} - activitiy {activities_labels[activity]}: Reject normal distributions | p-value={p_value}')
        else:
            print(
                f'Variable {variable} - activitiy {activities_labels[activity]}: Normal distribution | p-value={p_value}')
        activity_data.hist(alpha=0.4, label=activities_labels[activity])

    plt.legend(bbox_to_anchor=(1.04, 1), loc="upper left")
    plt.show()


def kruskal_wallis_test(data, threshold=0.05):
    """
    Test the data with the Kruskal-Wallis algorithm to check if all variables have the same mean
    @param data: dataframe to analyse
    @param threshold: threshold to validate the p-value calculated with the test
    """
    """ Test the data using the Kruskal-Wallis algorithm"""
    dfs = []  # to store activities data
    for activity in data.activity.unique():
        # divide by activities
        dfs.append(data[data.activity == activity])

    variables = ['accelerometer_module', 'gyroscope_module', 'magnetometer_module']
    for variable in variables:
        # calculates the Kruskal-wallis correlation of all activities of each variable
        print(f'Kruscal - Wallis test - {variable}')
        statistic, p_value = kruskal(
            dfs[0][variable],
            dfs[1][variable],
            dfs[2][variable],
            dfs[3][variable],
            dfs[4][variable],
            dfs[5][variable],
            dfs[6][variable],
            dfs[7][variable],
            dfs[8][variable],
            dfs[9][variable],
            dfs[0][variable],
            dfs[11][variable],
            dfs[12][variable],
            dfs[13][variable],
            dfs[14][variable],
            dfs[15][variable],
        )
        print(f'p-value: {p_value}')
        if p_value < threshold:
            print('Reject!')
        else:
            print('Accept!')
