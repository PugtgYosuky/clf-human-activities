import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def calculate_alphas(data, p):
    rows = len(data) - p
    cols = p + 1
    x = np.ones((rows, cols))
    for i in range(p):
        x[:, p - i] = data[i : len(data)-p + i]
    x_plus = np.linalg.pinv(x)
    del x
    a = x_plus.dot(data[p:])
    del x_plus
    return a

def predict(data, p, a=None):
    """ receives only the previous points and returns the predicted value based on the previous p points"""
    size = len(data)
    if a is None:
        a = calculate_alphas(data, p)
    prev_values = np.ones(p+1)
    prev_values[1:] = data[:size-p-1:-1]
    y_pred = prev_values.dot(a)
    del prev_values
    return y_pred

def rmse (y_real, y_pred):
    """https://www.askpython.com/python/examples/rmse-root-mean-square-error"""
    mse = np.square(y_real - y_pred).mean()
    return np.sqrt(mse)

def define_best_p_value(data, pred_index, min_p_value=10, max_step=100, step=100):
    """Leave one-out technique to determine the best number of previous values to predict a value"""
    p_possible_values = range(min_p_value, max_step, step)
    real_value = data[pred_index]
    errors = []
    for p in p_possible_values:
        d = data[:pred_index]
        y_pred = predict(d, p)
        # print('P: ', p)
        # print('Real: ', real_value)
        # print('Pred: ', y_pred)
        errors.append(rmse(real_value, y_pred))
    plt.figure()
    plt.plot(p_possible_values, errors, '*')

    # for i in range(len(errors)):
    #     plt.annotate(errors[i], (p_possible_values[i], errors[i] + 0.2))

    plt.xlabel('Previous p values')
    plt.ylabel('RMSE')
    plt.title(f'Errors obtained with RMSE using the Leave-one-out technique to determine the best p value')
    plt.show()
    
    return p_possible_values[np.array(errors).argmin()], errors

def plot_results(data, predicted, outliers_indexes):
    #x_axis = np.arange(len(data))
    plt.scatter(outliers_indexes, data[outliers_indexes], s=4, label='real values')
    plt.scatter(outliers_indexes, predicted, color='r', s=4, label='predicted values')
    plt.legend()
    plt.xlabel('Index')
    plt.ylabel('Value')
    plt.title('Comparison between the real values and the predicted values')
    plt.show()

def plot_original(data, title):
    plt.figure()
    plt.scatter(range(len(data)), data, s=4)
    plt.xlabel('Index')
    plt.ylabel('Value')
    plt.title(title)
    plt.show()

def predict_in_the_middle(data_all, p, split_index, a_pred, a_post):
    # if split_index == len(data_all)-1:
    #     # predict the last index of the data
    #     return predict(data_all[:-1], p, a_prev)
    # if split_index == 0:
    #     # predict the first index of the data
    #     return predict(data_all[1:][::-1], a_post)
    # if split_index < p + 1:
    #     return predict(data_all[:split_index], split_index-1)
    # if len(data_all) - split_index - 1< p + 1:
    #     return predict(data_all[:split_index], p)
    prev_data = data_all[:split_index]
    post_data = data_all[split_index+1:]

    prev_pred = predict(prev_data, p, a_pred)
    post_pred = predict(post_data[::-1], p, a_post)

    return (prev_pred + post_pred) / 2 # mean of the two predictions

