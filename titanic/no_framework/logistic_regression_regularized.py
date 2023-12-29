import numpy as np
import pandas as pd

def split_X_and_Y(data):
    x_train = data[:, 2:]
    y_train = data[:, 0:1]

    return x_train, y_train

def calculate_cost(x, y, model):
    cost = 0
    # CALCULATE COST

    return cost

def initialize_model(x_train):
    model = {
        'weights': np.zeros(x_train.shape[1]),
        "bias": 0.0
    }

    return model

def calculate_gradient(x_train, y_train, model):
    dw_dx = np.zeros(x_train.shape[1])
    db_dx = 0
    #CALCULATE GRADIENT

    return {
        'weights': dw_dx,
        'bias': db_dx
    }

def gradient_descent(model, learning_rate, gradient):
    return {
        'weights': model['weights'] - gradient['weights'] * learning_rate,
        'bias': model['bias'] - gradient['bias'] * learning_rate
    }

def predict(model, x_test):
    #CALCULATE PREDICTION
    return np.zeros(x_test.shape()[0])

def z_score_normalization(x):
    #Perform z-score normalization

    return (0, 0, 0)


pd.set_option('display.max_columns', None)
pd.set_option('display.width', 200)
data = pd.read_csv('../data/train.csv', sep=',', header=0)
print(data)
# x_train, y_train = split_X_and_Y(data)
# print(x_train[:5])
#TRAIN THE MODEL AND TEST IT(DON'T FORGET TO REGULARIZE)
