import numpy as np
import matplotlib.pyplot as plt

def split_train_and_test_data(data):
    np.random.shuffle(data)
    split_index = int(0.8 * len(data))
    train_data = data[:split_index]
    test_data = data[split_index:]
    x_train = train_data[:, 1:-1]
    y_train = train_data[:, -1]
    x_test = test_data[:, 1:-1]
    y_test = test_data[:, -1]

    return x_train, y_train, x_test, y_test

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


data = np.genfromtxt('insert_filename.csv', delimiter=',', skip_header=1)
#TRAIN THE MODEL AND TEST IT(DON'T FORGET TO REGULARIZE)
