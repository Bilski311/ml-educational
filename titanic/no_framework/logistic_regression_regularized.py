import numpy as np
import pandas as pd
import math
import matplotlib.pyplot as plt
from my_custom_educational_framework import z_score_normalization

def split_X_and_Y(data):
    X_train = data.iloc[:, 2:]
    Y_train = data.iloc[:, 1]

    return X_train, Y_train.to_numpy()


def prepare_X(X_train):
    X_train = X_train.drop(columns=['Name', 'Ticket'])
    gender_dict = {
        'male': 1.0,
        'female': 0.0
    }
    X_train['Pclass'] = X_train['Pclass'].apply(convert_pclass)
    X_train['Sex'] = X_train['Sex'].apply(convert_sex)
    X_train['Age'].fillna(X_train['Age'].mean(), inplace=True)
    X_train['SibSp'].fillna(X_train['SibSp'].mean(), inplace=True)
    X_train['Parch'].fillna(X_train['Parch'].mean(), inplace=True)
    X_train['Fare'].fillna(X_train['Fare'].mean(), inplace=True)
    X_train['Cabin'] = X_train['Cabin'].apply(convert_cabin)
    X_train['Embarked'] = X_train['Embarked'].apply(convert_embarked)

    return X_train.to_numpy()


def convert_pclass(pclass):
    if pd.isna(pclass):
        return 2.0
    else:
        return pclass


def convert_sex(sex):
    if pd.isna(sex):
        return 0.5
    elif sex == 'male':
        return 1.0
    elif sex == 'female':
        return 0.0


def convert_cabin(cabin_number):
    if pd.isna(cabin_number):
        return 0.0
    elif cabin_number.startswith('A'):
        return 1.0
    elif cabin_number.startswith('B'):
        return 2.0
    elif cabin_number.startswith('C'):
        return 3.0
    else:
        return 0.0


def convert_embarked(embarked):
    if pd.isna(embarked):
        return 0.0
    elif embarked == 'S':
        return 1.0
    elif embarked == 'C':
        return 2.0
    elif embarked == 'Q':
        return 3.0


def initialize_model(x_train):
    model = {
        'weights': np.zeros(x_train.shape[1]),
        "bias": 0.0
    }

    return model

def calculate_cost(X, Y, model, lambda_):
    cost = 0
    sum_of_squared_weights = (model['weights'] ** 2).sum()
    regularization_cost = lambda_ / (2 * X.shape[0]) * sum_of_squared_weights
    # CALCULATE COST
    for i, X_i in enumerate(X):
        z = np.dot(X_i, model['weights']) + model['bias']
        prediction = sigmoid(z)
        cost += calculate_loss(prediction, Y[i])

    cost = cost / len(X)

    return cost + regularization_cost


def sigmoid(z):
    z = np.array(z)

    return 1 / (1 + np.exp(-z))


def calculate_loss(prediction, y):
    return -y * (math.log(prediction + 0.0000000001)) - (1 - y) * (math.log(1 - prediction + 0.000000001))


def calculate_gradient(X_train, Y_train, model, lambda_):
    dw_dx = np.zeros(X_train.shape[1])
    db_dx = 0
    for i, x_i in enumerate(X_train):
        z = np.dot(x_i, model['weights']) + model['bias']
        prediction = sigmoid(z)
        error = prediction - Y_train[i]
        dw_dx += error * x_i
        dw_dx += (lambda_ / len(X_train)) * model['weights']
        db_dx += error

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


pd.set_option('display.max_columns', None)
pd.set_option('display.width', 200)
data = pd.read_csv('../data/train.csv', sep=',', header=0)
print(data)
X_train, Y_train = split_X_and_Y(data)
X_train = prepare_X(X_train)
X_train, means, standard_deviations = z_score_normalization(X_train)
print(X_train)
print(X_train.shape)
print(Y_train.shape)

#TRAIN THE MODEL
model = initialize_model(X_train)
print(model)
learning_rate = 0.003
lambda_ = 1
initial_cost = calculate_cost(X_train, Y_train, model, lambda_)
print(f'Initial cost: {initial_cost}')
history = [initial_cost]
for i in range(1000):
    gradient = calculate_gradient(X_train, Y_train, model, lambda_)
    model = gradient_descent(model, learning_rate, gradient)
    if (i % 10 == 0):
        cost = calculate_cost(X_train, Y_train, model, lambda_)
        print(f'Cost at iteration {i}: {cost}')
        history.append(cost)
plt.plot(history)
plt.title('Cost')
plt.xlabel('Iteration')
plt.ylabel('Cost')
plt.show()
print(f'Final cost: {history[-1]}')
