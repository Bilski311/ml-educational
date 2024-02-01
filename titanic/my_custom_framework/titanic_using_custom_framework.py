import numpy as np
import pandas as pd
import math
import matplotlib.pyplot as plt
from my_custom_educational_framework import z_score_normalization
from my_custom_educational_framework import NeuralNetwork, Layer

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
neural_network = NeuralNetwork(X_train.shape[1], [
    Layer(units=64, activation='relu'),
    Layer(units=64, activation='relu'),
    Layer(units=1, activation='sigmoid')
])
print(neural_network)
neural_network.predict(X_train[0:2])