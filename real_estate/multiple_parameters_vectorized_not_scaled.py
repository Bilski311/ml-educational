import numpy as np

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
    for i, x_i in enumerate(x):
        cost += ((np.dot(model['weights'], x_i) + model['bias']) - y[i]) ** 2
    cost = cost / len(x)

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
    for i, x_i in enumerate(x_train):
        error = np.dot(model['weights'], x_i) - y_train[i]
        db_dx += error
        dw_dx += error * x_i
    dw_dx = dw_dx / len(x_train)
    db_dx = db_dx / len(x_train)

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
    return np.array([np.dot(model['weights'], x_i) + model['bias'] for x_i in x_test])

data = np.genfromtxt('real_estate.csv', delimiter=',', skip_header=1)
x_train, y_train, x_test, y_test = split_train_and_test_data(data)
model = initialize_model(x_train)
learning_rate = 0.0000001
print(f'Cost for initial model: {calculate_cost(x_train, y_train, model)}')
for i in range(1000):
    gradient = calculate_gradient(x_train, y_train, model)
    model = gradient_descent(model, learning_rate, gradient)
    cost = calculate_cost(x_train, y_train, model)
    if (i % 100 == 0 or i == 0):
        print(model)
        print(cost)
prediction = predict(model, x_test)
print(prediction)
print(y_test)
print(calculate_cost(x_test, y_test, model))

