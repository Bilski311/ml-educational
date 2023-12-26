import numpy as np

def split_train_and_test_data(data):
    np.random.shuffle(data)
    split_index = int(0.8 * len(data))
    train_data = data[:split_index]
    test_data = data[split_index:]
    x_train = train_data[:, 0]
    y_train = train_data[:, 1]
    x_test = test_data[:, 0]
    y_test = test_data[:, 1]

    return x_train, y_train, x_test, y_test

def calculate_cost(x, y, params):
    cost = 0
    print(params)
    for i, x_i in enumerate(x):
        cost += (((params[0] * x_i) + params[1]) - y[i]) ** 2
    cost = cost / len(x)

    return cost

def initialize_params(x_train):
    params = [[np.random.rand(), np.random.rand()] for _ in range(x_train.ndim)]

    return params[0]

def calculate_gradient(x_train, y_train, params):
    dw_dx = 0
    db_dx = 0
    for i, x in enumerate(x_train):
        dw_dx += (x * params[0] + params[1] -  y_train[i]) * x
        db_dx += (x * params[0] + params[1] - y_train[i])
    dw_dx = dw_dx / len(x_train)
    db_dx = db_dx / len(y_train)

    return np.array([dw_dx, db_dx])

def gradient_descent(params, learning_rate, gradient):
    return params - gradient * learning_rate

def predict(params, x_test):
    return np.array([x_i * params[0] + params[1] for x_i in x_test])

data = np.genfromtxt('real_estate/simple_real_estate.csv', delimiter=',', skip_header=1)
x_train, y_train, x_test, y_test = split_train_and_test_data(data)
params = initialize_params(x_train)
learning_rate = 0.0001
for i in range(10000):
    cost = calculate_cost(x_train, y_train, params)
    print(cost)
    gradient = calculate_gradient(x_train, y_train, params)
    params = gradient_descent(params, learning_rate, gradient)
    cost = calculate_cost(x_train, y_train, params)
    print(params)
    print(cost)
prediction = predict(params, x_test)
print(prediction)
print(y_test)
print(calculate_cost(x_test, y_test, params))

