import numpy as np
from linear_regression import LinearRegression

def split_train_test(data, test_ratio):
    data = np.random.permutation(data)
    print("Initial data number of rows: {}".format(len(data)))
    print("Initial data shape: {}".format(data.shape))

    test_size = int(len(data) * test_ratio)
    print('Test size: {}'.format(test_size))
    train_data = data[test_size:, :]
    print('Train data shape: {}'.format(train_data.shape))
    test_data = data[:test_size, :]

    X_train = train_data[:, 1:-1]
    print('X_train shape: {}'.format(X_train.shape))
    X_test = test_data[:, 1:-1]
    Y_train = train_data[:, -1]
    Y_test = test_data[:, -1]

    return X_train, X_test, Y_train, Y_test


if __name__ == "__main__":
    data = np.genfromtxt('real_estate.csv', delimiter=',', skip_header=1)
    X_train, X_test, Y_train, Y_test = split_train_test(data, 0.2)
    model = LinearRegression()
    model.fit(X_train, Y_train)
    print(model.predict(X_test[:10]))
    print(Y_test[:10])
    
