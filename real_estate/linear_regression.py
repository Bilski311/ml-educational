import numpy as np
from random import randint

class LinearRegression:
    def __init__(self):
        self.weights = None
        self.bias = None

    def fit(self, X, y, learning_rate = 0.0000003, num_of_iterations=100000
            ):
        self.weights = np.zeros(X.shape[1])
        self.bias = 0
        for _ in range(num_of_iterations):
            y_hat = self.predict(X)
            gradient_w_from_la, gradient_b_from_la = self.calculate_gradient_using_linear_algebra(X, y_hat, y)
            self.weights = self.weights - learning_rate * gradient_w_from_la
            self.bias = self.bias - learning_rate * gradient_b_from_la
        print('Cost: {}'.format(self.calculate_cost_function(y_hat, y)))


    def predict(self, X):
        return np.dot(X, self.weights) + self.bias
    
    def calculate_cost_function(self, y_hat, y):
        differences = y - y_hat
        squared_differences = differences ** 2
        sum_of_squared_diffreneces = np.sum(squared_differences)
        double_sample_size = 2 * len(y)

        return sum_of_squared_diffreneces / double_sample_size
    
    def calculate_gradient_from_loops(self, X, y_hat, y):
        derivates_for_weights = np.zeros(X.shape[1])
        derivative_for_bias = 0.
        for i, example in enumerate(X):
            error = y_hat[i] - y[i]
            for j, feature in enumerate(example):
                derivates_for_weights[j] = derivates_for_weights[j] + error * feature
            derivative_for_bias = derivative_for_bias + error
        
        derivates_for_weights = derivates_for_weights / X.shape[0]
        derivative_for_bias = derivative_for_bias / X.shape[0]
        return derivates_for_weights, derivative_for_bias
    
    def calculate_gradient_using_linear_algebra(self, X, y_hat, y):
        errors = y_hat - y
        return np.dot(errors, X) / len(errors), np.mean(errors)
