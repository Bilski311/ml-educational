import numpy as np

class Layer:
    def __init__(self, activation, units=0):
        self.units = units
        self.activation = activation
        self.weights = None
        self.biases = None

    def initialize_layer(self, number_of_inputs=None, weights=None, biases=None):
        if number_of_inputs is not None:
            self.weights = np.random.rand(number_of_inputs, self.units)
            self.biases = np.random.rand(self.units)
        if number_of_inputs is None and weights is not None and biases is not None:
            print(weights)
            print(biases)
            self.units = len(biases)
            self.weights = weights
            self.biases = biases

    def compute_output_for_layer(self, X):
        print(f'X: {X}')
        print(f'Weights: {self.weights}')
        print(self.weights.shape)
        product = np.matmul(X, self.weights)
        print(product)
        return product + self.biases