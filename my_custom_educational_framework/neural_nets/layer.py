import numpy as np

class Layer:
    def __init__(self, units, activation):
        self.units = units
        self.activation = activation

    def initialize_layer(self, number_of_inputs):
        self.weights = np.zeros((number_of_inputs, self.units))
        self.biases = np.zeros((self.units,))

    def compute_output_for_layer(self, X):
        print(X)
        print(self.weights)
        print(self.biases)