import numpy as np

class Layer:
    def __init__(self, units, activation):
        self.units = units
        self.activation = activation

    def initialize_layer(self, number_of_inputs):
        self.parameters = np.zeros((self.units, number_of_inputs))