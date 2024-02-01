import unittest
import numpy as np
from neural_nets import NeuralNetwork

class NeuralNetworkTest(unittest.TestCase):
    def test_neural_network(self):
        layer_1_weights = np.array([[[1, 2, 3], [2, 2, 2], [1, 1, 1], [3, 3, 3]]])
        layer_1_biases = np.array([[1, 2, 3]])
        neural_network = NeuralNetwork(layers_weights=layer_1_weights, layers_biases=layer_1_biases)
        print(neural_network)