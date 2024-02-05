import unittest
import numpy as np
from numpy.testing import assert_array_equal
from neural_nets import NeuralNetwork

class NeuralNetworkTest(unittest.TestCase):
    def test_neural_network(self):
        layer_1_weights = np.array([[1, 2, 3], [2, 2, 2], [1, 1, 1], [3, 3, 3]])
        layer_1_biases = np.array([1, 2, 3])
        layer_2_weights = np.array([[2], [2], [2]])
        layer_2_biases = np.array([1])
        X = np.array([[3, 2, 1, 0], [1, 2, 3, 4]])
        neural_network = NeuralNetwork(layers_weights=[layer_1_weights, layer_2_weights], layers_biases=[layer_1_biases, layer_2_biases])
        forward_prop = neural_network.predict(X)
        #expected_output_first_layer = np.array([[9, 13, 17], [21, 23, 25]])
        expected_output_second_layer = np.array([[18 + 26 + 34 + 1], [42 + 46 + 50 + 1]])
        assert_array_equal(expected_output_second_layer, forward_prop)