import numpy as np

from .layer import Layer
class NeuralNetwork:
    def __init__(self, number_of_inputs=None, layers=[], layers_weights=None, layers_biases=None):
        self.number_of_inputs = number_of_inputs
        self.layers = layers
        for layer_number, layer in enumerate(self.layers):
            if layer_number == 0:
                layer.initialize_layer(self.number_of_inputs)
            else:
                layer.initialize_layer(self.layers[layer_number - 1].units)
        if self._is_initialized_from_arrays(layers, layers_weights, layers_biases):
            for layer_number, layer_weights in enumerate(layers_weights):
                layer = Layer(activation='relu')
                layer.initialize_layer(weights=layer_weights, biases=layers_biases[layer_number])
                self.layers.append(layer)

    def _is_initialized_from_arrays(self, layers, layers_weights, layers_biases):
        if layers:
            return False
        if len(layers_weights) != len(layers_biases):
            return False
        for layer_number, layer_weights in enumerate(layers_weights):
            if layer_weights.shape[1] != len(layers_biases[layer_number]):
                return False

        return True

    def predict(self, X):
        input_to_next_layer = X
        for layer_number, layer in enumerate(self.layers):
            input_to_next_layer = layer.compute_output_for_layer(input_to_next_layer)

        return input_to_next_layer



    def __str__(self):
        title = f"Neural network with {len(self.layers)} layers"
        layers = ""
        for index, layer in enumerate(self.layers):
            layers += f"- Layer {index}: \n \tUnits: {layer.units}, Activation: {layer.activation}\n\tWeights: {layer.weights.shape}\n"

        return f"{title}\nNumber of inputs: {self.number_of_inputs}\n{layers}"
