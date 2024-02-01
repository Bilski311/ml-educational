import numpy as np
class NeuralNetwork:
    def __init__(self, number_of_inputs, layers):
        self.number_of_inputs = number_of_inputs
        self.layers = layers
        for layer_number, layer in enumerate(self.layers):
            if layer_number == 0:
                layer.initialize_layer(self.number_of_inputs)
            else:
                layer.initialize_layer(self.layers[layer_number - 1].units)

    def predict(self, X):
        print(X)
        input_to_next_layer = X
        for layer_number, layer in enumerate(self.layers):
            print(f'Input to layer: {input_to_next_layer}\n of shape {input_to_next_layer.shape}')
            print(f'Layer: {layer.weights}\n of shape {layer.weights.shape}')
            input_to_next_layer = np.matmul(input_to_next_layer, layer.weights)
            print(input_to_next_layer)



    def __str__(self):
        title = f"Neural network with {len(self.layers)} layers"
        layers = ""
        for index, layer in enumerate(self.layers):
            layers += f"- Layer {index}: \n \tUnits: {layer.units}, Activation: {layer.activation}\n\tWeights: {layer.weights.shape}\n"

        return f"{title}\nNumber of inputs: {self.number_of_inputs}\n{layers}"