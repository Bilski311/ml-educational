class NeuralNetwork:
    def __init__(self, number_of_inputs, layers):
        self.number_of_inputs = number_of_inputs
        self.layers = layers
        for layer_number, layer in enumerate(self.layers):
            if layer_number == 0:
                layer.initialize_layer(self.number_of_inputs)
            else:
                layer.initialize_layer(self.layers[layer_number - 1].units)

    def __str__(self):
        title = f"Neural network with {len(self.layers)} layers"
        layers = ""
        for index, layer in enumerate(self.layers):
            layers += f"- Layer {index}: \n \tUnits: {layer.units}, Activation: {layer.activation}\n\tParameters: {layer.parameters.shape}\n"

        return f"{title}\nNumber of inputs: {self.number_of_inputs}\n{layers}"