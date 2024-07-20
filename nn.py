class Neuron:
    def __init__(self, weights, bias):
        self.weights = weights
        self.bias = bias

class Layer:
    def __init__(self, no_of_neurons):
        self.neurons = [] 

class MLP:
    def __init__(self, no_of_layers):
        self.layers = []
