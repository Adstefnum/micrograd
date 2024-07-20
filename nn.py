import random
from engine import Value

class Neuron:
    """
    A simple Neuron class that holds weights and bias
    """
    def __init__(self, no_of_neurons_from_next_layer):
        self.weights = [Value(random.uniform(-1, 1)) for _ in range(no_of_neurons_from_next_layer)]
        self.bias = Value(0)

    def parameters(self): #returns the weights and bias of the neuron in a list
        return self.weights + [self.bias]

    def __repr__(self):
        return f"Neuron(weights={self.weights}, bias={self.bias})"
    
    def __call__(self, xs): #w1*x1 + w2*x2 + ... + wn*xn + b, then apply tanh
        activation = [w*x for w, x in zip(self.weights, xs)] + [self.bias]
        return Value(sum(activation)).tanh()

class Layer:
    """
    A simple Layer class that holds a list of neurons
    """
    def __init__(self, no_of_neurons_in_cur_layer, no_of_neurons_from_next_layer):
        self.neurons = [Neuron(no_of_neurons_from_next_layer) for _ in range(no_of_neurons_in_cur_layer)] 
    
    def parameters(self):
        return [param for neuron in self.neurons for param in neuron.parameters()]

    def __repr__(self):
        return f"Layer(neurons={self.neurons})"
    
    def __call__(self, xs):
        out = [neuron(xs) for neuron in self.neurons] #just apply the neuron to the input for all neurons in the layer
        return out

class MLP:
    """
    A simple Multi-Layer Perceptron class
    """
    def __init__(self, no_of_neurons_in_first_layer, layers_list):#layers list contains the number of neurons in each layer
        full_layers_list = [no_of_neurons_in_first_layer] + layers_list #[3, 4, 5, 1]
        self.layers = [Layer(full_layers_list[i], full_layers_list[i+1]) for i in range(len(full_layers_list)-1)] #this will create the layers up to n-1
        self.layers += [Layer(full_layers_list[-1], 1)]

    def parameters(self):
        return [param for layer in self.layers for param in layer.parameters()]

    def __repr__(self):
        return f"MLP(layers={self.layers})"
    
    def __call__(self, xs):
        for layer in self.layers: 
            #just apply input to the layer for all layers in the MLP, this will run down to run the neurons in each layer which will return a list 
            #for each layer which we handle outside the MLP which is why we aren't aggregating it inside the MLP
            xs = layer(xs)
        return xs
