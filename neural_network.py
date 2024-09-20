import numpy as np


#Define network parameters.
input_neurons = 5
hidden_neurons = 10
output_neurons = 3

hidden_layer_count = 3

#Layer objects

class Layer_dense:
    def __init__(self, input_neurons, neurons):
        #init random weight and biases
        self.weight = np.random.rand(input_neurons, output_neurons)
        self.bias = np.random.rand((1, neurons))
    def forward(self, input_layer):
        self.output = np.dot(input_layer, self.weights) + self.bais
        
        
