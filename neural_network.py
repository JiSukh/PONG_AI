import numpy as np


#Define network parameters.
input_neurons_number = 5
hidden_neurons_number = 10
output_neurons_number = 3

hidden_layer_count = 3

#Layer objects


#3,4 input layer

input_layer = [[2,3,4,5],
               [.2,.3,.4,.5],
               [-2,-3,-4,-5]]

class LayerDense:
    """Dense layer object for neural network"""
    def __init__(self, n_input, n_neurons):
        """Create dense layer"""
        self.weight = np.random.rand(n_input, n_neurons)
        self.bias = np.random.rand(1,n_neurons)
        self.output = np.zeros((1,n_neurons))
    def forward(self, input_neurons):
        """Forward pass for dense Layer, using input neurons"""
        #claculate output of forward pass from dense layer.
        self.output = np.dot(input_neurons, self.weight) + self.bias
        
#Activation function

class ActivationReLU:
    """ReLU activation function for entire neuron layer"""
    def forward(self, input_neurons):
        """Forward pass for ReLU activation"""
        self.output = input_neurons * (input_neurons > 0)

layer1 = LayerDense(4,3)
layer2 = LayerDense(3,2)

layer1.forward(input_layer)

activate = ActivationReLU()
activate.forward(layer1.output)



print(layer1.output)
print(activate.output)
