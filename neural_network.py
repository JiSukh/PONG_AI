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

targets = np.array([0,1,2])

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
        
class ActivationStableSoftMax:
    """Soft Max activation function for entire neuron layer. Function is stable, preventing under/over flow."""
    def forward(self, input_neurons):
        """Forward pass for Stable Soft Max activation"""
        z = input_neurons - np.max(input_neurons, axis=-1, keepdims=True)
        numerator = np.exp(z)
        denominator = np.sum(numerator, axis=-1, keepdims=True)
        self.output = numerator/denominator
        
        
class CategoricalCrossEntropy:
    """Calculate the loss of function using Categorical Cross Entropy"""
    def calculate_loss(self, ypred, ytrue):
        batch_loss = np.mean(self.forward(ypred, ytrue))
        return batch_loss
        
        
    def forward(self, ypred, ytrue):
        clip_ypred = np.clip(ypred, 1e-15, 1-1e-15) #prevent log(0)
        #handle multiple types of targets (one-hots and class target)
        if len(ytrue.shape) == 1:
            confidences = clip_ypred[range(len(ypred)), ytrue] #class target
        else:
            confidences = np.sum(clip_ypred * ytrue, axis=1) #One hot
            
        return -np.log(confidences)

        

layer1 = LayerDense(4,3)
layer2 = LayerDense(3,2)

layer1.forward(input_layer)

activate = ActivationStableSoftMax()
activate.forward(layer1.output)
loss = CategoricalCrossEntropy()



print(layer1.output)
print(activate.output)
print(loss.calculate_loss(activate.output, targets))
