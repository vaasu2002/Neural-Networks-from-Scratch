# !pip install nnfs
import numpy as np 
import nnfs
from nnfs.datasets import spiral_data

class Layer_Dense:
    def __init__(self, n_inputs, n_neurons): # number of inputs = number of neurons in prevous layer , n_neurons = no. of neurons in layer we making
        self.weights = 0.01 * np.random.randn(n_inputs, n_neurons)  # shape -> (n_inputs , n_neurons)     ## random init of weight
        self.biases = np.zeros((1, n_neurons))                      # shape -> (1 , n_neurons)            ## random init of bias
    def forward(self, inputs):
        self.output = np.dot(inputs, self.weights) + self.biases    # shape ->  (number of samples , number of neurons in current layer)
        
        
X, y = spiral_data(samples=100, classes=3) # X -> (3000, 2)

# layer 1
dense1 = Layer_Dense(X.shape[1], 3) # passing (2,3) number of featues/ input / number of neurons in prevous laye
dense1.forward(X) # performing forward propogation (dot product of input and weights)

# layer 2 
dense2 = Layer_Dense(dense1.output.shape[1], 1)
dense2.forward(dense1.output)
