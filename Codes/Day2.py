!pip install nnfs
import numpy as np
import nnfs
from nnfs.datasets import spiral_data
X, y = spiral_data(samples=100, classes=3)
print(f"X = {X.shape}")
print(f"y = {y.shape}")

class Layer_Dense:
  def __init__(self, n_inputs, n_neurons):
    self.weights = 0.01 * np.random.randn(n_inputs, n_neurons)
    self.biases = np.zeros((1, n_neurons))
  def forward(self, inputs):
    self.output = np.dot(inputs, self.weights) + self.biases
    
class Activation_ReLU:
  # Forward pass
  def forward(self, inputs):
    # Calculate output values from inputs
    self.output = np.maximum(0, inputs)
    
class Activation_Softmax:
  # Forward pass
  def forward(self, inputs):
    # Get unnormalized probabilities
    exp_values = np.exp(inputs - np.max(inputs, axis=1,
    keepdims=True))
    # Normalize them for each sample
    probabilities = exp_values / np.sum(exp_values, axis=1,
    keepdims=True)
    self.output = probabilities

dense1 = Layer_Dense(2, 3)
activation1 = Activation_ReLU()
dense2 = Layer_Dense(3, 3)
activation2 = Activation_Softmax()

dense1.forward(X)
activation1.forward(dense1.output)
dense2.forward(activation1.output)
activation2.forward(dense2.output)
