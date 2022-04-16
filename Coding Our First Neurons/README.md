![Three Neural](https://dagshub.com/vaasu2002/DeepLearning/raw/cef9ce8090ff467d16406cc28e1a69fe9a8c1eaa/Three%20Neural.PNG)
```ruby
inputs  = [1,2,3,2.5]
weight1 = [3.1,2.1,1.7,1.0]
weight2 = [0.6,0.4,0.7,0.2]
weight3 = [0.26,0.56,0.89,0.23]
bias1 = 2
bias2 = 3
bias3 = 0.5

output =  [inputs[0]*weight1[0] + inputs[1]*weight1[1] + inputs[2]*weight1[2] + inputs[3]*weight1[3] + bias1,
           inputs[0]*weight2[0] + inputs[1]*weight2[1] + inputs[2]*weight2[2] + inputs[3]*weight2[3] + bias2,
           inputs[0]*weight3[0] + inputs[1]*weight3[1] + inputs[2]*weight3[2] + inputs[3]*weight3[3] + bias3]
print(output)
```
----------------------
```ruby
import numpy as np

# 3 nuerons and 4 inputs for each neuron 

# 3 x 4     
weights = [[3.1,2.1,1.7,1.0],     # neuron 1
           [0.6,0.4,0.7,0.2],     # neuron 2
           [0.26,0.56,0.89,0.23]] # neuron 3 

# 3 x 4
X = [[1, 2, 3, 2.5],          # sample 1
     [2.0, 5.0, -1.0, 2.0],   # sample 2
     [-1.5, 2.7, 3.3, -0.8]]  # sample 3
 
bias = [10,20,30]

layer_output = np.dot(X , np.array(weights).T) + bias
print(layer_output)
```


----------------------------
```ruby
import numpy as np 

np.random.seed(0)
# 3 x 4 (number of samples x number of features(number of inputs))
X = [[1, 2, 3, 2.5],
     [2.0, 5.0, -1.0, 2.0],
     [-1.5, 2.7, 3.3, -0.8]]
     
class Layer_Dense:
    def __init__(self, n_inputs, n_neurons):
        self.weights = 0.10 * np.random.randn(n_inputs, n_neurons) # random weight init  (n_inputs,n_neurons)
        self.biases = np.zeros((1, n_neurons))                     # (1,n_neurons)
    def forward(self, inputs):
        self.output = np.dot(inputs, self.weights) + self.biases   # (number of inputs x number of neurons)

layer1 = Layer_Dense(4,5)

layer1.forward(X)

layer1.output.shape 
```
