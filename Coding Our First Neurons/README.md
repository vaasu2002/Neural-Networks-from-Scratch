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
