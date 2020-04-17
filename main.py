import numpy as np
from network import NeuralNetwork
from neuron import Neuron

def error_loss(y, t):
  return 0.5*((y - t)**2)

def train(input_x,t, epsilon, gamma, weights, epochs):
  local_weights = weights
  new_weights = local_weights.copy()
  for f in range(0, epochs):
    for k in range(len(input_x)):
      for i in range(0, len(local_weights)):
        l = local_weights.copy()
        l[i] = local_weights[i] + epsilon
        y1 = NeuralNetwork(l).feedforward(input_x[k])
        l[i] = local_weights[i] - epsilon
        y2 = NeuralNetwork(l).feedforward(input_x[k])
        new_weights[i] = local_weights[i] - gamma * (error_loss(y1,t[k]) - error_loss(y2, t[k]))/ 2 * epsilon

      local_weights = new_weights.copy()
  return local_weights
      
if __name__ == '__main__':
    
    weights = np.random.sample(9)
    input_x = np.array([[0,0], [0,1], [1,0], [1,1]])
    input_t = np.array([0, 1, 1, 0])
    gamma = 0.1
    epsilon = 0.2
    epochs = 100000
    new_weights = train(input_x, input_t, epsilon, gamma, weights, epochs)

    xor = NeuralNetwork(new_weights)
    print('input: 0 0 output:', xor.feedforward([0, 0]))
    print('input: 0 1 output:',xor.feedforward([0, 1]))
    print('input: 1 0 output:',xor.feedforward([1, 0]))
    print('input: 1 1 output:',xor.feedforward([1, 1]))
