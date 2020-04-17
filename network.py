import numpy as np
from neuron import Neuron

class NeuralNetwork:
    def __init__(self, weights):
        self.weights = weights
        self.a = Neuron(weights[0 : 2], weights[2])
        self.b = Neuron(weights[3 : 5] ,weights[5])
        self.c = Neuron(np.array(weights[6:8]),weights[8])
    
    def update_weights(self):
      self.weights = self.new_weidhts

 
    def feedforward(self, x):
        feed_a = self.a.feedforward(x)
        feed_b = self.b.feedforward(x)
        out_o1 = self.c.feedforward(np.array([feed_a, feed_b]))
        return out_o1
