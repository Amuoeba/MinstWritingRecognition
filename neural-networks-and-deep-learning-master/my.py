# -*- coding: utf-8 -*-
import numpy as np
import random

class Network(object):
    """Initialized with a list of number of neurons. [4,3,2] --> network with 4 neurons in 1st layer, 3 in 2nd layer
    and 2 in the output layer"""
    
    def __init__(self,sizes):
        self.numLayers  = len(sizes)
        self.sizes = sizes
        self.biases = [np.random.rand(y,1) for y in sizes[1:]]
        self.weights = [np.random.rand(y,x) for x,y in zip(sizes[:-1],sizes[1:])]
        
    def feedForward(self,a):
        """Return the output of the network if "a" is the input."""
        for b, w in zip(self.biases, self.weights):
            a = sigmoid(np.dot(w,a)+b)
        return a
    
    def SGD (self,train_data,epochs,mini_batch_size,eta,test_data=None):
        """- The training data is a list of tuples (x,y) representing the training inputs and coresponding outputs
           - eta us the learning rate """
        if test_data:
            n_test = len(test_data)
        n = len(train_data)
        
        for epoch in range(epochs):
            random.shuffle(train_data)
            mini_batches = [train_data[k:k+mini_batch_size] for k in range(0,n,mini_batch_size)]
            for mini_batch in mini_batches:
                self.update_mini_batch(mini_batch,eta)
            if test_data:
                print ("epoch {0}: {1} / {2}".format(epoch,self.evaluate(test_data),n_test))
            else:
                print("Epoch {0} complete".format(epoch))
    
    def update_mini_batch(self,mini_batch,eta):
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        
        
        for x,y in mini_batch:
            delta_nabla_b, delta_nabla_w =self.backprop(x,y)
            nabla_b = [nb+dnb for nb, dnb in zip(nabla_b,delta_nabla_b)]
            nabla_w = [nw+dnw for nw, dnw in zip(nabla_w,delta_nabla_w)]
        
#        self.weights = [w-(eta/len(mini_batch))*nw for w,nw in zip(weights,nabla_w)]
        
    
def sigmoid(z):
    return 1.0/(1.0+np.exp(-z))

