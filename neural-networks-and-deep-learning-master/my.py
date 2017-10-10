# -*- coding: utf-8 -*-
import numpy as np

class Network(object):
    """Initialized with a list of number of neurons. [4,3,2] --> network with 4 neurons in 1st layer, 3 in 2nd layer
    and 2 in the output layer"""
    
    def __init__(self,sizes):
        self.numLayers  = len(sizes)
        self.sizes = sizes
        self.biases = [np.random.rand(y,1) for y in sizes[1:]]
        self.weights = [np.random.rand(y,x) for x,y in zip(sizes[:-1],sizes[1:])]