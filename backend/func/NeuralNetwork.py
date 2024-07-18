import pandas as pd
import math
import numpy as np
import time

from keras.datasets import mnist
from decimal import Decimal
np.set_printoptions(threshold=np.inf)


class Input():
    def __init__(self):
        self.input_values = np.empty((0,), dtype= np.float64)
        self.randomness_coefficient = None
        self.input_shape_history = None
        self.input_data_shape = None
        self.input_values = 'cicaj ma'####
        
    @staticmethod
    def rescaling_values(data,  SCALING_FACTOR=0):
        return data / np.float64(SCALING_FACTOR)
        
    def set_input_values(self, data, SCALING_FACTOR = 0):
        for feature in range(data.shape[1]): # for column in range(X_train.shape[1]):  
            if SCALING_FACTOR != 0 :    
                value = self.rescaling_values(data[feature][0], SCALING_FACTOR)
            else:
                value = data[feature][0]
            self.input_values = np.append(self.input_values, value)  

class Output():
    def __init__(self):
        self.output = np.empty((0,), dtype = np.float64)
        self.output_data_shape = None
        self.output_shape_history = None
    def set_ouput_shape(self, shape):
        pass

class ExceptionList(Exception):
    pass

class WeightsFunctions():
    def __init__(self):
        self.randomness_coefficient = None
    
    def XavierInitialization(self):
        if self.input_shape_history != None:
            return math.sqrt(6/self.input_data_shape)
        else: return "Error"
    
    def HeInitialization(self):
        if self.input_shape_history != None and self.output_shape_history != None:
            return math.sqrt(2/(self.input_data_shape + self.output_data_shape))
        else: return "Error"
###MAKE IT ADAPT WITH LAYERS

class Layer(Input, WeightsFunctions):
    def __init__(self):
        super().__init__()
        self.weights = np.empty((0,), dtype=np.float64)
        self.biases = np.empty((0,), dtype=np.float64)
        self.layer = None
        
    def settings(self, *args, **kwargs):
        if kwargs:
            directory_looker(kwargs)
        def directory_looker(dictionary:dict):
            for index, obj in enumerate(dictionary):
                print(index, obj)
    
    def random_weights(self, input_values):
        return np.random.uniform(-(self.randomness_coefficient), self.randomness_coefficient, (input_values)).astype(np.float64) 
        
    def random_bias(self):
        return np.array(np.random.uniform(-(self.randomness_coefficient), self.randomness_coefficient)).astype(np.float64)
        
    def create_neuron(cls):
        #if cls.settings[0] == 'XavierInitialization':
        #neuron = tuple([cls.random_weights(cls.input_values, cls.XavierInitialization(cls.input_shape_history, )), cls.random_bias(cls.XavierInitialization(cls.input_shape_history, ))])
        #if cls.settings[0] == 'HeInitialization':
        return tuple([cls.random_weights(cls.input_data_shape), cls.random_bias()])
    
    @staticmethod
    def create_layer(cls, hidden_layer_neurons:int, input_shape:int = None):
        if cls.input_shape_history == None:
            cls.input_data_shape = input_shape
            cls.input_shape_history = True
            cls.randomness_coefficient = cls.XavierInitialization()
        layer = []
        for neuron in range(hidden_layer_neurons):
            if cls.neuron_count == 0: 
                layer = [[cls.create_neuron()]]
            else:
                layer.append(cls.create_neuron())
            cls.neuron_count += 1
        cls.neuron_count = 0
        return layer  
    
class CostFunctions(DataPrep):
    def __init__(self):
        super().__init__()
    @staticmethod
    def MeanSquaredError(predicted_data, prediction_data):
        if len(predicted_data) == len(prediction_data):
            results = []
            for sample in range(len(predicted_data)):
                result = math.pow(prediction_data[sample] - predicted_data[sample], 2)
                results.append(result)
        else:
            return "Error"
        array = np.array(results, dtype=np.float64)
        return array
    @staticmethod
    def EvalueatingCost(self, array):
        return self.find_average(array)
        
    @staticmethod
    def CrossEntropyLoss(input_values):
        pass
        
    @staticmethod
    def BinaryCrossEntropyLoss(input_values):
        pass

class Activacions():
    def __init_(self):
        self.act_type = {
            'relu':self.Relu(),
            'all_sigmoid':self.Sigmoid(),
        }
    @staticmethod
    def Relu(input_values:list):
        relu_value = np.maximum(0, input_values)
        return relu_value
        
    @staticmethod
    def Sigmoid(input_values:list):
        output = 1 / (1 + np.exp(-(input_values)))
        return output 

class AdditionalFunctions():
    def __init__(self):
        pass
        
    @staticmethod
    def measure_runtime(func):
        def wrapper(*args, **kwargs):
            start_time = time.time()
            result = func(start_time)
            end_time = time.time()
            
        return wrapper

class NeuronNetwork(Layer, AdditionalFunctions, FrontEnd):
    def __init__(self):
        super().__init__()
        self.neuron_count = 0
        self.layers = None
        self.hidden_layer_neurons = 0 #Work
        self.layers_count = 0
        self.review_heavy = None #[perIteration[perLayer[weights, biases], settings]]
        self.review_light = None #[perLayer[weights, biases], settings]
        
    def vector_multiply(self, input_values:list, layer:int, neuron):
        return np.dot(input_values, self.layers[layer][neuron][0]) + self.layers[layer][neuron][1]
      
    def forward_propagation(self, input_values):
        output = np.empty((0,), dtype=np.float64)
        for layer in range(len(self.layers)):
            layer_output = []
            for neuron in range(len(self.layers[layer])):
                neuron_output = self.vector_multiply(input_values, layer, neuron)
                layer_output.append(neuron_output)
            input_values = np.array(layer_output)  # Pass output of this layer as input to the next layer
            output = np.append(output, layer_output)
        return output

        #return [10, [99, 0]] #delete
    #@FrontEnd.ShowBoard 
    def fit(self, X_data, Y_data, num_batches, activision_type:str = None):
        input_data = np.empty((0,))
        output_data = np.empty((0,))
        self.set_input_values(X_data, 255) #Rework add rescaling addaptivity
        
        return [10, [99, 0]]
        
    def evaluate(self, X_data, Y_data):
        # [[weights, bias], settinggs, ]
        return diagnose
        
    def add(self, func):
        if self.layers == None:
            self.layers = func
        else:
            self.layers.append(func)
        
############################################################################################################################################################################
model = NeuronNetwork()

model.add(Layer.create_layer(model, 1, input_shape = 784))
model.add(Layer.create_layer(model, 3))
model.add(Layer.create_layer(model, 1))
#print(len(model.layers[1][0][0][0]))
#print(model.layers)

model.fit(X_train, Y_train, 10)