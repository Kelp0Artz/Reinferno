import pandas as pd
import math
import numpy as np
import time
import sys

from keras.datasets import mnist
from decimal import Decimal
np.set_printoptions(threshold=np.inf)


class Input():      
    def __init__(self):
        self.input_values = np.empty((0,), dtype= np.float64)
        self.randomness_coefficient = None
        self.input_shape_history = None  #[INPUT, INPUT_SHAPE, MEAN, MAX, MIN, SCALING_FACTOR, SCALING_TYPE]
        self.input_data_shape = None
        
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

    def set_input_history(self, size):
        for data in size:
            if size >= size:
                self.input_shape_history = True
    

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
        for neuron in range(0, hidden_layer_neurons):
            if cls.neuron_count == 0: 
                layer = [[cls.create_neuron()]]
            else:
                layer.append(cls.create_neuron())
            cls.neuron_count += 1
        cls.neuron_count = 0
        return layer  
    
class CostFunctions():
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

class NeuralNetwork(Layer, AdditionalFunctions):
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
        self.forward_propagation(X_data)
        return [10, [99, 0]]
        
    def evaluate(self, X_data, Y_data):
        # [[weights, bias], settinggs, ]
        pass

    def add(self, func):
        if self.layers == None:
            self.layers = func
        else:
            self.layers.append(func)
        
class LinearRegression(Input, Output):
    def __init__(self):
        self.slopes = None #[[m1, b1], ...]
    
    #### Somehow store values dataX, dataY
    @staticmethod
    def sumlist(data):
        sum_value = 0
        for value in data:
            sum_value += value
        return sum_value
    
    @staticmethod
    def squaredlist(data):
        squared_value = 0 
        for value in data:
            squared_value += value**2
        return squared_value

    @staticmethod
    def multiplylist(dataX, dataY):
        if len(dataX) == len(dataY):
            multiplied_value = 0 
            for index in range(len(dataX)):
                multiplied_value += dataX[index]*dataY[index]
            return multiplied_value
        else:
            print('Error')
            
    def find_slope(self, dataX, dataY):
        dataX = np.array(dataX)
        print(dataX.shape)
        if len(dataX.shape) == 1:
            m_numerator = len(dataX) * self.multiplylist(dataX, dataY) - self.sumlist(dataX) * self.sumlist(dataY)
            m_denominator = len(dataX) * self.squaredlist(dataX) - self.sumlist(dataX)**2
            m = m_numerator/m_denominator
            b_numerator = self.sumlist(dataY)- m*self.sumlist(dataX)
            b = b_numerator/len(dataX)
            self.slopes = [m, b]
        else:
            array = np.empty((0,), dtype = np.float64)
            # add support for 3d data
            if len(dataX.shape) == 2:
                for value in range(0, dataX.shape[-1]):
                    np.append(array, [dataX[:, value]])
            print(array)
    import numpy as np

def linearRegression(dataX, dataY, learningRate):
    listOfSquaredResiduals = []
    squaredResidual = float('inf')
    slope = 1.0
    intercept = 0.0
    NUM_ITERATION = 1000
    iteration = 0

    def calculateSquaredResidual(dataX, dataY, slope, intercept):
        residual = np.sum((dataY - (slope * dataX + intercept))**2)
        return residual

    while squaredResidual > 0.0001 and iteration < NUM_ITERATION:
        # Calculate gradients
        predictions = slope * dataX + intercept
        intercept_gradient = -2 * np.sum(dataY - predictions)
        slope_gradient = -2 * np.sum((dataY - predictions) * dataX)

        # Update parameters
        intercept -= learningRate * intercept_gradient
        slope -= learningRate * slope_gradient

        # Calculate and store squared residuals
        squaredResidual = calculateSquaredResidual(dataX, dataY, slope, intercept)
        listOfSquaredResiduals.append(squaredResidual)
        
        iteration += 1
    
    return slope, intercept, listOfSquaredResiduals
"""
# Sample data
dataX = np.array([1, 2, 3, 4, 5])
dataY = np.array([2, 3, 5, 7, 11])
learningRate = 0.01

# Perform linear regression
slope, intercept, residuals = linearRegression(dataX, dataY, learningRate)

print("Slope:", slope)
print("Intercept:", intercept)
print("Residuals:", residuals)
"""
            
"""
    def fit(self, dataX, dataY):
        self.slope = self.find_slope(dataX, dataY) #Error in the function
        predicted_values = np.empty((0,), dtype = np.float64)
        for index in range(0,len(dataX)):
            np.append(predicted_values, dataX[index]*self.slope[0]+self.slope[1])
        return predicted_values

    def evaluate(self, dataX, dataY):
        if self.slope == None:
            return "Error"
        
    def predict(self, dataX):
        if self.slope == None:
            return "Error"
        else:
            predictions = np.empty((0,), dtype = np.float64)
            for value in dataX:
                predictions.append(value*self.slope[0]+self.slope[1])
            return predictions
"""