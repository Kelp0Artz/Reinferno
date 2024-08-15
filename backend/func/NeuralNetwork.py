import pandas as pd
import math
import numpy as np
import time
import sys

from keras.datasets import mnist
from decimal import Decimal
np.set_printoptions(threshold=np.inf)


class Input():      
    """
    This class contain Inputs data and operations for processing input.
    """
    def __init__(self):
        self.input_values = np.empty((0), dtype= np.float64)
        self.randomness_coefficient = None
        self.input_shape_history = None  #[INPUT, INPUT_SHAPE, MEAN, MAX, MIN, SCALING_FACTOR, SCALING_TYPE]
        self.input_data_shape = None
        
    @staticmethod
    def rescaling_values(data,  SCALING_FACTOR=0):
        return data / np.float64(SCALING_FACTOR)
        
    def set_input_values(self, data, SCALING_FACTOR = 0): 
        """
        With this function, you can set input data for training model.

        PARAMETERS
        ----------
        data : array-like
            Data. It can be a 0-dim array or multi-dim array of numpy type or python's basic list.

        SCALING_FACTOR : 0-dim array
            SCALING_FACTOR. It sets a scaling factor, that will process data.
        """
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
        self.output_values = np.empty((0,), dtype = np.float64)
        self.output_data_shape = None
        self.output_shape_history = None
    
    def set_output_values(self, data, SCALING_FACTOR = 0): #MAKE A SCALING FACTOR
        """
        With this function, you can set output data for training model.

        PARAMETERS
        ----------
        data : array-like
            Data. It can be a 0-dim array or multi-dim array of numpy type or python's basic list.

        SCALING_FACTOR : 0-dim array
            SCALING_FACTOR. It sets a scaling factor, that will process data.
        """
        self.set_output_values = data

    def set_ouput_shape(self, shape):
        pass

class ExceptionList(Exception):
    """
    This class contain all custom Exceptions for Reinferno library.

    EXAMPLES
    --------
    Such as: GYATT:>
    """
    def __init__(self):
        super().__init__()

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

class Layer(Input, WeightsFunctions, Output):
    """
    PURPOSE
    ------
    Calls a Layer class.

    Is it used for adding layers to the model, and has lot of functions for processing data in layers.

    EXAMPLES
    --------
    First you need to create a model.
    >>> model = NeuralNetwork() 
    #Then add some Layers.
    >>> model.add(Layer.create_layer(model, 128, input_shape = 10)) 
    >>> model.add(Layer.create_layer(model, 3, activaion = "Softmax"))
    """
    def __init__(self):
        super().__init__()
        self.weights = np.empty((0,), dtype=np.float64)
        self.biases = np.empty((0,), dtype=np.float64)
        self.layer = None
    """    
    def settings(self, *args, **kwargs):
        if kwargs:
            directory_looker(kwargs)
        def directory_looker(dictionary:dict):
            for index, obj in enumerate(dictionary):
                print(index, obj)
    """
    def random_weights(self, input_values):
        return np.random.uniform(-(self.randomness_coefficient), self.randomness_coefficient, (input_values)).astype(np.float64) 
        
    def random_bias(self):
        return np.array(np.random.uniform(-(self.randomness_coefficient), self.randomness_coefficient)).astype(np.float64)
        
    def create_neuron(cls):
        #if cls.settings[0] == 'XavierInitialization':
        #neuron = tuple([cls.random_weights(cls.input_values, cls.XavierInitialization(cls.input_shape_history, )), cls.random_bias(cls.XavierInitialization(cls.input_shape_history, ))])
        #if cls.settings[0] == 'HeInitialization':
        if cls.layers != None:
            return tuple([cls.random_weights(len(cls.layers[-1][0])), cls.random_bias()])
        else:
            return tuple([cls.random_weights(cls.input_data_shape), cls.random_bias()])
    @staticmethod
    def create_layer(cls, hidden_layer_neurons:int, activation:str = "Sigmoid", input_shape:int = None):
        """
        INFO 
        ----
        With this function, you can create layers that can be used in your model.
        It should be primary used with a NeuralNetwork.add() function.

        PARAMETERS
        ----------
        hidden_layer_neurons : int
            hidden_layer_neurons. It is used for creating hidden neurons inside a layer.

        activation : str
            activation. It sets a activation function, that will be used inside a layer.

        input_shape : int
            input_shape. It assign a input shape to a first layer inside the model.
        
        EXAMPLES
        ---------
        >>> 
        """
        if cls.input_shape_history == None:
            cls.input_data_shape = input_shape
            cls.input_shape_history = True
            cls.randomness_coefficient = cls.XavierInitialization()
        # MAKES A RANDOMNESS COEFECIENT DEPENDING ON INPUT SHAPE 
        layer = []
        for neuron in range(0, hidden_layer_neurons):
            #if cls.layers == None:
            layer.append(cls.create_neuron())
        return [layer, activation]
    
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
            'relu': self.Relu(),
            'sigmoid': self.Sigmoid(),
            'softmax': self.Sigmoid()
        }

    @staticmethod
    def Relu(input_values):
        """
        The ReLU function returns the input value if it is greater than zero; otherwise, it returns zero.

        Parameters
        ----------
        input_values : array_like
            Input values. Can be a single number, a list, or a NumPy array.

        Returns
        -------
        ndarray
            An array where each element is the result of applying the ReLU function to the corresponding element in the input.
        """
        relu_value = np.maximum(0, input_values)
        return relu_value
        
    @staticmethod
    def Sigmoid(input_values):
        """
        The Sigmoid function maps input values to a value between 0 and 1, using the logistic function.

        Parameters
        ----------
        input_values : array_like
            Input values. Can be a single number, a list, or a NumPy array.

        Returns
        -------
        ndarray
            An array where each element is the result of applying the Sigmoid function to the corresponding element in the input.
        """
        output = 1 / (1 + np.exp(-(input_values)))
        return output 

    @staticmethod
    def Softmax(input_values):
        """
        The Softmax function converts a vector of raw scores into a probability distribution, where each value is in the range (0, 1) and the sum of all values is 1.

        Parameters
        ----------
        input_values : array_like
            Input values. A 1-D array.

        Returns
        -------
        ndarray
            An array where each element is the result of applying the Softmax function to the corresponding element in the input. The output values are normalized probabilities.
        """
        output = np.empty((0,), dtype=np.float64)
        for value in input_values:
            output = np.append(output, np.exp(value)/ np.exp(input_values))
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

class Settings():
    def __init__(self):
        self.activation_opt = {
            True: True
        }
        self.loss_type = None
        self.loss_opt = {
            True: True
        }
        self.accuracy_type = None
        self.accuracy_opt = {
            True: True
        }
        self.activation_type = None

    def settings(self): #WORK ON THIS PLEASE
        pass
class NeuralNetwork(Layer, AdditionalFunctions, Settings):
    """
    PURPOSE
    ------
    Calls a Neural Network class.

    EXAMPLES
    --------
    >>> model = NeuralNetwork()
    >>> model.add(Layer.create_layer(model, 128, input_shape = 10)) 
    >>> model.add(Layer.create_layer(model, 3, activaion = "Softmax"))
    >>> model.fit(X_train, Y_train)
    >>> model.predict(X_train)
    """
    def __init__(self):
        super().__init__()
        self.neuron_count = 0
        self.layers = None
        self.activation_layers = None
        self.hidden_layer_neurons = 0 
        self.review_heavy = None #[perLayer[weights, biases],[accuracy, settings, statistics]]
        self.review_light = None #[perLayer[weights, biases], [accuracy]]
        
    def save_model(self, file_path:str):
        with open(file_path, 'w') as file:
            for layer in self.layers:
                file.write({layer})

        """# Read the list back from the file
        with open('list.txt', 'r') as file:
            loaded_list = [line.strip() for line in file]"""
        # [[weights, bias], settinggs, ]
    def settings(self, activation_type, loss_type, accuracy_type):
        pass
        
    
    def vector_multiply(self, input_values, neuron): #REWORK docstring and PURPOSE!!!
        """
        PURPOSE
        ------
        Function that can be used for multiplying matrices.

        EXAMPLES
        --------
        >>> X = vector_multiply([1,2,3], [2,7,6], 10)
        >>> print(X)
        44
        """
        print("Neuron",neuron)
        #return np.dot(input_values, neuron[0]) + neuron[1]
      
    def forward_propagation(self, input_values): #REWORK docstring
        """
        LAYERS []
        NEURONS [][]
        Weights [][][0]
        """                               
        OUTPUT = np.empty((0,))
        for layer in range(0, len(self.layers)):
            LAYER_OUTPUT = np.empty((0,))
            for neuron in range(0, len(self.layers[layer])):
                LAYER_OUTPUT = np.append(LAYER_OUTPUT, self.vector_multiply(input_values, self.layers[layer][neuron]))
                print("Layer Output", LAYER_OUTPUT)
            OUTPUT = np.append(OUTPUT, LAYER_OUTPUT)
        print("Output", OUTPUT)
        """output = np.empty((0,), dtype=np.float64)
        for layer in range(len(self.layers)):
            layer_output = np.empty((0,), dtype=np.float64)
            for neuron in range(0, len(self.layers[layer])):
                neuron_output = self.vector_multiply(input_values, layer, neuron)
                np.append(layer_output, [neuron_output])
            print("OUTPUT", layer_output)
            input_values = np.array(layer_output)  # Pass output of this layer as input to the next layer
            output = np.append(output, layer_output)
        return output"""

        #return [10, [99, 0]] #delete
    #@FrontEnd.ShowBoard 
    def fit(self, X_data, Y_data, num_batches, activision_type:str = None):
        input_data = np.empty((0,))
        output_data = np.empty((0,))
        self.set_input_values(X_data, 255) #Rework add rescaling addaptivity
        self.set_output_values(Y_data)
        self.forward_propagation(X_data)
        return [10, [99, 0]]
        
    def evaluate(self, X_data, Y_data):
        # [[weights, bias], settinggs, ]
        pass

    def add(self, func):
        if self.layers == None:
            self.layers = [func[0]]
            self.activation_layers = [func[1]]
        else:
            self.layers.append(func[0])
            self.activation_layers.append(func[1])
class linearRegression(Input, Output):
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
                    array = np.append(array, [dataX[:, value]])
            print(array)
    import numpy as np

def LinearRegression(dataX, dataY, learningRate):
    """
    PURPOSE
    ------
    Class used for simple Linear Regression model training.

    EXAMPLES
    --------
    >>> model = LinearRegression()
    >>> model.fit(X_train, Y_train)
    >>> model.predict(X_predict)
    >>> weights_biases = model.save_model()
    """
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