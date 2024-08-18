import pandas as pd
import math
import numpy as np
import time
import sys
import matplotlib
from keras.datasets import mnist
from decimal import Decimal
np.set_printoptions(threshold=np.inf)

class DataPrep():
    def __init__(self):
        self.rescaling = None ##Update
        
    @staticmethod  
    def list_sum(array:list) -> list:
        sum_value = 0
        for value in array:
            sum_value += value
        return sum_value
        
    @staticmethod  
    def find_index_value(array:list, position:list):
        for dim in position:
            array = array[dim]
        return array
     
    def flatten(self, input_values, flattened_list = None):
        if flattened_list == None:
            flattened_list = []
        if isinstance(input_values, (list, tuple)):
            for element in input_values:
                self.flatten(element, flattened_list)
        else:
            flattened_list.append(input_values)
        return flattened_list
        
    def control_shape_LT(self, obj, size, count=None):
        if count is None:
            count = []
        count.append(len(obj))
        for part in obj:
            print(part)
            if isinstance(part, (list, tuple)):
                self.control_shape_LT(part, size, count)
            else:
                obj
        if len(count) == len(size):
            pass
     
    def find_average(self, input_values):
        average = 0
        value_list = self.flatten(input_values)
        for value in value_list:
            average += value
        return average / len(value_list)
    
    def find_max(self, input_values:list):     
        if not isinstance(input_values, (list, tuple)):
            return input_values
        max_value = float('-inf')
        for element in input_values:
            max_value_elemnet = self.find_max(element)
            if max_value_elemnet > max_value:
                max_value = max_value_elemnet
        return max_value
        
    def find_min(self, input_values:list):     
        if not isinstance(input_values, (list, tuple)):
            return input_values
        min_value = float('inf')
        for element in input_values:
            min_value_elemnet = self.find_min(element)
            if min_value_elemnet < min_value:
                min_value = min_value_elemnet
        return min_value
        
    def find_shape_LT(self, array, size=None) -> list:  # Add self parameter
        if size is None:
            size = []
        if isinstance(array, (list, tuple)):
            size.append(len(array))
            if isinstance(array[0], (list, tuple)):  # Correct type checking
                self.find_shape_LT(array[0], size)
        return size
    
class Input():      
    """
    This class contain Inputs data and operations for processing input.
    """
    def __init__(self):
        self.input_values = np.empty((0), dtype= np.float64)
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
        pass 
class InconsistentArrayError(ExceptionList):
    def __init__(self, message="Inconsistent array"):
        self.message = message
        super().__init__(self.message)

class WeightsFunctions():
    """
    PURPOSE
    ------
    Class ussed for creating a Randomness Coefficient, that is used in creating weights and biases.

    RANDOMNESS COEFFICIENT - 
    """
    def __init__(self):
        super().__init__()
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

class Layer(Input, Output, WeightsFunctions):
    """
    PURPOSE
    ------
    Calls a Layer class.

    Is it used for adding layers to the model, and has lot of functions for processing data in layers.

    STRUCTURE OF LAYERS
    -------------------
    LAYERS [ ]\n
    NEURONS [ ][ ]\n
    Weights [ ][ ][0]\n
    Bias [ ][ ][1]

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
        self.layer_data_shape = None

    def random_weights(self, input_values):
        return np.random.uniform(-(self.randomness_coefficient), self.randomness_coefficient, input_values).astype(np.float64) 
        
    def random_bias(self):
        return np.array(np.random.uniform(-(self.randomness_coefficient), self.randomness_coefficient)).astype(np.float64)
        
    def create_neuron(cls):
        #if cls.settings[0] == 'XavierInitialization':
        #neuron = tuple([cls.random_weights(cls.input_values, cls.XavierInitialization(cls.input_shape_history, )), cls.random_bias(cls.XavierInitialization(cls.input_shape_history, ))])
        #if cls.settings[0] == 'HeInitialization':
        if cls.layers != None:
            return tuple([cls.random_weights(cls.layer_data_shape), cls.random_bias()])
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
        cls.layer_data_shape = hidden_layer_neurons
        return [layer, activation]
    
    
class CostFunctions():
    def __init__(self):
        pass

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

class Settings(): #REMAKE DOCSTRING
    """
    PURPOSE
    ------
    Class used for storing setting used in models.

    Is it used for adding layers to the model, and has lot of functions for processing data in layers.
    
    TYPES OF ACTIVATIONS:
    ---------------------
    Adam - 
    SGD - 

    TYPES of LOSSES:
    ----------------
    MSE - 
    EvalueatingCost - 
    CrossEntropyLoss - 

    EXAMPLE
    --------
    >>> your_model.settings("Adam",
                            "MSE",
                            "precision"                 
        )
    """
    def __init__(self):
        self.activation_opt = {
            "adam": self.Adam,
            "sgd": self.SGD
        }
        self.activation_type = None
        self.loss_opt = {
            "mse": self.MeanSquaredError
        }
        self.loss_type = None
        self.accuracy_type = None
        self.accuracy_opt = {
            "precision": self.precision
        }

    def settings(self, activation_type:str=None, loss_type:str=None, accuracy_type:str=None): # STILL WORK ON 
        try:
            self.activation_type = self.accuracy_opt[accuracy_type.lower()]
            self.loss_type = self.loss_opt[loss_type.lower()]
            self.activation_type = self.accuracy_opt[activation_type.lower()]
        except: #CREATE CUSTOM EXCEPTIONS!!!
            pass

class Activations():
    def __init__(self): 
        self.activations_type = {"sigmoid":self.Sigmoid, 
                                 "relu":self.Relu, 
                                 "softmax":self.Softmax
                                }
    def check_activation(self):
        """
        PURPOSE
        -------
        Function for checking if activations are spelled correctly.
        """
        for activation in self.activation_layers:
            if activation in self.activations_type:
                pass
            else:
                raise Exception("Cicina jedna")
            
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
        exp_values = np.exp(input_values - np.max(input_values))
        return exp_values / np.sum(exp_values)
    
class NeuralNetwork(Layer, CostFunctions, AdditionalFunctions,  Settings, Activations):
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
        Activations.__init__(self)
        super().__init__()
        self.layers = None
        self.activation_layers = None
        self.hidden_layer_neurons = 0 
        #self.review = None #[perLayer[weights, biases],[accuracy, settings, statistics]] WORK ON!!
        self.review = None #[perLayer[weights, biases], [accuracy]] WORK ON!!
        
    def save_model(self, file_path:str):
        with open(file_path, 'w') as file:
            for layer in self.layers:
                file.write({layer})
        # [[weights, bias], settinggs, ]
    
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
        return np.dot(input_values, neuron[0]) + neuron[1]
      
    def forward_propagation(self, input_values): #REWORK docstring
        """
        LAYERS []
        NEURONS [][]
        Weights [][][0]
        """           
        data_history = None              
        OUTPUT = []
        for layer in range(0, len(self.layers)):
            LAYER_OUTPUT = np.empty((0,))
            for neuron in range(0, len(self.layers[layer])):
                if data_history == None:
                    compute_weighted_sum = self.vector_multiply(input_values, self.layers[layer][neuron])
                else:
                    compute_weighted_sum = self.vector_multiply(OUTPUT[-1], self.layers[layer][neuron])
                LAYER_OUTPUT = np.append(LAYER_OUTPUT, compute_weighted_sum)
            LAYER_OUTPUT = self.activations_type[self.activation_layers[layer]](LAYER_OUTPUT)
            data_history = True
            OUTPUT.append([LAYER_OUTPUT])
        return OUTPUT
    
    #@FrontEnd.ShowBoard 
    def fit(self, X_data, Y_data, num_batches):
        self.set_input_values(X_data, 255) #Rework add rescaling addaptivity
        self.set_output_values(Y_data)
        for example in X_data:
            output_data = self.forward_propagation(example)
            output_data = output_data.append(self.MeanSquaredError(output_data[-1], Y_data))
        
    def evaluate(self, X_data, Y_data):
        # [[weights, bias], settinggs, ]
        pass

    def predict(self, X_data):
        """
        PURPOSE
        ------
        Function that predicts output, thanks to pretrained model.

        Parameters
        ----------
        X_data : array_like
            Input values. Can be a single number, a list, or a NumPy array with multiple examples.

        EXAMPLES
        --------
        
        """
        prediction = np.array([])
        if not isinstance(X_data, np.ndarray):
            try:
                X_data = np.array(X_data)
            except InconsistentArrayError as e:
                print(f"{e}")
        if len(X_data.shape) > 1:
            for example in X_data:
                prediction = np.append(prediction, self.forward_propagation(example)[-1])
        else:
            prediction = np.append(prediction, self.forward_propagation(X_data)[-1])
        return prediction

    def add(self, func):
        if self.layers == None:
            self.layers = [func[0]]
            self.activation_layers = [func[1]]
        else:
            self.layers.append(func[0])
            self.activation_layers.append(func[1])
        self.check_activation()
        
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
    >>> model.evaluate
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