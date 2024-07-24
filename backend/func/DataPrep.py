import pandas as pd
import numpy as np

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