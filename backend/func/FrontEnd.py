import time
import matplotlib as plt

class FrontEnd():
    def __init__():
        pass
    @staticmethod
    def ShowBoard(func, settings= None):
        def wrapper(*args, **kwargs):
            start_time = time.time()
            result = func(*args, **kwargs)
            settings = ['Accuraccy', 'Los']
            time.sleep(1)
            end_time = time.time()
            print(f"Step: {result[0]} Output: {result[1]} {settings[0]}: {result[1][0]} {settings[1]}: {result[1][1]}\nTime taken: {end_time - start_time}\n--------------------------------------------")
            return result
        return wrapper
    
