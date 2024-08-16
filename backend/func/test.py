
import numpy as np
list = [1,2,3,4,5,6,7,8,9]
OUTPUT = np.empty((0,1))
for a in list:
    OUTPUT = np.append(OUTPUT, [[a]], axis= 0)
print(OUTPUT)

