
import numpy as np
list = np.array([1,2,3,4,5,6,7])
ar = np.empty((0))
for a in list:
    np.append(ar, a)
print(ar)
