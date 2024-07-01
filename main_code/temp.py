
import numpy as np
a = np.array([[]])
print(len(a.flatten()))
b = np.array([[[1.23,2.45],[3.11,4.21],[5.34,6.66]]])
print(" ".join([str(i) for i in b[0].flatten()]))
