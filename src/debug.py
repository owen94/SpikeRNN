import matplotlib.pyplot as plt
import numpy as np

a = np.ones((4,5))
b = np.arange(4)
b = b[:, np.newaxis]
print(b)
print(a * b)