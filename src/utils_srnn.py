'''
This is the functions need for spike rnn
'''
import numpy as np


def rmse(a, b):
    return np.sqrt(np.mean((a-b)**2))

