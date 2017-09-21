'''
This is the functions need for spike rnn
'''
import numpy as np


def unravelparam(theta, d = 4):
    """
    Restores a vector of parameters into matrix form, which
    consists of W and b
    """
    W = theta[0: d** 2].reshape(d, d)
    b = theta[d**2:]
    return W, b

def rmse(a, b):
    return np.sqrt(np.mean((a-b)**2))

def abs_error(a,b):
    return np.sum(np.abs(a-1))

def computeNumericalGradient(J, theta):
    EPSILON = 0.0001
    numgrad = np.zeros(np.shape(theta))
    num_para = theta.shape[0]
    e_plus = EPSILON * np.eye(num_para)
    e_minus = - e_plus
    theta_e_plus = e_plus + theta
    theta_e_minus = e_minus + theta

    for i in range(num_para):
        numgrad[i] = (J(theta_e_plus[i, :]) - J(theta_e_minus[i, :]))/ (2 * EPSILON)

    num_Wgrad, num_bgrad = unravelparam(numgrad)

    return num_Wgrad, num_bgrad


