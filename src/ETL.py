'''
This file we define the Energy transition learning  process.
Basically, we define the conditional log-likelihood, the gradient based on transition,
check the gradient, and finally do the learning.
'''

import numpy as np
import matplotlib.pyplot as plt

class SRNN(object):
    def __init__(self, n_units, W = None, b= None, input= None, time_steps= None,  np_rng = None, temp =1, lr = 0.001):
        if np_rng is None:
            self.rng = np.random.RandomState(1234567)
        else:
            self.rng = np_rng

        self.n_units = n_units
        self.temp = temp

        if W is None:
            self.W = self.rng.normal(0, 1/self.n_units,size=(self.n_units,self.n_units))
        else:
            self.W = W

        #np.fill_diagonal(self.W,0)

        if b is None:
            self.b = self.rng.normal(0, 1/self.n_units, size=(self.n_units,))
        else:
            self.b = b

        self.input = input

        self.lr = lr
        self.time_step = time_steps


    def check_transition(self, x_1, x_2):
        # check whether there exists transitions between two consecutive states
        # here we only has 1-bit flip, and one sample
        is_transit = True
        diff = x_1 - x_2
        indices = np.argmax(np.abs(diff)) # return which bit has a flip
        if indices == 0 and x_1[0] - x_2[0] == 0:
            is_transit = False

        return is_transit, indices

    def get_cost_update(self, x_1, x_2, delat_t):
        # get the update of the gradient based on one step transition

        is_transit, indices = self.check_transition(x_1, x_2)

        if is_transit:
            # compute the loss
            s_j = 1 - 2 * x_1[indices]
            z_j = np.dot(x_1, self.W[:,indices]) + self.b[indices]
            loss = np.log(delat_t) + 1/(2*self.temp) * s_j * z_j

            # compute the gradient
            w_grad = 1/(2*self.temp) * s_j * x_1
            b_grad = 1/(2*self.temp) * s_j

        else:
            # this part is not correct now
            print('no transition')
            s_j = 1 - 2 * x_1[indices]
            z = np.dot(self.W, x_1[indices]) + self.b
            Gamma = np.exp(1/(2*self.temp) * s_j * z)
            sum_Gamma = np.sum(Gamma)
            loss = -delat_t * sum_Gamma

            w_grad = - 1/(2*self.temp) * s_j * x_1 * Gamma  * delat_t
            b_grad = - 1/(2*self.temp) * s_j *       Gamma  * delat_t

        return loss, w_grad, b_grad, indices

    def update_params(self, x_1, x_2, delat_t):
        # do gradient descent
        loss, w_grad, b_grad, index = self.get_cost_update(x_1=x_1, x_2=x_2, delat_t=delat_t)
        #print(self.W[:,index].shape)
        self.W[:,index] -= self.lr * w_grad
        #np.fill_diagonal(self.W, 0)
        self.b[index] -= self.lr * b_grad


EPOCHS = 100
LR = 0.001


def learn_ETL(data, time_steps, lr):



    num_samples = data.shape[0]
    srnn = SRNN(n_units=data.shape[1])

    for epoch in range(EPOCHS):
        for i in range(num_samples-1):
            x_1 = data[i,:]
            x_2 = data[i+1,:]
            delta_t = time_steps[i]
            srnn.update_params(x_1=x_1, x_2=x_2, delat_t= delta_t)

    ret_W = srnn.W
    print(ret_W.shape)
    ret_b = srnn.b

    ori_w = np.load('../data/sanity_w.npy')
    print(ori_w.shape)
    ori_b = np.load('../data/sanity_b.npy')

    print(ret_W)

    print(ori_w)
    plt.plot(ret_W.ravel())
    plt.plot(ori_w.ravel())
    plt.legend(['recover', 'origin'])
    plt.show()

    plt.plot(ret_b)
    plt.plot(ori_b)
    plt.show()


if __name__ == '__main__':

    data = np.load('../data/sanity_data.npy')
    time_steps = np.load('../data/sanity_time_steps.npy')
    learn_ETL(data, time_steps, lr = LR)
