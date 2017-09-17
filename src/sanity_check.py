'''

'''

import matplotlib.pyplot as plt
from src.utils_srnn import *

class SRNN(object):
    def __init__(self, n_units, W = None, b= None, input= None, time_steps= None, batch_sz =1,
                 np_rng = None, temp =1, lr = 0.001):
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

        np.fill_diagonal(self.W,0)
        assert self.W[0, 0] == 0

        if b is None:
            self.b = self.rng.normal(0, 1/self.n_units, size=(self.n_units,))
        else:
            self.b = b

        self.input = input
        # if batch_sz == 1:
        #     self.input = self.input[:,np.newaxis]
        #self.input_for_update = self.input[:-1]
        self.d = self.n_units

        self.lr = lr
        self.time_step = time_steps
        self.batch_sz = batch_sz


    def check_transition(self):
        # return a matrix, each item ij indicate that from state i to state i+1 does node j flipped or not

        return np.array(1 - (self.input[:-1] == self.input[1:]))


    def get_loss(self, theta):

        self.is_transit = self.check_transition()
        self.W, self.b = unravelparam(theta, d = self.d )

        s =  1  - 2 * np.sum( self.is_transit * self.input[:-1], axis=1) # get 1 - 2 x_j for each sample x, usually
        # the sum is 1 or -1. the other elements in each row are zero
        b_j = np.dot(self.is_transit,self.b)
        w_j = np.dot(self.is_transit,self.W) # return an array, where each row is the corresponding weights w_ji
        #from node i to node j, where j is the node that flipped during transition
        z = np.sum(w_j * self.input[:-1], axis=1) + b_j
        # w_j = np.dot(self.W, self.is_transit.T)
        # z = np.dot(self.input[:-1], w_j) + b_j

        loss = 0.5/self.temp * np.sum(s*z)

        return loss

    def get_loss_no_transit(self, theta):
        self.is_transit = self.check_transition()
        self.W, self.b = unravelparam(theta, d = self.d)
        '''
        since there is no transition, actually we need to update all the w_ji pairs in this stage.
        '''
        s = 1 - 2 * self.input[:-1]

        Gamma = np.exp( 0.5/self.temp * (np.dot(self.input[:-1], self.W.T) + self.b) * s )
        sum_Gamma = - np.sum(Gamma, axis = 1) # \sum_j \lambda_j
        loss = np.dot(self.time_step, sum_Gamma) # \delta_t * sum_Gamma

        loss /= self.batch_sz

        return loss

    def get_update(self):

        self.is_transit = self.check_transition()
        s = (1 - 2 * self.input[:-1]) * self.is_transit
        w_grad = 0.5/self.temp * np.dot(s.T, self.input[:-1]) # here we compute gradient for all the
        # samples and add together by vectorization, reaching a d*d matrix
        b_grad = 0.5/self.temp * np.sum(s, axis = 0 )

        np.fill_diagonal(w_grad, 0)

        return w_grad, b_grad

    def get_update_no_transit(self):
        self.is_transit = self.check_transition()
        s = (1 - 2 * self.input[:-1])
        Gamma = np.exp( 0.5/self.temp * (np.dot(self.input[:-1], self.W.T) + self.b) * s )
        #new_s = (1 - 2 * self.input[:-1]) * Gamma * self.time_step[:, np.newaxis]
        new_s = (1 - 2 * self.input[:-1]) * Gamma * self.time_step
        w_grad = - 0.5/self.temp * np.dot(new_s.T, self.input[:-1])
        b_grad = - 0.5/self.temp * np.sum(new_s, axis = 0)

        w_grad /= self.batch_sz
        b_grad /= self.batch_sz
        np.fill_diagonal(w_grad,0)

        return w_grad, b_grad

    def learn(self):
        w_grad, b_grad = self.get_update()
        self.W -= self.lr * w_grad
        self.b -= self.lr * b_grad

    def learn_no_transit(self):
        w_grad, b_grad = self.get_update_no_transit()
        self.W -= self.lr * w_grad
        self.b -= self.lr * b_grad


EPOCHS = 10
LR = 0.01
BATCH_SIZE = 1


def learn_ETL(data, time_steps, lr):

    num_samples = data.shape[0]
    srnn = SRNN(n_units=data.shape[1])

    for epoch in range(EPOCHS):
        for i in range(num_samples-1):
            srnn.input = data[i:i+BATCH_SIZE,:]
            srnn.time_step = time_steps[i]
            srnn.learn()
            #srnn.learn_no_transit()

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
    #
    # plt.plot(ret_b)
    # plt.plot(ori_b)
    # plt.show()


if __name__ == '__main__':

    data = np.load('../data/sanity_data.npy')
    time_steps = np.load('../data/sanity_time_steps.npy')[:-1]
    learn_ETL(data,time_steps,LR)


    #srnn = SRNN(n_units = 4, input=data, time_steps= time_steps)

    # w_grad, b_grad = srnn.get_update_no_transit()
    # theta = np.concatenate( (srnn.W.ravel(), srnn.b) )
    # num_Wgrad3, num_bgrad3 = computeNumericalGradient(lambda x: srnn.get_loss_no_transit(x), theta)
    #
    #
    # print(num_Wgrad3)
    # print(w_grad)












