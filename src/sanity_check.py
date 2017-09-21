'''

'''

import matplotlib.pyplot as plt
from src.utils_srnn import *
from scipy.optimize import fmin_l_bfgs_b as minimize
import os

class SRNN(object):
    def __init__(self, n_units, W = None, b= None, input= None, time_steps= None, batch_sz =1,
                 np_rng = None, temp =1, lr_1 = 0.001, lr_2 = 0.001):
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
            #self.b = self.rng.normal(0, 1/self.n_units, size=(self.n_units,))
            self.b = np.zeros((self.n_units))
        else:
            self.b = b

        self.input = input
        # if batch_sz == 1:
        #     self.input = self.input[:,np.newaxis]
        #self.input_for_update = self.input[:-1]
        self.d = self.n_units

        self.lr_1 = lr_1
        self.lr_2 = lr_2
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
        z = np.dot(self.input[:-1], self.W.T) + self.b
        gamma = ( 0.5/self.temp * z * s )
        '''
        This Gamma can be very large if we use a large lr for no_transit case.
        The reason is because that the time-step in some cases is very large, which is taking into account
        to compute the gradients.
        '''
        #print(gamma)
        Gamma = np.exp(gamma)
        #new_s = (1 - 2 * self.input[:-1]) * Gamma * self.time_step[:, np.newaxis]
        new_s = (1 - 2 * self.input[:-1]) * Gamma * self.time_step
        #print(np.max(new_s))
        w_grad = - 0.5/self.temp * np.dot(new_s.T, self.input[:-1])
        b_grad = - 0.5/self.temp * np.sum(new_s, axis = 0)

        w_grad /= self.batch_sz
        b_grad /= self.batch_sz
        np.fill_diagonal(w_grad,0)

        return w_grad, b_grad

    def learn(self):
        w_grad, b_grad = self.get_update()
        self.W += self.lr_1 * w_grad
        #self.b += self.lr_1 * b_grad

    def learn_no_transit(self):
        w_grad, b_grad = self.get_update_no_transit()
        self.W += self.lr_2 * w_grad
        #self.b += self.lr_2 * b_grad


EPOCHS = 500
# LR1 = 0.001
# LR2 = 0.0001
BATCH_SIZE = 1


def learn_ETL(data, time_steps, lr1, lr2):

    ori_w = np.load('../data/sanity_w.npy')
    num_samples = data.shape[0]
    srnn = SRNN(n_units=data.shape[1], lr_1= lr1, lr_2=lr2)

    path = '../result/lr1_' + str(lr1) + '/lr2_' + str(lr2)

    if not os.path.exists(path):
        os.makedirs(path)
    rmse_error = []

    for epoch in range(EPOCHS):
        for i in range(num_samples-1):
            srnn.input = data[i:i+BATCH_SIZE+1,:]
            srnn.time_step = time_steps[i]
            srnn.learn_no_transit()
            srnn.learn()

        ret_W = srnn.W
        error = abs_error(ret_W, ori_w)
        rmse_error += [error]

        if (epoch + 1) % 20 == 0:
            plt.figure()
            plt.plot(ret_W.ravel())
            plt.plot(ori_w.ravel())
            plt.legend(['Recovered weights', 'Original weights'])
            weight_path = path + '/weights_epoch_' + str(epoch) + '.pdf'
            plt.savefig(weight_path)
            #lr1 /= 10


    plt.figure()
    plt.plot(np.arange(len(rmse_error)), rmse_error)
    plt.xlabel('Number of Epcohes')
    plt.ylabel('Error')
    error_path = path + '/Abs.pdf'
    plt.savefig(error_path)
    #print(rmse_error)

    # plt.plot(ret_b)
    # plt.plot(ori_b)
    # plt.show()


if __name__ == '__main__':

    lr_1_list = [0.00001, 0.0001, 0.001]
    lr_2_list = [0.000001, 0.00001, 0.0001, 0.001]

    for LR1 in lr_1_list:
        for LR2 in lr_2_list:
            data = np.load('../data/sanity_data.npy')
            time_steps = np.load('../data/sanity_time_steps.npy')[:-1]
            print(np.max(time_steps))
            print(time_steps)
            learn_ETL(data,time_steps,LR1, LR2)

    #srnn = SRNN(n_units = 4, input=data, time_steps= time_steps)

    # w_grad, b_grad = srnn.get_update_no_transit()
    # theta = np.concatenate( (srnn.W.ravel(), srnn.b) )
    # num_Wgrad3, num_bgrad3 = computeNumericalGradient(lambda x: srnn.get_loss_no_transit(x), theta)
    #
    #
    # print(num_Wgrad3)
    # print(w_grad)












