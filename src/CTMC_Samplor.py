import numpy as np
import os

class CTMC_Samplor(object):
    def __init__(self, num_units, W = None, b = None, np_rng = None, temp=1):

        if np_rng is None:
            self.rng = np.random.RandomState(123)
        else:
            self.rng = np_rng

        self.n_units = num_units
        self.temp = temp

        if W is None:
            self.W = self.rng.normal(0, 1/self.n_units,size=(self.n_units,self.n_units))
        else:
            self.W = W

        np.fill_diagonal(self.W,0)

        if b is None:
            #self.b = self.rng.normal(0, 1/self.n_units, size=(self.n_units,))
            self.b = np.zeros((self.n_units))
        else:
            self.b = b
            # Note here b is a 1-d tensor, so if we changed self.x to a 2-d array, we need to change b respectively.

        # Initialize a state as starting point
        self.x = self.rng.randint(2,size=(self.n_units))


    def samplor(self):
        # In this function, we compute the necessary variables for sampling.
        s = 1 - 2 * self.x
        z = np.dot(self.W, self.x) + self.b
        Gamma = np.exp(1/(2*self.temp) * s * z)
        sum_Gamma = np.sum(Gamma)
        transit_prob = Gamma/sum_Gamma

        return s, z, Gamma, sum_Gamma, transit_prob


    def CTMC_simulation(self, n_samples):
        # Generate n-samples with the CTMC samplor
        # Return the samples (n_samples,  n_units )
        # Return the time intervals (n_units, )
        samples = np.zeros(shape=(n_samples,self.n_units))
        time_intervals = np.zeros(n_samples)

        for i in range(n_samples):

            samples[i] = self.x

            s, z, Gamma, sum_Gamma, transit_prob = self.samplor()
            hold_time = np.random.exponential(scale = sum_Gamma)
            time_intervals[i] = hold_time

            flip_bit = np.random.choice(a=self.n_units, size=1,p=transit_prob)
            self.x[flip_bit] = 1 - self.x[flip_bit]
        return samples, time_intervals

if __name__ == '__main__':

    ctmc_Samplor = CTMC_Samplor(num_units=10)
    samples,time_intervals = ctmc_Samplor.CTMC_simulation(n_samples=10000)

    weight = ctmc_Samplor.W
    bias = ctmc_Samplor.b

    path = '../data'
    if not os.path.exists(path):
        os.makedirs(path)


    np.save('../data/sanity_data.npy', samples)
    np.save('../data/sanity_time_steps.npy', time_intervals)
    np.save('../data/sanity_w.npy', weight)
    np.save('../data/sanity_b.npy', bias)

    print(samples.shape)
    print(time_intervals.shape)




















