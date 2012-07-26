

__author__="Christoph Dann <cdann@cdann.de>"
import numpy as np
import util
class LinearContinuous(object):
    
    def __init__(self, theta=None, noise=None, dim_S=None, dim_A=None):
        if theta is None:
            self.dim_S = dim_S
            self.dim_A = dim_A
            self.theta = np.zeros((self.dim_A,self.dim_S))
        else:
            self.theta = np.atleast_2d(theta)
            self.dim_A, self.dim_S = self.theta.shape
            

        if noise is None:
            self.noise = np.zeros((self.dim_A, self.dim_A))
        else:
            self.noise = noise
        if np.all(self.noise == 0):
            self.approx_noise = np.eye(self.dim_A)*10e-50
        else:
            self.approx_noise = self.noise
        self.precision = np.linalg.pinv(self.approx_noise)
        
    def __call__(self, s):
        return np.random.multivariate_normal(np.array(np.dot(self.theta, s)).flatten(), self.noise)
        
    def p(self,s,a):
        s = np.array(s).flatten()
        m_a = np.array(np.dot(self.theta, s)).flatten() - a
        return np.exp(-0.5*np.dot(m_a, np.dot(self.precision, m_a)))/ ((2*np.pi)**(float(self.dim_A) / 2)) / np.sqrt(np.trace(self.approx_noise))

class Discrete(object):

    def __init__(self, prop_table):
        self.tab = prop_table
        self.dim_S, self.dim_A = self.tab.shape
        
    def __call__(self, s):
        return  util.multinomial_sample(1,self.tab[s, :])
        
    def p(self,s,a):
        return self.tab[s,a]

class DiscreteUniform(Discrete):

    def __init__(self, dim_S, dim_A):
        self.tab = np.ones((dim_S, dim_A)) / float(dim_A)
        self.dim_S, self.dim_A = dim_S, dim_A


