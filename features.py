import mdp
import numpy as np


class gaussians(object):

    def __repr__(self):
        return "gaussians("+repr(self.means)+","+repr(self.sigmasq)+")"

    def expectation(self, x, Sigma):
        
        sig = Sigma + self.sigmasq[:, None]
        detsig = np.sqrt(sig.prod(axis=1))
        return np.exp(-(np.power(x-self.means, 2)/sig).sum(axis=1) / 2.) \
                / np.power(2*np.pi, len(x) /2.) / detsig      

    def __init__(self, means, sigmas):
        assert(means.shape[0] == sigmas.shape[0])
        self.means = means
        self.sigmasq = np.power(sigmas,2)

    def __call__(self, x):
        return np.exp(-np.power(x-self.means, 2).sum(axis=1) / self.sigmasq / 2.) \
                / np.power(self.sigmasq*2*np.pi, len(x) /2.)

class squared_full(object):

    def __repr__(self):
        return "squared_full("+repr(self.normalization)+")"    
    
    def __init__(self, normalization=None):
        self.normalization = normalization

    def __call__(self, state):
        a = np.outer(state, state)
        r= np.concatenate((a.flatten(), [1])) 
        if self.normalization is not None:
            assert self.normalization.shape == r.shape
            r /= self.normalization
        return r
    def param_back(self, theta):
        """ transform theta back to P,b """
        if self.normalization is not None:
            theta = theta / self.normalization
        x = theta
        return (x[:-1].reshape(int(np.sqrt(len(x[:-1]))), int(np.sqrt(len(x[:-1])))), x[-1])

    def expectation(self, x, Sigma):
        a = np.outer(x, x)
        a += np.diag(Sigma)
        r= np.concatenate((a.flatten(), [1]))
        if self.normalization is not None:
            assert self.normalization.shape == r.shape
            r /= self.normalization
        return r

    def param_forward(self, P, b):
        """ transform P,b to theta """
        r = np.concatenate((np.array(P).ravel(), [b]))
        if self.normalization is not None:
            r *= self.normalization
        return r

class squared_tri(object):

    def __repr__(self):
        return "squared_tri("+repr(self.normalization)+")"       
    
    def __init__(self, normalization=None):
        self.normalization = normalization

    def __call__(self, state):
        iu1 = np.triu_indices(len(state))
        a = np.outer(state, state)
        a = a* (2-np.eye(len(state)))
        r = np.concatenate((a[iu1], [1]))
        if self.normalization is not None:
            assert self.normalization.shape == r.shape
            r /= self.normalization
        return r

    def expectation(self, x, Sigma):

        iu1 = np.triu_indices(len(x))
        a = np.outer(x,x)
        a += np.diag(Sigma)
        a = a* (2-np.eye(len(x)))
        r = np.concatenate((a[iu1], [1]))
        if self.normalization is not None:
            assert self.normalization.shape == r.shape
            r /= self.normalization
        return r

    def param_back(self, theta):
        """ transform theta back to P,b """
        if self.normalization is not None:
            theta = theta / self.normalization
        b = theta[-1]
        p = theta	[:-1]
        l = 1 if len(p) == 1 else (-1 + np.sqrt(1 + 8*len(p)))/2
        iu = np.triu_indices(l)
        il = np.tril_indices(l)
        a = np.empty((l,l))
        a[iu] = p
        a[il] = a.T[il]
        return a, b

    def param_forward(self, P, b):
        """ transform P,b to theta """
        iu1 = np.triu_indices(P.shape[0])
        r = np.concatenate((np.array(P[iu1]).ravel(), [b]))
        if self.normalization is not None:
            r *= self.normalization
        return r
class squared_diag(object):

    def __repr__(self):
        return "squared_diag("+repr(self.normalization)+")"   
    
    def __init__(self, normalization=None):
        self.normalization = normalization

    def __call__(self, state):
        r = state * state

        if self.normalization is not None:
            assert self.normalization.shape == r.shape
            r /= self.normalization
        return r

    def expectation(self, x, Sigma):
        r = x * x + Sigma
        if self.normalization is not None:
            assert self.normalization.shape == r.shape
            r /= self.normalization
        return r

    def param_back(self, theta):
        """ transform theta back to P,b """
        if self.normalization is not None:
            theta = theta / self.normalization
        return np.diag(theta), 0

    def param_forward(self, P, b):
        """ transform P,b to theta """
        r = np.diag(P)
        if self.normalization is not None:
            r *= self.normalization
        return r
