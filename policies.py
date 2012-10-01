__author__ = "Christoph Dann <cdann@cdann.de>"
import numpy as np
import util

@util.memory.cache(hashfun={"policy": repr})
def mean_action_trajectory(policy, states):
    ret = np.empty((states.shape[0], policy.dim_A))
    for i in xrange(states.shape[0]):
        ret[i] = policy.mean(states[i])
    return ret

class NoisyContinuous(object):

    def __init__(self, noise=None, dim_A=None):
        if noise is None:
            self.dim_A = dim_A
            self.noise = np.zeros(self.dim_A)
        else:
            self.noise = noise
            self.dim_A = len(noise)
        self.approx_noise = self.noise.copy()
        self.approx_noise[self.approx_noise == 0] = 10e-50
        self.precision = 1. / self.approx_noise

    def __call__(self, s, n_samples=1):
        m = self.mean(s)
        noise = np.sqrt(self.noise[None, :]) * np.random.randn(n_samples, self.dim_A)
        if n_samples == 1:
            return (m + noise).flatten()
        else:
            return m + noise

    def p(self, s, a, mean=None):
        m_a = mean -a if mean is not None else self.mean(s) - a
        return np.exp(-.5 * (m_a * m_a * self.precision).sum()) / \
            ((2 * np.pi) ** (float(self.dim_A) / 2)) / np.sqrt(
                self.approx_noise.sum())


class LinearContinuous(NoisyContinuous):

    def __repr__(self):
        return "LinearContinuous(" + repr(self.theta) + "," + repr(self.noise) + ")"

    def __init__(self, theta=None, noise=None, dim_S=None, dim_A=None):
        NoisyContinuous.__init__(self, noise=noise, dim_A=dim_A)
        if theta is None:
            self.dim_S = dim_S
            self.dim_A = dim_A
            self.theta = np.zeros((self.dim_A, self.dim_S))
        else:
            self.theta = np.atleast_2d(theta)
            self.dim_A, self.dim_S = self.theta.shape

    def mean(self, s):
        return np.array(np.dot(self.theta, s)).flatten()


class Discrete(object):
    def __repr__(self):
        return "Discrete(" + repr(self.tab) + ")"

    def __init__(self, prop_table):
        self.tab = prop_table
        prop_table /= prop_table.sum(axis=1)[:,None]
        self.dim_S, self.dim_A = self.tab.shape

    def __call__(self, s):
        return  util.multinomial_sample(1, self.tab[int(s), :])

    def mean(self, s):
        return np.sum(self.tab[int(s),:] * np.arange(self.dim_A))

    def p(self, s, a, mean=None):
        return self.tab[int(s), int(a)]


class DiscreteUniform(Discrete):

    def __init__(self, dim_S, dim_A):
        self.tab = np.ones((dim_S, dim_A)) / float(dim_A)
        self.dim_S, self.dim_A = dim_S, dim_A



class MarcsPolicy(NoisyContinuous):

    def __init__(self, filename="./mlab_cartpole/policy.mat", noise=None, dim_A=1):
        NoisyContinuous.__init__(self, dim_A=dim_A, noise=noise)
        try:
            from mlabwrap import mlab
            self.mlab = mlab
            self.mlab._autosync_dirs = False
            self.mlab.addpath("./mlab_cartpole")
            self.mlab.load(filename)
        except Exception:
            self.mlab = None
        self.filename = filename

    def __repr__(self):
        return "MarcsPolicy (fn=" + self.filename + ",noise=" + repr(self.noise) + ")"

    def __getstate__(self):
        res = self.__dict__.copy()
        del res["mlab"]
        return res

    def __setstate__(self, state):
        self.__dict__ = state
        try:
            from mlabwrap import mlab

            self.mlab = mlab
            self.mlab._autosync_dirs = False
            self.mlab.addpath("./mlab_cartpole")
            self.mlab.load(self.filename)
        except Exception:
            try:
                import mlabwrap
                self.mlab = mlabwrap.MlabWrap()
                self.mlab._autosync_dirs = False
                self.mlab.addpath("./mlab_cartpole")
                self.mlab.load(self.filename)
            except Exception, e:
                print e
                print mlab._session
                self.mlab = None


    def mean(self, s):
        lst = [str(a) for a in s[:-1]] + [str(np.sin(s[-1])), str(
            np.cos(s[-1]))]
        strrep = ",".join(lst)
        r = self.mlab._do("policy.fcn(policy,[" + strrep + "]', zeros(5,5))")
        return np.ones(1) * float(r)
