import numpy as np

def make_slice(l, u, n):
    return slice(l, u + float(u - l) / (n - 1) / 2., float(u - l) / (n - 1))

def make_grid(n_slices, bounds):
    s = [make_slice(b[0], b[1], n) for b, n in zip(bounds, n_slices)]
    bounds = np.array(bounds, dtype="float")
    means = np.mgrid[s[0], s[1], s[2], s[3]].reshape(4, -1).T

    sigmas = np.ones_like(means) * (
        (bounds[:, 1] - bounds[:, 0]) / 2. / (np.array(n_slices) - 1)).flatten()
    return means, sigmas


class gaussians(object):

    def __repr__(self):
        return "gaussians(" + repr(self.constant) + repr(self.normalization) \
            + "," + repr(self.means) + "," + repr(self.sigmasq) + ")"

    def expectation(self, x, Sigma):

        sig = Sigma + self.sigmasq
        phi = np.exp(-(np.power(x - self.means, 2) / sig).sum(axis=1) / 2.)
        return phi / np.sum(phi)

    def __init__(self, means, sigmas, constant=False, normalization=None):
        self.dim = means.shape[0]
        if constant:
            self.dim +=1
        self.normalization = normalization
        assert(means.shape[0] == sigmas.shape[0])
        self.means = means
        self.sigmasq = np.power(sigmas, 2)
        self.constant = constant

    def __call__(self, x):
        #detsig = np.sqrt(self.sigmasq.prod(axis=1))
        phi = np.exp(
            -(np.power(x - self.means, 2) / self.sigmasq).sum(axis=1) / 2.)
        if self.constant:
            b = np.ones(phi.shape[0] + 1)
            b[:-1] = phi / np.sum(phi)
        else:
            b = phi / np.sum(phi)
        if hasattr(self, "normalization") and self.normalization is not None:
            b /= self.normalization
        return b


class linear_blended(object):
    """
        official approximation function for baird star example

        taken from: Maei, H. R. (2011). Gradient Temporal-Difference Learning
                Algorithms. Machine Learning. University of Alberta.
                p. 17
    """

    def __repr__(self):
        return "linear_blended(" + repr(self.n_states) + ")"

    def __call__(self, x):
        state = int(x)
        n_corners = self.n_states - 1
        result = np.zeros(self.dim)
        if state == n_corners:
            result[-1] = 2
            result[-2] = 1
        else:
            result[-1] = 1
            result[state] = 2
        return result

    def __init__(self, n_states):
        self.dim = n_states + 1
        self.n_states = n_states


class squared_full(object):

    def __repr__(self):
        return "squared_full(" + repr(self.normalization) + ")"

    def __init__(self, dim, normalization=None):
        self.dim = dim
        self.normalization = normalization

    def __call__(self, state):
        a = np.outer(state, state)
        r = np.concatenate((a.flatten(), [1]))
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
        r = np.concatenate((a.flatten(), [1]))
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
        return "squared_tri(" + repr(self.normalization) + ")"

    def __init__(self, dim, normalization=None):
        self.dim = dim
        self.normalization = normalization

    def __call__(self, state):
        iu1 = np.triu_indices(len(state))
        a = np.outer(state, state)
        a = a * (2 - np.eye(len(state)))
        r = np.concatenate((a[iu1], [1]))
        if self.normalization is not None:
            assert self.normalization.shape == r.shape
            r /= self.normalization
        return r

    def expectation(self, x, Sigma):

        iu1 = np.triu_indices(len(x))
        a = np.outer(x, x)
        a += np.diag(Sigma)
        a = a * (2 - np.eye(len(x)))
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
        p = theta[:-1]
        l = 1 if len(p) == 1 else (-1 + np.sqrt(1 + 8 * len(p))) / 2
        iu = np.triu_indices(l)
        il = np.tril_indices(l)
        a = np.empty((l, l))
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


class corrupted_rbfs(object):

    def __init__(self, n_S, n_random, n_rbfs):
        self.n_S = n_S
        self.n_random = n_random
        self.n_rbfs = n_rbfs
        self.dim = n_rbfs + 1 + n_random
        self.rbf_mean = np.linspace(0, n_S, n_rbfs)
        self.rbf_sigma = n_S / (n_rbfs - 1.)
        self.constant=True
        self.offset = np.zeros(self.dim)
        self.scaling = np.ones(self.dim)

    def __call__(self, state):
        phi = np.empty(self.dim)
        phi[self.dim - 1] = 1.
        rbf = np.exp(-np.power(state - self.rbf_mean, 2)
                     / (self.rbf_sigma ** 2) / 2.)
        phi[:self.n_rbfs] = rbf / np.sum(rbf)
        phi[self.n_rbfs:-1] = np.random.normal(size=self.n_random)
        return (phi - self.offset) / self.scaling

    def normalization(self, samples):
        rbfs = np.exp(-np.power(samples - self.rbf_mean[None, :], 2)
                      / (self.rbf_sigma[None, :] ** 2) / 2.)
        rbfs = rbfs / np.sum(rbfs, axis=1)[:, None]
        #self.offset[0] = -1.
        self.offset = np.zeros(self.dim)
        self.scaling = np.ones(self.dim)
        self.offset[:self.n_rbfs] = np.mean(rbfs, axis=0)
        rbfs -= np.mean(rbfs, axis=0)
        self.scaling[:self.n_rbfs] = np.std(rbfs, axis=0)

"""    def expectation(self, state):
        phi = self(state)
        phi[1 + self.n_rbfs:] = 0.
        return (phi - self.offset) / self.scaling
"""

class spikes(object):

    def __init__(self, n, dim_S):

        self.dim = dim_S
        self.n = n

    def __call__(self, state):
        n = self.dim
        a = (n - 1.) / (self.n - 1)
        r = 1 - abs((state + 1 - np.linspace(1, n, self.n)) / a)
        r[r < 0] = 0
        return r


class eye(object):

    def __init__(self, dim_S):

        self.dim = dim_S

    def __call__(self, state):
        ret = np.zeros(self.dim)
        ret[int(state)] = 1
        return ret


class lin_random(object):
    def __repr__(self):
        return "lin_random(" + repr(self.A) + ")"

    def __init__(self, dim, dim_S, seed=1, constant=False):

        self.dim = dim
        self.dim_S = dim_S
        self.seed = seed
        np.random.seed(seed)
        if constant:
            self.A = np.ones((dim_S, dim))
            self.A[:, :-1] = np.random.rand(dim_S, dim - 1)
        else:
            self.A = np.random.rand(dim_S, dim)

    def __call__(self, state):
        return self.A[int(state)]


class squared_diag(object):

    def __repr__(self):
        return "squared_diag(" + repr(self.normalization) + ")"

    def __init__(self, dim, normalization=None):
        self.dim = dim
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
