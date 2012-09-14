import numpy as np
from joblib import Memory


memory = Memory(cachedir="./cache", verbose=0)
#memory = Memory(cachedir="/BS/latentCRF/nobackup/td", verbose=50)


class cached_property(object):
    '''A read-only @property that is only evaluated once. The value is cached
    on the object itself rather than the function or class; this should prevent
    memory leakage.'''
    def __init__(self, fget, doc=None):
        self.fget = fget
        self.__doc__ = doc or fget.__doc__
        self.__name__ = fget.__name__
        self.__module__ = fget.__module__

    def __get__(self, obj, cls):
        if obj is None:
            return self
        obj.__dict__[self.__name__] = result = self.fget(obj)
        return result


def multinomial_sample(n, p):
    """
    draw n random samples of integers with probabilies p
    """
    if len(p.shape) < 2:
        p.shape = (1, p.shape[0])
    p_accum = np.add.accumulate(p, axis=1)
    n_v, n_c = p_accum.shape
    rnd = np.random.rand(n, n_v, 1)
    m = rnd < p_accum.reshape(1, n_v, n_c)

    m2 = np.zeros(m.shape, dtype='bool')
    m2[:, :, 1:] = m[:, :, :-1]
    np.logical_xor(m, m2, out=m)
    ind_mat = np.arange(n_c, dtype='uint8').reshape(1, 1, n_c)
    mask = np.multiply(ind_mat, m, dtype="uint8")
    S = np.add.reduce(mask, 2, dtype='uint8').squeeze()
    return S


def apply_rowise(f, arr):
    if len(arr) <= 0:
        return arr

    n = len(f(arr[0]))
    res = np.empty((arr.shape[0], n))
    for i in xrange(arr.shape[0]):
        res[i, :] = f(arr[i])
    #import ipdb; ipdb.set_trace()
    return res


def normalize_phi_mean(phi, s_samples):
    Phi = apply_rowise(phi, s_samples)
    m = np.mean(Phi, axis=0)
    stdd = np.std(Phi, axis=0)
    phi_n = lambda x: (phi(x) - m) / stdd
    if hasattr(phi, "retransform"):
        phi_n.__dict__["retransform"] = lambda x: phi.retransform((x / stdd))
    return phi_n


def normalize_phi(phi, s_samples):
    Phi = apply_rowise(phi, s_samples)
    stdd = np.std(Phi, axis=0)
    phi_n = lambda x: (phi(x)) / stdd
    if hasattr(phi, "retransform"):
        phi_n.__dict__["retransform"] = lambda x: phi.retransform((x / stdd))
    return phi_n


class GrowingMat(object):

    def __init__(self, shape, capacity, grow_factor=4):
        self.data = np.zeros(capacity)
        self.shape = shape
        self.capacity = capacity
        self.grow_factor = grow_factor

    def expand(self, cols=None, rows=None, block=None):
        if cols is not None and rows is not None:
            cols = np.atleast_2d(cols)
            rows = np.atleast_2d(rows)
            new_shape = (
                self.shape[0] + rows.shape[0], self.shape[1] + cols.shape[1])
            new_capacity = (
                self.capacity[0] * self.grow_factor if new_shape[
                    0] > self.capacity[0] else self.capacity[0],
                self.capacity[1] * self.grow_factor if new_shape[1] > self.capacity[1] else self.capacity[1])
            if new_capacity != self.capacity:
                # grow array
                newdata = np.zeros(new_capacity)
                newdata[:self.shape[0], :self.shape[1]] = self.data
                self.capacity = new_capacity

            self.data[self.shape[0]:new_shape[0], :self.shape[1]] = rows
            self.data[:self.shape[0], self.shape[1]:new_shape[1]] = cols
            if block is not None:
                self.data[self.shape[0]:new_shape[0],
                          self.shape[1]:new_shape[1]] = block

            self.shape = new_shape
            #print "New shape", new_shape, self.shape, self.view.shape, #self.finalized.shape
        elif cols is not None:
            cols = np.atleast_2d(cols)
            new_shape = (self.shape[0], self.shape[1] + cols.shape[1])
            new_capacity = (self.capacity[0],
                            self.capacity[1] * self.grow_factor if new_shape[1] > self.capacity[1] else self.capacity[1])
            if new_capacity != self.capacity:
                # grow array
                newdata = np.zeros(new_capacity)
                newdata[:self.shape[0], :self.shape[1]] = self.data
                self.capacity = new_capacity

            self.data[:self.shape[0], self.shape[1]:new_shape[1]] = cols
            self.shape = new_shape

        elif rows is not None:

            rows = np.atleast_2d(rows)
            new_shape = (self.shape[0] + rows.shape[0], self.shape[1])
            new_capacity = (
                self.capacity[0] * self.grow_factor if new_shape[
                    0] > self.capacity[0] else self.capacity[0],
                self.capacity[1])
            if new_capacity != self.capacity:
                # grow array
                newdata = np.zeros(new_capacity)
                newdata[:self.shape[0], :self.shape[1]] = self.data
                self.capacity = new_capacity

            self.data[self.shape[0]:new_shape[0], :self.shape[1]] = rows
            self.shape = new_shape

    @property
    def view(self):
        return self.data[:self.shape[0], :self.shape[1]]

    @view.setter
    def view(self, d):
        self.data[:self.shape[0], :self.shape[1]] = d

    @property
    def finalized(self):
        data = self.view
        return np.reshape(data, newshape=self.shape)


class GrowingVector(object):

    def __init__(self, size, capacity=100, grow_factor=4):
        self.data = np.zeros(capacity)
        self.size = size
        self.capacity = capacity
        self.grow_factor = grow_factor

    def expand(self, rows):

        rows = np.atleast_1d(rows)
        new_size = self.size + rows.shape[0]
        new_capacity = self.capacity * \
            self.grow_factor if new_size > self.capacity else self.capacity
        if new_capacity != self.capacity:
            # grow array
            newdata = np.zeros(new_capacity)
            newdata[:self.size] = self.data
            self.capacity = new_capacity

        self.data[self.size:new_size] = rows
        self.size = new_size

    @property
    def view(self):
        return self.data[:self.size]

    @view.setter
    def view(self, d):
        self.data[:self.size] = d

    @property
    def finalized(self):
        data = self.view
        return np.reshape(data, newshape=(self.size))
