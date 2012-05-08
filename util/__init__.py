import numpy as np

def apply_rowise(f, arr):
    if len(arr) <= 0:
        return arr
    
    n = len(f(arr[0]))
    res = np.empty((arr.shape[0],n))
    for i in xrange(arr.shape[0]):
        res[i,:] = f(arr[i])
    #import ipdb; ipdb.set_trace()
    return res