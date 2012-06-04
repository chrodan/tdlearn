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
    
    
def normalize_phi_mean(phi, s_samples):
    Phi = apply_rowise(phi, s_samples)
    m = np.mean(Phi, axis=0)
    stdd = np.std(Phi, axis=0)
    phi_n = lambda x: (phi(x)-m)/stdd
    if hasattr(phi, "retransform"):
        phi_n.__dict__["retransform"] = lambda x: phi.retransform((x / stdd))
    return phi_n
def normalize_phi(phi, s_samples):
    Phi = apply_rowise(phi, s_samples)
    stdd = np.std(Phi, axis=0)
    phi_n = lambda x: (phi(x))/stdd
    if hasattr(phi, "retransform"):
        phi_n.__dict__["retransform"] = lambda x: phi.retransform((x / stdd))
    return phi_n