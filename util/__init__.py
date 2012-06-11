import numpy as np

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
