import numpy as np
cimport numpy as np
cimport cython
cdef extern from "math.h":
    double cos(double)
    double sin(double)

def ode_wrap(*args):
    return ode(*args)

def ode_jac_wrap(*args):
    return ode_jac_wrap(*args)

@cython.boundscheck(False)
#@cython.wraparound(False)
def ode(np.ndarray[np.double_t, ndim=1] s, double t, double a,
        double m, double l, double M, double b):
    cdef double g, c3, s3
    s3 = sin(s[3])
    c3 = cos(s[3])
    g = 9.81
    cdef np.ndarray[np.double_t, ndim=1] ds = np.zeros(4)
    ds[0] = s[1]
    ds[1] = (2 * m * l * s[2] ** 2 * s3 + 3 * m * g * s3 * c3 + 4 * a - 4 * b * s[1])\
        / (4 * (M + m) - 3 * m * c3 ** 2)
    ds[2] = (-3 * m * l * s[2] ** 2 * s3*c3 - 6 * (M + m) * g * s3 - 6 * (a - b * s[1]) * c3)\
        / (4 * l * (m + M) - 3 * m * l * c3 ** 2)
    ds[3] = s[2]
    return ds

#@cython.boundscheck(False)
#@cython.wraparound(False)
def ode_jac(np.ndarray[np.double_t, ndim=1] s, double t, double a, double m,
        double l, double M, double b):
    cdef double g, c3, s3
    g = 9.81
    c3 = cos(s[3])
    s3 = sin(s[3])
    c = 4 * (M + m) - 3 * m * c3 ** 2
    cdef np.ndarray[np.double_t, ndim=2] jac = np.zeros(4, 4)
    jac[0, 1] = 1.
    jac[3, 2] = 1.
    jac[1, 1] = -4 * b / c
    jac[1, 2] = 4 * m * l * s[2] / c
    jac[1, 3] = m / c * (2 * l * s[2] ** 2 * c3 + 3 * g * (1 - 2 * s3 ** 2)) \
        - 6 * m * s3 * c3 / c / c * (2 * m * l * s[2] ** 2 * s3 + 3 * m * g * s3 * c3 + 4 * a -
                                        4 * b * s[1])
    jac[2, 1] = 6 * b * c3 / c / l
    jac[2, 2] = -6 * m * l * s[2] * c3 * s3 / c
    jac[2, 3] = (3 * m * l * s[2] ** 2 * (2 * s3 - 1) - 6 * (M + m) * g * s3 + 6 * (a - b * s[1]) * s3) / l / l / c \
        + 3 / l / l / c / c * (m * l * s[2] ** 2 * s3 * c3 - 2 * (
                                M + m) * g * s3 + 2*(a - b * s[1]) * c3) * 6 * m * c3 * s3
    return jac

@cython.boundscheck(False)
def squared_tri(np.ndarray[np.double_t, ndim=1] s, int n,
                np.ndarray[np.double_t, ndim=1] normalization):
    cdef unsigned int n_feat, u, i, j
    n_feat = ((n+1)*n / 2)+1
    cdef np.ndarray[np.double_t, ndim=1] f = np.empty(n_feat)
    u = 0
    for i in range(n):
        for j in range(i,n):
            f[u] = s[i]*s[j]
            if i != j:
                f[u] *= 2
            f[u] /= normalization[u]
            u+=1
    f[u] = 1.
    return f
