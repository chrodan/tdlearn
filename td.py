        # -*- coding: utf-8 -*-
"""
Temporal Difference Learning for finite MDPs

Created on Sun Dec 11 01:06:00 2011

@author: Christoph Dann <cdann@cdann.de>
"""

import numpy as np
import itertools
import logging
import copy
import time
import util

#logging.basicConfig(level=logging.DEBUG)


class ValueFunctionPredictor(object):
    """
        predicts the value function of a MDP for a given policy from given
        samples
    """

    def __init__(self, gamma=1, **kwargs):
        self.gamma = gamma
        self.time = 0
        if not hasattr(self, "init_vals"):
            self.init_vals = {}

    def update_V(self, s0, s1, r, V, **kwargs):
        raise NotImplementedError("Each predictor has to implement this")

    def reset(self):
        self.time = 0
        self.reset_trace()
        for k, v in self.init_vals.items():
            self.__setattr__(k, copy.copy(v))

    def reset_trace(self):
        if hasattr(self, "z"):
            if "z" in self.init_vals:
                self.z = self.init_vals["z"]
            else:
                del self.z

    def _assert_iterator(self, p):
        try:
            return iter(p)
        except TypeError:
            return itertools.repeat(p)

    def _tic(self):
        self._start = time.clock()

    def _toc(self):
        self.time += (time.clock() - self._start)


class LinearValueFunctionPredictor(ValueFunctionPredictor):
    """
        base class for value function predictors that predict V as a linear
        approximation, i.e.:
            V(x) = theta * phi(x)
    """
    def __init__(self, phi, theta0=None, **kwargs):

        ValueFunctionPredictor.__init__(self, **kwargs)

        self.phi = phi
        if theta0 is None:
            self.init_vals['theta'] = np.array([0])
        else:
            self.init_vals['theta'] = theta0

    def V(self, theta=None):
        """
        returns a the approximate value function for the given parameter
        """
        if theta is None:
            if not hasattr(self, "theta"):
                raise Exception("no theta available, has to be specified"
                                " by parameter")
            theta = self.theta

        return lambda x: np.dot(theta, self.phi(x))

    def _compute_detTD_updates(self, task):
        T = np.matrix(task.mdp.policy_P(task.target_policy))
        Phi = task.Phi
        Phi = np.matrix(Phi)
        D = np.diag(task.beh_mu)
        F = np.array(Phi.T * np.matrix(D) * (self.gamma * T) * Phi)
        Cmat = np.array(Phi.T * np.matrix(D) * Phi)
        R = task.mdp.P * task.mdp.r * task.target_policy[:, :, np.newaxis]
        R = np.sum(R, axis=1)  # sum over all A
        R = np.sum(R, axis=1)  # sum over all S'
        b = np.array(Phi) * R[:, np.newaxis] * task.beh_mu[:, np.newaxis]
        b = np.array(np.sum(b, axis=0)).flatten()
        return F, Cmat, b


class LambdaValueFunctionPredictor(ValueFunctionPredictor):
    """
        base class for predictors that have the lambda parameter as a tradeoff
        parameter for bootstrapping and sampling
    """
    def __init__(self, lam, z0=None, **kwargs):
        """
            z0: optional initial value for the eligibility trace
        """
        ValueFunctionPredictor.__init__(self, **kwargs)
        self.lam = lam
        if z0 is not None:
            self.init_vals["z"] = z0


class OffPolicyValueFunctionPredictor(ValueFunctionPredictor):
    """
        base class for value function predictors for a MDP given target and
        behaviour policy
    """

    def update_V_offpolicy(
        self, s0, s1, r, a, beh_pi, target_pi, f0=None, f1=None, theta=None,
            **kwargs):
        """
        off policy training version for transition (s0, a, s1) with reward r
        which was sampled by following the behaviour policy beh_pi.
        The parameters are learned for the target policy target_pi

         beh_pi, target_pi: S x A -> R
                numpy array of shape (n_s, n_a)
                *_pi(s,a) is the probability of taking action a in state s
        """
        rho = target_pi.p(s0, a) / beh_pi.p(s0, a)
        kwargs["rho"] = rho
        if not np.isfinite(rho):
            import ipdb
            ipdb.set_trace()
        return self.update_V(s0, s1, r, f0=f0, f1=f1, theta=theta, **kwargs)


class GTDBase(LinearValueFunctionPredictor, OffPolicyValueFunctionPredictor):
    """ Base class for GTD, GTD2 and TDC algorithm """

    def __init__(self, alpha, beta, **kwargs):
        """
            alpha:  step size. This can either be a constant number or
                    an iterable object providing step sizes
            beta:   step size for weights w. This can either be a constant
                    number or an iterable object providing step sizes
            gamma: discount factor
        """
        LinearValueFunctionPredictor.__init__(self, **kwargs)
        OffPolicyValueFunctionPredictor.__init__(self, **kwargs)

        self.init_vals['alpha'] = alpha
        self.init_vals['beta'] = beta

        self.reset()

    def clone(self):
        o = self.__class__(self.init_vals['alpha'], self.init_vals[
                           'beta'], gamma=self.gamma, phi=self.phi)
        return o

    def __getstate__(self):
        res = self.__dict__
        for n in ["alpha", "beta"]:
            if isinstance(res[n], itertools.repeat):
                res[n] = res[n].next()
        return res

    def __setstate__(self, state):
        self.__dict__ = state
        self.alpha = self._assert_iterator(self.init_vals['alpha'])
        self.beta = self._assert_iterator(self.init_vals['beta'])

    def reset(self):
        LinearValueFunctionPredictor.reset(self)
        self.alpha = self._assert_iterator(self.init_vals['alpha'])
        self.beta = self._assert_iterator(self.init_vals['beta'])
        self.w = np.zeros_like(self.init_vals['theta'])
        if hasattr(self, "A"):
            del(self.A)
        if hasattr(self, "b"):
            del(self.b)

        if hasattr(self, "F"):
            del(self.F)
        if hasattr(self, "Cmat"):
            del(self.Cmat)

    def init_deterministic(self, task):
        self.F, self.Cmat, self.b = self._compute_detTD_updates(task)
        self.A = np.array(self.F - self.Cmat)


class GTD(GTDBase):
    """
    GTD algorithm with linear function approximation
    for details see

    Maei, H. R. (2011). Gradient Temporal-Difference Learning Algorithms.
    (p. 36)
    """
    def update_V(self, s0, s1, r, f0=None, f1=None, rho=1, theta=None, **kwargs):
        """
            rho: weight for this sample in case of off-policy learning
        """
        w = self.w
        if theta is None:
            theta = self.theta
        if f0 is None or f1 is None:
            f0 = self.phi(s0)
            f1 = self.phi(s1)

        self._tic()

        delta = r + self.gamma * np.dot(theta, f1) - np.dot(theta, f0)
        a = np.dot(f0, w)

        w += self.beta.next() * rho * (delta * f0 - w)
        theta += self.alpha.next() * rho * (f0 - self.gamma * f1) * a

        self.w = w
        self.theta = theta

        self._toc()
        return theta

    def deterministic_update(self, theta=None):
        w = self.w
        if theta is None:
            theta = self.theta

        w_d = w + self.beta.next() * (np.dot(self.A, theta) - w + self.b)
        theta_d = theta + self.alpha.next() * (- np.dot(self.A.T, w) + self.b)
        self.theta = theta_d
        self.w = w_d
        return self.theta


class GTD2(GTDBase):
    """
    GTD2 algorithm with linear function approximation
    for details see

    Maei, H. R. (2011). Gradient Temporal-Difference Learning Algorithms.
    (p. 38)
    """

    def update_V(self, s0, s1, r, f0=None, f1=None, rho=1, theta=None, **kwargs):
        """
            rho: weight for this sample in case of off-policy learning
        """
        w = self.w
        if theta is None:
            theta = self.theta
        if f0 is None or f1 is None:
            f0 = self.phi(s0)
            f1 = self.phi(s1)

        self._tic()

        delta = r + self.gamma * np.dot(theta, f1) - np.dot(theta, f0)
        a = np.dot(f0, w)

        w += self.beta.next() * (rho * delta - a) * f0
        theta += self.alpha.next() * rho * a * (f0 - self.gamma * f1)

        self.w = w
        self.theta = theta
        self._toc()
        return theta

    def deterministic_update(self, theta=None):
        w = self.w
        if theta is None:
            theta = self.theta

        w_d = w + self.beta.next(
        ) * (np.dot(self.A, theta) - np.dot(self.Cmat, w) + self.b)
        theta_d = theta + self.alpha.next() * (- np.dot(self.A.T, w) + self.b)
        self.theta = theta_d
        self.w = w_d
        return self.theta


class TDC(GTDBase):
    """
    TDC algorithm with linear function approximation
    for details see

    Maei, H. R. (2011). Gradient Temporal-Difference Learning Algorithms.
    (p. 38)
    """

    def update_V(self, s0, s1, r, f0=None, f1=None, rho=1, theta=None, **kwargs):
        """
            rho: weight for this sample in case of off-policy learning
        """
        w = self.w
        if theta is None:
            theta = self.theta
        if f0 is None or f1 is None:
            f0 = self.phi(s0)
            f1 = self.phi(s1)

        self._tic()
        delta = r + self.gamma * np.dot(theta, f1) - np.dot(theta, f0)
        a = np.dot(f0, w)

        w += self.beta.next() * (rho * delta - a) * f0
        theta += self.alpha.next() * rho * (delta * f0 - self.gamma * f1 * a)
        self.w = w
        self.theta = theta
        self._toc()
        return theta

    def deterministic_update(self, theta=None):
        w = self.w
        if theta is None:
            theta = self.theta

        w_d = w + self.beta.next(
        ) * (np.dot(self.A, theta) - np.dot(self.Cmat, w) + self.b)
        theta_d = theta + self.alpha.next(
        ) * (np.dot(self.A, theta) - np.dot(self.F.T, w) + self.b)
        self.theta = theta_d
        self.w = w_d
        return self.theta


class TDCLambda(GTDBase, LambdaValueFunctionPredictor):
    """
    TDC algorithm with linear function approximation
    for details see

    Maei, H. R. (2011). Gradient Temporal-Difference Learning Algorithms.
    (p. 74)
    """

    def __init__(self, **kwargs):

        GTDBase.__init__(self, **kwargs)
        LambdaValueFunctionPredictor.__init__(self, **kwargs)

        self.reset()

    def update_V(self, s0, s1, r, f0=None, f1=None, rho=1, theta=None, **kwargs):
        """
            rho: weight for this sample in case of off-policy learning
        """
        w = self.w
        if theta is None:
            theta = self.theta

        if f0 is None or f1 is None:
            f0 = self.phi(s0)
            f1 = self.phi(s1)

        if not hasattr(self, "z"):
            self.z = np.zeros_like(f0)
        self._tic()
        self.z = rho * (f0 + self.gamma * self.lam * self.z)

        delta = r + self.gamma * np.dot(theta, f1) - np.dot(theta, f0)
        a = np.dot(f0, w)

        theta += self.alpha.next() * (delta * self.z - self.gamma *
                                      (1 - self.lam) * np.dot(self.z, w) * f1)
        w += self.beta.next() * (delta * self.z - a * f0)
        self.w = w
        self.theta = theta
        self._toc()
        return theta


class GeriTDC(TDC):
    """
    the TDC algorithm except that the pseudo-stationary guess for off-policy estimation is computed differently
    """

    def update_V(self, s0, s1, r, f0=None, f1=None, rho=1, theta=None, **kwargs):
        """
            rho: weight for this sample in case of off-policy learning
        """
        w = self.w
        if theta is None:
            theta = self.theta

        if f0 is None or f1 is None:
            f0 = self.phi(s0)
            f1 = self.phi(s1)

        self._tic()
        delta = r + self.gamma * np.dot(theta, f1) - np.dot(theta, f0)
        a = np.dot(f0, w)

        w += self.beta.next() * rho * (delta - a) * f0
        theta += self.alpha.next() * rho * (delta * f0 - self.gamma * f1 * a)
        self.w = w
        self.theta = theta
        self._toc()
        return theta


class KTD(LinearValueFunctionPredictor):
    """ Kalman Temporal Difference Learning

        for details see Geist, M. (2010).
            Kalman temporal differences. Journal of artificial intelligence research, 39, 483-532.
            Retrieved from http://www.aaai.org/Papers/JAIR/Vol39/JAIR-3911.pdf
            Algorithm 5 (XKTD-V)
    """
    def __init__(self, kappa=1., theta_noise=0.001, eta=None, P_init=10, reward_noise=0.001, **kwargs):
        LinearValueFunctionPredictor.__init__(self, **kwargs)
        self.kappa = kappa
        self.P_init = P_init
        self.reward_noise = reward_noise
        self.eta = eta
        if eta is not None and theta_noise is not None:
            print "Warning, eta and theta_noise are complementary"
        self.theta_noise = theta_noise
        self.reset()

    def reset(self):
        LinearValueFunctionPredictor.reset(self)
        self.p = len(self.theta)
        if self.theta_noise is not None:
            self.P_vi = np.eye(self.p) * self.theta_noise
        self.P = np.eye(self.p + 2) * self.P_init
        self.x = np.zeros(self.p + 2)
        self.x[:self.p] = self.theta
        self.F = np.eye(self.p + 2)
        self.F[-2:, -2:] = np.array([[0., 0.], [1., 0.]])

    def sample_sigma_points(self, mean, variance):
        n = len(mean)
        X = np.empty((2 * n + 1, n))
        X[:, :] = mean[None, :]
        C = np.linalg.cholesky((self.kappa + n) * variance)
        for j in range(n):
            X[j + 1, :] += C[:, j]
            X[j + n + 1, :] -= C[:, j]
        W = np.ones(2 * n + 1) * (1. / 2 / (self.kappa + n))
        W[0] = (self.kappa / (self.kappa + n))
        return X, W

    def update_V(self, s0, s1, r, f0=None, f1=None, theta=None, rho=1, **kwargs):
        if f0 is None or f1 is None:
            f0 = self.phi(s0)
            f1 = self.phi(s1)
        self._tic()
        if theta is not None:
            print "Warning, setting theta by hand is not valid"

        # Prediction Step
        xn = np.dot(self.F, self.x)
        Pn = np.dot(self.F, np.dot(self.P, self.F.T))
        if self.eta is not None:
            self.P_vi = self.eta * self.P[:-2, :-2]
        Pn[:-2, :-2] += self.P_vi
        Pn[-2:, -2:] += np.array([[1., -self.gamma], [-self.gamma,
                                 self.gamma ** 2]]) * self.reward_noise

        # Compute Sigma Points
        X, W = self.sample_sigma_points(xn, Pn)
        R = (np.dot(f0, X[:, :-2].T) - self.gamma * np.dot(f1, X[:,
             :-2].T) + X[:, -1].T).flatten()

        # Compute statistics of interest
        rhat = (W * R).sum()
        Pxr = ((W * (R - rhat))[:, None] * (X - xn)).sum(axis=0)
        Pr = max((W * (R - rhat) * (R - rhat)).sum(), 10e-5)  # ensure a minimum amount of noise to avoid numerical instabilities

        # Correction Step
        K = Pxr * (1. / Pr)
        #try:
        #    np.linalg.cholesky(Pn - np.outer(K,K)*Pr)
        #except Exception:
        #    import ipdb
        #    ipdb.set_trace()

        self.P = Pn - np.outer(K, K) * Pr

        self.x = xn + K * (r - rhat)
        self.theta = self.x[:-2]
        self._toc()


class GPTDP(LinearValueFunctionPredictor):
    """
    Parametric GPTD
    for details see
     Engel, Y. (2005). Algorithms and Representations for Reinforcement Learning. Hebrew University.
    Algorithm 18
    """
    def __init__(self, sigma=0.05, **kwargs):
        LinearValueFunctionPredictor.__init__(self, **kwargs)
        self.sigma = sigma
        self.init_vals["sinv"] = 0
        self.init_vals["d"] = 0

        self.reset()

    def reset(self):
        LinearValueFunctionPredictor.reset(self)
        n = len(self.theta)
        self.p = np.zeros(n)
        self.P = np.eye(n)

    def update_V(self, s0, s1, r, f0=None, f1=None, theta=None, rho=1, **kwargs):
        if f0 is None or f1 is None:
            f0 = self.phi(s0)
            f1 = self.phi(s1)
        self._tic()
        if theta is not None:
            print "Warning, setting theta by hand is not valid"

        df = f0 - self.gamma * f1
        c = self.gamma * self.sinv * (self.sigma ** 2)
        a = c * self.p
        self.p = a + np.dot(self.P, df)
        self.d = c * self.d + r - np.inner(df, self.theta)
        s = self.sigma ** 2 + self.gamma ** 2 * self.sigma ** 2 - self.sigma **2 * self.gamma * c + \
            np.inner(a + self.p, df)
        self.sinv = 1. / s
        self.theta += self.sinv * self.d * self.p
        self.P -= self.sinv * np.outer(self.p, self.p)
        self._toc()


class GPTD(ValueFunctionPredictor):
    """
        Gaussian Process Temporal Difference Learning implementation
        with online sparsification
        for details see
        Engel, Y., Mannor, S., & Meir, R. (2005). Reinforcement learning with Gaussian processes.
         Proceedings of the 22nd international conference on Machine learning - ICML  ’05,
         201-208. New York, New York, USA: ACM Press. doi:10.1145/1102351.1102377
         Table 1
         and Engel's PhD thesis
    """

    def __init__(self, phi, nu=1, sigma0=0.05, **kwargs):
        """
            kernel: a mercer kernel function as a python function
                that takes 2 arguments, i.e. gauss kernel
            nu: threshold for sparsification test
        """
        ValueFunctionPredictor.__init__(self, **kwargs)
        self.nu = nu
        self.sigma0 = sigma0
        self.kernel = np.frompyfunc(lambda x, y: np.dot(phi(x), phi(y)), 2, 1)
        self.init_vals["D"] = []
        self.init_vals["C"] = util.GrowingMat((0, 1), (100, 100))
        self.init_vals["c"] = util.GrowingVector(0)
        self.init_vals["alpha"] = util.GrowingVector(0)
        self.init_vals["d"] = 0
        self.init_vals["sinv"] = 0
        self.init_vals["Kinv"] = util.GrowingMat((0, 1), (100, 100))
        self.reset()

    def V(self, x):
        return float(np.inner(self.kernel(self.D, x), self.alpha.view))

    def update_V(self, s0, s1, r, f0=None, f1=None, theta=None, rho=1, **kwargs):
        """
            rho: weight for this sample in case of off-policy learning
        """

        self._tic()
        # first observation?
        if len(self.D) == 0:
            self.D.append(s0)
            self.Kinv.expand(rows=np.array([[1. / self.kernel(s0, s0)]]))
            self.C.expand(rows=np.array(0))
            self.a = np.array(1)
            self.c.expand(rows=np.array(0))
            self.alpha.expand(rows=np.array(0))

        k = self.kernel(self.D, s1)
        a = np.array(np.dot(self.Kinv.view, k)).flatten()
        ktt = float(self.kernel(s1, s1))
        dk = self.kernel(self.D, s0) - self.gamma * self.kernel(self.D, s1)
        delta = ktt - float(np.inner(k.T, a))
        self.d = self.d * self.sinv * self.gamma * self.sigma0 ** 2 + \
            r - float(np.inner(dk, self.alpha.view.flatten()))
        # sparsification test
        if delta > self.nu:
            #import ipdb; ipdb.set_trace()
            dk2 = np.array((self.kernel(self.D, s0) - 2 *
                           self.gamma * self.kernel(self.D, s1))).flatten()
            self.D.append(s1)
            # update K^-1
            self.Kinv.view = delta * self.Kinv.view + np.outer(a, a)
            self.Kinv.expand(cols=-a.reshape(
                -1, 1), rows=-a.reshape(1, -1), block=np.array([[1]]))
            self.Kinv.view /= delta
            #print "inverted Kernel matrix:", self.Kinv.view

            a = np.zeros(self.Kinv.shape[0])
            a[-1] = 1

            hbar = np.zeros_like(a)
            hbar[:-1] = self.a
            hbar[-1] = - self.gamma

            dktt = float(np.inner(self.a, dk2)) + self.gamma ** 2 * ktt

            cm1 = self.c.view.copy().flatten()
            self.c.view = self.c.view.flatten() * self.sinv * self.gamma * self.sigma0 ** 2 + self.a - np.dot(self.C.view, dk)
            self.c.expand(rows=np.array(- self.gamma))

            s = (1 + self.gamma ** 2) * self.sigma0 ** 2 - self.sinv * self.gamma ** 2 * self.sigma0 ** 4 + dktt - \
                np.dot(dk, np.dot(self.C.view, dk)) + 2 * self.sinv * \
                self.gamma * self.sigma0 ** 2 * np.dot(cm1, dk)

            self.alpha.expand(rows=np.array([[0]]))

            self.C.expand(rows=np.zeros(
                (1, self.C.shape[1])), cols=np.zeros((self.C.shape[0], 1)))

        else:
            self.hbar = self.a - self.gamma * a
            #dktt = np.dot(hbar, dk)

            cm1 = self.c.view.copy()

            self.c.view = self.c.view.flatten() * self.sinv * self.gamma * self.sigma0 ** 2 + self.hbar - np.dot(self.C.view, dk)

            s = (1 + self.gamma ** 2) * self.sigma0 ** 2 - self.sinv * self.gamma ** 2 * self.sigma0 ** 4 + \
                np.dot(dk, self.c.view +
                       self.gamma * self.sigma0 ** 2 * self.sinv * cm1)

        self.sinv = 1 / s
        self.alpha.view += self.sinv * self.d * self.c.view
        self.C.view += self.sinv * np.outer(self.c.view, self.c.view)
        self.a = a

        self._toc()


class LSTDLambda(OffPolicyValueFunctionPredictor, LambdaValueFunctionPredictor, LinearValueFunctionPredictor):
    """
        Implementation of Least Squared Temporal Difference Learning
         LSTD(\lambda) with linear function approximation, also works in the
         off-policy case and uses eligibility traces

        for details see Yu, H. (2010). Least Squares Temporal Difference Methods :
         An Analysis Under General Conditions. (8)+(9)+(10)
    """

    def __init__(self, init_theta=0., **kwargs):
        """
            lam: lambda in [0, 1] specifying the tradeoff between bootstrapping
                    and MC sampling
            gamma:  discount factor
        """
        LinearValueFunctionPredictor.__init__(self, **kwargs)
        OffPolicyValueFunctionPredictor.__init__(self, **kwargs)
        LambdaValueFunctionPredictor.__init__(self, **kwargs)
        self.init_theta = init_theta
        #import ipdb; ipdb.set_trace()
        #self.init_vals["C"] = np.zeros(len(self.init_vals["theta"]))
        self.reset()

    def clone(self):
        o = self.__class__(lam=self.lam, gamma=self.gamma, phi=self.phi)
        return o

    def reset(self):
        self.reset_trace()

        self.init_vals["C"] = self.init_theta * np.eye(len(
            self.init_vals["theta"]))
#        self.init_vals["k2"] = self.init_theta * np.eye(len(
#            self.init_vals["theta"]))
#        self.init_vals["k1"] = self.init_theta * np.eye(len(
#            self.init_vals["theta"]))
        self.init_vals["b"] = -self.init_vals["theta"] * self.init_theta
        for k, v in self.init_vals.items():
            if k == "theta":
                continue
            self.__setattr__(k, copy.copy(v))
        self.t = 0

    @property
    def theta(self):
        return -np.dot(np.linalg.pinv(self.C), self.b)

    @theta.setter
    def theta_set(self, val):
        pass

    def update_V(self, s0, s1, r, f0=None, f1=None, theta=None, rho=1, **kwargs):
        """
            rho: weight for this sample in case of off-policy learning
        """
        if f0 is None or f1 is None:
            f0 = self.phi(s0)
            f1 = self.phi(s1)
        self._tic()
        if theta is None:
            theta = self.theta
        if not hasattr(self, "z"):
            self.z = np.zeros_like(f0)
        self.z = self.gamma * self.lam * rho * self.z + f0
        alpha = 1. / (1 + self.t + 1)
        self.t += 1
        self.b = (1 - alpha) * self.b + alpha * self.z * rho * r
        self.C = (1 - alpha) * self.C + alpha * np.outer(self.z,
                                                         self.gamma * rho * f1 - f0)
 #       self.k1 = (1 - alpha) * self.k1 + alpha * np.outer(f0, rho * f0)
 #       self.k2 = (1 - alpha) * self.k2 + alpha * np.outer(f0, f0)
        self._toc()
        #print self.C
        #print self.b


class LSTDLambdaJP(LSTDLambda):
    """
        Implementation of Least Squared Temporal Difference Learning
         LSTD(\lambda) with linear function approximation, also works in the
         off-policy case and uses eligibility traces

        for details see Yu, H. (2010). Least Squares Temporal Difference Methods :
         An Analysis Under General Conditions. (8)+(9)
         Important difference: The update of C is different!
    """

    def update_V(self, s0, s1, r, f0=None, f1=None, theta=None, rho=1, **kwargs):
        """
            rho: weight for this sample in case of off-policy learning
        """
        if f0 is None or f1 is None:
            f0 = self.phi(s0)
            f1 = self.phi(s1)
        self._tic()
        if theta is None:
            theta = self.theta
        if not hasattr(self, "z"):
            self.z = np.zeros_like(f0)
        self.z = self.gamma * self.lam * rho * self.z + f0
        alpha = 1. / (1 + self.t + 1)
        self.t += 1
        self.b = (1 - alpha) * self.b + alpha * self.z * rho * r
        self.C = (1 - alpha) * self.C + alpha * rho * np.outer(
            self.z, self.gamma * f1 - f0)

        self._toc()
        #print self.C
        #print self.b


class RecursiveLSTDLambdaJP(OffPolicyValueFunctionPredictor, LambdaValueFunctionPredictor, LinearValueFunctionPredictor):
    """
        recursive Implementation of Least Squared Temporal Difference Learning
         LSTD(\lambda) with linear function approximation, also works in the
         off-policy case and uses eligibility traces

        for details see Scherrer, B., & Geist, M. (EWRL 2011). :
            Recursive Least-Squares Learning with Eligibility Traces.
            Algorithm 1
    """

    def __init__(self, eps=100, **kwargs):
        """
            lam: lambda in [0, 1] specifying the tradeoff between bootstrapping
                    and MC sampling
            gamma:  discount factor
        """
        LinearValueFunctionPredictor.__init__(self, **kwargs)
        OffPolicyValueFunctionPredictor.__init__(self, **kwargs)
        LambdaValueFunctionPredictor.__init__(self, **kwargs)
        self.eps = eps
        #import ipdb; ipdb.set_trace()
        self.init_vals["C"] = np.eye(len(self.init_vals["theta"])) * eps
        self.reset()

    def clone(self):
        o = self.__class__(
            eps=self.eps, lam=self.lam, gamma=self.gamma, phi=self.phi)
        return o

    def reset(self):
        self.reset_trace()
        self.init_vals["C"] = np.eye(len(self.init_vals["theta"])) * self.eps
        for k, v in self.init_vals.items():
            self.__setattr__(k, copy.copy(v))

    def update_V(self, s0, s1, r, f0=None, f1=None, theta=None, rho=1, **kwargs):
        """
            rho: weight for this sample in case of off-policy learning
        """
        if f0 is None or f1 is None:
            f0 = self.phi(s0)
            f1 = self.phi(s1)
        self._tic()
        if theta is None:
            theta = self.theta
        if not hasattr(self, "z"):
            self.z = f0

        L = np.dot(self.C, self.z)
        deltaf = f0 - self.gamma * f1
        K = rho * L / (1 + rho * np.dot(deltaf, L))

        theta += K * (r - np.dot(deltaf, theta))
        self.C -= np.outer(K, np.dot(deltaf, self.C))
        self.z = self.gamma * self.lam * rho * self.z + f1
        self.theta = theta
        self._toc()


class RecursiveLSPELambda(OffPolicyValueFunctionPredictor, LambdaValueFunctionPredictor, LinearValueFunctionPredictor):
    """
        recursive Implementation of Least Squared Policy Evaluation
         LSPE(\lambda) with linear function approximation, also works in the
         off-policy case and uses eligibility traces

        for details see Scherrer, B., & Geist, M. (EWRL 2011). :
            Recursive Least-Squares Learning with Eligibility Traces.
            Algorithm 2
    """

    def __init__(self, alpha=1, eps=100, **kwargs):
        """
            lam: lambda in [0, 1] specifying the tradeoff between bootstrapping
                    and MC sampling
            gamma:  discount factor
        """
        LinearValueFunctionPredictor.__init__(self, **kwargs)
        OffPolicyValueFunctionPredictor.__init__(self, **kwargs)
        LambdaValueFunctionPredictor.__init__(self, **kwargs)
        self.eps = eps
        #import ipdb; ipdb.set_trace()
        n = len(self.init_vals["theta"])
        self.init_vals["A"] = np.zeros((n, n))
        self.init_vals["b"] = np.zeros(n)
        self.init_vals["N"] = np.eye(n) * eps
        self.init_vals['alpha'] = alpha
        self.init_vals["i"] = 0
        self.reset()

    def clone(self):
        o = self.__class__(
            eps=self.eps, lam=self.lam, gamma=self.gamma, phi=self.phi)
        return o

    def reset(self):
        self.reset_trace()
        n = len(self.init_vals["theta"])
        self.init_vals["A"] = np.zeros((n, n))
        self.init_vals["b"] = np.zeros(n)
        self.init_vals["N"] = np.eye(n) * self.eps
        for k, v in self.init_vals.items():
            self.__setattr__(k, copy.copy(v))
        self.alpha = self._assert_iterator(self.init_vals['alpha'])

    def update_V(self, s0, s1, r, f0=None, f1=None, theta=None, rho=1, **kwargs):
        """
            rho: weight for this sample in case of off-policy learning
        """
        if f0 is None or f1 is None:
            f0 = self.phi(s0)
            f1 = self.phi(s1)
        self._tic()
        if theta is None:
            theta = self.theta
        if not hasattr(self, "z"):
            self.z = f0

        L = np.dot(f0, self.N)
        self.N -= np.outer(np.dot(self.N, f0), L) / (1 + np.dot(L, f0))
        deltaf = f0 - self.gamma * rho * f1
        self.A += np.outer(self.z, deltaf)

        self.b += rho * self.z * r
        self.i += 1
        theta += self.alpha.next() * np.dot(self.N, (self.b - np.dot(self.A, theta)))
        self.theta = theta
        self.z = self.gamma * self.lam * rho * self.z + f1
        self._toc()


class RecursiveLSTDLambda(OffPolicyValueFunctionPredictor, LambdaValueFunctionPredictor, LinearValueFunctionPredictor):
    """
        recursive Implementation of Least Squared Temporal Difference Learning
         LSTD(\lambda) with linear function approximation, also works in the
         off-policy case and uses eligibility traces

        for details see Scherrer, B., & Geist, M. (EWRL 2011). :
            Recursive Least-Squares Learning with Eligibility Traces.
            Algorithm 1
    """

    def __init__(self, eps=100, **kwargs):
        """
            lam: lambda in [0, 1] specifying the tradeoff between bootstrapping
                    and MC sampling
            gamma:  discount factor
        """
        LinearValueFunctionPredictor.__init__(self, **kwargs)
        OffPolicyValueFunctionPredictor.__init__(self, **kwargs)
        LambdaValueFunctionPredictor.__init__(self, **kwargs)
        self.eps = eps
        #import ipdb; ipdb.set_trace()
        self.init_vals["C"] = np.eye(len(self.init_vals["theta"])) * eps
        self.reset()

    def clone(self):
        o = self.__class__(
            eps=self.eps, lam=self.lam, gamma=self.gamma, phi=self.phi)
        return o

    def reset(self):
        self.reset_trace()
        self.init_vals["C"] = np.eye(len(self.init_vals["theta"])) * self.eps
        for k, v in self.init_vals.items():
            self.__setattr__(k, copy.copy(v))

    def update_V(self, s0, s1, r, f0=None, f1=None, theta=None, rho=1, **kwargs):
        """
            rho: weight for this sample in case of off-policy learning
        """
        if f0 is None or f1 is None:
            f0 = self.phi(s0)
            f1 = self.phi(s1)
        self._tic()
        if theta is None:
            theta = self.theta
        if not hasattr(self, "z"):
            self.z = f0

        L = np.dot(self.C, self.z)
        deltaf = f0 - self.gamma * rho * f1
        K = L / (1 + np.dot(deltaf, L))

        theta += K * (rho * r - np.dot(deltaf, theta))
        self.C -= np.outer(K, np.dot(deltaf, self.C))
        self.z = self.gamma * self.lam * rho * self.z + f1
        self.theta = theta
        self._toc()


class LinearTDLambda(OffPolicyValueFunctionPredictor, LambdaValueFunctionPredictor, LinearValueFunctionPredictor):
    """
        TD(\lambda) with linear function approximation
        for details see Szepesvári (2009): Algorithms for Reinforcement
        Learning (p. 30)
    """

    def __init__(self, alpha, **kwargs):
        """
            alpha:  step size. This can either be a constant number or
                    an iterable object providing step sizes
            lam: lambda in [0, 1] specifying the tradeoff between bootstrapping
                    and MC sampling
            gamma:  discount factor
        """
        LinearValueFunctionPredictor.__init__(self, **kwargs)
        OffPolicyValueFunctionPredictor.__init__(self, **kwargs)
        LambdaValueFunctionPredictor.__init__(self, **kwargs)
        self.init_vals['alpha'] = alpha
        self.reset()

    def clone(self):
        o = self.__class__(self.init_vals['alpha'], lam=self.lam,
                           gamma=self.gamma, phi=self.phi)
        return o

    def reset(self):
        LinearValueFunctionPredictor.reset(self)
        self.alpha = self._assert_iterator(self.init_vals['alpha'])
        if hasattr(self, "A"):
            del(self.A)
        if hasattr(self, "b"):
            del(self.b)

    def init_deterministic(self, task):
        assert self.lam == 0
        F, Cmat, self.b = self._compute_detTD_updates(task)
        self.A = np.array(F - Cmat)

    def update_V(self, s0, s1, r, f0=None, f1=None, theta=None, rho=1, **kwargs):

        if f0 is None or f1 is None:
            f0 = self.phi(s0)
            f1 = self.phi(s1)
        if theta is None:
            theta = self.theta
        if not hasattr(self, "z"):
            self.z = f0
        else:
            self.z = rho * (f0 + self.lam * self.gamma * self.z)
        self._tic()
        delta = r + self.gamma * np.dot(theta, f1) \
            - np.dot(theta, f0)

        theta_d = theta + self.alpha.next() * delta * self.z
        self.theta = theta_d
        self._toc()
        return theta

    def deterministic_update(self, theta=None):
        if theta is None:
            theta = self.theta
        theta_d = theta + self.alpha.next() * np.dot(self.A, theta) + self.b
        self.theta = theta_d
        return self.theta

    def __getstate__(self):
        res = self.__dict__
        for n in ["alpha"]:
            if isinstance(res[n], itertools.repeat):
                res[n] = res[n].next()
        return res

    def __setstate__(self, state):
        self.__dict__ = state
        self.alpha = self._assert_iterator(self.init_vals['alpha'])


class RMalpha(object):
    """
    step size generator of the form
        alpha = c*t^{-mu}
    """
    def __init__(self, c, mu):
        self.mu = mu
        self.c = c
        self.t = 0.

    def __iter__(self):
        return self

    def next(self):
        self.t += 1.
        return self.c * self.t ** (-self.mu)


class ResidualGradient(OffPolicyValueFunctionPredictor, LinearValueFunctionPredictor):
    """
        Residual Gradient algorithm with linear function approximation
        for details see Baird, L. (1995): Residual Algorithms : Reinforcement :
        Learning with Function Approximation.
    """

    def __init__(self, alpha, **kwargs):
        """
            alpha:  step size. This can either be a constant number or
                    an iterable object providing step sizes

            gamma:  discount factor
        """
        LinearValueFunctionPredictor.__init__(self, **kwargs)
        OffPolicyValueFunctionPredictor.__init__(self, **kwargs)
        self.init_vals['alpha'] = alpha
        self.reset()

    def clone(self):
        o = self.__class__(
            alpha=self.init_vals['alpha'], gamma=self.gamma, phi=self.phi)
        return o

    def reset(self):
        LinearValueFunctionPredictor.reset(self)
        self.alpha = self._assert_iterator(self.init_vals['alpha'])

    def update_V(self, s0, s1, r, f0=None, f1=None, theta=None, rho=1, **kwargs):

        if f0 is None or f1 is None:
            f0 = self.phi(s0)
            f1 = self.phi(s1)
        if theta is None:
            theta = self.theta

        self._tic()
        delta = r + self.gamma * np.dot(theta, f1) \
            - np.dot(theta, f0)
        theta += self.alpha.next() * delta * rho * (f0 - self.gamma * f1)
        self.theta = theta
        self._toc()
        return theta

    def __getstate__(self):
        res = self.__dict__
        for n in ["alpha"]:
            if isinstance(res[n], itertools.repeat):
                res[n] = res[n].next()
        return res

    def __setstate__(self, state):
        self.__dict__ = state
        self.alpha = self._assert_iterator(self.init_vals['alpha'])


class LinearTD0(LinearValueFunctionPredictor, OffPolicyValueFunctionPredictor):
    """
    TD(0) learning algorithm for on- and off-policy value function estimation
    with linear function approximation
    for details on off-policy importance weighting formulation see
    Maei, H. R. (2011). Gradient Temporal-Difference Learning Algorithms.
    University of Alberta. (p. 31)
    """

    def __init__(self, alpha, **kwargs):
        """
            alpha:  step size. This can either be a constant number or
                    an iterable object providing step sizes
            gamma:  discount factor
        """
        LinearValueFunctionPredictor.__init__(self, **kwargs)
        self.init_vals['alpha'] = alpha
        self.reset()

    def clone(self):
        o = self.__class__(
            alpha=self.init_vals['alpha'], gamma=self.gamma, phi=self.phi)
        return o

    def reset(self):
        LinearValueFunctionPredictor.reset(self)
        self.alpha = self._assert_iterator(self.init_vals['alpha'])

    def __getstate__(self):
        res = self.__dict__
        for n in ["alpha"]:
            if isinstance(res[n], itertools.repeat):
                res[n] = res[n].next()
        return res

    def __setstate__(self, state):
        self.__dict__ = state
        self.alpha = self._assert_iterator(self.init_vals['alpha'])

    def update_V(self, s0, s1, r, f0=None, f1=None, theta=None, rho=1, **kwargs):
        """
        adapt the current parameters theta given the current transition
        (s0 -> s1) with reward r and (a weight of rho)
        returns the next theta
        """
        if theta is None:
            theta = self.theta

        if f0 is None or f1 is None:
            f0 = self.phi(s0)
            f1 = self.phi(s1)
        self._tic()
        delta = r + self.gamma * np.dot(theta, f1) \
            - np.dot(theta, f0)
        #if np.isnan(delta):
        #    import ipdb; ipdb.set_trace()
        #print theta, delta
        logging.debug("TD Learning Delta {}".format(delta))
        #print theta
        #print f0, f1
        al = self.alpha.next()
        #if isinstance(self.alpha,  RMalpha):
        #    print al, self.alpha.t
        theta += al * delta * rho * f0
        self.theta = theta
        self._toc()
        return theta


class TabularTD0(ValueFunctionPredictor):
    """
        Tabular TD(0)
        for details see Szepesvári (2009): Algorithms for Reinforcement
        Learning (p. 19)
    """

    def __init__(self, alpha, gamma=1):
        """
            alpha: step size. This can either be a constant number or
                an iterable object providing step sizes
            gamma: discount factor
        """
        try:
            self.alpha = iter(alpha)
        except TypeError:
            self.alpha = itertools.repeat(alpha)

        self.gamma = gamma

    def update_V(self, s0, s1, r, V, **kwargs):
        self._tic()
        delta = r + self.gamma * V[s1] - V[s0]
        V[s0] += self.alpha.next() * delta
        self._toc()
        return V


class TabularTDLambda(ValueFunctionPredictor):
    """
        Tabular TD(\lambda)
        for details see Szepesvári (2009): Algorithms for Reinforcement
        Learning (p. 25)
    """

    def __init__(self, alpha, lam, gamma=1, trace_type="replacing"):
        """
            alpha: step size. This can either be a constant number or
                an iterable object providing step sizes
            gamma: discount factor
            lam:  lambda parameter controls the tradeoff between
                        bootstraping and MC sampling
            trace_type: controls how the eligibility traces are updated
                this can either be "replacing" or "accumulating"
        """
        try:
            self.alpha = iter(alpha)
        except TypeError:
            self.alpha = itertools.repeat(alpha)

        self.trace_type = trace_type
        assert trace_type in ("replacing", "accumulating")
        self.gamma = gamma
        self.lam = lam
        self.time = 0

    def update_V(self, s0, s1, r, V, **kwargs):
        if "z" in kwargs:
            z = kwargs["z"]
        elif hasattr(self, "z"):
            z = self.z
        else:
            z = np.zeros_like(V)
        self._tic()
        delta = r + self.gamma * V[s1] - V[s0]
        z = self.lam * self.gamma * z
        if self.trace_type == "replacing":
            z[s0] = 1
        elif self.trace_type == "accumulating":
            z[s0] += 1
        V += self.alpha.next() * delta * z
        self.z = z
        self._toc()
        return V
