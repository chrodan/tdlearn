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
        self.reset_trace()
        for k,v in self.init_vals.items():
            self.__setattr__(k,copy.copy(v))
        
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
        F = np.array(Phi.T * np.matrix(D) * (self.gamma*T) * Phi)
        Cmat = np.array(Phi.T * np.matrix(D) * Phi)
        R = task.mdp.P * task.mdp.r * task.target_policy[:, :, np.newaxis]
        R = np.sum(R, axis=1) # sum over all A
        R = np.sum(R, axis=1) # sum over all S'
        b = np.array(Phi) * R[:,np.newaxis] * task.beh_mu[:,np.newaxis]
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
    
    def update_V_offpolicy(self, s0, s1, r, a, beh_pi, target_pi, theta=None, 
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
        #print rho
        return self.update_V(s0, s1, r, theta=theta, **kwargs)
        

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
        o = self.__class__(self.init_vals['alpha'], self.init_vals['beta'], gamma=self.gamma, phi=self.phi)
        return o


    def reset(self):
        LinearValueFunctionPredictor.reset(self)
        self.alpha = self._assert_iterator(self.init_vals['alpha'])
        self.beta = self._assert_iterator(self.init_vals['beta'])
        self.w = np.zeros_like(self.init_vals['theta'])
        if hasattr(self, "A"): del(self.A)
        if hasattr(self, "b"): del(self.b)

        if hasattr(self, "F"): del(self.F)
        if hasattr(self, "Cmat"): del(self.Cmat)

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
    def update_V(self, s0, s1, r, rho=1, theta=None, **kwargs):
        """
            rho: weight for this sample in case of off-policy learning
        """
        w = self.w
        if theta is None:
            theta = self.theta
        
        f0 = self.phi(s0)
        f1 = self.phi(s1)

        self._tic()
        # TODO check if rho is used correctly
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

    def update_V(self, s0, s1, r, rho=1, theta=None, **kwargs):
        """
            rho: weight for this sample in case of off-policy learning
        """
        w = self.w
        if theta is None:
            theta = self.theta
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

        w_d = w + self.beta.next() * (np.dot(self.A, theta) - np.dot(self.Cmat, w) + self.b)
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

    def update_V(self, s0, s1, r, rho=1, theta=None, **kwargs):
        """
            rho: weight for this sample in case of off-policy learning
        """
        w = self.w
        if theta is None:
            theta = self.theta

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

        w_d = w + self.beta.next() * (np.dot(self.A, theta) - np.dot(self.Cmat, w) + self.b)
        theta_d = theta + self.alpha.next() * (np.dot(self.A, theta) - np.dot(self.F.T, w) + self.b)
        self.theta = theta_d
        self.w = w_d
        return self.theta

class GeriTDC(TDC):
    """
    the TDC algorithm except that the pseudo-stationary guess for off-policy estimation is computed differently
    """

    def update_V(self, s0, s1, r, rho=1, theta=None, **kwargs):
        """
            rho: weight for this sample in case of off-policy learning
        """
        w = self.w
        if theta is None:
            theta = self.theta

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

class GPTD(ValueFunctionPredictor):
    """
        Gaussian Process Temporal Difference Learning implementation
        with online sparsification
        for details see Engel, Y., Mannor, S., & Meir, R. (ICML 2003). 
            Bayes Meets Bellman: The Gaussian Process Approach To 
            Temporal Difference Learning.
            Table 1
    """

    def __init__(self, kernel, nu=1, sigma0=0.05, **kwargs):
        """
            kernel: a mercer kernel function as a python function
                that takes 2 arguments, i.e. gauss kernel
            nu: threshold for sparsification test
        """
        ValueFunctionPredictor.__init__(self, **kwargs)
        self.nu = nu
        self.sigma0 = sigma0
        self.kernel = np.frompyfunc(kernel,2,1)
        self.init_vals["D"] = []
        self.init_vals["C"] = 0
        self.init_vals["theta"] = 0
        self.init_vals["Kinv"] = np.empty(())
        self.reset()


#    def update_V(self, s0, s1, r, theta=None, rho=1, **kwargs):
#        """
#            rho: weight for this sample in case of off-policy learning
#        """
#        
#
#        self._tic()        
#        # first observation?
#        if len(self.D) == 0:
#            first=True
#            self.D.append(s0)
#            self.Kinv = np.matrix([[1./ self.kernel(s0, s0)]])
#            self.K = np.matrix([[self.kernel(s0, s0)]])
#            self.Hbar = None
#            self.A = None            
#        else:
#            first=False
#        if theta is None: theta=self.theta
#        k = self.kernel(self.D,s1)
#        a = self.Kinv * k
#        
#        
#        eta = self.kernel(s1, s1) - float(k.T * a)
#
#        if first:
#
#
#
#        # sparsification test        
#        elif eta > self.nu:
#            self.D.append(s1)
#            
#            # update K^-1
#            Kinv = np.matrix(np.ones((self.Kinv.shape[0]+1,self.Kinv.shape[1]+1)))
#            Kinv[:-1, :-1] = self.Kinv * eta + a*a.T
#            Kinv[-1,:-1] = -a.T
#            Kinv[:-1,-1] = -a
#            self.Kinv = Kinv / eta
#                         
#            print "inverted Kernel matrix:", self.Kinv
#            
#            theta 
#        else:
#            da = self.Kinv * self.kernel(self.D, s0) - self.gamma * a
#            da = da.reshape(1, -1)
#            self.Hbar =np.vstack(self.Hbar, da)   
#            self.A = np.vstack(self.A, a)  
#            
#            dk = self.kernel(D,s0) - self.gamma * self.kernel(D, s1)
#            h  = self.Kinv * self.kernel(self.D, s0) - self.gamma * a            
#            c =  - h
#            theta += c / s * (dk * theta - r)
#        self.theta = theta
#        self._toc()
#        return theta

        

class LSTDLambda(OffPolicyValueFunctionPredictor, LambdaValueFunctionPredictor, LinearValueFunctionPredictor):
    """
        recursive Implementation of Least Squared Temporal Difference Learning
         LSTD(\lambda) with linear function approximation, also works in the
         off-policy case and uses eligibility traces
        
        for details see Scherrer, B., & Geist, M. (EWRL 2011). :
            Recursive Least-Squares Learning with Eligibility Traces.
            Algorithm 1
    """

    def __init__(self, eps=1, **kwargs):
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
        self.init_vals["C"] = np.eye(len(self.init_vals["theta"]))*eps
        self.reset()

    def clone(self):
        o = self.__class__(eps=self.eps, lam=self.lam, gamma=self.gamma, phi=self.phi)
        return o

    def reset(self):
        self.reset_trace()   
        self.init_vals["C"] = np.eye(len(self.init_vals["theta"]))*self.eps
        for k,v in self.init_vals.items():
            self.__setattr__(k,copy.copy(v))

    def update_V(self, s0, s1, r, theta=None, rho=1, **kwargs):
        """
            rho: weight for this sample in case of off-policy learning
        """
        f0 = self.phi(s0)
        f1 = self.phi(s1)
        self._tic()
        if theta is None: theta=self.theta
        if not hasattr(self, "z"):
            self.z = f0
        
        L = np.dot(self.C,self.z)
        deltaf = f0 - self.gamma * rho * f1
        K = L / (1+ np.dot(deltaf,L))
        
        theta += K * (rho*r - np.dot(deltaf, theta))
        self.C -= np.outer(K, np.dot(deltaf, self.C))
        self.z = self.gamma * self.lam * rho * self.z + f1
        self.theta = theta
        self._toc()
        return theta

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
        o = self.__class__(self.init_vals['alpha'], lam=self.lam, gamma=self.gamma, phi=self.phi)
        return o

    def reset(self):
        LinearValueFunctionPredictor.reset(self)
        self.alpha = self._assert_iterator(self.init_vals['alpha'])
        if hasattr(self, "A"): del(self.A)
        if hasattr(self, "b"): del(self.b)

    def init_deterministic(self, task):
        assert self.lam == 0
        F, Cmat, self.b = self._compute_detTD_updates(task)
        self.A = np.array(F - Cmat)
    def update_V(self, s0, s1, r, theta=None, rho=1, **kwargs):

        f0 = self.phi(s0)
        f1 = self.phi(s1)
        if theta is None: theta=self.theta
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
        if theta is None: theta=self.theta
        theta_d = theta +  self.alpha.next() * np.dot(self.A, theta) + self.b
        self.theta = theta_d
        return self.theta

class RMalpha(object):
    """
    step size generator of the form
        alpha = c*t^{-mu}
    """
    def __init__(self, c,  mu):
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
        o = self.__class__(alpha=self.init_vals['alpha'], gamma=self.gamma, phi=self.phi)
        return o

    def reset(self):
        LinearValueFunctionPredictor.reset(self)
        self.alpha = self._assert_iterator(self.init_vals['alpha'])

    def update_V(self, s0, s1, r, theta=None, rho=1, **kwargs):

        f0 = self.phi(s0)
        f1 = self.phi(s1)
        if theta is None: theta=self.theta

        self._tic()
        delta = r + self.gamma * np.dot(theta, f1) \
                               - np.dot(theta, f0)
        theta += self.alpha.next() * delta * rho * (f0 - self.gamma * f1)
        self.theta = theta
        self._toc()
        return theta


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
        o = self.__class__(alpha=self.init_vals['alpha'], gamma=self.gamma, phi=self.phi)
        return o

    def reset(self):
        LinearValueFunctionPredictor.reset(self)
        self.alpha = self._assert_iterator(self.init_vals['alpha'])



    def update_V(self, s0, s1, r, theta=None, rho=1, **kwargs):
        """
        adapt the current parameters theta given the current transition
        (s0 -> s1) with reward r and (a weight of rho)
        returns the next theta
        """
        if theta is None:
            theta = self.theta
            
        f0 = self.phi(s0)
        f1 = self.phi(s1)
        self._tic()
        delta = r + self.gamma * np.dot(theta, f1) \
                               - np.dot(theta, f0)
        #import ipdb; ipdb.set_trace()
        logging.debug("TD Learning Delta {}".format(delta))
        #print theta
        #print f0, f1
        theta += self.alpha.next() * delta * rho * f0
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
        self.time=0

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


