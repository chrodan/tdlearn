# -*- coding: utf-8 -*-
"""
Temporal Difference Learning for finite MDPs

Created on Sun Dec 11 01:06:00 2011

@author: Christoph Dann <cdann@cdann.de>
"""

import numpy as np
import itertools
import logging
#logging.basicConfig(level=logging.DEBUG)

class ValueFunctionPredictor(object):
    """
        predicts the value function of a MDP for a given policy from given
        samples
    """

    def update_V(self, s0, s1, r, V, **kwargs):
        raise NotImplementedError("Each predictor has to implement this")

    def reset(self):
        if hasattr(self, 'theta'):
            del self.theta
        

    def _assert_iterator(self, p):
        try:
            return iter(p)
        except TypeError:
            return itertools.repeat(p)


class GTDBase(ValueFunctionPredictor):
    """ Base class for GTD, GTD2 and TDC algorithm """

    def __init__(self, alpha, beta, phi, gamma=1):
        """
            alpha:  step size. This can either be a constant number or
                    an iterable object providing step sizes
            beta:   step size for weights w. This can either be a constant
                    number or an iterable object providing step sizes
            gamma: discount factor
        """

        self.alpha = self._assert_iterator(alpha)
        self.beta = self._assert_iterator(beta)
        self.phi = phi
        self.gamma = gamma

    def reset(self):
        if hasattr(self, 'w'):
            del self.w
        if hasattr(self, 'theta'):
            del self.theta

    def update_V_offpolicy(self, s0, s1, r, a, theta,
                           beh_pi, target_pi, **kwargs):
        """
        off policy training version for transition (s0, a, s1) with reward r
        which was sampled by following the behaviour policy beh_pi.
        The parameters are learned for the target policy target_pi

         beh_pi, target_pi: S x A -> R
                numpy array of shape (n_s, n_a)
                *_pi(s,a) is the probability of taking action a in state s
        """
        rho = target_pi[s0, a] / beh_pi[s0, a]
        return self.update_V(s0, s1, r, theta, rho=rho, **kwargs)

class GTD(GTDBase):
    """
    GTD algorithm with linear function approximation
    for details see

    Maei, H. R. (2011). Gradient Temporal-Difference Learning Algorithms.
    (p. 36)
    """
    def update_V(self, s0, s1, r, theta, rho=1, **kwargs):
        """
            rho: weight for this sample in case of off-policy learning
        """
        if "w" in kwargs:
            w = kwargs["w"]
        elif hasattr(self, "w"):
            w = self.w
        else:
            w = np.zeros_like(theta)

        f0 = self.phi(s0)
        f1 = self.phi(s1)

        # TODO check if rho is used correctly
        delta = r + self.gamma * np.dot(theta, f1) - np.dot(theta, f0)
        a = np.dot(f0, w)

        w += self.beta.next() * rho * (delta * f0 - w)
        theta += self.alpha.next() * rho * (f0 - self.gamma * f1) * a

        self.w = w
        self.theta = theta
        return theta
    

class GTD2(GTDBase):
    """
    GTD2 algorithm with linear function approximation
    for details see

    Maei, H. R. (2011). Gradient Temporal-Difference Learning Algorithms.
    (p. 38)
    """

    def update_V(self, s0, s1, r, theta, rho=1, **kwargs):
        """
            rho: weight for this sample in case of off-policy learning
        """
        if "w" in kwargs:
            w = kwargs["w"]
        elif hasattr(self, "w"):
            w = self.w
        else:
            w = np.zeros_like(theta)

        f0 = self.phi(s0)
        f1 = self.phi(s1)

        # TODO check if rho has to be in here
        delta = r + self.gamma * np.dot(theta, f1) - np.dot(theta, f0)
        a = np.dot(f0, w)

        w += self.beta.next() * (rho * delta - a) * f0
        theta += self.alpha.next() * rho * a * (f0 - self.gamma * f1)

        self.w = w
        self.theta = theta
        return theta


class TDC(GTDBase):
    """
    TDC algorithm with linear function approximation
    for details see

    Maei, H. R. (2011). Gradient Temporal-Difference Learning Algorithms.
    (p. 38)
    """

    def update_V(self, s0, s1, r, theta, rho=1, **kwargs):
        """
            rho: weight for this sample in case of off-policy learning
        """
        if "w" in kwargs:
            w = kwargs["w"]
        elif hasattr(self, "w"):
            w = self.w
        else:
            w = np.zeros_like(theta)

        f0 = self.phi(s0)
        f1 = self.phi(s1)

        delta = r + self.gamma * np.dot(theta, f1) - np.dot(theta, f0)
        a = np.dot(f0, w)
        #import ipdb; ipdb.set_trace()

        w += self.beta.next() * (rho * delta - a) * f0
        logging.debug("Weight: {}".format(w))
        theta += self.alpha.next() * rho * (delta * f0 - self.gamma * f1 * a)
        logging.debug("Theta:  {}".format(theta))
        self.w = w
        self.theta = theta
        return theta



class LinearTDLambda(ValueFunctionPredictor):
    """
        TD(\lambda) with linear function approximation
        for details see Szepesvári (2009): Algorithms for Reinforcement
        Learning (p. 30)
    """

    def __init__(self, alpha, lam, phi, gamma=1):
        """
            alpha:  step size. This can either be a constant number or
                    an iterable object providing step sizes
            lam: lambda in [0, 1] specifying the tradeoff between bootstrapping
                    and MC sampling
            gamma:  discount factor
        """
        self.alpha = self._assert_iterator(alpha)
        self.phi = phi
        self.lam = lam
        self.gamma = gamma

    def reset(self):
        if hasattr(self, 'z'):
            del self.z
        if hasattr(self, 'theta'):
            del self.theta

    def update_V(self, s0, s1, r, theta, **kwargs):
        if "z" in kwargs:
            z = kwargs["z"]
        elif hasattr(self, "z"):
            z = self.z
        else:
            z = np.zeros_like(theta)

        delta = r + self.gamma * np.dot(theta, self.phi(s1)) \
                               - np.dot(theta, self.phi(s0))
        z = self.phi(s0) + self.lam * self.gamma * z
        theta += self.alpha.next() * delta * z
        self.z = z
        self.theta = theta
        return theta


class LinearTD0(LinearTDLambda):
    """
    TD(0) learning algorithm for on- and off-policy value function estimation
    with linear function approximation
    for details on off-policy importance weighting formulation see
    Maei, H. R. (2011). Gradient Temporal-Difference Learning Algorithms.
    University of Alberta. (p. 31)
    """

    def __init__(self, alpha, phi, gamma=1):
        """
            alpha:  step size. This can either be a constant number or
                    an iterable object providing step sizes
            gamma:  discount factor
        """
        self.alpha = self._assert_iterator(alpha)
        self.phi = phi
        self.gamma = gamma

    def update_V_offpolicy(self, s0, s1, r, a, theta,
                           beh_pi, target_pi, **kwargs):
        """
        off policy training version for transition (s0, a, s1) with reward r
        which was sampled by following the behaviour policy beh_pi.
        The parameters are learned for the target policy target_pi

         beh_pi, target_pi: S x A -> R
                numpy array of shape (n_s, n_a)
                *_pi(s,a) is the probability of taking action a in state s
        """
        rho = target_pi[s0, a] / beh_pi[s0, a]
        logging.debug("Off-Policy weight {}".format(rho))
        return self.update_V(s0, s1, r, theta, rho=rho, **kwargs)

    def update_V(self, s0, s1, r, theta, rho=1, **kwargs):
        """
        adapt the current parameters theta given the current transition
        (s0 -> s1) with reward r and (a weight of rho)
        returns the next theta
        """
        delta = r + self.gamma * np.dot(theta, self.phi(s1)) \
                               - np.dot(theta, self.phi(s0))
        logging.debug("TD Learning Delta {}".format(delta))
        theta += self.alpha.next() * delta * rho * self.phi(s0)
        self.theta = theta
        return theta



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
        delta = r + self.gamma * V[s1] - V[s0]
        V[s0] += self.alpha.next() * delta
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

    def update_V(self, s0, s1, r, V, **kwargs):
        if "z" in kwargs:
            z = kwargs["z"]
        elif hasattr(self, "z"):
            z = self.z
        else:
            z = np.zeros_like(V)

        delta = r + self.gamma * V[s1] - V[s0]
        z = self.lam * self.gamma * z
        if self.trace_type == "replacing":
            z[s0] = 1
        elif self.trace_type == "accumulating":
            z[s0] += 1
        V += self.alpha.next() * delta * z
        self.z = z
        return V


