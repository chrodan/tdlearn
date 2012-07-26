# -*- coding: utf-8 -*-
"""
Created on Wed Feb 22 18:39:00 2012

@author: Christoph Dann <cdann@cdann.de>
"""


import measures
import td
import dynamic_prog
import numpy as np
from util.progressbar import ProgressBar
import util
from joblib import delayed, Parallel
import matplotlib.pyplot as plt
import policies
import features
from util import cached_property

class LinearValuePredictionTask(object):
    """ Base class for LQR and discrete case tasks """
    
    def _init_methods(self, methods):
        for method in methods:
            method.phi=self.phi
            method.init_vals["theta"] = self.theta0
            method.gamma = self.gamma
            method.reset()
            
   
        
        
    def min_error(self, methods, n_eps=10000, n_samples=1000, seed=None, criterion="MSE"):

        self._init_methods(methods)
        err_f = self._init_error_fun(criterion)
        min_errors = np.ones(len(methods))*np.inf
        
        for i in xrange(n_eps):
            for m in methods: m.reset_trace()
            cur_seed = i+n_samples*seed if seed is not None else None
            for s, a, s_n, r in self.mdp.sample_transition(n_samples, 
                                                           with_restart=False, 
                                                           seed=cur_seed):
                for k, m in enumerate(methods):
                    if self.off_policy:
                        m.update_V_offpolicy(s, s_n, r, a,
                                                              self.behavior_policy, 
                                                              self.target_policy)
                    else:
                        m.update_V(s, s_n, r)
                    cur_theta = m.theta
                    min_errors[k] = min(min_errors[k], err_f(cur_theta))
               

        return min_errors

    def avg_error_traces(self, methods, n_indep, n_eps=None, verbose=False, n_jobs=1, stationary=False, **kwargs):

        res = []
        if n_jobs==1:
            with ProgressBar(enabled=verbose) as p:
                
                for seed in range(n_indep):
                    p.update(seed, n_indep, "{} of {} seeds".format(seed, n_indep))
                    kwargs['seed']=seed
                    if stationary:
                        res.append(self.stationary_error_traces(methods, **kwargs))
                    elif n_eps is None:
                        res.append(self.ergodic_error_traces(methods, **kwargs))
                    else:
                        res.append(self.episodic_error_traces(methods, n_eps=n_eps, **kwargs))
        else:
            jobs = []
            for seed in range(n_indep):
                kwargs['seed']=seed
                curmethods = [m.clone() for m in methods]
                kwargs["curmdp"] = self.mdp
                if n_eps is None:
                    jobs.append((LinearLQRValuePredictionTask.ergodic_error_traces,["", curmethods], kwargs))
                    
                else:
                    kwargs["n_eps"] = n_eps
                    jobs.append((LinearLQRValuePredictionTask.episodic_error_traces,[methods], kwargs))
            res = Parallel(n_jobs=n_jobs, verbose=verbose)(jobs)
        res = np.array(res).swapaxes(0,1)
        return np.mean(res, axis=1), np.std(res, axis=1), res

    def deterministic_error_traces(self, methods, n_samples, criterion="MSPBE"):
        
        self._init_methods(methods)
        err_f = self._init_error_fun(criterion)
        errors = np.ones((len(methods), n_samples))*np.inf
        for m in methods:
            m.init_deterministic(self)

        for i in xrange(n_samples):
            for j,m in enumerate(methods):
                cur_theta = m.deterministic_update()
                errors[j,i] = err_f(cur_theta)
        return errors

    def deterministic_parameter_traces(self, methods, n_samples, criterion="MSPBE"):
        
        self._init_methods(methods)
        param = np.ones((len(methods), n_samples) + self.theta0.shape)*np.inf
        for m in methods:
            m.init_deterministic(self)

        for i in xrange(n_samples):
            for j,m in enumerate(methods):
                cur_theta = m.deterministic_update()
                param[j,i,:] = cur_theta
        return param


    def stationary_error_traces(self, methods, n_samples=1000, seed=None, criterion="MSE", error_every=1):
        self._init_methods(methods)
        err_f = self._init_error_fun(criterion)
        errors = np.ones((int(np.ceil(float(n_samples)/error_every)),len(methods)))*np.inf
        mu = self.mu
        rands = np.random.randint(mu.shape[0], size=n_samples)
        for i in xrange(n_samples):

            s,a,s_n, r = self.mdp.sample_step(mu[rands[i], :], policy=self.behavior_policy)
            for k, m in enumerate(methods):
                m.reset_trace()
                if self.off_policy:
                    m.update_V_offpolicy(s, s_n, r, a,
                        self.behavior_policy,
                        self.target_policy)
                else:
                    m.update_V(s, s_n, r)
                if i % error_every == 0:
                    cur_theta = m.theta
                    errors[int(i/error_every),k] = err_f(cur_theta)


        return errors[:i,:].T
        
    def ergodic_error_traces(self, methods, curmdp=None, n_samples=1000, 
                             seed=None, criterion="MSE", error_every=1, with_trace=False):
        if curmdp is None:
            curmdp = self.mdp
        self._init_methods(methods)
        err_f = self._init_error_fun(criterion)
        err_f_gen = self._init_error_fun(criterion, general=True)
        errors = np.ones((int(np.ceil(float(n_samples)/error_every)),len(methods)))*np.inf
        if with_trace:
            states = np.zeros((int(np.ceil(float(n_samples)/error_every)),curmdp.dim_S))
            rewards = np.zeros(int(np.ceil(float(n_samples)/error_every)))
        
        for m in methods: m.reset_trace()
        i = 0
        for s, a, s_n, r in curmdp.sample_transition(n_samples,
                                                       policy=self.behavior_policy,
                                                       with_restart=False, 
                                                       seed=seed):
                        
            for k, m in enumerate(methods):
                if self.off_policy:
                    m.update_V_offpolicy(s, s_n, r, a,
                                                          self.behavior_policy, 
                                                          self.target_policy)
                else:
                    m.update_V(s, s_n, r)
                if i % error_every == 0:
                    if isinstance(m, td.LinearValueFunctionPredictor):
                        cur_theta = m.theta
                        errors[int(i/error_every),k] = err_f(cur_theta)
                    else:
                        errors[int(i/error_every),k] = err_f_gen(m.V)
            if i % error_every == 0 and with_trace:
                rewards[int(i/error_every)] = r
                states[int(i/error_every),:] = s
            i += 1
        if with_trace:
            return errors[:i,:], rewards[:i], states[:i]
        else:
            return errors[:i,:].T


    def parameter_search(self, methods, n_eps=None, n_samples=1000, seed=None):

        self._init_methods(methods)
        param = [None] * len(methods)
        if n_eps is None: n_eps=1
            
        for s in range(n_eps):
            cur_seed = s*seed if seed is not None else None
            for m in methods: m.reset_trace()
                
            for s, a, s_n, r in self.mdp.sample_transition(n_samples, policy=self.behavior_policy,
                                                               with_restart=False, 
                                                               seed=cur_seed):
                for k, m in enumerate(methods):
                    if self.off_policy:
                        m.update_V_offpolicy(s, s_n, r, a,
                                                                  self.behavior_policy, 
                                                                  self.target_policy)
                    else:
                        m.update_V(s, s_n, r)
                    param[k] = m.theta
                
        return param 
    
    def parameter_traces(self, methods, n_samples=1000, seed=None):
        # deprecated
        pass
        self._init_methods(methods)
        
        param = np.empty((n_samples,len(methods)) + self.theta0.shape)
        param[0,:,:] = self.theta0
        i = 1
        while i < n_samples:
            
            for m in methods: m.reset_trace()
            cur_seed = i*seed if seed is not None else None
            for s, a, s_n, r in self.mdp.sample_transition(n_samples, policy=self.behavior_policy,
                                                           with_restart=False, 
                                                           seed=cur_seed):
                for k, m in enumerate(methods):
                    if self.off_policy:
                        m.update_V_offpolicy(s, s_n, r, a,
                                                              self.behavior_policy, 
                                                              self.target_policy)
                    else:
                        m.update_V(s, s_n, r)
                    param[i,k] = m.theta
                i += 1
                    
                if i >= n_samples: break
        return param      
      
    def episodic_error_traces(self, methods, n_eps=10000, error_every=1, n_samples=1000, seed=None, criterion="MSE"):

        self._init_methods(methods)
        err_f = self._init_error_fun(criterion)
        errors = np.ones((int(np.ceil(n_samples/error_every)),len(methods)))*np.inf
        err_f_gen = self._init_error_fun(criterion, general=True)

        for i in xrange(n_eps):
            for m in methods: m.reset_trace()
            cur_seed = i+n_samples*seed if seed is not None else None
            for s, a, s_n, r in self.mdp.sample_transition(n_samples, 
                                                           policy=self.behavior_policy,
                                                           with_restart=False, 
                                                           seed=cur_seed):
                for k, m in enumerate(methods):
                    if self.off_policy:
                        m.update_V_offpolicy(s, s_n, r, a,
                                                              self.behavior_policy, 
                                                              self.target_pi)
                    else:
                        m.update_V(s, s_n, r)
                    if i % error_every == 0:
                        if isinstance(m, td.LinearValueFunctionPredictor):
                            cur_theta = m.theta
                            errors[int(i/error_every),k] = err_f(cur_theta)
                        else:
                            errors[int(i/error_every),k] = err_f_gen(m.V)
               
        return errors.T

    def _init_error_fun(self, criterion, general=False):
        if criterion is "MSE":
            err_f = self.MSE
        elif criterion is "RMSE":
            err_o = self.MSE
            err_f = lambda x: np.sqrt(err_o(x))
        elif criterion is "MSPBE":
            err_f = self.MSPBE
        elif criterion is "MSBE":
            err_f = self.MSBE
        elif criterion is "RMSPBE":
            err_o = self.MSPBE
            err_f = lambda x: np.sqrt(err_o(x))
        elif criterion is "RMSBE":
            err_o = self.MSBE
            err_f = lambda x: np.sqrt(err_o(x))
        return err_f


        
        
        
class LinearDiscreteValuePredictionTask(LinearValuePredictionTask):
    """
    A task to perform value function prediction of an mdp. It provides handy 
    methods to evaluate different algorithms on the same problem setting.
    """
    
    def __init__(self, mdp, gamma, phi, theta0, policy="uniform", target_policy=None):
        self.mdp = mdp
        self.gamma = gamma
        self.phi = phi
        self.theta0 = theta0
        if policy == "uniform":
            policy = policies.DiscreteUniform(len(self.mdp.states),len(self.mdp.actions))
        self.behavior_policy = policy 

        if target_policy is not None:
            self.off_policy = True
            if target_policy == "uniform":
                target_policy = mdp.uniform_policy()
            self.target_policy = target_policy
        else:
            self.target_policy =policy
            self.off_policy = False
        
    def __getattr__(self, name):
        """
        some attribute such as state distribution or the true value function
        are very costly to compute, so they are only evaluated, if really needed
        """
        if name is "mu":        
            self.mu = self.mdp.stationary_distribution(seed=50, iterations=100000, policy=self.target_policy)
            return self.mu
        elif name is "beh_mu":        
            self.beh_mu = self.mdp.stationary_distribution(seed=50, iterations=100000, policy=self.behavior_policy)
            return self.beh_mu
        elif name is "V_true":            
            self.V_true = dynamic_prog.estimate_V_discrete(self.mdp, policy=self.target_policy, gamma=self.gamma)
            return self.V_true
        else:
            raise AttributeError(name)


    def projection_operator(self):

        D = np.diag(self.mu)
        Pi = self.Phi * np.linalg.pinv(self.Phi.T * D * self.Phi) * self.Phi.T * D
        return Pi

    @property
    def Phi(self):
        """
        produce the feature representation of all states of mdp as a vertically
        stacked matrix,
            mdp:    Markov Decision Process, instance of mdp.MDP
            phi:    feature function: S -> R^d given as python function

            returns: numpy matrix of shape (n_s, dim(phi)) where
                    Phi[i,:] = phi(S[i])
        """
        if not hasattr(self, "Phi_"):
            Phil = []
            for s in self.mdp.states:
                f = self.phi(s)
                Phil.append(f)
            Phi = np.matrix(np.vstack(Phil))
            self.Phi_ = Phi
        return self.Phi_



    def bellman_operator(self, V, policy="behavior"):
        """
        the bellman operator
            T(V) = R + gamma * P * V

        details see Chapter 3 of
        Sutton, R. S., Maei, H. R., Precup, D., Bhatnagar, S., Silver, D.,
        Szepesvari, C., & Wiewiora, E. (2009).: Fast gradient-descent methods for
        temporal-difference learning with linear function approximation.
        """
        if policy == "behavior":
            policy = self.behavior_policy
        elif policy == "target":
            policy = self.target_policy
        if hasattr(self, "R"):
            R = self.R
        else:
            R = self.mdp.P * self.mdp.r * policy.tab[:, :, np.newaxis]
            R = np.sum(R, axis=1) # sum over all A
            R = np.sum(R, axis=1) # sum over all S'
            self.R = R

        if hasattr(self, "T_P"):
            P = self.T_P
        else:
            P = self.mdp.P * policy.tab[:, :, np.newaxis]
            P = np.sum(P, axis=1) # sum over all A => p(s' | s)
            self.T_P = P

        return R + self.gamma * np.dot(P, V)

    def MSE(self, theta):
        return np.sum(((theta * np.asarray(self.Phi)).sum(axis=1) - self.V_true)**2 * self.mu)

    def MSBE(self, theta):

        V = (theta * np.asarray(self.Phi)).sum(axis=1)
        v = np.asarray(V - self.bellman_operator(V))
        return np.sum(v**2 * self.mu)

    def MSPBE(self, theta):

        V = (theta * np.asarray(self.Phi)).sum(axis=1)
        v = np.asarray(V - np.dot(self.projection_operator,self.bellman_operator(V)))
        return np.sum(v**2 * self.mu)



class LinearContinuousValuePredictionTask(LinearValuePredictionTask):
    """
    A task to perform value function prediction of an mdp. It provides handy
    methods to evaluate different algorithms on the same problem setting.
    """

    def __init__(self, mdp, gamma, phi, theta0, policy, target_policy=None, normalize_phi=False):
        self.mdp = mdp
        self.gamma = gamma
        self.phi = phi
        self.seed = None
        self.theta0 = theta0
        self.behavior_policy = policy

        if target_policy is not None:
            self.off_policy = True
            self.target_policy = target_policy
        else:
            self.target_policy =policy
            self.off_policy = False
        if normalize_phi:
            Phi = util.apply_rowise(phi, self.mu)
            phi.normalization = np.std(Phi, axis=0)
            phi.normalization[phi.normalization == 0] = 1.

    def projection_operator(self):
        if hasattr(self, "Pi"):
            return self.Pi
        else:
            self.Pi = np.matrix(self.mu_phi) * np.linalg.pinv(np.matrix(self.mu_phi).T * np.matrix(self.mu_phi)) * np.matrix(self.mu_phi).T
        return self.Pi

    def kl_policy(self):
        """ computes the KL Divergence between the behavioral and target policy
        while assuming that the steady state distribution is the state distribution of the
        behavioral policy!
        """
        r = .5 * (np.trace(np.dot(self.behavior_policy.precision, self.target_policy.noise))\
                  - self.behavior_policy.dim_A - np.log(np.linalg.det(self.target_policy.noise) / np.linalg.det(self.behavior_policy.noise)))

        dtheta = (self.behavior_policy.theta - self.target_policy.theta)
        da = np.dot(dtheta, self.mu.T)
        m = float(np.sum(da * np.dot(self.target_policy.precision, da))) / self.mu.shape[0]

        r += .5 * m
        return r

    def __getattr__(self, name):
        """
        some attribute such as state distribution or the true value function
        are very costly to compute, so they are only evaluated, if really needed
        """
        if name is "mu" or name is "mu_next" or name is "mu_r":
            self.mu,_, self.mu_r, self.mu_next  = self.mdp.samples( n_iter=1000, n_restarts=5,
                policy=self.target_policy, no_next_noise=True)
            return self.__dict__[name]
        elif name is "mu_phi_full":
            self.mu_phi_full = util.apply_rowise(features.squared_tri(), self.mu)
            return self.mu_phi_full
        elif name is "mu_phi":
            self.mu_phi = util.apply_rowise(self.phi, self.mu)
            return self.mu_phi
        elif name is "mu_phi_next":
            self.mu_phi_next = util.apply_rowise(lambda x: self.phi.expectation(x, self.mdp.Sigma), self.mu_next)
            return self.mu_phi_next
        else:
            raise AttributeError(name)


    def MSPBE(self, theta):
        """ Mean Squared Bellman Error """
        V = np.array((theta * self.mu_phi).sum(axis=1))
        V2 = self.gamma * np.array((theta * self.mu_phi_next).sum(axis=1))
        return np.mean(np.array((V-np.dot(self.projection_operator(),V2+self.mu_r)))**2)

    def MSBE(self, theta):
        """ Mean Squared Bellman Error """
        V = np.array((theta * self.mu_phi).sum(axis=1))
        V2 = self.gamma * np.array((theta * self.mu_phi_next).sum(axis=1))
        #import ipdb
        #ipdb.set_trace()
        return np.mean((V-V2-self.mu_r)**2)


class LinearLQRValuePredictionTask(LinearContinuousValuePredictionTask):


    def __getattr__(self, name):
        """
        some attribute such as state distribution or the true value function
        are very costly to compute, so they are only evaluated, if really needed
        """
        if name is "V_true":
            self.V_true = dynamic_prog.estimate_V_LQR(self.mdp, lambda x,y: self.bellman_operator(x,y, policy="target"),
                                                                                        gamma=self.gamma)
            return self.V_true
        else:
            return LinearContinuousValuePredictionTask.__getattr__(self,name)





    def bellman_operator(self, P, b, policy="behavior"):
        """
        the bellman operator for the behavioral policy
        as a python function which takes the value function s^T P s + b represented as a numpy
        squared array P
            T(P,b) = R + theta_p^T Q theta_p + gamma * (A + B theta_p)^T P (A + B theta_p), gamma * (b + tr(P * Sigma))
        
        """
        Q = np.matrix(self.mdp.Q)
        R = np.matrix(self.mdp.R)
        A = np.matrix(self.mdp.A)
        B = np.matrix(self.mdp.B)
        Sigma = np.matrix(np.diag(self.mdp.Sigma))

        if policy == "behavior":
            theta = np.matrix(self.behavior_policy.theta)
            noise = self.behavior_policy.noise
            if hasattr(self, "S"):
                S = self.S
            else:
                S = A+ B * theta
                self.S = S
            if hasattr(self, "C"):
                C = self.C
            else:
                C = Q + theta.T * R * theta
                self.C = C
        elif policy == "target":
            theta = np.matrix(self.target_policy.theta)
            noise = self.target_policy.noise
            if hasattr(self, "S_target"):
                S = self.S_target
            else:
                S = A+ B * theta
                self.S_target = S
            if hasattr(self, "C_target"):
                C = self.C_target
            else:
                C = Q + theta.T * R * theta
                self.C_target = C
        else:
            theta = np.matrix(policy)
            noise = policy.noise
            S = A+ B * theta
            C = Q + theta.T * R * theta

        Pn = C + self.gamma * (S.T * np.matrix(P) * S)
        bn = self.gamma * (b + np.trace(np.matrix(P)*np.matrix(Sigma) )) \
                + np.trace((R + self.gamma * B.T * np.matrix(P)*B)* np.matrix(noise))
        return Pn, bn

    def MSE(self, theta):
        p = features.squared_tri().param_forward(*self.phi.param_back(theta)) -\
                features.squared_tri().param_forward(*self.V_true)
        return np.mean((p * self.mu_phi_full).sum(axis=1)**2)


    def MSBE(self, theta):
        """ Mean Squared Bellman Error """
        V = np.array((theta * self.mu_phi).sum(axis=1))
        theta_trans = features.squared_tri().param_forward(*self.bellman_operator(*self.phi.param_back(theta)))
        V2 = np.array((theta_trans * self.mu_phi_full).sum(axis=1))
        #import ipdb
        #ipdb.set_trace()
        return np.mean((V-V2)**2)

    def MSPBE(self, theta):
        #return LinearContinuousValuePredictionTask.MSPBE(self, theta)
        """ Mean Squared Projected Bellman Error """
        #print "Wrong!"
        V = np.matrix((theta * np.asarray(self.mu_phi)).sum(axis=1)).T
        theta_trans = features.squared_tri().param_forward(*self.bellman_operator(*self.phi.param_back(theta)))
        v = np.asarray(V - self.projection_operator() * np.matrix(self.mu_phi_full) * np.matrix(theta_trans).T)
        return np.mean(v**2)