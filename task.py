# -*- coding: utf-8 -*-
"""
Created on Wed Feb 22 18:39:00 2012

@author: Christoph Dann <cdann@cdann.de>
"""


import measures
import dynamic_prog
import numpy as np
from util.progressbar import ProgressBar

class LinearValuePredictionTask(object):
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
            policy = mdp.uniform_policy()
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
            self.mu = self.mdp.stationary_distrubution(seed=50, iterations=100000)
            return self.mu
        elif name is "V_true":            
            self.V_true = dynamic_prog.estimate(self.mdp, policy=self.target_policy, gamma=self.gamma)
            return self.V_true
        else:
            raise AttributeError
            
            
    def _init_methods(self, methods):
        for method in methods:
            method.phi=self.phi
            method.init_vals["theta"] = self.theta0
            method.gamma = self.gamma
            method.reset()
            
    def _init_error_fun(self, criterion):
        if criterion is "MSE":
            err_f = measures.prepare_MSE(self.mu, self.mdp, self.phi, self.V_true)    
        elif criterion is "RMSE":
            err_o = measures.prepare_MSE(self.mu, self.mdp, self.phi, self.V_true)  
            err_f = lambda x: np.sqrt(err_o(x))
        elif criterion is "MSPBE":
            err_f = measures.prepare_MSPBE(self.mu, self.mdp, self.phi, self.gamma, self.target_policy)
        elif criterion is "RMSPBE":
            err_o = measures.prepare_MSPBE(self.mu, self.mdp, self.phi, self.gamma, self.target_policy)
            err_f = lambda x: np.sqrt(err_o(x))
        elif criterion is "MSBE":
            err_f = measures.prepare_MSBE(self.mu, self.mdp, self.phi, self.gamma, self.target_policy)
        elif criterion is "RMSBE":
            err_o = measures.prepare_MSBE(self.mu, self.mdp, self.phi, self.gamma, self.target_policy)
            err_f = lambda x: np.sqrt(err_o(x))
        return err_f
        
        
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
                        cur_theta = m.update_V_offpolicy(s, s_n, r, a, 
                                                              self.behavior_policy, 
                                                              self.target_policy)
                    else:
                        cur_theta = m.update_V(s, s_n, r)
                    min_errors[k] = min(min_errors[k], err_f(cur_theta))
               

        return min_errors

    def avg_error_traces(self, methods, n_indep, n_eps=None, **kwargs):

        res = []
        with ProgressBar() as p:
            
            for seed in range(n_indep):
                p.update(seed, n_indep, "{} of {} seeds".format(seed, n_indep))
                kwargs['seed']=seed
                if n_eps is None:
                    res.append(self.ergodic_error_traces(methods, **kwargs))
                else:
                    res.append(self.episodic_error_traces(methods, n_eps=n_eps, **kwargs))
        res = np.array(res).swapaxes(0,1)
        return np.mean(res, axis=1), np.std(res, axis=1), res

    def ergodic_error_traces(self, methods, n_samples=1000, seed=None, criterion="MSE"):

        self._init_methods(methods)
        err_f = self._init_error_fun(criterion)
        errors = np.ones((n_samples,len(methods)))*np.inf
        
        for m in methods: m.reset_trace()
        i=0
        for s, a, s_n, r in self.mdp.sample_transition(n_samples, 
                                                       with_restart=True, 
                                                       seed=seed):
            for k, m in enumerate(methods):
                if self.off_policy:
                    cur_theta = m.update_V_offpolicy(s, s_n, r, a, 
                                                          self.behavior_policy, 
                                                          self.target_policy)
                else:
                    cur_theta = m.update_V(s, s_n, r)
                errors[i,k] = err_f(cur_theta)
            i += 1
               
        return errors.T
      
    def parameter_traces(self, methods, n_samples=1000, seed=None, criterion="MSE"):

        self._init_methods(methods)
        
        param = np.empty((n_samples,len(methods)) + self.theta0.shape)
        param[0,:,:] = self.theta0
        i = 1
        while i < n_samples:
            
            for m in methods: m.reset_trace()
            cur_seed = i*seed if seed is not None else None
            for s, a, s_n, r in self.mdp.sample_transition(n_samples, 
                                                           with_restart=False, 
                                                           seed=cur_seed):
                for k, m in enumerate(methods):
                    if self.off_policy:
                        cur_theta = m.update_V_offpolicy(s, s_n, r, a, 
                                                              self.behavior_policy, 
                                                              self.target_pi)
                    else:
                        cur_theta = m.update_V(s, s_n, r)
                    param[i,k] = cur_theta
                i += 1
                    
                if i >= n_samples: break
        return param      
      
    def episodic_error_traces(self, methods, n_eps=10000, n_samples=1000, seed=None, criterion="MSE"):

        self._init_methods(methods)
        err_f = self._init_error_fun(criterion)
        errors = np.ones((n_eps,len(methods)))*np.inf
        
        for i in xrange(n_eps):
            for m in methods: m.reset_trace()
            cur_seed = i+n_samples*seed if seed is not None else None
            for s, a, s_n, r in self.mdp.sample_transition(n_samples, 
                                                           with_restart=False, 
                                                           seed=cur_seed):
                for k, m in enumerate(methods):
                    if self.off_policy:
                        cur_theta = m.update_V_offpolicy(s, s_n, r, a, 
                                                              self.behavior_policy, 
                                                              self.target_pi)
                    else:
                        cur_theta = m.update_V(s, s_n, r)
                    errors[i,k] = err_f(cur_theta)
               
        return errors.T