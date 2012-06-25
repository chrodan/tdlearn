# -*- coding: utf-8 -*-
"""
Created on Wed Feb 22 18:39:00 2012

@author: Christoph Dann <cdann@cdann.de>
"""


import measures
import dynamic_prog
import numpy as np
from util.progressbar import ProgressBar
import util
from joblib import delayed, Parallel
import matplotlib.pyplot as plt
import policies
import features 

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
    def ergodic_error_traces(self, methods, curmdp=None, n_samples=1000, seed=None, criterion="MSE", error_every=1):
        if curmdp is None:
            curmdp = self.mdp
        self._init_methods(methods)
        err_f = self._init_error_fun(criterion)
        errors = np.ones((int(np.ceil(float(n_samples)/error_every)),len(methods)))*np.inf
        
        for m in methods: m.reset_trace()
        i=0
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
                    cur_theta = m.theta
                    errors[int(i/error_every),k] = err_f(cur_theta)
            i += 1
               
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
                        cur_theta = m.theta
                        errors[int(i / error_every),k] = err_f(cur_theta)
               
        return errors.T
        
        
        
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
            self.mu = self.mdp.stationary_distrubution(seed=50, iterations=100000, policy=self.target_policy)
            return self.mu
        elif name is "beh_mu":        
            self.beh_mu = self.mdp.stationary_distrubution(seed=50, iterations=100000, policy=self.behavior_policy)
            return self.beh_mu
        elif name is "V_true":            
            self.V_true = dynamic_prog.estimate_V_discrete(self.mdp, policy=self.target_policy, gamma=self.gamma)
            return self.V_true
        elif name is "Phi":
            self.Phi = measures.Phi_matrix(self.mdp, self.phi)
            return self.Phi
        else:
            raise AttributeError(name)
            
            
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
        elif criterion is "SMSPBE":
            err_f = measures.prepare_SMSPBE(self.mu, self.mdp, self.phi, self.gamma, self.target_policy)
        elif criterion is "SRMSPBE":
            err_o = measures.prepare_SMSPBE(self.mu, self.mdp, self.phi, self.gamma, self.target_policy)
            err_f = lambda x: np.sqrt(err_o(x))
        elif criterion is "MSBE":
            err_f = measures.prepare_MSBE(self.mu, self.mdp, self.phi, self.gamma, self.target_policy)
        elif criterion is "RMSBE":
            err_o = measures.prepare_MSBE(self.mu, self.mdp, self.phi, self.gamma, self.target_policy)
            err_f = lambda x: np.sqrt(err_o(x))
        return err_f




class LinearLQRValuePredictionTask(LinearValuePredictionTask):
    """
    A task to perform value function prediction of an mdp. It provides handy 
    methods to evaluate different algorithms on the same problem setting.
    """
    
    def __init__(self, mdp, gamma, phi, theta0, policy="linear", target_policy=None, normalize_phi=False):
        self.mdp = mdp
        self.gamma = gamma
        self.phi = phi
        self.seed = None
        self.theta0 = theta0
        if policy == "linear":
            policy = mdp.linear_policy()
        self.behavior_policy = policy 

        if target_policy is not None:
            self.off_policy = True
            if target_policy == "linear":
                target_policy = mdp.linear_policy()
            self.target_policy = target_policy
        else:
            self.target_policy =policy
            self.off_policy = False
        if normalize_phi:
            Phi = util.apply_rowise(phi, self.mu)
            phi.normalization = np.std(Phi, axis=0)
            phi.normalization[phi.normalization == 0] = 1.
            
    def __getattr__(self, name):
        """
        some attribute such as state distribution or the true value function
        are very costly to compute, so they are only evaluated, if really needed
        """
        if name is "V_true":            
            self.V_true = dynamic_prog.estimate_V_LQR(self.mdp, policy=self.target_policy, gamma=self.gamma)
            return self.V_true
#        elif name is "normalized_V_true":
#            return self.V_true * np.std(self.mu_phi_full).reshape(self.mdp.dim_S, self.mdp.dim_S)
        elif name is "mu":
            self.mu = self.mdp.state_samples(self.phi, n_iter=500, n_restarts=5,
                                        policy=self.target_policy, seed=self.seed,  verbose=False)
            return self.mu
        elif name is "mu_phi_full":
            self.mu_phi_full = util.apply_rowise(features.squared_tri(), self.mu)
            return self.mu_phi_full
        elif name is "mu_phi":
            self.mu_phi = util.apply_rowise(self.phi, self.mu)
            return self.mu_phi
#        elif name is "mu_beh_phi":
#            self.mu_phi = self.mdp.stationary_feature_distribution(self.phi, n_iter=100, n_restarts=10, 
#                                        policy=self.behavior_policy, seed=50, n_jobs=-2,
#                                        parallel=True, verbose=False)
#            return self.mu_phi
        else:
            raise AttributeError(name)
            
    def kl_policy(self):
        """ computes the KL Divergence between the behavioral and target policy
        while assuming that the steady state distribution is the state distribution of the
        behavioral policy!
        """
        r = .5 * (np.trace(np.dot(self.behavior_policy.precision, self.target_policy.noise)) \
            - self.behavior_policy.dim_A - np.log(np.linalg.det(self.target_policy.noise) / np.linalg.det(self.behavior_policy.noise)))

        dtheta = (self.behavior_policy.theta - self.target_policy.theta)
        da = np.dot(dtheta, self.mu.T)
        m = float(np.sum(da * np.dot(self.target_policy.precision, da))) / self.mu.shape[0]
        #import ipdb; ipdb.set_trace()
        r += .5 * m
        return r


    def _init_error_fun(self, criterion):
        if criterion is "MSE":
            err_f = measures.prepare_MSE(self.mu_phi_full, self.mdp, self.phi, self.V_true)    
        elif criterion is "RMSE":
            err_o = measures.prepare_MSE(self.mu_phi_full, self.mdp, self.phi, self.V_true)  
            err_f = lambda x: np.sqrt(err_o(x))
        elif criterion is "MSPBE":
            err_f = measures.prepare_MSPBE((self.mu_phi_full, self.mu_phi), self.mdp, self.phi, self.gamma, self.target_policy)
        elif criterion is "MSBE":
            err_f = measures.prepare_MSBE((self.mu_phi_full, self.mu_phi), self.mdp, self.phi, self.gamma, self.target_policy)
        elif criterion is "RMSPBE":
            err_o = measures.prepare_MSPBE((self.mu_phi_full, self.mu_phi), self.mdp, self.phi, self.gamma, self.target_policy)
            err_f = lambda x: np.sqrt(err_o(x))
        elif criterion is "RMSBE":
            err_o = measures.prepare_MSBE((self.mu_phi_full, self.mu_phi), self.mdp, self.phi, self.gamma, self.target_policy)
            err_f = lambda x: np.sqrt(err_o(x))
        return err_f
