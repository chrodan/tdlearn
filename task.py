# -*- coding: utf-8 -*-
"""
Created on Wed Feb 22 18:39:00 2012

@author: Christoph Dann <cdann@cdann.de>
"""


import td
import dynamic_prog
import numpy as np
from util.progressbar import ProgressBar
import util
from joblib import Parallel
import policies
import features
import mdp


def tmp(cl, *args, **kwargs):
    return cl.error_traces(*args, **kwargs)


def tmp2(cl, *args, **kwargs):
    return cl.episodic_error_traces(*args, **kwargs)


def tmp3(cl, *args, **kwargs):
    return cl.error_data_budget(*args, **kwargs)


class LinearValuePredictionTask(object):
    """ Base class for LQR and discrete case tasks """

    def _init_methods(self, methods):
        for method in methods:
            method.phi = self.phi
            method.init_vals["theta"] = self.theta0
            method.gamma = self.gamma
            method.reset()

    def min_error(self, methods, n_eps=10000, n_samples=1000, seed=None, criterion="MSE"):

        self._init_methods(methods)
        err_f = self._init_error_fun(criterion)
        min_errors = np.ones(len(methods)) * np.inf

        for i in xrange(n_eps):
            for m in methods:
                m.reset_trace()
            cur_seed = i + n_samples * seed if seed is not None else None
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

    def avg_error_traces(self, methods, n_indep, verbose=False, n_jobs=1, **kwargs):
        res = []
        if n_jobs == 1:
            with ProgressBar(enabled=verbose) as p:

                for seed in range(n_indep):
                    p.update(
                        seed, n_indep, "{} of {} seeds".format(seed, n_indep))
                    kwargs['seed'] = seed
                    res.append(self.error_traces(methods, **kwargs))
        else:
            jobs = []
            for seed in range(n_indep):
                kwargs = kwargs.copy()
                kwargs['seed'] = seed
                self.projection_operator()
                jobs.append((tmp, [self, methods], kwargs))
            res = Parallel(n_jobs=n_jobs, verbose=verbose)(jobs)
            res = np.array(res)
        return np.mean(res, axis=0), np.std(res, axis=0), res

    def deterministic_error_traces(self, methods, n_samples, criterion="MSPBE"):

        self._init_methods(methods)
        err_f = self._init_error_fun(criterion)
        errors = np.ones((len(methods), n_samples)) * np.inf
        for m in methods:
            m.init_deterministic(self)

        for i in xrange(n_samples):
            for j, m in enumerate(methods):
                cur_theta = m.deterministic_update()
                errors[j, i] = err_f(cur_theta)
        return errors

    def deterministic_parameter_traces(self, methods, n_samples, criterion="MSPBE"):

        self._init_methods(methods)
        param = np.ones((len(methods), n_samples) + self.theta0.shape) * np.inf
        for m in methods:
            m.init_deterministic(self)

        for i in xrange(n_samples):
            for j, m in enumerate(methods):
                cur_theta = m.deterministic_update()
                param[j, i, :] = cur_theta
        return param

    def parameter_search(self, methods, n_eps=None, n_samples=1000, seed=None):

        self._init_methods(methods)
        param = [None] * len(methods)
        if n_eps is None:
            n_eps = 1

        for s in range(n_eps):
            cur_seed = s * seed if seed is not None else None
            for m in methods:
                m.reset_trace()

            for s, a, s_n, r in self.mdp.sample_transition(
                n_samples, policy=self.behavior_policy,
                with_restart=False,
                    seed=cur_seed):
                f0 = self.phi(s)
                f1 = self.phi(s_n)
                for k, m in enumerate(methods):
                    if self.off_policy:
                        m.update_V_offpolicy(s, s_n, r, a,
                                             self.behavior_policy,
                                             self.target_policy,
                                             f0=f0, f1=f1)
                    else:
                        m.update_V(s, s_n, r, f0=f0, f1=f1)
                    param[k] = m.theta

        return param

    def parameter_traces(self, methods, n_samples=1000, seed=None, override_terminal=0):
        # deprecated
        self._init_methods(methods)

        param = np.empty((n_samples, len(methods)) + self.theta0.shape)
        param[0, :, :] = self.theta0
        i = 1
        while i < n_samples:

            for m in methods:
                m.reset_trace()
            cur_seed = i * seed if seed is not None else None
            for s, a, s_n, r in self.mdp.sample_transition(
                n_samples, policy=self.behavior_policy,
                with_restart=False,
                    seed=cur_seed):
                    #override_terminal=override_terminal):
                f0 = self.phi(s)
                f1 = self.phi(s_n)
                #print s, a, s_n, r, f0, f1
                for k, m in enumerate(methods):
                    if self.off_policy:
                        m.update_V_offpolicy(s, s_n, r, a,
                                             self.behavior_policy,
                                             self.target_policy,
                                             f0=f0, f1=f1)
                    else:
                        m.update_V(s, s_n, r, f0=f0, f1=f1)
                    param[i, k] = m.theta
                i += 1

                if i >= n_samples:
                    break
        return param

    def avg_error_data_budget(self, methods, n_indep, verbose=False, n_jobs=1, **kwargs):

        res = []
        if n_jobs == 1:
            with ProgressBar(enabled=verbose) as p:

                for seed in range(n_indep):
                    p.update(
                        seed, n_indep, "{} of {} seeds".format(seed, n_indep))
                    kwargs['seed'] = seed
                    res.append(self.error_data_budget(methods, **kwargs))

        else:
            jobs = []
            for seed in range(n_indep):
                kwargs = kwargs.copy()
                kwargs['seed'] = seed
                self.projection_operator()
                jobs.append((tmp3, [self, methods], kwargs))

            res = Parallel(n_jobs=n_jobs, verbose=verbose)(jobs)
        errors, times = zip(*res)

        errors = np.array(errors).swapaxes(0, 1)
        return np.mean(errors, axis=1), np.std(errors, axis=1), np.mean(times, axis=0)

    def error_data_budget(self, methods, passes, n_samples=1000, n_eps=1,
                          seed=1, criterion="MSE"):

        self._init_methods(methods)
        err_f = self._init_error_fun(criterion)
        errors = np.ones((passes, len(methods))) * np.inf
        for m in methods:
            m.reset_trace()

        s, a, r, s_n, restarts, f0, f1 = self.mdp.samples_featured(
            phi=self.phi, n_iter=n_samples,
            n_restarts=n_eps,
            policy=self.behavior_policy,
            seed=seed)
        for p in range(passes):
            for k, m in enumerate(methods):
                m.reset_trace()
                for i in xrange(n_samples * n_eps):
                    if restarts[i]:
                        m.reset_trace()
                    if self.off_policy:
                        m.update_V_offpolicy(s[i], s_n[i], r[i], a[i],
                                             self.behavior_policy,
                                             self.target_policy,
                                             f0=f0[i], f1=f1[i])
                    else:
                        m.update_V(s[i], s_n[i], r[i], f0=f0[i], f1=f1[i])
                errors[p, k] = err_f(m.theta)
        times = [m.time for m in methods]
        return errors, times

    def fill_trajectory_cache(self, seeds, n_samples=1000, n_eps=1):
        for seed in seeds:
            s, a, r, s_n, restarts = self.mdp.samples_cached(n_iter=n_samples,
                                                             n_restarts=n_eps,
                                                             policy=self.behavior_policy,
                                                             seed=seed)

    def error_traces(self, methods, n_samples=1000, n_eps=1,
                             seed=1, criteria=["RMSBE"], error_every=1, episodic=False):

        # Intialization
        self._init_methods(methods)
        err_f = [self._init_error_fun(criterion) for criterion in criteria]
        err_f_gen = [self._init_error_fun(criterion, general=True) for criterion in criteria]

        if episodic:
            n_e = n_eps
            k_e = 0
        else:
            n_e = int(np.ceil(float(n_samples * n_eps) / error_every))

        errors = np.ones((len(methods), len(criteria), n_e)) * np.inf
        for m in methods:
            m.reset_trace()

        # Generate trajectories
        s, a, r, s_n, restarts = self.mdp.samples_cached(n_iter=n_samples,
                                                         n_restarts=n_eps,
                                                         policy=self.behavior_policy,
                                                         seed=seed)
        if self.off_policy:
            m_a_beh = policies.mean_action_trajectory(self.behavior_policy, s)
            m_a_tar = policies.mean_action_trajectory(self.target_policy, s)
            rhos = np.zeros_like(r)
            self.rhos = rhos

        # Method learning
        for i in xrange(n_samples * n_eps):
            f0 = self.phi(s[i])
            f1 = self.phi(s_n[i])
            if restarts[i]:
                for k, m in enumerate(methods):
                    m.reset_trace()
                    if episodic:
                        cur_theta = m.theta
                        for i_e in range(len(criteria)):
                            if isinstance(m, td.LinearValueFunctionPredictor):
                                errors[k, i_e, k_e] = err_f[i_e](cur_theta)
                            else:
                                errors[k, i_e, k_e] = err_f_gen[i_e](m.V)

                if episodic:
                    k_e += 1
                    if k_e >= n_e:
                        break

            for k, m in enumerate(methods):
                if self.off_policy:
                    rhos[i] = self.target_policy.p(s[i], a[i], mean=m_a_tar[i]) / self.behavior_policy.p(s[i], a[i], mean=m_a_beh[i])
                    m.update_V(s[i], s_n[i], r[i],
                               rho=rhos[i],
                               f0=f0, f1=f1)
                else:
                    m.update_V(s[i], s_n[i], r[i], f0=f0, f1=f1)
                if i % error_every == 0 and not episodic:
                    cur_theta = m.theta
                    for i_e in range(len(criteria)):
                        if isinstance(m, td.LinearValueFunctionPredictor):
                            errors[k, i_e, int(i / error_every)] = err_f[i_e](cur_theta)
                        else:
                            errors[k, i_e, int(i / error_every)] = err_f_gen[i_e](m.V)

        return errors

    def regularization_paths(self, methods, n_samples=1000, n_eps=1,
                             seed=1, criteria=["RMSBE"]):

        # Intialization
        self._init_methods(methods)
        err_f = [self._init_error_fun(criterion) for criterion in criteria]

        errors = dict([(crit, [[] for m in methods]) for crit in criteria])
        for m in methods:
            m.reset_trace()

        # Generate trajectories
        s, a, r, s_n, restarts = self.mdp.samples_cached(n_iter=n_samples,
                                                         n_restarts=n_eps,
                                                         policy=self.behavior_policy,
                                                         seed=seed)
        if self.off_policy:
            m_a_beh = policies.mean_action_trajectory(self.behavior_policy, s)
            m_a_tar = policies.mean_action_trajectory(self.target_policy, s)
            rhos = np.zeros_like(r)
            self.rhos = rhos

        # Method learning
        for i in xrange(n_samples * n_eps):
            f0 = self.phi(s[i])
            f1 = self.phi(s_n[i])
            if restarts[i]:
                for k, m in enumerate(methods):
                    m.reset_trace()

            for k, m in enumerate(methods):
                if self.off_policy:
                    rhos[i] = self.target_policy.p(s[i], a[i], mean=m_a_tar[i]) / self.behavior_policy.p(s[i], a[i], mean=m_a_beh[i])
                    m.update_V(s[i], s_n[i], r[i],
                               rho=rhos[i],
                               f0=f0, f1=f1)
                else:
                    m.update_V(s[i], s_n[i], r[i], f0=f0, f1=f1)
        for i,m in enumerate(methods):
            v = m.regularization_path()
            for tau, theta in v:
                for i_e, crit in enumerate(criteria):
                    errors[crit][i].append((tau, theta, err_f[i_e](theta)))

        return errors

    def _init_error_fun(self, criterion, general=False):
        if criterion == "MSE":
            err_f = self.MSE
        elif criterion == "RMSE":
            err_o = self.MSE
            err_f = lambda x: np.sqrt(err_o(x))
        elif criterion == "MSPBE":
            err_f = self.MSPBE
        elif criterion == "MSBE":
            err_f = self.MSBE
        elif criterion == "RMSPBE":
            err_o = self.MSPBE
            err_f = lambda x: np.sqrt(err_o(x))
        elif criterion == "RMSBE":
            err_o = self.MSBE
            err_f = lambda x: np.sqrt(err_o(x))
        elif criterion == "MSPBE_tar":
            err_f = self.MSPBE_tar
        elif criterion == "MSBE_tar":
            err_f = self.MSBE_tar
        elif criterion == "RMSPBE_tar":
            err_o = self.MSPBE_tar
            err_f = lambda x: np.sqrt(err_o(x))
        elif criterion == "RMSBE_tar":
            err_o = self.MSBE_tar
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
            policy = policies.DiscreteUniform(
                len(self.mdp.states), len(self.mdp.actions))
        self.behavior_policy = policy

        if target_policy is not None:
            self.off_policy = True
            self.target_policy = target_policy
        else:
            self.target_policy = policy
            self.off_policy = False

    def __getattr__(self, name):
        """
        some attribute such as state distribution or the true value function
        are very costly to compute, so they are only evaluated, if really needed
        """
        if name == "mu":
            self.mu = self.mdp.stationary_distribution(
                seed=1000, iterations=100000, policy=self.target_policy)
            return self.mu
        elif name == "beh_mu":
            self.beh_mu = self.mdp.stationary_distribution(
                seed=1000, iterations=100000, policy=self.behavior_policy)
            return self.beh_mu
        elif name == "V_true":
            self.V_true = dynamic_prog.estimate_V_discrete(
                self.mdp, policy=self.target_policy, gamma=self.gamma)
            return self.V_true
        else:
            raise AttributeError(name)

    def projection_operator(self):

        D = np.diag(self.mu)
        Pi = self.Phi * np.linalg.pinv(self.Phi.T * D * self.Phi) * \
            self.Phi.T * D
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
                if hasattr(self.phi, "expectation"):
                    f = self.phi.expectation(s)
                else:
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
            R = np.sum(R, axis=1)  # sum over all A
            R = np.sum(R, axis=1)  # sum over all S'
            self.R = R

        if hasattr(self, "T_P"):
            P = self.T_P
        else:
            P = self.mdp.P * policy.tab[:, :, np.newaxis]
            P = np.sum(P, axis=1)  # sum over all A => p(s' | s)
            self.T_P = P

        return R + self.gamma * np.dot(P, V)

    def MSE(self, theta):
        return np.sum(((theta * np.asarray(self.Phi)).sum(axis=1) - self.V_true) ** 2 * self.beh_mu)

    def MSBE(self, theta):

        V = (theta * np.asarray(self.Phi)).sum(axis=1)
        v = np.asarray(V - self.bellman_operator(V))
        return np.sum(v ** 2 * self.beh_mu)

    def MSPBE(self, theta):

        V = (theta * np.asarray(self.Phi)).sum(axis=1)
        v = np.asarray(
            V - np.dot(self.projection_operator(), self.bellman_operator(V)))
        return np.sum(v ** 2 * self.beh_mu)


class LinearContinuousValuePredictionTask(LinearValuePredictionTask):
    """
    A task to perform value function prediction of an mdp. It provides handy
    methods to evaluate different algorithms on the same problem setting.
    """

    def __init__(
        self, mdp, gamma, phi, theta0, policy, target_policy=None, normalize_phi=False, mu_iter=1000,
            mu_restarts=5, mu_seed=1000, mu_subsample=1, mu_next=20):
        self.mdp = mdp
        self.mu_n_next = mu_next
        self.mu_iter = mu_iter
        self.mu_seed = mu_seed
        self.mu_restarts = mu_restarts
        self.gamma = gamma
        self.phi = phi
        self.theta0 = theta0
        self.behavior_policy = policy
        self.mu_subsample = mu_subsample
        if target_policy is not None:
            self.off_policy = True
            self.target_policy = target_policy
        else:
            self.target_policy = policy
            self.off_policy = False
        if normalize_phi:
            mu, _, _, _, _ = self.mdp.samples_cached(policy=self.target_policy,
                                                     n_iter=self.mu_iter,
                                                     n_restarts=self.mu_restarts,
                                                     no_next_noise=True,
                                                     seed=self.mu_seed)
            Phi = util.apply_rowise(phi, mu)
            phi.normalization = np.std(Phi, axis=0)
            phi.normalization[phi.normalization == 0] = 1.

    def projection_operator(self):
        if hasattr(self, "Pi"):
            return self.Pi
        else:
            self.Pi = self.proj(self.mu_phi)
        return self.Pi

    @staticmethod
    #@memory.cache
    def proj(mu):
        m = np.matrix(mu)
        minv = np.linalg.pinv(m.T * m)
        return m * minv * m.T

    def kl_policy(self):
        """ computes the KL Divergence between the behavioral and target policy
        while assuming that the steady state distribution is the state distribution of the
        behavioral policy!
        """
        r = .5 * (np.trace(np.dot(self.behavior_policy.precision, self.target_policy.noise))
                  - self.behavior_policy.dim_A - np.log(np.linalg.det(self.target_policy.noise) / np.linalg.det(self.behavior_policy.noise)))

        dtheta = (self.behavior_policy.theta - self.target_policy.theta)
        da = np.dot(dtheta, self.mu.T)
        m = float(np.sum(
            da * np.dot(self.target_policy.precision, da))) / self.mu.shape[0]

        r += .5 * m
        return r

    def __getattr__(self, name):
        """
        some attribute such as state distribution or the true value function
        are very costly to compute, so they are only evaluated, if really needed
        """
        if name == "mu" or name == "mu_next" or name == "mu_r" or name == "mu_phi" or name == "mu_phi_next":
            self.mu, self.mu_r, self.mu_next, self.mu_phi, self.mu_phi_next = mdp.samples_distribution(self.mdp, policy=self.target_policy,
                                                                                                       policy_traj=self.behavior_policy,
                                                                                                       phi=self.phi,
                                                                                                       n_next=self.mu_n_next,
                                                                                                       n_iter=self.mu_iter,
                                                                                                       n_restarts=self.mu_restarts,
                                                                                                       seed=self.mu_seed,
                                                                                                       n_subsample=self.mu_subsample)
            return self.__dict__[name]
        elif name == "mu_tar" or name == "mu_next_tar" or name == "mu_r_tar" or name == "mu_phi_tar" or name == "mu_phi_next_tar":
            self.mu_tar, self.mu_r_tar, self.mu_next_tar, self.mu_phi_tar, self.mu_phi_next_tar = mdp.samples_distribution(self.mdp, policy=self.target_policy,
                                                                                                                           phi=self.phi,
                                                                                                                           n_next=self.mu_n_next,
                                                                                                                           n_iter=self.mu_iter,
                                                                                                                           n_restarts=self.mu_restarts,
                                                                                                                           seed=self.mu_seed,
                                                                                                                           n_subsample=self.mu_subsample)
            return self.__dict__[name]

        else:
            raise AttributeError(name)

    def MSPBE_tar(self, theta):
        """ Mean Squared Bellman Error """
        V = np.array((theta * self.mu_phi_tar).sum(axis=1))
        V2 = self.gamma * np.array((theta * self.mu_phi_next_tar).sum(axis=1))
        return np.mean(np.array((V - np.dot(self.projection_operator(), V2 + self.mu_r_tar))) ** 2)

    def MSPBE(self, theta):
        """ Mean Squared Bellman Error """
        V = np.array((theta * self.mu_phi).sum(axis=1))
        V2 = self.gamma * np.array((theta * self.mu_phi_next).sum(axis=1))
        return np.mean(np.array((V - np.dot(self.projection_operator(), V2 + self.mu_r))) ** 2)

    def MSBE_tar(self, theta):
        """ Mean Squared Bellman Error """
        V = np.array((theta * self.mu_phi_tar).sum(axis=1))
        V2 = self.gamma * np.array((theta * self.mu_phi_next_tar).sum(axis=1))
        return np.mean((V - V2 - self.mu_r_tar) ** 2)

    def MSBE(self, theta):
        """ Mean Squared Bellman Error """
        V = np.array((theta * self.mu_phi).sum(axis=1))
        V2 = self.gamma * np.array((theta * self.mu_phi_next).sum(axis=1))
        return np.mean((V - V2 - self.mu_r) ** 2)


class LinearLQRValuePredictionTask(LinearContinuousValuePredictionTask):

    def __getattr__(self, name):
        """
        some attribute such as state distribution or the true value function
        are very costly to compute, so they are only evaluated, if really needed
        """
        if name == "V_true":
            self.V_true = dynamic_prog.estimate_V_LQR(
                self.mdp, lambda x, y: self.bellman_operator(
                    x, y, policy="target"),
                gamma=self.gamma)
            return self.V_true
        elif name == "mu_phi_full":
            self.mu_phi_full = util.apply_rowise(
                features.squared_tri(self.mdp.dim_S), self.mu)
            return self.mu_phi_full
        else:
            return LinearContinuousValuePredictionTask.__getattr__(self, name)

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
                S = A + B * theta
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
                S = A + B * theta
                self.S_target = S
            if hasattr(self, "C_target"):
                C = self.C_target
            else:
                C = Q + theta.T * R * theta
                self.C_target = C
        else:
            theta = np.matrix(policy)
            noise = policy.noise
            S = A + B * theta
            C = Q + theta.T * R * theta

        Pn = C + self.gamma * (S.T * np.matrix(P) * S)
        bn = self.gamma * (b + np.trace(np.matrix(P) * np.matrix(Sigma))) \
            + np.trace((R + self.gamma * B.T *
                        np.matrix(P) * B) * np.matrix(np.diag(noise)))
        return Pn, bn

    def expected_reward_operator(self, P, b, policy="behavior"):
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
                S = A + B * theta
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
                S = A + B * theta
                self.S_target = S
            if hasattr(self, "C_target"):
                C = self.C_target
            else:
                C = Q + theta.T * R * theta
                self.C_target = C
        else:
            theta = np.matrix(policy)
            noise = policy.noise
            S = A + B * theta
            C = Q + theta.T * R * theta

        Pn = C
        bn = np.trace((R) * np.matrix(noise))
        return Pn, bn

    def MSE(self, theta):
        p = features.squared_tri(self.mdp.dim_S).param_forward(*self.phi.param_back(theta)) -\
            features.squared_tri(self.mdp.dim_S).param_forward(*self.V_true)
        return np.mean((p * self.mu_phi_full).sum(axis=1) ** 2)

    def MSBE(self, theta):
        """ Mean Squared Bellman Error """
        V = np.array((theta * self.mu_phi).sum(axis=1))
        theta_trans = features.squared_tri(self.mdp.dim_S).param_forward(
            *self.bellman_operator(*self.phi.param_back(theta)))
        V2 = np.array((theta_trans * self.mu_phi_full).sum(axis=1))
        return np.mean((V - V2) ** 2)

    def MSPBE(self, theta):
        """ Mean Squared Projected Bellman Error """
#       return LinearContinuousValuePredictionTask.MSPBE(self, theta)
        V = np.matrix((theta * np.asarray(self.mu_phi)).sum(axis=1)).T
        theta_trans = features.squared_tri(self.mdp.dim_S).param_forward(
            *self.bellman_operator(*self.phi.param_back(theta)))
        v = np.asarray(V - self.projection_operator(
        ) * np.matrix(self.mu_phi_full) * np.matrix(theta_trans).T)
        return np.mean(v ** 2)
