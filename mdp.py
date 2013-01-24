# -*- coding: utf-8 -*-
"""
Markov Decision Processes with finite state and action spaces

Created on Fri Dec  9 19:01:41 2011

@author: Christoph Dann <cdann@cdann.de>
"""
import numpy as np
from util import multinomial_sample, memory, apply_rowise
from util.progressbar import ProgressBar
from joblib import Parallel

def _false(x):
    return False


@memory.cache(hashfun={"mymdp": repr, "policy": repr}, ignore=["verbose"])
def samples_cached(mymdp, policy, n_iter=1000, n_restarts=100,
                   no_next_noise=False, seed=1., verbose=0.):
    assert(seed is not None)
    states = np.ones([n_restarts * n_iter, mymdp.dim_S])
    states_next = np.ones([n_restarts * n_iter, mymdp.dim_S])
    actions = np.ones([n_restarts * n_iter, mymdp.dim_A])
    rewards = np.ones(n_restarts * n_iter)
    np.random.seed(seed)

    restarts = np.zeros(n_restarts * n_iter, dtype="bool")
    k = 0

    with ProgressBar(enabled=(verbose > 2.)) as p:
        while k < n_restarts * n_iter:
            restarts[k] = True
            for s, a, s_n, r in mymdp.sample_transition(
                    n_iter, policy, with_restart=False, seed=None):
                states[k, :] = s
                states_next[k, :] = s_n
                rewards[k] = r
                actions[k, :] = a

                k += 1
                p.update(k, n_restarts * n_iter)
                if k >= n_restarts * n_iter:
                    break
    return states, actions, rewards, states_next, restarts


@memory.cache(hashfun={"mymdp": repr, "policy": repr})
def samples_cached_transitions(mymdp, policy, states, seed=2):
    assert(seed is not None)
    n = states.shape[0]
    states_next = np.ones([n, mymdp.dim_S])
    actions = np.ones([n, mymdp.dim_A])
    rewards = np.ones(n)
    np.random.seed(seed)

    for k in xrange(n):
        _, a, s_n, r = mymdp.sample_step(states[k], policy=policy)
        states_next[k, :] = s_n
        rewards[k] = r
        actions[k, :] = a

    return actions, rewards, states_next


@memory.cache(hashfun={"mymdp": repr, "policy": repr}, ignore=["verbose"])
def samples_distribution_from_states(mymdp, policy, phi, states, n_next=20, seed=1, verbose=True):
    n = states.shape[0]
    states_next = np.ones([n, mymdp.dim_S])
    feat = np.zeros((n, phi.dim))
    feat_next = np.zeros_like(feat)
    rewards = np.ones(n)
    np.random.seed(seed)

    with ProgressBar(enabled=verbose) as p:
        for k in xrange(n):
            p.update(k, n, "Sampling MDP Distribution")
            s = states[k, :]
            s0, a, s1, r = mymdp.sample_step(
                s, policy=policy, n_samples=n_next)
            states[k, :] = s0
            feat[k, :] = phi(s0)
            fn = apply_rowise(phi, s1)
            feat_next[k, :] = np.mean(fn, axis=0)
            states_next[k, :] = np.mean(s1, axis=0)
            rewards[k] = np.mean(r)

    return states, rewards, states_next, feat, feat_next


@memory.cache(hashfun={"mymdp": repr, "policy": repr, "policy_traj": repr}, ignore=["verbose"])
def samples_distribution(mymdp, policy, phi, policy_traj=None, n_subsample=1,
                         n_iter=1000, n_restarts=100, n_next=20, seed=1, verbose=True):
    assert(n_subsample == 1)  # not implemented, do that if you need it
    states = np.ones([n_restarts * n_iter, mymdp.dim_S])
    if policy_traj is None:
        policy_traj = policy
    states_next = np.ones([n_restarts * n_iter, mymdp.dim_S])
    feat = np.zeros((n_restarts * n_iter, phi.dim))
    feat_next = np.zeros_like(feat)
    rewards = np.ones(n_restarts * n_iter)
    np.random.seed(seed)

    k = 0
    s = mymdp.start()
    c = 0
    with ProgressBar(enabled=verbose) as p:
        for k in xrange(n_restarts * n_iter):
            if mymdp.terminal_f(s) or c >= n_iter:
                s = mymdp.start()
                c = 0
            p.update(k, n_restarts * n_iter, "Sampling MDP Distribution")
            s0, a, s1, r = mymdp.sample_step(
                s, policy=policy, n_samples=n_next)
            states[k, :] = s0
            feat[k, :] = phi(s0)
            fn = apply_rowise(phi, s1)
            feat_next[k, :] = np.mean(fn, axis=0)
            states_next[k, :] = np.mean(s1, axis=0)
            rewards[k] = np.mean(r)
            _, _, s, _ = mymdp.sample_step(s, policy=policy_traj, n_samples=1)
            c += 1

    return states, rewards, states_next, feat, feat_next


def run1(*args, **kwargs):
    return accum_reward_for_states(*args, **kwargs)

@memory.cache(hashfun={"mymdp": repr, "policy": repr}, ignore=["verbose", "n_jobs"])
def accum_reward_for_states(mymdp, policy, states, gamma, n_eps, l_eps, seed, verbose=3, n_jobs=24):
    n = states.shape[0]
    rewards = np.ones(n)
    if n_jobs == 1:
        with ProgressBar(enabled=(verbose >= 1)) as p:
            for k in xrange(n):
                p.update(k, n, "Sampling acc. reward")
                np.random.seed(seed)
                r = mymdp.sample_accum_reward(states[k], gamma, policy, n_eps=n_eps, l_eps=l_eps)
                rewards[k] = np.mean(r)
    else:
        jobs = []
        b = int(n / n_jobs)+1
        k = 0
        while k < n:
            kp = min(k+b, n)
            jobs.append((run1, [mymdp, policy, states[k:kp], gamma, n_eps, l_eps, seed], {"verbose": verbose-1, "n_jobs": 1}))
            k = kp
        res = Parallel(n_jobs=n_jobs, verbose=verbose)(jobs)
        rewards = np.concatenate(res, axis=0).mean(axis=1)
    return rewards


class ContinuousMDP(object):

    def __init__(self, sf, rf, dim_S, dim_A, start, terminal_f=None, Sigma=0.):
        self.sf = sf
        self.rf = rf
        self.dim_S = dim_S
        self.dim_A = dim_A
        if terminal_f is None:
            terminal_f = _false
        self.terminal_f = terminal_f
        if not hasattr(start, '__call__'):
            self.start_state = start
            startf = lambda: self.start_state.copy()
        else:
            self.start_state = None
            startf = start
        self.start = startf
        if isinstance(Sigma, (float, int, long)):
            self.Sigma = Sigma * np.ones(self.dim_S)
        else:
            assert Sigma.shape == (self.dim_S,)
            self.Sigma = Sigma
        self.__setstate__(self.__dict__)

    def __getstate__(self):
        res = self.__dict__.copy()
        if "start_state" in res:
            del res["start"]
        #del res["samples_featured"]
        #del res["samples_cached"]
        return res

    def __setstate__(self, state):
        self.__dict__ = state
        if "start" not in state:
            self.start = lambda: self.start_state.copy()

    def samples(self, policy, n_iter=1000, n_restarts=100,
                no_next_noise=False, seed=None):
        states = np.empty((n_restarts * n_iter, self.dim_S))
        states_next = np.empty((n_restarts * n_iter, self.dim_S))
        actions = np.empty((n_restarts * n_iter, self.dim_A))
        rewards = np.empty((n_restarts * n_iter))
        k = 0
        for i in xrange(n_restarts):
            for s, a, s_n, r in self.sample_transition(
                n_iter, policy, with_restart=False,
                    no_next_noise=no_next_noise, seed=seed):
                states[k, :] = s
                states_next[k, :] = s_n
                rewards[k] = r
                actions[k, :] = a
                k += 1

        return states[:k, :], actions[:k, :], rewards[:k], states_next[:k, :]

    def samples_cached(self, *args, **kwargs):
        return samples_cached(self, *args, **kwargs)

    def samples_cached_transitions(self, *args, **kwargs):
        return samples_cached_transitions(self, *args, **kwargs)

    def samples_featured(self, phi, policy, n_iter=1000, n_restarts=100,
                         n_next=1, seed=None, n_subsample=1):
        assert(seed is not None)
        s, a, r, sn, = samples_distribution(self, policy=policy,
                                            n_iter=n_iter, n_next=n_next, n_restarts=n_restarts, seed=seed)

        n_feat = len(phi(np.zeros(self.dim_S)))
        feats = np.empty(
            [int(n_restarts * n_iter / float(n_subsample)), n_feat])
        feats_next = np.empty(
            [int(n_restarts * n_iter / float(n_subsample)), n_feat])
        i = 0
        l = range(0, n_restarts * n_iter * n_next, n_subsample)
        for k in xrange(n_iter * n_restarts):
            if k % n_subsample == 0:
                feats[i, :] = phi(s[k])
                feats_next[i, :] = phi(sn[k])
                i += 1
        return s[l], a[l], r[l], sn[l], feats, feats_next

    def sample_transition(self, max_n, policy, seed=None, with_restart=False, no_next_noise=False):
        """
        generator that samples from the MDP
        be aware that this chains can be infinitely long
        the chain is restarted if the policy changes

            max_n: maximum number of samples to draw

            policy: python function S -> A that gets the current state and
                returns the action to take

            seed: optional seed for the random generator to generate
                deterministic samples

            returns a transition tuple (X_n, A, X_n+1, R)
        """

        if seed is not None:
            np.random.seed(seed)

        rands = np.random.randn(
            max_n, self.dim_S) * np.sqrt(self.Sigma[None, :])
        i = 0
        while i < max_n:
            s0 = self.start()
            while i < max_n:
                if self.terminal_f(s0):
                    if with_restart:
                        break
                    else:
                        return
                a = policy(s0)
                mean = self.sf(s0, a)
                s1 = mean + rands[i]

                r = self.rf(s0, a)
                yield (s0, a, s1, r)
                i += 1
                s0 = s1

    def sample_accum_reward(self, s0, gamma, policy, n_eps=10, l_eps=200):
        r = np.zeros(n_eps)
        for n in xrange(n_eps):
            s = s0
            g = 1.
            rands = np.random.randn(
                l_eps, self.dim_S) * np.sqrt(self.Sigma[None, :])
            for l in xrange(l_eps):
                a = policy(s, 1)
                s = self.sf(s,a)
                r[n] += self.rf(s,a) * g
                g *= gamma
        return r

    def sample_step(self, s0, policy, seed=None, n_samples=1):
        """
        samples one step from the MDP
        returns a transition tuple (X_n, A, X_n+1, R)
        """

        if seed is not None:
            np.random.seed(seed)

        rands = np.random.randn(
            n_samples, self.dim_S) * np.sqrt(self.Sigma[None, :])
        a = policy(s0, n_samples)
        if n_samples == 1:
            mean = self.sf(s0, a)
            s1 = mean + rands.flatten()
            r = self.rf(s0, a)
        else:
            s1 = np.zeros((n_samples, self.dim_S))
            r = np.zeros(n_samples)
            for i in xrange(n_samples):
                s1[i, :] = self.sf(s0, a[i])
                r[i] = self.rf(s0, a[i])
            s1 += rands

        return (s0, a, s1, r)


class LQRMDP(ContinuousMDP):
    """
        Linear Quadratic MDP with continuous states and actions
        but time discrete transitions

    """
    def __init__(self, A, B, Q, R, start, Sigma, terminal_f=_false):
        """The MDP is defined by the state transition kernel:
                s' ~ Normal(As + Ba, Sigma)
            and the reward
                r(s,a) = s^T Q s + a^T R a
            terminal_f: python function S -> Bool that returns True exactly if
                s is a terminal state
            start_f: start state as ndarray
        """

        self.dim_S = A.shape[0]
        self.dim_A = B.shape[1]
        self.A = A
        self.B = B
        self.Q = Q
        self.R = R
        self.terminal_f = terminal_f
        if isinstance(Sigma, (float, int, long)):
            self.Sigma = np.ones(self.dim_S) * Sigma
            #self.Sigma = np.eye(self.dim_S) * float(Sigma)
        else:
            assert Sigma.shape == (self.dim_S,)
            self.Sigma = Sigma

        if not hasattr(start, '__call__'):
            self.start_state = start
            startf = lambda: self.start_state.copy()
        else:
            self.start_state = None
            startf = start
        self.start = startf
        assert A.shape[1] == self.dim_S
        assert B.shape[0] == self.dim_S
        self.__setstate__(self.__dict__)

    def statefun(self, s0, a):
        return np.dot(self.A, s0) + np.dot(self.B, a)

    def rewardfun(self, s0, a):
        return np.dot(s0.T, np.dot(self.Q, s0)) + np.dot(a.T, np.dot(self.R, a))

    def __getstate__(self):
        res = ContinuousMDP.__getstate__(self)
        del res["rf"]
        del res["sf"]
        return res

    def __setstate__(self, state):
        ContinuousMDP.__setstate__(self, state)
        self.sf = self.statefun
        self.rf = self.rewardfun


class MDP(object):
    """
    Markov Decision Process

    consists of:
        states S:       list or n_s dimensional numpy array of states
        actions A:      list or n_a dimensional numpy array of actions
        reward_function r: S x A x S -> R
                            numpy array of shape (n_s, n_a, n_s)

                            r(s,a,s') assigns a real valued reward to the
                            transition from state s taking action a and going
                            to state s'

        state_transition_kernel P: S x A x S -> R
                            numpy array of shape (n_s, n_a, n_s)
                            p(s,a,s') assign the transition from s to s' by
                            taking action a a probability

                            sum_{s'} p(s,a,s') = 0 if a is not a valid action
                            in state s, otherwise 1
                            if p(s,a,s) = 1 for each a, s is a terminal state

        start distribution P0: S -> R
                            numpy array of shape (n_s,)
                            defines the distribution of initial states
    """

    def __init__(self, states, actions, reward_function,
                 state_transition_kernel,
                 start_distribution):
        self.state_names = states
        self.states = np.arange(len(states))
        self.action_names = actions
        self.actions = np.arange(len(actions))
        self.r = reward_function
        self.Phi = {}

        # start distribution testing
        self.P0 = np.asanyarray(start_distribution)
        assert np.abs(np.sum(self.P0) - 1) < 1e-12
        assert np.all(self.P0 >= 0)
        assert np.all(self.P0 <= 1)

        self.dim_S = 1
        self.dim_A = 1
        # transition kernel testing
        self.P = np.asanyarray(state_transition_kernel)
        assert np.all(self.P >= 0)
        assert np.all(self.P <= 1)

        # extract valid actions and terminal state information
        sums_s = np.sum(self.P, axis=2)
        assert np.all(np.bitwise_or(np.abs(sums_s - 1) < 0.0001,
                                    np.abs(sums_s) < 0.0001))
        self.valid_actions = np.abs(sums_s - 1) < 0.0001

        self.s_terminal = np.asarray([np.all(self.P[s, :, s] == 1)
                                      for s in self.states])

    def extract_transitions(self, episode):
        """
        takes an episode (X_0, A_0, X_1, A_1, ..., X_n) of the MDP and
        procudes a list of tuples for each transition containing
         (X_n, A, X_n+1, R)
             X_n: previous state
             X_n+1: next state
             A: action
             R: associated reward
        """
        transitions = []
        for i in xrange(0, len(episode) - 2, 2):
            s, a, s_n = tuple(episode[i:i + 3])
            transitions.append((s, a, s_n, self.r[s, a, s_n]))

        return transitions

    def stationary_distribution(self, iterations=10000,
                                seed=None, avoid0=False, policy="uniform"):
        """
        computes the stationary distribution by sampling
        """
        cnt = np.zeros(len(self.states), dtype='uint64')
        for s, _, _, _ in self.sample_transition(max_n=iterations,
                                                 policy=policy, seed=seed):
            cnt[s] += 1
        if avoid0 and np.any(cnt == 0):
            cnt += 1
        mu = (cnt).astype("float")
        mu = mu / mu.sum()
        return mu

    def samples_cached(self, policy, n_iter=1000, n_restarts=100,
                       no_next_noise=False, seed=None):
        if seed is not None:
            np.random.seed(seed)
        assert (not no_next_noise)
        assert(seed is not None)
        states = np.ones([n_restarts * n_iter, self.dim_S])

        states_next = np.ones([n_restarts * n_iter, self.dim_S])
        actions = np.ones([n_restarts * n_iter, self.dim_A])
        rewards = np.ones(n_restarts * n_iter)

        restarts = np.zeros(n_restarts * n_iter, dtype="bool")
        k = 0
        while k < n_restarts * n_iter:
            restarts[k] = True
            for s, a, s_n, r in self.sample_transition(
                    n_iter, policy, with_restart=False):
                states[k, :] = s
                states_next[k, :] = s_n
                rewards[k] = r
                actions[k, :] = a

                k += 1
                if k >= n_restarts * n_iter:
                    break
        return states, actions, rewards, states_next, restarts

    def reward_samples(self, policy, n_iter=1000, n_restarts=100, seed=None):
        if seed is not None:
            np.random.seed(seed)
        rewards = np.zeros((len(self.states), n_restarts, n_iter))
        for s0 in self.states:
            for k in range(n_restarts):
                i = 0
                for s, a, s_n, r in self.sample_transition(
                        n_iter, policy, with_restart=False, s_start=s0):
                    rewards[s0, k, i] = r
                    i += 1

        return rewards

    def samples_cached_transitions(self, policy, states, seed=None):
        n = states.shape[0]
        sn = np.zeros_like(states)
        a = np.ones([n, self.dim_A])
        r = np.ones(n)
        for i in xrange(n):
            a[i] = policy(states[i])
            sn[i] = multinomial_sample(1, self.P[int(states[i]), int(a[i])])
            r[i] = self.r[int(states[i]), int(a[i]), int(sn[i])]
        return a, r, sn

    def samples_featured(self, phi, policy, n_iter=1000, n_restarts=100,
                         no_next_noise=False, seed=1, n_subsample=1):
        assert(seed is not None)
        s, a, r, sn, restarts = self.samples_cached(
            policy, n_iter, n_restarts, no_next_noise, seed)

        n_feat = len(phi(0))
        feats = np.empty([n_restarts * n_iter, n_feat])
        feats_next = np.empty([n_restarts * n_iter, n_feat])

        for k in xrange(n_iter * n_restarts):

            feats[k, :] = phi(s[k])
            feats_next[k, :] = phi(sn[k])

        return s, a, r, sn, restarts, feats, feats_next

    def synchronous_sweep(self, seed=None, policy="uniform"):
        """
        generate samples from the MDP so that exactly one transition from each
        non-terminal-state is yielded

        Parameters
        -----------
            policy pi: policy python function

            seed: optional seed for the random generator to generate
                deterministic samples

        Returns
        ---------
            transition tuple (X_n, A, X_n+1, R)
        """
        if seed is not None:
            np.random.seed(seed)
        if policy is "uniform":
            policy = self.uniform_policy()

        for s0 in self.states:
            if self.s_terminal[s0]:
                break
            a = policy(s0)
            s1 = multinomial_sample(1, self.P[s0, a])
            r = self.r[s0, a, s1]
            yield (s0, a, s1, r)

    def sample_transition(self, max_n, policy, seed=None,
                          with_restart=True, s_start=None):
        """
        generator that samples from the MDP
        be aware that this chains can be infinitely long
        the chain is restarted if the policy changes

            max_n: maximum number of samples to draw

            policy pi: policy python function

            seed: optional seed for the random generator to generate
                deterministic samples

            with_restart: determines whether sampling with automatic restart:
                is used

            returns a transition tuple (X_n, A, X_n+1, R)
        """

        if seed is not None:
            np.random.seed(seed)

        i = 0
        while i < max_n:
            if s_start is None:
                s0 = multinomial_sample(1, self.P0)
            else:
                s0 = s_start
            while i < max_n:
                if self.s_terminal[s0]:
                        break
                a = policy(s0)
                s1 = multinomial_sample(1, self.P[s0, a])
                r = self.r[s0, a, s1]
                yield (s0, a, s1, r)
                i += 1
                s0 = s1
            if not with_restart:
                break

    def policy_P(self, policy="uniform"):
        if policy is "uniform":
            policy = self.uniform_policy()
        T = self.P * policy.tab[:, :, np.newaxis]
        T = np.sum(T, axis=1)
        return T
