# -*- coding: utf-8 -*-
"""
Markov Decision Processes with finite state and action spaces

Created on Fri Dec  9 19:01:41 2011

@author: Christoph Dann <cdann@cdann.de>
"""
import numpy as np
import matplotlib.pyplot as plt
from joblib import Parallel, delayed
import policies
from util import multinomial_sample
from joblib import Memory

#memory = Memory(cachedir="./cache", verbose=20)
memory = Memory(cachedir="/BS/latentCRF/nobackup/td", verbose=50)


def _false(x):
    return False

@memory.cache(hashfun={"mymdp": repr, "policy": repr})
def samples_cached(mymdp, policy, n_iter=1000, n_restarts=100,
                 no_next_noise=False, seed=1):
    assert(seed is not None)
    states = np.ones([n_restarts * n_iter, mymdp.dim_S])
    states_next = np.ones([n_restarts * n_iter, mymdp.dim_S])
    actions = np.ones([n_restarts * n_iter, mymdp.dim_A])
    rewards = np.ones(n_restarts * n_iter)
    np.random.seed(seed)

    restarts = np.zeros(n_restarts * n_iter, dtype="bool")
    k=0
    while k < n_restarts * n_iter:
        restarts[k] = True
        for s,a,s_n, r in mymdp.sample_transition(n_iter, policy, with_restart=False, 
                                                 no_next_noise=no_next_noise, seed=None):
            states[k,:] = s
            states_next[k,:] = s_n
            rewards[k] = r
            actions[k,:] = a

            k+=1
            if k >= n_restarts * n_iter:
                break
    return states ,actions, rewards, states_next, restarts

class ContinuousMDP(object):


    def __init__(self, sf, rf, dim_S, dim_A, start, terminal_f = None, Sigma=0.):
        self.sf = sf
        self.rf = rf
        self.dim_S = dim_S
        self.dim_A = dim_A
        if terminal_f is None:
            terminal_f = _false
        self.terminal_f = terminal_f
        if not hasattr(start, '__call__'):
            self.start_state=start
            startf = lambda: self.start_state.copy()
        else:
            self.start_state=None
            startf = start
        self.start = startf
        if isinstance(Sigma, (float, int, long)):
            self.Sigma = Sigma*np.ones(self.dim_S)
        else:
            assert Sigma.shape == (self.dim_S,)
            self.Sigma = Sigma
        self.__setstate__(self.__dict__)

    def __getstate__(self):
        res = self.__dict__.copy()
        if "start_state" in res:
            del res["start"]
        del res["samples_featured"]        
        #del res["samples_cached"]
        return res

    def __setstate__(self, state):
        self.__dict__ = state
        if "start" not in state:
            self.start = lambda: self.start_state.copy()
        self.samples_featured = memory.cache(self.samples_featured)   

    def samples(self, policy,n_iter=1000, n_restarts=100,
                     no_next_noise=False, seed=None):
        states = np.empty((n_restarts * n_iter, self.dim_S))
        states_next = np.empty((n_restarts * n_iter, self.dim_S))
        actions = np.empty((n_restarts * n_iter, self.dim_A))
        rewards = np.empty((n_restarts * n_iter))
        k=0
        for i in xrange(n_restarts):
            for s,a,s_n, r in self.sample_transition(n_iter, policy, with_restart=False,
                                                    no_next_noise=no_next_noise, seed=seed):
                states[k,:] = s
                states_next[k,:] = s_n
                rewards[k] = r
                actions[k,:] = a
                k+=1

        return states[:k,:],actions[:k,:], rewards[:k], states_next[:k,:]

    def samples_cached(self, *args, **kwargs):
        return samples_cached(self, *args, **kwargs)
       
    def samples_featured(self, phi, policy, n_iter=1000, n_restarts=100,
                     no_next_noise=False, seed=1, n_subsample=1):
        assert(seed is not None)
        s,a,r,sn,restarts = self.samples_cached(policy, n_iter, n_restarts, no_next_noise, seed)        
             
        n_feat = len(phi(np.zeros(self.dim_S)))
        feats = np.empty([int(n_restarts * n_iter / float(n_subsample)), n_feat])
        feats_next = np.empty([int(n_restarts * n_iter / float(n_subsample)),n_feat])  
        i=0
        l = range(0, n_restarts * n_iter, n_subsample)
        for k in xrange(n_iter * n_restarts):
            if k % n_subsample == 0:
                
                feats[i,:] = phi(s[k])
                feats_next[i,:] = phi(sn[k])
                i += 1                
        return s[l] ,a[l], r[l], sn[l], restarts[l], feats, feats_next


    def sample_transition(self, max_n, policy, seed=None, with_restart = False, no_next_noise=False):
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

        rands = np.random.multivariate_normal(np.zeros(self.dim_S), np.diag(self.Sigma), max_n)
        i=0
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
                if no_next_noise:
                    yield (s0, a, np.array(mean).flatten(), r)
                else:
                    yield (s0, a, s1, r)
                i+=1
                s0 = s1

    def sample_step(self, s0 , policy=None, seed=None, no_next_noise=False):
        """
        samples one step from the MDP
        returns a transition tuple (X_n, A, X_n+1, R)
        """

        if seed is not None:
            np.random.seed(seed)

        rands = np.random.multivariate_normal(np.zeros(self.dim_S), np.diag(self.Sigma), 1)
        a = policy(s0)
        mean = self.sf(s0, a)
        if not no_next_noise:    
            s1 = mean + rands[0]
        else:
            s1 = mean
        r = self.rf(s0, a)
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
            self.Sigma = np.ones(self.dim_S)*Sigma
            #self.Sigma = np.eye(self.dim_S) * float(Sigma)
        else:
            assert Sigma.shape == (self.dim_S,)
            self.Sigma = Sigma
            
        if not hasattr(start, '__call__'):
            self.start_state=start
            startf = lambda: self.start_state.copy()
        else:
            self.start_state=None
            startf = start
        self.start = startf
        assert A.shape[1] == self.dim_S
        assert B.shape[0] == self.dim_S
        self.__setstate__(self.__dict__)

    def statefun(self, s0, a):
        return np.dot(self.A,s0) + np.dot(self.B,a)
    
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

    def tabular_phi(self, state):
        """
        feature function that makes linear approximation equivalent to
        tabular algorithms
        """
        result = np.zeros(len(self.states))
        result[state] = 1.
        return result


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

    def uniform_policy(self):
        """
        returns the uniform policy in the form of a numpy array of shape 
        (n_s, n_a)        
        """
        policy = np.zeros((len(self.states), len(self.actions)))
        policy[self.valid_actions] = 1
        policy /= policy.sum(axis=1).reshape(-1, 1)
        return self.make_policy_fun(policy)

    def make_policy_fun(self, policy_table):
        """
        generates a python function for a policy given as a probability table
                S x A -> R
                numpy array of shape (n_s, n_a)
                pi(s,a) is the probability of taking action a in state
        """
        o = lambda s: multinomial_sample(1, policy_table[s, :])
        o.tab = policy_table
        return o


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
        mu =  mu / mu.sum()
        return mu
        
    def samples_cached(self, policy, n_iter=1000, n_restarts=100,
                     no_next_noise=False, seed=1):
                         
        assert (no_next_noise==False)
        assert(seed is not None)
        states = np.ones([n_restarts * n_iter, self.dim_S])
        states_next = np.ones([n_restarts * n_iter, self.dim_S])
        actions = np.ones([n_restarts * n_iter, self.dim_A])
        rewards = np.ones(n_restarts * n_iter)
       
        restarts = np.zeros(n_restarts * n_iter, dtype="bool")
        k=0
        while k < n_restarts * n_iter:
            restarts[k] = True
            for s,a,s_n, r in self.sample_transition(n_iter, policy, with_restart=False, 
                                                     seed=seed):
                states[k,:] = s
                states_next[k,:] = s_n
                rewards[k] = r
                actions[k,:] = a

                k+=1
                if k >= n_restarts * n_iter:
                    break
        return states ,actions, rewards, states_next, restarts
       
    def samples_featured(self, phi, policy, n_iter=1000, n_restarts=100,
                     no_next_noise=False, seed=1, n_subsample=1):
        assert(seed is not None)
        s,a,r,sn,restarts = self.samples_cached(policy, n_iter, n_restarts, no_next_noise, seed)        
             
        n_feat = len(phi(0))
        feats = np.empty([n_restarts * n_iter, n_feat])
        feats_next = np.empty([n_restarts * n_iter,n_feat])  
        
        for k in xrange(n_iter * n_restarts):
                
            feats[k,:] = phi(s[k])
            feats_next[k,:] = phi(sn[k])
                
        return s ,a, r, sn, restarts, feats, feats_next
                                                                

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


    def sample_transition(self, max_n, policy="uniform", seed=None,
                                    with_restart=True):
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
        if policy is "uniform":
            policy = self.uniform_policy()

        i=0
        while i < max_n:        
            s0 = multinomial_sample(1, self.P0)
            while i < max_n:  
                if self.s_terminal[s0]:
                        break
                a = policy(s0)
                s1 = multinomial_sample(1, self.P[s0, a])
                r = self.r[s0, a, s1]
                yield (s0, a, s1, r)
                i+=1
                s0 = s1
            if not with_restart:
                break

    def policy_P(self, policy="uniform"):
        if policy is "uniform":
            policy = self.uniform_policy()
        T = self.P * policy.tab[:, :, np.newaxis]
        T = np.sum(T, axis=1)
        return T

    def sample_episodes(self, n, max_len=1000, policy="uniform"):
        """
        generate a n markov chain realizations (episodes) by sampling

            n: numer of episodes
            max_len: maximum length of each episode
            policy pi: S x A -> R
                numpy array of shape (n_s, n_a)
                pi(s,a) is the probability of taking action a in state s
        """
        episodes = []
        s0 = multinomial_sample(n, self.P0)

        if policy is "uniform":
            policy = self.uniform_policy()

        cur_eps = np.zeros(max_len * 2 + 1, dtype="uint")
        for i_ep in xrange(n):
            cur_eps[0] = s = s0[i_ep]
            for i in xrange(max_len):
                if self.s_terminal[s]:
                    break
                a = policy(s)
                s = multinomial_sample(1, self.P[s, a])
                cur_eps[2 * i + 1:2 * i + 3] = a, s

            episodes.append(cur_eps[:2 * i + 1].copy())
        return episodes


