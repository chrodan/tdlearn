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

def test_lqrmdp():
    A = np.eye(2)
    B = np.eye(2)*0.5
    Q = np.eye(2)
    R = np.zeros((2,2))
    uut = LQRMDP(A,B,Q,R, 3)
    sc = []
    for s in uut.sample_transition(1000000):
        sc.append(s[0])
    d = np.vstack(tuple(sc))
    plt.plot(d[:,0], d[:,1], "o-")
    print d.shape
    plt.show()
    
def test_pole_balancing():
    r"""
    S = [\alpha, \dot \alpha, x, \dot x]
    A = [\ddot x]
    """
    dt = .01
    m = 1.
    l = 2.
    g = 9.81
    mu = 0.01
    A = np.array([[1., dt, 0, 0],
                  [g/l, 1 - (mu*dt)/m/l/l, 0 ,0],
                  [0., 0, 1, dt],
                  [0, 0, 0, 1]])
    B = np.array([0., dt/l, 0, dt]).reshape(4,1)
    Q = np.diag([-2., -.5, -0.1, 0])
    terminal_f = lambda x: np.abs(x[0]) > 1
    R = np.zeros((1,1))
    sigma = np.zeros((4,4))
    sigma[-1,-1] = 0.1
    uut = LQRMDP(A,B,Q,R, sigma, terminal_f=terminal_f, start_f=lambda : np.array([0.0001, 0, 0, 0]))
    sc = []
    for s in uut.sample_transition(1000, policy=lambda x: np.array([0])):
        sc.append(s[0])
    d = np.vstack(tuple(sc))
    plt.figure()    
    plt.plot(d[:,0], "b-")
    from dynamic_prog import solve_LQR
    theta, P =  solve_LQR(uut, n_iter=int(1e5))
    sc = []
    for s in uut.sample_transition(1000, policy=lambda x: np.dot(theta, x)):
        sc.append(s[0])
    d = np.vstack(tuple(sc))
    plt.plot(d[:,0], "r-")
    plt.show()

_false = lambda x: False

class LQRMDP(object):
    """
        Linear Quadratic MDP with continuous states and actions
        but time discrete transitions
        
    """
    def __init__(self, A, B, Q, R, Sigma, terminal_f=_false, start_f="zero"):
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
        self.start_f = start_f
        self.terminal_f = terminal_f
        if isinstance(Sigma, (float, int, long)):
            self.Sigma = np.eye(self.dim_S) * float(Sigma)
        else:
            assert Sigma.shape == (self.dim_S, self.dim_S)
            self.Sigma = Sigma
        assert A.shape[1] == self.dim_S
        assert B.shape[0] == self.dim_S
        

    def samples(self, phi, n_iter=1000, n_restarts=100,
                                        policy="linear", seed=None,  verbose=False):
        states = np.empty((n_restarts * n_iter, self.dim_S))
        actions = np.empty((n_restarts * n_iter, self.dim_A))
        rewards = np.empty((n_restarts * n_iter))
        k=0
        for i in xrange(n_restarts):
            for s,a,s_n, r in self.sample_transition(n_iter, policy, with_restart=False):
                states[k,:] = s
                rewards[k] = r
                actions[k,:] = a
                k+=1
        return states[:k,:],actions[:k,:], rewards[:k]

    def state_samples(self, phi, n_iter=1000, n_restarts=100,
                policy="linear", seed=None,  verbose=False):
        return self.samples(phi, n_iter, n_restarts, policy, seed, verbose)[0]

    def stationary_feature_distribution(self, phi, n_iter=1000, n_restarts=100, 
                                        policy="linear", seed=None,  verbose=False):
        n_feat = len(phi(np.zeros(self.dim_S)))
        result = np.empty((n_restarts * n_iter, n_feat))
        k=0
        for i in xrange(n_restarts):
            for s,a,s_n, r in self.sample_transition(n_iter, policy, with_restart=False):
                result[k,:] = phi(s)
                k+=1
        return result


    def sample_step(self, s0 , policy=None, seed=None, with_restart = False):
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

        rands = np.random.multivariate_normal(np.zeros(self.dim_S), self.Sigma, 1)
        a = policy(s0)
        mean = np.dot(self.A,s0) + np.dot(self.B,a)
        s1 = mean + rands[0]
        r = np.dot(s0.T, np.dot(self.Q, s0)) + np.dot(a.T, np.dot(self.R, a))
        return (s0, a, s1, r)

    def sample_transition(self, max_n, policy="linear", seed=None, with_restart = False):
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
        if policy is "linear":
            policy = self.linear_policy()

        rands = np.random.multivariate_normal(np.zeros(self.dim_S), self.Sigma, max_n)
        i=0
        while i < max_n:        
            s0 = self.start_f
            while i < max_n:  
                if self.terminal_f(s0):
                    if with_restart: 
                        break
                    else:
                        return
                a = policy(s0)
                mean = np.dot(self.A,s0) + np.dot(self.B,a)
                s1 = mean + rands[i]
              
                #s1 = np.random.multivariate_normal(mean, self.Sigma)
                #import ipdb; ipdb.set_trace()
                r = np.dot(s0.T, np.dot(self.Q, s0)) + np.dot(a.T, np.dot(self.R, a))
                yield (s0, a, s1, r)
                i+=1
                s0 = s1


                  
        
        

    

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


