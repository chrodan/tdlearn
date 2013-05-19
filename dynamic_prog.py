# -*- coding: utf-8 -*-
"""
Dynamic programming value function estimation
"""
__author__ = "Christoph Dann <cdann@cdann.de>"
import numpy as np
import logging
from joblib import Memory
memory = Memory(cachedir="./cache", verbose=20)


def estimate_V_discrete(mdp, n_iter=100000, policy="uniform", gamma=1.):
    if policy == "uniform":
        policy = mdp.uniform_policy()

    P = mdp.P * policy.tab[:, :, np.newaxis]
    P = P.sum(axis=1)
    P /= P.sum(axis=1)[:, np.newaxis]

    r = mdp.r * policy.tab[:, :, np.newaxis]
    r = r.sum(axis=1)
    V = np.zeros(len(mdp.states))
    for i in xrange(n_iter):
        V_n = (P * (gamma * V + r)).sum(axis=1)
        if np.linalg.norm(V - V_n) < 1e-22:
            V = V_n
            logging.info("Convergence after {} iterations".format(i + 1))
            break
        V = V_n
    return V


def estimate_V_LQR(lqmdp, bellman_op, n_iter=100000, gamma=1., eps=1e-14):
    """ Evaluate the value function exactly fora given Linear-quadratic MDP
        the value function has the form
        V = s^T P s

        for the policy

        as_fun: returns V as a python function instead of P"""

    T = bellman_op
    P = np.matrix(np.zeros((lqmdp.dim_S, lqmdp.dim_S)))
    b = 0.
    for i in xrange(n_iter):
        P_n, b_n = T(P, b)  # Q + theta.T * R * theta + gamma * (A+ B * theta).T * P * (A + B * theta)
        if np.linalg.norm(P - P_n) < eps and np.abs(b - b_n) < eps:
            print "Converged estimating V after ", i, "iterations"
            break
        P = P_n
        b = b_n
    return np.array(P), b


def bellman_operator(mdp, P, b, theta, noise=0., gamma=1.):
    """
    the bellman operator for the behavioral policy
    as a python function which takes the value function s^T P s + b represented as a numpy
    squared array P
        T(P,b) = R + theta_p^T Q theta_p + gamma * (A + B theta_p)^T P (A + B theta_p), gamma * (b + tr(P * Sigma))

    """
    Q = np.matrix(mdp.Q)
    R = np.matrix(mdp.R)
    A = np.matrix(mdp.A)
    B = np.matrix(mdp.B)
    Sigma = np.matrix(np.diag(mdp.Sigma))
    theta = np.matrix(theta)
    if noise == 0.:
        noise = np.zeros((theta.shape[0]))
    S = A + B * theta
    C = Q + theta.T * R * theta

    Pn = C + gamma * (S.T * np.matrix(P) * S)
    bn = gamma * (b + np.trace(np.matrix(P) * np.matrix(Sigma)))\
        + np.trace(
            (R + gamma * B.T * np.matrix(P) * B) * np.matrix(np.diag(noise)))
    return Pn, bn


@memory.cache
def solve_LQR(lqmdp, n_iter=100000, gamma=1., eps=1e-14):
    """ Solves exactly the Linear-quadratic MDP with
        the value function has the form
        V* = s^T P* s and policy a = theta* s

        returns (theta*, P*)"""


    P = np.matrix(np.zeros((lqmdp.dim_S, lqmdp.dim_S)))
    R = np.matrix(lqmdp.R)
    b = 0.
    theta = np.matrix(np.zeros((lqmdp.dim_A, lqmdp.dim_S)))
    A = np.matrix(lqmdp.A)
    B = np.matrix(lqmdp.B)
    for i in xrange(n_iter):
        theta_n = - gamma * np.linalg.pinv(R + gamma * B.T * P *
                                           B) * B.T * P * A
        P_n, b_n = bellman_operator(lqmdp, P, b, theta, gamma=gamma)  # Q + theta.T * R * theta + gamma * (A+ B * theta).T * P * (A + B * theta)
        if np.linalg.norm(P - P_n) < eps and np.abs(b - b_n) < eps and np.linalg.norm(theta - theta_n) < eps:
            print "Converged estimating V after ", i, "iterations"
            break
        P = P_n
        b = b_n
        theta = theta_n
    return np.asarray(theta), P, b
