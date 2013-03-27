# -*- coding: utf-8 -*-
"""
Set of Example MDPs that are widely used in the TD Learning community.

Created on Tue Dec 20 21:04:24 2011

@author: Christoph Dann <cdann@cdann.de>
"""
import mdp
import numpy as np
import scipy.integrate
import matplotlib.animation as animation
import re
import matplotlib.pyplot as plt
#import pyximport; pyximport.install()
import swingup_ode

class GridWorld(mdp.MDP):
    """
    Gridworld for paper plot
    """

    def __init__(self):
        world = """       W
G

       WW
     WWWWWW


 WWW
WWWW
WWWWWWWW
            WW
W            W
             W
           A

"""
        world = ["{:<15}".format(l) for l in world.splitlines()]
        self.world = world
        n = 15
        actions = ["l", "u", "r", "d"]
        states = range(n ** 2)
        s = "".join(world)
        assert(len(s) == n ** 2)

        d0 = np.zeros(n ** 2)
        d0[s.find("A")] = 1.

        r = np.ones((n ** 2, 4, n ** 2)) * (-.0)
        g = s.find("G")
        r[:, :, g] = 10.
        r[g, :, :] = 0.

        gi = int(g / n)
        gj = g % n

        self.pol = np.zeros((n, n, 4))
        for i in range(n):
            for j in range(n):
                if i == gi and j == gj:
                    self.pol[i, j, :] = .25
                    continue
                if gi < i:
                    self.pol[i, j, 1] = 1.
                elif gi > i:
                    self.pol[i, j, 3] = 1.
                if gj < j:
                    self.pol[i, j, 0] = 1.
                elif gj > j:
                    self.pol[i, j, 2] = 1.
                self.pol[i, j, :] /= self.pol[i, j, :].sum()
        self.pol = self.pol.reshape(n ** 2, 4)

        for m in re.finditer('W', s):
            r[:, :, m.start()] = -2.

        P = np.zeros((n, n, 4, n, n))
        for i in range(n):
            for j in range(n):
                k = -1
                for d in [(0, -1, 0), (-1, 0, 1), (0, 1, 2), (1, 0, 3)]:
                    ni = min(n - 1, max(i + d[0], 0))
                    nj = min(n - 1, max(j + d[1], 0))
                    #print i,j,ni,nj
                    P[i, j, :, ni, nj] = 0.2 / 3
                    P[i, j, d[2], ni, nj] = .8
                    if ni == i and nj == j:
                        if k > -1:
                            P[i, j, k, ni, nj] = .8
                            P[i, j, :, ni, nj] += 0.2 / 3
                        else:
                            k = d[2]
        P = P.reshape(n ** 2, 4, n ** 2)
        P[g, :, :] = 0
        P[g, :, g] = 1.
        mdp.MDP.__init__(self, states, actions, r, P, d0)


class MiniLQMDP(mdp.LQRMDP):
    """
    toy LQR MDP
    """

    def __init__(self, dt=.01, sigma=0.1):
        """
        mass: point mass of the pendulum
        length: length of the stick / pendulum
        mu: friction coefficient
        dt: time step
        """
        A = np.array([[1., dt],
                      [0., 1.]])
        B = np.array([0., dt]).reshape(2, 1)
        Q = np.diag([-1., 0.])
        #terminal_f = lambda x: np.abs(x[0]) > 10

        R = np.ones((1, 1)) * (-1)

        mdp.LQRMDP.__init__(self, A, B, Q, R, sigma,
                            start=np.array([0.0001, 0]))


class NLinkPendulumMDP(mdp.LQRMDP):
    """

    a 2D Pendulum with n links each link has an actuated joint. The pendulum is
    linearized around the balance point.

    """

    def __repr__(self):
        return "<NLinkPendulumMDP(" + repr([self.masses, self.lengths, self.dt,
                                            self.n, self.Sigma, self.R, self.Q]) + ")>"

    def __init__(self, masses, lengths, dt=.01, sigma=0.01, penalty=0.01, action_penalty=0.):
        self.lengths = lengths
        self.dt = dt
        self.masses = masses
        self.n = len(masses)
        n = self.n
        m = np.cumsum(self.masses[::-1])[::-1]
        Upp = -9.81 * self.lengths * m
        self.M = np.outer(
            self.lengths, self.lengths) * np.minimum(m[:, None], m)
        Minv = np.linalg.pinv(self.M)
        A = np.eye(2 * n)
        A[:n, n:] += np.eye(n) * self.dt
        A[n:, :n] -= self.dt * Minv * Upp[None, :]
        B = np.zeros((2 * n, n))
        B[n:, :] += Minv * self.dt
        Q = np.zeros((2 * n, 2 * n))
        Q[:n, :n] += np.eye(n) * penalty
        R = np.eye(n) * action_penalty
        mdp.LQRMDP.__init__(self, A, B, Q, R, Sigma=sigma,
                            start=np.zeros(2 * n))


class PoleBalancingMDP(mdp.LQRMDP):
    """
    Linear Quadratic MDP which models the pole balancing task
    i.e. 2D inverted pendulum linearly approximated around the
    balance point.

        S = [\alpha, \dot \alpha, x, \dot x]
        A = [\ddot x]

    \alpha is the angle of the pendulum.
    x is the position of the cart / hand
    """

    def __repr__(self):
        return "<PoleBalancingMDP(" + repr([self.length, self.dt, self.A, self.B, self.Q, self.R, self.Sigma]) + ")>"

    def __init__(self, m=.5, M=.5, length=.6, b=0.1,dt=.01, sigma=0.):
        """
        mass: point mass of the pendulum
        length: length of the stick / pendulum
        mu: friction coefficient
        dt: time step
        """
        self.length = length
        self.dt = dt
        g = 9.81
        k = 4.*M-m
        A = np.array([[1., dt, 0, 0],
                      [dt*3*(M+m)/k/length, 1 + dt*3*b/k, 0, 0],
                      [0., 0, 1, dt],
                      [dt*3*m*g/k, -4*b/k, 0, 1]])
        B = np.array([0., -3*dt /length/k, 0, dt *4/k]).reshape(4, 1)
        Q = np.diag([-100., 0., -1, 0])
        #terminal_f = lambda x: np.abs(x[0]) > 10

        R = np.ones((1, 1)) * (-0.1)

        mdp.LQRMDP.__init__(
            self, A, B, Q, R, np.array([0.0001, 0, 0, 0]), sigma)

    def animate_trace(self, state_trace, action_trace=None):
        fig = plt.figure()
        off = np.max(np.abs(state_trace[:, 2])) + self.length
        ax = fig.add_subplot(111, autoscale_on=False, xlim=(-off,
                             off), ylim=(-1, self.length * 1.1))
        ax.grid()
        line, = ax.plot([], [], 'o-', lw=2)
        time_template = 'time = %.1fs'
        time_text = ax.text(0.05, 0.9, '', transform=ax.transAxes)

        psi_template = r'$\psi$ = %.2g'
        psi_text = ax.text(0.05, 0.8, '', transform=ax.transAxes)
        if action_trace is not None:
            line_a, = ax.plot([], [], 'r-*', lw=2)

        def init():
            line.set_data([], [])
            time_text.set_text('')
            psi_text.set_text('')

            if action_trace is not None:
                line_a.set_data([], [])
                return line, line_a, time_text, psi_text
            return line, time_text, psi_text

        def anim(i):
            thisx = [state_trace[i, 2], state_trace[i, 2] +
                     self.length * np.sin(10 * state_trace[i, 0])]
            thisy = [0, self.length * np.cos(10 * state_trace[i, 0])]

            line.set_data(thisx, thisy)
            time_text.set_text(time_template % (i * self.dt))

            psi_text.set_text(psi_template % (state_trace[i, 0]))
            if action_trace is not None:
                line_a.set_data([0., action_trace[i, 0] /
                                np.max(np.abs(action_trace))], [-0.9, -0.9])
                return line, line_a, time_text, psi_text
            return line, time_text, psi_text

        ani = animation.FuncAnimation(
            fig, anim, np.arange(state_trace.shape[0]),
            interval=1000. * self.dt * 10, blit=True, init_func=init)
        return ani


class ActionMDP(mdp.MDP):
    """
    MDP with n states and actions where the action deterministically determines the next state.
    The rewward is only dependent on the current state and is different for each state
    """

    def __init__(self, n, reward_level=1.):
        """
        n: number of states and actions
        reward_level: the general amplitude of rewards
        """
        n_s = n
        n_a = n
        actions = range(n_a)
        states = range(n_s)

        d0 = np.ones((n_s))
        d0 = d0 / d0.sum()

        r = np.ones((n_s, n_a, n_s)) * reward_level
        r *= np.arange(1., n_s + 1.)[:, None, None] / n_s
        P = np.zeros((n_s, n_s, n_s))
        for i in range(n_s):
            P[:, i, i] = 1.
        mdp.MDP.__init__(self, states, actions, r, P, d0)


class RandomMDP(mdp.MDP):
    """
    Random MDP with uniformly distributed transition probabilities and
    reward function.
    """

    def __init__(self, n_states, n_actions, seed=None, reward_level=1.):
        """
            n_states: number of states
            seed: random seed to generate the transition and reward function
        """
        if seed is not None:
            np.random.seed(seed)
        n_s = n_states
        n_a = n_actions
        actions = range(n_a)
        states = range(n_s)

        d0 = np.random.rand(n_s) + 1e-5
        d0 = d0 / d0.sum()
        r = np.ones((n_s, n_a, n_s)) * reward_level
        r *= np.random.rand(n_a, n_s)[None, :, :]
        #r[:, :, n_s - 1] = 1
        P = np.random.rand(n_s, n_a, n_s) + 1e-5
        P /= P.sum(axis=2)[:, :, np.newaxis]

        mdp.MDP.__init__(self, states, actions, r, P, d0)


class PendulumSwingUpCartPole(mdp.ContinuousMDP):
    """
    Simulation of a Pendulum mounted on a cart that can move along a line. This is a standard benchmark
    in Reinforcement Learning. The task is to find a policy that swings up the pendulum. The only actuation is
    the Force on the cart.

    The task has the following parameters:
    M: Mass of the cart [kg]
    m: Mass of the pendulum [kg]
    l: Length of the pendulum [m]
    dt:Time step [s]
    b: Friction coefficient between cart and ground [N/m/s]

    The state is 4 dimensional:
        position of the cart
        velocity of the cart
        angular velocity of the pendulum
        angular position of the pendulum  (in [-pi, +pi[)

    """
    def __init__(self, M=0.5, l=0.6, m=0.5, dt=0.15, b=0.1, Sigma=0., start_amp=2.0):
        self.l = l
        self.M = M
        self.m = m
        self.dt = dt
        self.b = b
        self.start_amp=start_amp
        mdp.ContinuousMDP.__init__(self, self.statefun, self.rewardfun, 4,
                                   1, lambda: self.__class__.randstart(start_amp), Sigma=Sigma)

    def __repr__(self):
        return "<PendulumSwingUpCartPole(" + repr([self.l, self.M, self.m, self.dt, self.b, self.Sigma, self.start_amp]) + ")>"

    @staticmethod
    def randstart(a):
        return np.array([0., 0., 0., (np.random.rand() - .5) * a * np.pi])

    def ode(self, s, t, a, m, l, M, b):
        g = 9.81
        ds = np.zeros(4)
        ds[0] = s[1]
        ds[1] = (2 * m * l * s[2] ** 2 * np.sin(s[3]) + 3 * m * g * np.sin(s[3]) * np.cos(s[3]) + 4 * a - 4 * b * s[1])\
            / (4 * (M + m) - 3 * m * np.cos(s[3]) ** 2)
        ds[2] = (-3 * m * l * s[2] ** 2 * np.sin(s[3]) * np.cos(s[3]) - 6 * (M + m) * g * np.sin(s[3]) - 6 * (a - b * s[1]) * np.cos(s[3]))\
            / (4 * l * (m + M) - 3 * m * l * np.cos(s[3]) ** 2)
        ds[3] = s[2]
        return ds

    def ode_jac(self, s, t, a, m, l, M, b):
        print "jAC CALLED"
        g = 9.81
        c3 = np.cos(s[3])
        s3 = np.sin(s[3])
        c = 4 * (M + m) - 3 * m * c3 ** 2
        jac = np.zeros(4, 4)
        jac[0, 1] = 1.
        jac[3, 2] = 1.
        jac[1, 1] = -4 * b / c
        jac[1, 2] = 4 * m * l * s[2] / c
        jac[1, 3] = m / c * (2 * l * s[2] ** 2 * c3 + 3 * g * (1 - 2 * s3 ** 2)) \
            - 6 * m * s3 * c3 / c / c * (2 * m * l * s[2] ** 2 * s3 + 3 * m * g * s3 * c3 + 4 * a -
                                         4 * b * s[1])
        jac[2, 1] = 6 * b * c3 / c / l
        jac[2, 2] = -6 * m * l * s[2] * c3 * s3 / c
        jac[2, 3] = (3 * m * l * s[2] ** 2 * (2 * s3 - 1) - 6 * (M + m) * g * s3 + 6 * (a - b * s[1]) * s3) / l / l / c \
            + 3 / l / l / c / c * (m * l * s[2] ** 2 * s3 * c3 - 2 * (
                                   M + m) * g * s3 + 2(a - b * s[1] * c3)) * 6 * m * c3 * s3
        return jac

    def statefun(self, s, a):
        s1 = scipy.integrate.odeint(
            swingup_ode.ode, s, [0., self.dt], args=(a, self.m, self.l, self.M, self.b),
            #Dfun=ode_jac_wrap,
            printmessg=False)
        s1 = s1[-1, :].flatten()
        s1[-1] = ((s1[-1] + np.pi) % (2 * np.pi)) - np.pi
        return s1

    def rewardfun(self, s, a):
        l = self.l
        return -np.cos(s[-1]) * l - 1e-5 * np.abs(s[0])

    def __getstate__(self):
        res = mdp.ContinuousMDP.__getstate__(self)
        del res["rf"]
        del res["sf"]
        return res

    def __setstate__(self, state):
        mdp.ContinuousMDP.__setstate__(self, state)
        self.rf = self.rewardfun
        self.sf = self.statefun
        self.start = lambda: self.__class__.randstart(self.start_amp)

    def animate_trace(self, state_trace, action_trace=None):
        fig = plt.figure()
        off = np.max(np.abs(state_trace[:, 0])) + self.l
        ax = fig.add_subplot(111, autoscale_on=False, aspect='equal',
                             xlim=(-off, off), ylim=(-self.l * 2, self.l * 2))
        ax.grid()
        line, = ax.plot([], [], 'o-', lw=2)
        time_template = 'time = %.1fs'
        time_text = ax.text(0.05, 0.9, '', transform=ax.transAxes)

        psi_template = r'$\psi$ = %.2g'
        psi_text = ax.text(0.05, 0.8, '', transform=ax.transAxes)
        if action_trace is not None:
            line_a, = ax.plot([], [], 'r-*', lw=2)

        def init():
            line.set_data([], [])
            time_text.set_text('')
            psi_text.set_text('')

            if action_trace is not None:
                line_a.set_data([], [])
                return line, line_a, time_text, psi_text
            return line, time_text, psi_text

        def anim(i):
            thisx = [state_trace[i, 0], state_trace[i, 0] +
                     self.l * np.sin(state_trace[i, 3])]
            thisy = [0, - self.l * np.cos(state_trace[i, 3])]

            line.set_data(thisx, thisy)
            time_text.set_text(time_template % (i * self.dt))

            psi_text.set_text(psi_template % (state_trace[i, 3]))
            if action_trace is not None:
                line_a.set_data([0., action_trace[i, 0] /
                                np.max(np.abs(action_trace))], [-0.9, -0.9])
                return line, line_a, time_text, psi_text
            return line, time_text, psi_text

        ani = animation.FuncAnimation(
            fig, anim, np.arange(state_trace.shape[0]),
            interval=1000. * self.dt, blit=True, init_func=init)
        return ani


class CorruptedChain(mdp.MDP):
    """ Corrupted Chain example
        from Kolter & Ng: Regularized Least Squares Temporal Difference Learning (2009)
    """

    def __init__(self, n_states):
        n = n_states
        states = range(n)
        actions = ["l", "r"]
        d0 = np.ones(n)
        d0 /= d0.sum()
        r = np.zeros((n, 2, n))
        r[-1] = 1
        r[0] = 1
        P = np.zeros((n, 2, n))
        P[0, 0, 0] = 0.9
        P[0, 0, 1] = 0.1
        P[0, 1, 0] = 0.1
        P[0, 1, 1] = 0.9
        P[-1, 0, -1] = 0.1
        P[-1, 0, -2] = 0.9
        P[-1, 1, -1] = 0.9
        P[-1, 1, -2] = 0.1

        for s in states[1:-1]:
            P[s, 0, s - 1] = 0.9
            P[s, 1, s + 1] = 0.9
            P[s, 0, s + 1] = 0.1
            P[s, 1, s - 1] = 0.1
        mdp.MDP.__init__(self, states, actions, r, P, d0)


class RandomWalkChain(mdp.MDP):
    """ Random Walk chain example MDP """
    # TODO: explain + reference

    def __init__(self, n_states, p_minus=0.5, p_plus=0.5):
        """
            n_states: number of states including terminal ones
            p_minus: probability of going left
            p_plus: probability of going right
        """
        n_s = n_states
        states = range(n_s)
        actions = [0, ]
        d0 = np.zeros(n_s)
        d0[n_s / 2] = 1
        r = np.zeros((n_s, 1, n_s))
        r[n_s - 2, :, n_s - 1] = 1
        P = np.zeros((n_s, 1, n_s))
        P[0, :, 0] = 1
        P[n_s - 1, :, n_s - 1] = 1
        for s in np.arange(1, n_s - 1):
            P[s, :, s + 1] = p_plus
            P[s, :, s - 1] = p_minus

        mdp.MDP.__init__(self, states, actions, r, P, d0)

    def dependent_phi(self, state):
        """
        feature function that produces linear dependent features
        the feature of the middle state is an average over all positions
        """
        n = len(self.states)
        l = n / 2 + 1
        #print l
        if state >= l:
            res = state - l < np.arange(l)
        else:
            res = state >= np.arange(l)
        res = res.astype("float")
        res /= np.sqrt(np.sum(res))
        return res

    def linear_phi(self, state):
        n = len(self.states)
        a = (n - 1.) / (self.n_feat - 1)
        r = 1 - abs((state + 1 - np.linspace(1, n, self.n_feat)) / a)
        r[r < 0] = 0
        return r


class BoyanChain(mdp.MDP):
    """
    Boyan Chain example. All states form a chain with ongoing arcs and a
    single terminal state. From a state one transition with equal probability
    to either the direct or the second successor state. All transitions have
    a reward of -3 except going from second to last to last state (-2)
    """

    def __init__(self, n_states, n_feat):
        """
            n_states: number of states including terminal ones
            n_feat: number of features used to represent the states
                    n_feat <= n_states
        """
        assert n_states >= n_feat
        n_s = n_states
        self.n_feat = n_feat
        states = range(n_s)
        actions = [0, ]
        d0 = np.zeros(n_s)
        d0[0] = 1
        r = np.ones((n_s, 1, n_s)) * (-3)
        r[n_states - 2, :, n_states - 1] = -2
        r[n_states - 1:, :, :] = 0
        P = np.zeros((n_s, 1, n_s))
        P[-1, :, -1] = 1
        P[n_states - 2, :, n_states - 1] = 1
        for s in np.arange(n_states - 2):
            P[s, :, s + 1] = 0.5
            P[s, :, s + 2] = 0.5

        mdp.MDP.__init__(self, states, actions, r, P, d0, terminal_trans=n_s)


class BairdStarExample(mdp.MDP):
    """
    Baird's star shaped example for off-policy divergence of TD(\lambda)
    contains of states ordered in a star shape with 2 possible actions:
        - a deterministic one always transitioning in the star center and
        - a probabilistic one going to one of the star ends with uniform
            probability
    for details see Braid (1995): Residual Algorithms: Reinforcement Learning
        with Function Approximation
    """

    def __init__(self, n_corners):
        """
            n_corners: number of ends of the star
        """
        n_s = n_corners + 1
        actions = ["dotted", "solid"]
        r = np.zeros((n_s, 2, n_s))
        P = np.zeros((n_s, 2, n_s))
        # solid action always go to star center
        P[:, 1, n_corners] = 1
        # dotted action goes to star corners uniformly
        P[:, 0, :n_corners] = 1. / n_corners
        # start uniformly
        d0 = np.ones((n_s), dtype="double") / n_s

        mdp.MDP.__init__(self, range(1, n_corners + 1) + ["center", ],
                         actions, r, P, d0)
