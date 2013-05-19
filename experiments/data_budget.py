# -*- coding: utf-8 -*-
"""
Comparisoof CPU time of LSTD and TDC on a 100-link pendulum balancing task

Make sure to set the environment
variable OMP_NUM_THREADS=1 (only
use one cpu) to have a fair
comparison.
"""
__author__ = "Christoph Dann <cdann@cdann.de>"

import td
from joblib import Parallel
import examples
import numpy as np
import regtd
#import matplotlib.pyplot as plt
import features
import policies
from task import LinearLQRValuePredictionTask
import util
import dynamic_prog as dp


dim = 100
gamma = 0.95
sigma = np.ones(2 * dim) * 1.
dt = 0.1
mdp = examples.NLinkPendulumMDP(
    np.ones(dim) * .5, np.ones(dim) * .6, sigma=sigma, dt=dt)
phi = features.squared_tri((2 * dim) * (2 * dim + 1) / 2 + 1)

n_feat = phi.dim
print phi.dim, "features"
theta_p, _, _ = dp.solve_LQR(mdp, gamma=gamma)
theta_p = np.array(theta_p)
theta_o = theta_p.copy()
policy = policies.LinearContinuous(theta=theta_p, noise=np.ones(dim) * 0.4)
theta0 = 0. * np.ones(n_feat)

task = LinearLQRValuePredictionTask(mdp, gamma, phi, theta0,
                                    policy=policy,
                                    normalize_phi=True, mu_next=1000, mu_iter=1000,
                                    mu_restarts=8)


#states, _, _, _, _ = mdp.samples_cached(n_iter=1000, n_restarts=15,
#                                        policy=policy, seed=8000)


lam = 0.0
alpha = 0.00002
mu = .0002
tdc = td.TDCLambda(alpha=alpha, mu=mu, lam=lam, phi=phi)
tdc.name = r"TDC({}) $\alpha$={} $\mu$={}".format(lam, alpha, mu)

lam = 0.0
alpha = td.RMalpha(.00006, 0.02)
beta = td.RMalpha(.00001, 0.1)
tdcrm = td.TDCLambda(alpha=alpha, beta=beta, lam=lam, phi=phi)
tdcrm.name = r"TDC({}) $\alpha$={} $\mu$={}".format(lam, alpha, mu)


lam = 0.
eps = 10000
rlstd = td.RecursiveLSTDLambda(lam=lam, eps=eps, phi=phi)
rlstd.name = r"LSTD({}) $\epsilon$={}".format(lam, eps)

lam = 0.
eps = 1000000
lstd = td.LSTDLambda(lam=lam, eps=eps, phi=phi)
lstd.name = r"LSTD({}) $\epsilon$={}".format(lam, eps)

l = 1000
n_eps = 7
error_every = 100  # 4000
name = "link100_data_budget"
title = "100-link Pole Balancing"
n_indep = 10
episodic = False
criterion = "RMSE"
criteria = ["RMSE"]
eval_on_traces = False
n_samples_eval = 5000
verbose = 1
gs_ignore_first_n = 10000
gs_max_weight = 3.
max_t = 100.
min_diff = .5
disc_times = np.arange(min_diff, max_t, min_diff)


def  run(s):
    e = np.ones((len(disc_times), 3)) * np.nan
    t = np.ones((len(disc_times), 3)) * np.nan
    p = np.ones((len(disc_times), 3)) * np.nan
    lstd.time = 0.
    rlstd.time = 0.
    tdc.time = 0.
    tdcrm.time = 0.
#    el, tl = task.error_traces_cpu_time(
#        lstd, max_passes=1, max_t=100000000, min_diff=min_diff,
#        n_samples=l, n_eps=n_eps, verbose=0, seed=s, eval_on_traces=eval_on_traces,
#        n_samples_eval=n_samples_eval,
#        criteria=criteria, eval_once=True)
    e_, n_, t_ = task.error_traces_cpu_time(
        rlstd, max_passes=1, max_t=max_t, min_diff=min_diff,
        n_samples=l, n_eps=n_eps, verbose=4, seed=s, eval_on_traces=eval_on_traces,
        n_samples_eval=n_samples_eval,
        criteria=criteria)
    e[:len(e_), 0] = e_
    t[:len(t_), 0] = t_
    p[:len(n_), 0] = n_
    e_, n_, t_ = task.error_traces_cpu_time(
        tdc, max_passes=None, max_t=max_t, min_diff=min_diff,
        n_samples=l, n_eps=n_eps, verbose=4, seed=s, eval_on_traces=eval_on_traces,
        n_samples_eval=n_samples_eval,
        criteria=criteria)
    e[:len(e_), 1] = e_
    t[:len(t_), 1] = t_
    p[:len(n_), 1] = n_

    e_, n_, t_ = task.error_traces_cpu_time(
        tdcrm, max_passes=None, max_t=max_t, min_diff=min_diff,
        n_samples=l, n_eps=n_eps, verbose=4, seed=s, eval_on_traces=eval_on_traces,
        n_samples_eval=n_samples_eval,
        criteria=criteria)
    e[:len(e_), 2] = e_
    t[:len(t_), 2] = t_
    p[:len(n_), 2] = n_

    return e, p, t

if __name__ == "__main__":
    from experiments import *
    import matplotlib.pyplot as plt
    fn = "data/data_budget_link100_full.npz"
    n_jobs = 1
    if os.path.exists(fn):
        d = np.load(fn)
        globals().update(d)
    else:
        #task.fill_trajectory_cache(seeds=range(n_indep), n_eps=n_eps, n_samples=l, n_jobs=n_jobs)
        task.mu
        jobs = []
        for s in range(n_indep):
            jobs.append((run, [s], {}))
        res = Parallel(n_jobs=n_jobs, verbose=verbose)(jobs)
        res, proc, times = zip(*res)
        res = np.array(res)
        times = np.array(times)
        proc = np.array(proc)
        np.savez(fn, res=res, disc_times=disc_times, proc=proc, times=times,
                 title="100-Link Pendulum Balancing with {} features".format(phi.dim))
    plt.figure()
    times = np.array(times)
    proc = np.array(proc)
    r = np.ones((n_indep, len(disc_times), 3))
    for i in range(n_indep):
        for j in range(3):
            r[i, :, j] = np.interp(
                disc_times, times[i, :, j], res[i, :, j], left=np.nan)
    #for i in range(n_indep):

    ro = np.nansum(r, axis=0) / np.sum(np.isfinite(r), axis=0)
    u = (ro - r) ** 2
    std = np.sqrt(np.nansum(u, axis=0) / np.sum(np.isfinite(u), axis=0))
        #plt.plot(times[i,:,0], r[i,:,0], color="red", label="LSTD")
        #plt.plot(times[i,:,1], r[i,:,1], color="green", label="TDC const")
        #plt.plot(times[i,:,2], r[i,:,2], color="blue", label="TDC dec")
    plt.errorbar(disc_times, ro[:, 0], yerr=std[:, 0], errorevery=4,
                 color="red", label="LSTD")
    plt.errorbar(disc_times, ro[:, 1], yerr=std[:, 1], errorevery=4,
                 color="blue", label="TDC const.")
    plt.errorbar(disc_times, ro[:, 2], yerr=std[:, 2], errorevery=4,
                 color="green", label="TDC decr.")
    plt.ylabel("RMSE")
    plt.xlabel("Runtime in s")
    plt.title(title)
    plt.legend()
