# -*- coding: utf-8 -*-
"""
Comparison of graident-based and least-squares methods (TDC vs. LSTD)
in terms of CPU time on the discrete random MDP.
"""
__author__ = "Christoph Dann <cdann@cdann.de>"

import td
from joblib import Parallel
import examples
import numpy as np
import regtd
import features
import policies
from task import LinearDiscreteValuePredictionTask
import util

n = 800
n_a = 10
n_feat = 600
mdp = examples.RandomMDP(n, n_a)
phi = features.lin_random(n_feat, n, constant=True)
gamma = .95
np.random.seed(3)
beh_pol = policies.Discrete(np.random.rand(n, n_a))
tar_pol = beh_pol
task = LinearDiscreteValuePredictionTask(mdp, gamma, phi, np.zeros(phi.dim),
                                         policy=beh_pol)

lam = 0.0
alpha = 0.007
mu = .01
tdc = td.TDCLambda(alpha=alpha, mu=mu, lam=lam, phi=phi)
tdc.name = r"TDC({}) $\alpha$={} $\mu$={}".format(lam, alpha, mu)

lam = 0.0
alpha = td.RMalpha(.05, 0.2)
beta = td.RMalpha(.5, 0.25)
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

l = 8000
n_eps = 10
error_every = 600  # 4000
name = "swingup_data_budget"
title = "Cartpole Swingup Onpolicy"
n_indep = 3
episodic = False
criterion = "RMSPBE"
criteria = ["RMSPBE"]
eval_on_traces = False
n_samples_eval = 30000
verbose = 1
gs_ignore_first_n = 10000
gs_max_weight = 3.
max_t = 200.
min_diff = 1.
t = np.arange(min_diff, max_t, min_diff)
e = np.ones((len(t), 3)) * np.nan


def run(s):
    e = np.ones((len(t), 3)) * np.nan
    lstd.time = 0.
    rlstd.time = 0.
    tdc.time = 0.
    tdcrm.time = 0.
    el, tl = task.error_traces_cpu_time(
        lstd, max_passes=1, max_t=100000000, min_diff=min_diff,
        n_samples=l, n_eps=n_eps, verbose=0, seed=s, eval_on_traces=eval_on_traces,
        n_samples_eval=n_samples_eval,
        criteria=criteria, eval_once=True)

    e_, t_ = task.error_traces_cpu_time(
        rlstd, max_passes=1, max_t=max_t, min_diff=min_diff,
        n_samples=l, n_eps=n_eps, verbose=0, seed=s, eval_on_traces=eval_on_traces,
        n_samples_eval=n_samples_eval,
        criteria=criteria)
    e[:len(e_), 0] = e_
    e_, t_ = task.error_traces_cpu_time(
        tdc, max_passes=None, max_t=max_t, min_diff=min_diff,
        n_samples=l, n_eps=n_eps, verbose=0, seed=s, eval_on_traces=eval_on_traces,
        n_samples_eval=n_samples_eval,
        criteria=criteria)
    e[:len(e_), 1] = e_
    e_, t_ = task.error_traces_cpu_time(
        tdcrm, max_passes=None, max_t=max_t, min_diff=min_diff,
        n_samples=l, n_eps=n_eps, verbose=0, seed=s, eval_on_traces=eval_on_traces,
        n_samples_eval=n_samples_eval,
        criteria=criteria)
    e[:len(e_), 2] = e_
    return e, el, tl

if __name__ == "__main__":
    from .experiments import *
    import matplotlib.pyplot as plt
    fn = "data/budget_disc.npz"
    n_jobs = 1
    if os.path.exists(fn):
        d = np.load(fn)
        globals().update(d)
    else:
        # task.fill_trajectory_cache(seeds=range(n_indep), n_eps=n_eps,
        # n_samples=l, n_jobs=n_jobs)
        task.mu
        jobs = []
        for s in range(n_indep):
            jobs.append((run, [s], {}))
        res = Parallel(n_jobs=n_jobs, verbose=verbose)(jobs)
        res, el, tl = zip(*res)
        res = np.array(res)
        np.savez(fn, res=res, el=el, tl=tl,
                 title="Cartpole Swingup with {} features".format(phi.dim))
    plt.figure()
    for i in range(n_indep):
        o = res[i, 0, 0]
        for k in range(len(t)):
            if np.isnan(res[i, k, 0]):
                res[i, k, 0] = o
            o = res[i, k, 0]
    r = np.nansum(res, axis=0) / np.sum(np.isfinite(res), axis=0)
    u = (res - r) ** 2
    std = np.sqrt(np.nansum(u, axis=0) / np.sum(np.isfinite(u), axis=0))
    plt.errorbar(
        t, r[:, 0], yerr=std[:, 0], errorevery=20, color="red", label="LSTD")
    plt.errorbar(t, r[:, 1], yerr=std[:, 1], errorevery=20,
                 color="blue", label="TDC const.")
    plt.errorbar(t, r[:, 2], yerr=std[:, 2], errorevery=20,
                 color="green", label="TDC decr.")
    plt.ylabel("RMSPBE")
    plt.xlabel("Runtime in s")
    plt.title(title)
    plt.legend()
