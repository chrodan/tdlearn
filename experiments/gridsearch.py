import td
import os
import examples
import numpy as np
import matplotlib.pyplot as plt
import dynamic_prog as dp
import util
import features
import policies
import itertools
from task import LinearLQRValuePredictionTask
from joblib import Parallel, delayed
from matplotlib.colors import LogNorm
import pickle

from experiments.lqr_full import *

error_every = int(l * n_eps / 20)
n_indep = 3

ls_alphas = [0.1, 0.3, 0.5, 0.7, 0.8, 0.9, 1.0]
alphas = [0.0002, 0.0005] + list(np.arange(0.001, .01, 0.001)) + list(
    np.arange(0.01, 0.1, 0.01)) + [0.1, 0.2, 0.3, 0.4, 0.5]
mus = [0.0001, 0.001, 0.01, 0.01, 0.1, 0.5, 1, 2, 4, 8, 16]
lambdas = np.linspace(0., 1., 6)
sigmas = np.power(10, np.arange(-5., 2, .5))
reward_noises = np.power(10, np.arange(-5., 0, 1))
P_inits = [1., 10., 100.]
etas = [None, 1e-5, 1e-3]


def load_result_file(fn, maxerr=5):
    with open(fn) as f:
        d = pickle.load(f)
    print np.nanargmin(d["res"])
    best =  d["params"][np.nanargmin(d["res"])]
    for n,v in zip(d["param_names"], best):
        print n,v
    return d


def plot_2d_error_grid_file(fn, maxerr=5):
    with open(fn) as f:
        d = pickle.load(f)
    plot_2d_error_grid(d["res"], d["alphas"], d["mus"], maxerr=maxerr)


def plot_2d_error_grid(val, alphas, mus, maxerr=5):
    ferr = val.copy()
    ferr[val > maxerr] = np.nan
    plt.figure(figsize=(12, 10))
    plt.imshow(ferr, interpolation="nearest", cmap="hot", norm=LogNorm(
        vmin=np.nanmin(ferr), vmax=np.nanmax(ferr)))
    plt.yticks(range(len(alphas)), alphas)
    plt.xticks(range(len(mus)), mus, rotation=45, ha="right")
    plt.xlabel(r"$\mu$")
    plt.ylabel(r"$\alpha$")
    plt.colorbar()


def run(cls, param):
    np.seterr(all="ignore")
    m = cls(phi=task.phi, gamma=gamma, **param)
    mean, std, raw = task.avg_error_traces(
        [m], n_indep=n_indep, n_samples=l, n_eps=n_eps,
        error_every=error_every, criterion=criterion, verbose=False)
    val = np.mean(mean)
    return val


def make_rmalpha():
    c = list(np.arange(0.01, 0.1, 0.01)) + [0.1, 0.2, 0.3, 0.4, 0.5,
                                            0.6, 0.7, 0.8, 0.9, 1, 5, 10, 30]
    t = [0.01, 0.05, 0.1, 0.25, 0.5, 0.75, 1, 1.5]
    l = list(itertools.product(c, t))
    params = [td.RMalpha(ct, tt) for ct, tt in l]
    return params


def gridsearch(method, gs_name="", n_jobs=-1, **params):
    if gs_name != "":
        gs_name = "_" + gs_name
    param_names = params.keys()
    param_list = list(itertools.product(*[params[k] for k in param_names]))
    param_lengths = [len(params[k]) for k in param_names]
    k = (delayed(run)(method, dict(zip(param_names, p))) for p in param_list)
    res = Parallel(n_jobs=n_jobs, verbose=11)(k)
    res = np.array(res).reshape(*param_lengths)
    if not os.path.exists("data/{name}".format(name=name)):
        os.makedirs("data/{name}".format(name=name))
    with open("data/{}/{}{}.pck".format(name, method.__name__, gs_name), "w") as f:
        pickle.dump(dict(res=res, params = param_list, param_names=param_names, **params), f)
    print "Finished {}{}".format(method.__name__, gs_name)

if __name__ == "__main__":
    task.mu
    gridsearch(td.ResidualGradient, alpha=alphas)
    gridsearch(td.LinearTDLambda, alpha=alphas, lam=lambdas)

    gridsearch(td.TDCLambda, alpha=alphas, mu=mus, lam=lambdas)
    gridsearch(td.GTD, alpha=alphas, mu=mus)
    gridsearch(td.GTD2, alpha=alphas, mu=mus)
    gridsearch(td.TDC, alpha=alphas, mu=mus)

    gridsearch(td.RecursiveLSTDLambda, lam=lambdas)
    gridsearch(td.RecursiveLSPELambda, lam=lambdas, alpha=ls_alphas)
    gridsearch(td.FPKF, lam=lambdas, alpha=ls_alphas)

    gridsearch(td.GPTDP, sigma=sigmas)
    #gridsearch(td.KTD, reward_noise=reward_noises, eta=etas, P_init=P_inits)

    if task.off_policy:
        gridsearch(td.GeriTDC, alpha=alphas, mu=mus)
        gridsearch(td.RecursiveLSTDLambdaJP, lam=lambdas)
