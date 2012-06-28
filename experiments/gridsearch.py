import td
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
from lqr_offpolicy import *


def plot_2d_error_grid_file(fn, maxerr=5):
    with open(fn) as f:
        d = pickle.load(f)
    plot_error_grid(d["res"],d["alphas"], d["mus"], maxerr=maxerr)

def plot_2d_error_grid(val, alphas, mus, maxerr=5):
    ferr = val.copy()
    ferr[val > maxerr] = np.nan
    plt.figure(figsize=(12,10))
    plt.imshow(ferr, interpolation="nearest", cmap="hot", norm=LogNorm(vmin=np.nanmin(ferr), vmax=np.nanmax(ferr)))
    plt.yticks(range(len(alphas)), alphas)
    plt.xticks(range(len(mus)), mus, rotation=45, ha="right")
    plt.xlabel(r"$\mu$")
    plt.ylabel(r"$\alpha$")
    plt.colorbar()


def run_2d(alpha, mu, cls):
    np.seterr(all="ignore")
    m = cls(alpha=alpha, beta=mu*alpha, phi=task.phi, gamma=gamma)
    mean, std, raw = task.avg_error_traces([m], n_indep=3, n_samples=l, error_every=error_every, criterion="RMSPBE", verbose=False)
    val = np.mean(mean)
    return val

def run_1d(alpha, cls):
    np.seterr(all="ignore")
    m = cls(alpha=alpha, phi=task.phi, gamma=gamma)
    mean, std, raw = task.avg_error_traces([m], n_indep=3, n_samples=l, error_every=error_every, criterion="RMSPBE", verbose=False)
    val = np.mean(mean)
    return val

def run_lambda_1d(lam, cls):
    np.seterr(all="ignore")
    m = cls(lam=lam, phi=task.phi, gamma=gamma)
    mean, std, raw = task.avg_error_traces([m], n_indep=3, n_samples=l, error_every=error_every, criterion="RMSPBE", verbose=False)
    val = np.mean(mean)
    return val

def run_lambda_2d(lam, alpha, cls):
    np.seterr(all="ignore")
    m = cls(lam=lam, alpha=alpha, phi=task.phi, gamma=gamma)
    mean, std, raw = task.avg_error_traces([m], n_indep=3, n_samples=l, error_every=error_every, criterion="RMSPBE", verbose=False)
    val = np.mean(mean)
    return val

def run_lambda_3d(lam, alpha, mu, cls):
    np.seterr(all="ignore")
    m = cls(lam=lam, alpha=alpha, mu=mu, phi=task.phi, gamma=gamma)
    mean, std, raw = task.avg_error_traces([m], n_indep=3, n_samples=l, error_every=error_every, criterion="RMSPBE", verbose=False)
    val = np.mean(mean)
    return val


def gridsearch_2d():
    methods = [td.TDC, td.GeriTDC, td.GTD, td.GTD2]

    alphas = [0.0002, 0.0005] + list(np.arange(0.001, .01, 0.001)) + list(np.arange(0.01, 0.1, 0.01)) + [0.1, 0.2, 0.3, 0.4, 0.5]
    mus = [0.0001, 0.001, 0.01,0.01, 0.1, 0.5,1,2,4,8,16]
    params = list(itertools.product(alphas, mus))

    for m in methods:
        k = (delayed(run_2d)(*(list(p)+[m])) for p in params)
        res = Parallel(n_jobs=-1, verbose=11)(k)

        res = np.array(res).reshape(len(alphas), -1)
        with open("data/{}_{}_gs.pck".format(name, m.__name__), "w") as f:
            pickle.dump(dict(params=params, alphas=alphas, mus=mus, res=res), f)

def gridsearch_lambda():
    methods = [td.LSTDLambda, td.LSTDLambdaJP]

    alphas = [0.0002, 0.0005] + list(np.arange(0.001, .01, 0.001)) + list(np.arange(0.01, 0.1, 0.01)) + [0.1, 0.2, 0.3, 0.4, 0.5]
    mus = [0.0001, 0.001, 0.01,0.01, 0.1, 0.5,1,2,4,8,16]
    lambdas = np.linspace(0., 1., 10)
    params = lambdas

    for m in methods:
        k = (delayed(run_lambda_1d)(p,m) for p in params)
        res = Parallel(n_jobs=-1, verbose=11)(k)

        res = np.array(res).reshape(len(lambdas), -1)
        with open("data/{}_{}_gs_lam.pck".format(name, m.__name__), "w") as f:
            pickle.dump(dict(params=params, lambdas=lambdas, res=res), f)

    methods = [td.LinearTDLambda]


    params = list(itertools.product(lambdas, alphas))

    for m in methods:
        k = (delayed(run_lambda_2d)(*(list(p)+[m])) for p in params)
        res = Parallel(n_jobs=-1, verbose=11)(k)

        res = np.array(res).reshape(len(lambdas), -1)
        with open("data/{}_{}_gs_lam.pck".format(name, m.__name__), "w") as f:
            pickle.dump(dict(params=params, alphas=alphas, lambdas=lambdas, res=res), f)

    methods = [td.TDCLambda]
    params = list(itertools.product(lambdas, alphas, mus))

    for m in methods:
        k = (delayed(run_lambda_3d)(*(list(p)+[m])) for p in params)
        res = Parallel(n_jobs=-1, verbose=11)(k)

        res = np.array(res).reshape(len(lambdas), len(alphas), -1)
        with open("data/{}_{}_gs_lam.pck".format(name, m.__name__), "w") as f:
            pickle.dump(dict(params=params, alphas=alphas, lambdas=lambdas, mus=mus, res=res), f)

def gridsearch_1d():
    methods = [td.LinearTD0, td.ResidualGradient]

    alphas = [0.0002, 0.0005] + list(np.arange(0.001, .01, 0.001)) + list(np.arange(0.01, 0.1, 0.01)) + [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.]
    params = alphas

    for m in methods:
        k = (delayed(run_1d)(p,m) for p in params)
        res = Parallel(n_jobs=-1, verbose=11)(k)

        res = np.array(res).reshape(len(alphas), -1)
        with open("data/{}_{}_gs.pck".format(name, m.__name__), "w") as f:
            pickle.dump(dict(params=params, alphas=alphas, res=res), f)

if __name__ == "__main__":
    gridsearch_1d()
    gridsearch_2d()
    gridsearch_lambda()
