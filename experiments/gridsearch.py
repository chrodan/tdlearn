import td
import os
import numpy as np
import matplotlib.pyplot as plt
import itertools
from joblib import Parallel, delayed
from matplotlib.colors import LogNorm
import pickle
import argparse
parser = argparse.ArgumentParser(description='Do heady grid search.')
parser.add_argument('-e', '--experiment',
                   help='which experiment to test')
parser.add_argument("-b", "--batchsize", help="Number of parameter-settings to try per job",
                    type=int, default=5)
parser.add_argument("-n","--njobs", help="Number of cores to use", type=int, default=-2)
args = parser.parse_args()
if args.experiment != None:
    exec "from experiments."+args.experiment+" import *"
else:
    from experiments.swingup_gauss_onpolicy import *

ls_alphas = [0.001, 0.01, 0.1, 0.3, 0.5, 0.7, 0.8, 0.9, 1.0]
alphas = [0.0002, 0.0005] + list(np.arange(0.001, .01, 0.001)) + list(
    np.arange(0.01, 0.1, 0.01)) + [0.1, 0.2, 0.3, 0.4, 0.5]
mus = [0.0001, 0.001, 0.01, 0.05, 0.1, 0.5, 1, 2, 4, 8, 16]
lambdas = np.linspace(0., 1., 6)
ls_eps = [10**5, 10**3, 10**2, 10, 1, 0.1, 0.01]
sigmas = np.power(10, np.arange(-5., 4, .75))
reward_noises = np.power(10, np.arange(-5., 0, 1))
P_inits = [1., 10., 100.]
mins = [0, 500, 1000]
fpkf_alphas = [0.01, 0.1, 0.3, 0.5, 0.8, 1.0]
fpkf_beta = [1, 10, 100, 1000]
etas = [None, 1e-5, 1e-3]

try:
    gs_errorevery
except:
    gs_errorevery = int(l * n_eps / 20.)

try:
    gs_indep
except:
    gs_indep = 3

try:
    gs_max_weight
except:
    gs_max_weight = 2.

try:
    gs_ignore_first_n
except:
    gs_ignore_first_n = 0


def load_result_file(fn, maxerr=5):
    with open(fn) as f:
        d = pickle.load(f)
    for i in range(d["res"].shape[-1]):
        print d["criteria"][i], np.nanmin(d["res"][...,i])
        best = d["params"][np.nanargmin(d["res"][...,i])]
        for n, v in zip(d["param_names"], best):
            print n, v
    return d


def plot_2d_error_grid_file(fn, criterion, maxerr=5):
    with open(fn) as f:
        d = pickle.load(f)
    plot_2d_error_grid(criterion=criterion, maxerr=maxerr, **d)


def plot_2d_error_grid(criterion, res, param_names, params, criteria, maxerr=5, **kwargs):
    erri = criteria.index(criterion)
    ferr = res[..., erri].copy()
    ferr[ferr > maxerr] = np.nan
    plt.figure(figsize=(12, 10))
    plt.imshow(ferr, interpolation="nearest", cmap="hot", norm=LogNorm(
        vmin=np.nanmin(ferr), vmax=np.nanmax(ferr)))
    p1 = kwargs[param_names[0]]
    p2 = kwargs[param_names[1]]
    plt.yticks(range(len(p1)), p1)
    plt.xticks(range(len(p2)), p2, rotation=45, ha="right")
    plt.xlabel(param_names[1])
    plt.ylabel(param_names[0])
    plt.colorbar()


def run(cls, param):
    np.seterr(all="ignore")

    m = [cls(phi=task.phi, gamma=gamma, **p) for p in param]
    mean, std, raw = task.avg_error_traces(
        m, n_indep=gs_indep, n_samples=l, n_eps=n_eps,
        error_every=gs_errorevery, episodic=episodic, criteria=criteria, verbose=0)
    a = int(gs_ignore_first_n / gs_errorevery)
    mean = mean[... ,a:]
    weights = np.linspace(1., gs_max_weight, mean.shape[-1])

    val = (mean * weights).sum(axis=-1) / weights.sum()
    return val


def make_rmalpha():
    c = list(np.arange(0.01, 0.1, 0.01)) + [0.1, 0.2, 0.3, 0.4, 0.5,
                                            0.6, 0.7, 0.8, 0.9, 1, 5, 10, 30]
    t = [0.01, 0.05, 0.1, 0.25, 0.5, 0.75, 1, 1.5]
    l = list(itertools.product(c, t))
    params = [td.RMalpha(ct, tt) for ct, tt in l]
    return params


def gridsearch(method, gs_name="", njobs=-2, batchsize=3, **params):
    if gs_name != "":
        gs_name = "_" + gs_name
    fn = "data/{}/{}{}.pck".format(name, method.__name__, gs_name)
    if os.path.exists(fn):
        print "Already exists: {}{}".format(method.__name__, gs_name)
        load_result_file(fn)
        return
    param_names = params.keys()
    param_list = list(itertools.product(*[params[k] for k in param_names]))
    param_lengths = [len(params[k]) for k in param_names]
    k = []
    i = 0
    while i < len(param_list):
        j = min(i + batchsize, len(param_list)+1)
        par = [dict(zip(param_names, p)) for p in param_list[i:j]]
        k.append(delayed(run)(method, par))
        i = j

    print "Starting {} {}{}".format(name, method.__name__, gs_name)
    res1 = Parallel(n_jobs=njobs, verbose=11)(k)
    res = np.vstack(res1)
    for i,c in enumerate(criteria):
        j = np.nanargmin(res[:,i].flatten())
        print "best parameters for {} with value {}:".format(c, np.nanmin(res[:,i]))
        print zip(param_names, param_list[j])
    res = np.array(res).reshape(*(param_lengths + [len(criteria)]))
    if not os.path.exists("data/{name}".format(name=name)):
        os.makedirs("data/{name}".format(name=name))
    with open(fn, "w") as f:
        pickle.dump(dict(res=res, params=param_list, criteria=criteria,
                    param_names=param_names, **params), f)
    print "Finished {} {}{}".format(name, method.__name__, gs_name)


def gridsearch_cluster(method, experiment, filename=None, gs_name="", batchsize=3, **params):
    param_names = params.keys()
    param_list = list(itertools.product(*[params[k] for k in param_names]))
    param_lengths = [len(params[k]) for k in param_names]
    #k = (delayed(run)(method, dict(zip(param_names, p))) for p in param_list)
    if filename is None:
        filename = experiment + "_" + method.__name__
    if gs_name != "":
        filename += "_" + gs_name
    filename += "_joblist.sh"
    i = 0
    basestr = "python experiments/gs_cluster.py {method} -e {exp}".format(method=method.__name__,
                                                                          exp=experiment)
    param_l = zip(*param_list)
    with open(filename, "w") as f:

        while i < len(param_list):
            curstr = basestr
            curstr += " " + " ".join(["--id"] +
                                     [str(c) for c in range(i, min(i + batchsize, len(param_list)))])
            for j in range(len(param_names)):
                curstr += " " + " ".join(["--" + param_names[j]] +
                                         [repr(param_l[j][c]) for c in range(i, min(i + batchsize, len(param_list)))])
            print curstr
            f.write(curstr + "\n")
            i += batchsize


if __name__ == "__main__":
    njobs = args.njobs
    batchsize = args.batchsize
    task.mu
    task.fill_trajectory_cache(range(gs_indep), n_samples=l, n_eps=n_eps)
    gridsearch(td.ResidualGradient, alpha=alphas, batchsize=batchsize, njobs=njobs)
    gridsearch(td.ResidualGradientDS, alpha=alphas, batchsize=batchsize, njobs=njobs)
    gridsearch(td.LinearTDLambda, alpha=alphas, lam=lambdas, batchsize=batchsize, njobs=njobs)
    gridsearch(td.LinearTD0, alpha=make_rmalpha(), gs_name="rm", batchsize=batchsize, njobs=njobs)

    gridsearch(td.TDCLambda, alpha=alphas, mu=mus, lam=lambdas, batchsize=batchsize, njobs=njobs)
    #gridsearch(td.TDC, alpha=alphas, mu=mus)
    gridsearch(td.GTD, alpha=alphas, mu=mus, batchsize=batchsize, njobs=njobs)
    gridsearch(td.GTD2, alpha=alphas, mu=mus, batchsize=batchsize, njobs=njobs)

    gridsearch(td.RecursiveLSTDLambda, lam=lambdas, eps=ls_eps, batchsize=batchsize, njobs=njobs)
    gridsearch(td.RecursiveBRM, eps=ls_eps, batchsize=batchsize, njobs=njobs)
    gridsearch(td.RecursiveBRMDS, eps=ls_eps, batchsize=batchsize, njobs=njobs)
    gridsearch(td.RecursiveLSPELambda, lam=lambdas, alpha=ls_alphas, batchsize=batchsize, njobs=njobs)

    gridsearch(td.FPKF, lam=lambdas, alpha=fpkf_alphas, beta= fpkf_beta, mins=mins, batchsize=batchsize, njobs=njobs)

    #gridsearch(td.GPTDP, sigma=sigmas)
    #gridsearch(td.KTD, reward_noise=reward_noises, eta=etas, P_init=P_inits)

    if task.off_policy:
        #gridsearch(td.GeriTDC, alpha=alphas, mu=mus, batchsize=batchsize, njobs=njobs)
        gridsearch(td.GeriTDCLambda, alpha=alphas, mu=mus, lam=lambdas, batchsize=batchsize, njobs=njobs)
        gridsearch(td.RecursiveLSTDLambdaJP, lam=lambdas, eps=ls_eps, batchsize=batchsize, njobs=njobs)
        gridsearch(td.RecursiveLSPELambdaCO, lam=lambdas, alpha=ls_alphas,batchsize=batchsize, njobs=njobs)
    #else:
        #gridsearch(td.GPTDPLambda, tau=sigmas, lam=lambdas, batchsize=batchsize, njobs=njobs)
        #gridsearch(td.GPTDP, sigma=sigmas, batchsize=batchsize, njobs=njobs)
        #gridsearch(td.KTD, reward_noise=reward_noises, eta=etas, P_init=P_inits)
