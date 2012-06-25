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

gamma=0.9
sigma = np.zeros((4,4))
sigma[-1,-1] = 0.01

dt = 0.1
#mdp = examples.MiniLQMDP(dt=dt)
mdp = examples.PoleBalancingMDP(sigma=sigma, dt=dt)

phi = features.squared_diag()


n_feat = len(phi(np.zeros(mdp.dim_S)))
theta_p,_,_ = dp.solve_LQR(mdp, gamma=gamma)
print theta_p
theta_p = np.array(theta_p).flatten()
theta_o = theta_p.copy()
beh_policy = policies.LinearContinuous(theta=theta_o, noise=np.eye(1)*0.01)
target_policy = policies.LinearContinuous(theta=theta_p, noise=np.eye(1)*0.001)

#theta0 =  10*np.ones(n_feat)
theta0 =  0.*np.ones(n_feat)

task = LinearLQRValuePredictionTask(mdp, gamma, phi, theta0, policy=beh_policy, target_policy=target_policy,  normalize_phi=True)

#phi = task.phi
print "V_true", task.V_true
print "theta_true"
theta_true = phi.param_forward(*task.V_true)
print theta_true

l=20000
error_every=200




def plot_error_grid(val, alphas, mus, maxerr=5):
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

def gridsearch_2d():
    methods = [td.TDC, td.GeriTDC, td.GTD, td.GTD2]

    alphas = [0.0002, 0.0005] + list(np.arange(0.001, .01, 0.001)) + list(np.arange(0.01, 0.1, 0.01)) + [0.1, 0.2, 0.3, 0.4, 0.5]
    mus = [0.0001, 0.001, 0.01,0.01, 0.1, 0.5,1,2,4,8,16]
    params = list(itertools.product(alphas, mus))

    for m in methods:
        k = (delayed(run_2d)(*(list(p)+[m])) for p in params)
        res = Parallel(n_jobs=-1, verbose=11)(k)

        res = np.array(res).reshape(len(alphas), -1)
        with open("data/offpolicy_{}_gs.pck".format(m.__name__), "w") as f:
            pickle.dump(dict(params=params, alphas=alphas, mus=mus, res=res), f)


def gridsearch_1d():
    methods = [td.LinearTD0, td.ResidualGradient]

    alphas = [0.0002, 0.0005] + list(np.arange(0.001, .01, 0.001)) + list(np.arange(0.01, 0.1, 0.01)) + [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.]
    params = alphas

    for m in methods:
        k = (delayed(run_1d)(p,m) for p in params)
        res = Parallel(n_jobs=-1, verbose=11)(k)

        res = np.array(res).reshape(len(alphas), -1)
        with open("data/offpolicy_{}_gs.pck".format(m.__name__), "w") as f:
            pickle.dump(dict(params=params, alphas=alphas, res=res), f)

