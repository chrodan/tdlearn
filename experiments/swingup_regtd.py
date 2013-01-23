import td
import regtd
import examples
import numpy as np
import regtd
#import matplotlib.pyplot as plt
import features
import policies
from task import LinearContinuousValuePredictionTask
import util
gamma = 0.95
dt = 0.1


mdp = examples.PendulumSwingUpCartPole(
    dt=dt, Sigma=np.zeros(4), start_amp=2.)  # np.array([0., 0.005, 0.005, 0.]))
policy = policies.MarcsPolicy(noise=np.array([.05]))


states,_,_,_,_ = mdp.samples_cached(n_iter=200, n_restarts=30,
                                policy=policy,seed=8000)

n_slices = [3, 5, 7,10]
n_slices2 = [5, 5, 14,20]
bounds = [[0, 35], [-3, 4], [-12, 12], [-3, 3]]
means, sigmas = features.make_grid(n_slices, bounds)
means2, sigmas2 = features.make_grid(n_slices2, bounds)
means = np.vstack([means,means2])
sigmas = np.vstack([sigmas, sigmas2])
phi = features.gaussians(means, sigmas, constant=False)
A = util.apply_rowise(arr=states, f=phi)
a = np.nonzero(np.sum(A > 0.05, axis=0) > 5)[0]
phi = features.gaussians(means[a], sigmas[a], constant=True)
print phi.dim, "features are used"
theta0 = 0. * np.ones(phi.dim)

task = LinearContinuousValuePredictionTask(
    mdp, gamma, phi, theta0, policy=policy,
    normalize_phi=False, mu_seed=1100,
    mu_subsample=1, mu_iter=200,
    mu_restarts=150, mu_next=300)


methods = []
lam = 0.0
alpha = 0.3
mu = .1
tdc = td.TDCLambda(alpha=alpha, mu=mu, lam=lam, phi=phi, gamma=gamma)
tdc.name = r"TDC({}) $\alpha$={} $\mu$={}".format(lam, alpha, mu)
tdc.color = "b"
#methods.append(tdc)

lam = 0.
eps = 10000
lstd = td.LSTDLambda(lam=lam, eps=eps, phi=phi, gamma=gamma)
lstd.name = r"LSTD({}) $\epsilon$={}".format(lam, eps)
lstd.color = "g"
lstd.ls = "-."
#methods.append(lstd)
#
lam = 0.
eps = 10
lstd = td.LSTDLambda(lam=lam, eps=eps, phi=phi, gamma=gamma)
lstd.name = r"LSTD({}) $\epsilon$={}".format(lam, eps)
lstd.color = "g"
lstd.ls = "-."
#methods.append(lstd)

tau = 0.0002
lstd = regtd.DLSTD(tau=tau, lam=0, nonreg_ids=[phi.dim -1], phi=phi)
lstd.name = r"DLSTD({}) $\tau={}$".format(0, tau)
lstd.color = "b"
#methods.append(lstd)

tau = 0.0001
lstd = regtd.LSTDl1(tau=tau, lam=0, phi=phi)
lstd.name = r"LSTD-l1({}) $\tau={}$".format(0, tau)
#methods.append(lstd)

tau = 0.0001
beta= 0.01
lstd = regtd.LSTDl21(beta=beta, tau=tau, lam=0, phi=phi)
lstd.name = r"LSTD-l21({}) $\beta={}$ $\tau={}$".format(0, beta, tau)
#methods.append(lstd)

tau = 0.005
lstd = regtd.LarsTD(tau=tau, lam=0,  phi=phi)
lstd.name = r"LarsTD({}) $\tau={}$".format(0, tau)
methods.append(lstd)

lstd = regtd.LSTDRP(dim_lower=-1, lam=0,  phi=phi)
lstd.name = r"LSTD-RP({})$".format(0, tau)
#methods.append(lstd)


l = 200
n_eps = 100  # 1000
error_every = 600  # 4000
name = "swingup_regtd"
title = "Cartpole Swingup Onpolicy"
n_indep = 1
episodic = False
criterion = "RMSPBE"
criteria = ["RMSPBE", "RMSBE"]
eval_on_traces=False
n_samples_eval=10000
verbose=4
gs_ignore_first_n = 10000
gs_max_weight = 3.
if __name__ == "__main__":
    from experiments import *
    task.mu
    #task.set_mu_from_trajectory(n_samples=l, n_eps=n_eps, verbose=4.,
    #            seed=0,
    #            n_samples_eval=10000)
    errors = task.regularization_paths(methods, n_samples=l, n_eps=1, seed=0,
                                       criteria=criteria, verbose=4)

    #mean, std, raw = run_experiment(n_jobs=1, **globals())
    #save_results(**globals())
    #plot_errorbar(**globals())
