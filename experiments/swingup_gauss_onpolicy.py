import td
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


def make_slice(l, u, n):
    return slice(l, u + float(u - l) / (n - 1) / 2., float(u - l) / (n - 1))

mdp = examples.PendulumSwingUpCartPole(
    dt=dt, Sigma=np.zeros(4), start_amp=2.)  # np.array([0., 0.005, 0.005, 0.]))
policy = policies.MarcsPolicy(noise=np.array([.05]))


states,_,_,_,_ = mdp.samples_cached(n_iter=200, n_restarts=30,
                                policy=policy,seed=8000)

n_slices = [3, 5, 7,10]
bounds = [[0, 35], [-3, 4], [-12, 12], [-3, 3]]
s = [make_slice(b[0], b[1], n) for b, n in zip(bounds, n_slices)]
bounds = np.array(bounds, dtype="float")
means = np.mgrid[s[0], s[1], s[2], s[3]].reshape(4, -1).T

sigmas = np.ones_like(means) * (
    (bounds[:, 1] - bounds[:, 0]) / 2. / (np.array(n_slices) - 1)).flatten()
phi = features.gaussians(means, sigmas, constant=False)
A = util.apply_rowise(arr=states, f=phi)
a = np.nonzero(np.sum(A > 0.05, axis=0) > 20)[0]
phi = features.gaussians(means[a], sigmas[a], constant=True)
print phi.dim, "features are used"
theta0 = 0. * np.ones(phi.dim)

task = LinearContinuousValuePredictionTask(
    mdp, gamma, phi, theta0, policy=policy,
    normalize_phi=False, mu_seed=1100,
    mu_subsample=1, mu_iter=200,
    mu_restarts=150, mu_next=300)


methods = []
alpha = 0.5
mu = .01
gtd = td.GTD(alpha=alpha, beta=mu * alpha, phi=phi)
gtd.name = r"GTD $\alpha$={} $\mu$={}".format(alpha, mu)
gtd.color = "r"
methods.append(gtd)

alpha, mu = 0.5, 8.
gtd = td.GTD2(alpha=alpha, beta=mu * alpha, phi=phi)
gtd.name = r"GTD2 $\alpha$={} $\mu$={}".format(alpha, mu)
gtd.color = "orange"
methods.append(gtd)

alpha = td.RMalpha(10., 0.25)
lam = .0
td0 = td.LinearTDLambda(alpha=alpha, lam=lam, phi=phi, gamma=gamma)
td0.name = r"TD({}) $\alpha$={}".format(lam, alpha)
td0.color = "k"
methods.append(td0)

alpha = .5
lam = .2
td0 = td.LinearTDLambda(alpha=alpha, lam=lam, phi=phi, gamma=gamma)
td0.name = r"TD({}) $\alpha$={}".format(lam, alpha)
td0.color = "k"
methods.append(td0)

lam = 0.2
alpha = 0.2
mu = 8.
tdc = td.TDCLambda(alpha=alpha, mu=mu, lam=lam, phi=phi, gamma=gamma)
tdc.name = r"TDC({}) $\alpha$={} $\mu$={}".format(lam, alpha, mu)
tdc.color = "b"
methods.append(tdc)

alpha = .001
lam = 0.0
lstd = td.RecursiveLSPELambda(lam=lam, alpha=alpha, phi=phi, gamma=gamma)
lstd.name = r"LSPE({}) $\alpha$={}".format(lam, alpha)
lstd.color = "g"
methods.append(lstd)

lam = 0.
eps = 10000
lstd = td.RecursiveLSTDLambda(lam=lam, eps=eps, phi=phi, gamma=gamma)
lstd.name = r"LSTD({}) $\epsilon$={}".format(lam, eps)
lstd.color = "g"
lstd.ls = "-."
methods.append(lstd)
#
alpha = 0.0005
lam = .2
lstd = td.FPKF(lam=lam, alpha=alpha, phi=phi, gamma=gamma)
lstd.name = r"FPKF({}) $\alpha$={}".format(lam, alpha)
lstd.color = "g"
lstd.ls = "-."
#methods.append(lstd)

alpha = .5
rg = td.ResidualGradientDS(alpha=alpha, phi=phi, gamma=gamma)
rg.name = r"RG DS $\alpha$={}".format(alpha)
rg.color = "brown"
rg.ls = "--"
methods.append(rg)

alpha = .5
rg = td.ResidualGradient(alpha=alpha, phi=phi, gamma=gamma)
rg.name = r"RG $\alpha$={}".format(alpha)
rg.color = "brown"
methods.append(rg)

lam = .0
sigma = 31.
gptdp = td.GPTDPLambda(phi=phi, tau=sigma, lam=lam)
gptdp.name = r"GPTDP({}) $\sigma$={}".format(lam, sigma)
gptdp.ls = "--"
#methods.append(gptdp)

brm = td.RecursiveBRMDS(phi=phi, eps=eps)
brm.name = "BRMDS"
brm.color = "b"
brm.ls = "--"
methods.append(brm)

brm = td.RecursiveBRM(phi=phi, eps=eps)
brm.name = "BRM"
brm.color = "b"
methods.append(brm)

tau = 0.0001
lstd = regtd.LSTDl1(tau=tau, lam=0, phi=phi)
lstd.name = r"LSTD-l1({}) $\tau={}$".format(0, tau)
lstd.color = "b"
#methods.append(lstd)

tau = 0.005
lstd = regtd.LarsTD(tau=tau, lam=0, phi=phi)
lstd.name = r"LarsTD({}) $\tau={}$".format(0, tau)
lstd.color = "b"
#methods.append(lstd)


l = 200
n_eps = 300  # 1000
error_every = 600  # 4000
name = "swingup_gauss_onpolicy_perf"
title = "Cartpole Swingup Onpolicy stochasticity only in start dist."
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
    mean, std, raw = run_experiment(n_jobs=1, **globals())
    save_results(**globals())
    plot_errorbar(**globals())
