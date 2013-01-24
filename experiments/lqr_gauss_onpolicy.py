import td
import examples
import regtd
import numpy as np
import matplotlib.pyplot as plt
import dynamic_prog as dp
import features
import policies
from task import LinearLQRValuePredictionTask, LinearContinuousValuePredictionTask
import pickle
import util

gamma = 0.95
sigma = np.array([0.] * 3 + [0.01])
dt = 0.1
mdp = examples.PoleBalancingMDP(sigma=sigma, dt=dt)
theta_p, _, _ = dp.solve_LQR(mdp, gamma=gamma)
theta_p = np.array(theta_p).flatten()
policy = policies.LinearContinuous(theta=theta_p, noise=np.zeros((1)))

states,_,_,_,_ = mdp.samples_cached(n_iter=15000, n_restarts=1,
                                policy=policy,seed=8000)

def make_slice(l, u, n):
    return slice(l, u + float(u - l) / (n - 1) / 2., float(u - l) / (n - 1))

n_slices = [3, 5, 7,10]
bounds = [[-0.012, 0.012], [-0.02, 0.02], [-.6, .6], [-.6, .6]]
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



theta0 = np.zeros(phi.dim)

task = LinearContinuousValuePredictionTask(
    mdp, gamma, phi, theta0, policy=policy, normalize_phi=False, mu_next=200)

methods = []
alpha = 0.001
mu = .01
gtd = td.GTD(alpha=alpha, beta=mu * alpha, phi=phi)
gtd.name = r"GTD $\alpha$={} $\mu$={}".format(alpha, mu)
gtd.color = "r"
#methods.append(gtd)

alpha, mu = 0.006, 0.5
gtd = td.GTD2(alpha=alpha, beta=mu * alpha, phi=phi)
gtd.name = r"GTD2 $\alpha$={} $\mu$={}".format(alpha, mu)
gtd.color = "orange"
#methods.append(gtd)

alpha = td.RMalpha(0.6, .7)
lam = .0
td0 = td.LinearTDLambda(alpha=alpha, lam=lam, phi=phi, gamma=gamma)
td0.name = r"TD({}) $\alpha$={}".format(lam, alpha)
td0.color = "k"
#methods.append(td0)

alpha = .006
lam = .4
td0 = td.LinearTDLambda(alpha=alpha, lam=lam, phi=phi, gamma=gamma)
td0.name = r"TD({}) $\alpha$={}".format(lam, alpha)
td0.color = "k"
methods.append(td0)

lam = 0.2
alpha = 0.006
mu = 0.1
tdc = td.TDCLambda(alpha=alpha, mu = mu, lam=lam, phi=phi, gamma=gamma)
tdc.name = r"TDC({}) $\alpha$={} $\mu$={}".format(lam, alpha, mu)
tdc.color = "b"
methods.append(tdc)

alpha = 1.
lam = 0.2
lstd = td.RecursiveLSPELambda(lam=lam, alpha=alpha, phi=phi, gamma=gamma)
lstd.name = r"LSPE({}) $\alpha$={}".format(lam, alpha)
lstd.color = "g"
methods.append(lstd)

lam = 0.0
eps = 100000
lstd = td.RecursiveLSTDLambda(lam=lam, eps=eps, phi=phi, gamma=gamma)
lstd.name = r"LSTD({}) $\epsilon$={}".format(lam, eps)
lstd.color = "g"
lstd.ls = "-."
methods.append(lstd)
#
alpha = 0.5
beta=10.
mins=500
lam =.8
lstd = td.FPKF(lam=lam, alpha = alpha, beta=beta, mins=mins, phi=phi, gamma=gamma)
lstd.name = r"FPKF({}) $\alpha$={}".format(lam, alpha)
lstd.color = "g"
lstd.ls = "-."
#methods.append(lstd)

alpha = .02
rg = td.ResidualGradientDS(alpha=alpha, phi=phi, gamma=gamma)
rg.name = r"RG DS $\alpha$={}".format(alpha)
rg.color = "brown"
rg.ls = "--"
methods.append(rg)

alpha = .03
rg = td.ResidualGradient(alpha=alpha, phi=phi, gamma=gamma)
rg.name = r"RG $\alpha$={}".format(alpha)
rg.color = "brown"
methods.append(rg)

reward_noise = 1e-1
ktd = td.KTD(phi=phi, gamma=gamma, theta_noise=None, eta=1e-5, P_init=1.,
             reward_noise=reward_noise)
ktd.name = r"KTD $r_n={}$".format(reward_noise)
#methods.append(ktd)

brm = td.RecursiveBRMDS(phi=phi, eps=eps)
brm.name = "BRMDS"
brm.color = "b"
brm.ls = "--"
methods.append(brm)

brm = td.RecursiveBRM(phi=phi, eps=eps)
brm.name = "BRM"
brm.color = "b"
methods.append(brm)

tau=2e-9
lstd = regtd.LSTDl1(tau=tau, lam=0, phi=phi)
lstd.name = r"LSTD-l1({}) $\tau={}$".format(0,tau)
lstd.color = "b"
methods.append(lstd)

tau=1e-5
lstd = regtd.LarsTD(tau=tau, lam=0, phi=phi)
lstd.name = r"LarsTD({}) $\tau={}$".format(0,tau)
lstd.color = "b"
methods.append(lstd)


l = 15000
error_every = 500
n_indep = 1
n_eps = 1
episodic=False
criteria = ["RMSPBE", "RMSBE"]
criterion = "RMSPBE"
name = "lqr_gauss_onpolicy"
title = "4-dim. State Pole Balancing Onpolicy"


if __name__ == "__main__":
    from experiments import *
    task.mu
    mean, std, raw = run_experiment(n_jobs=1, verbose=4, **globals())
    save_results(**globals())
    plot_errorbar(**globals())
