import td
import regtd
import examples
import numpy as np
#import matplotlib.pyplot as plt
import features
import policies
from task import LinearContinuousValuePredictionTask

gamma = 0.95
dt = 0.1


def make_slice(l, u, n):
    return slice(l, u + float(u - l) / (n - 1) / 2., float(u - l) / (n - 1))

mdp = examples.PendulumSwingUpCartPole(
    dt=dt, Sigma=np.zeros(4))  # np.array([0., 0.005, 0.005, 0.]))
n_slices = [3, 5, 7, 10]
n_slices2 = [3, 3, 10, 30]
bounds = [[0, 20], [-3, 4], [-12, 12], [-3, 3]]
s = [make_slice(b[0], b[1], n) for b, n in zip(bounds, n_slices)]
bounds = np.array(bounds, dtype="float")
s2 = [make_slice(b[0], b[1], n) for b, n in zip(bounds, n_slices2)]


means1 = np.mgrid[s[0], s[1], s[2], s[3]].reshape(4, -1).T
means2 = np.mgrid[s2[0], s2[1], s2[2], s2[3]].reshape(4, -1).T
means = np.vstack((means1, means2))
sigmas = np.vstack((np.ones_like(means1) * (
    (bounds[:, 1] - bounds[:, 0]) / 2. / (np.array(n_slices) - 1)).flatten(),
np.ones_like(means2) * (
    (bounds[:, 1] - bounds[:, 0]) / 2. / (np.array(n_slices2) - 1)).flatten()))
phi = features.gaussians(means, sigmas)


n_feat = len(phi(np.zeros(mdp.dim_S)))
print "Number of features:", n_feat
theta_p = np.array([-0.1, 0., 0., 0.])

policy = policies.MarcsPolicy(noise=np.array([0.1]))
theta0 = 0. * np.ones(n_feat)

task = LinearContinuousValuePredictionTask(
    mdp, gamma, phi, theta0, policy=policy,
    normalize_phi=False,
    mu_subsample=1, mu_iter=200,
    mu_restarts=30, mu_next=500)


methods = []

alpha = .5
mu = 0.01  # optimal
gtd = td.GTD(alpha=alpha, beta=mu * alpha, phi=phi)
gtd.name = r"GTD $\alpha$={} $\mu$={}".format(alpha, mu)
gtd.color = "r"
methods.append(gtd)


alpha, mu = .5, 8.  # optimal
gtd = td.GTD2(alpha=alpha, beta=mu * alpha, phi=phi)
gtd.name = r"GTD2 $\alpha$={} $\mu$={}".format(alpha, mu)
gtd.color = "orange"
methods.append(gtd)

alpha = .5
td0 = td.LinearTD0(alpha=alpha, phi=phi, gamma=gamma)
td0.name = r"TD(0) $\alpha$={}".format(alpha)
td0.color = "k"
methods.append(td0)

c = 10.
mu = 0.25
alpha = td.RMalpha(c=c, mu=mu)
td0 = td.LinearTD0(alpha=alpha, phi=phi, gamma=gamma)
td0.name = r"TD(0) $\alpha={}\exp(-{}t)$".format(c, mu)
td0.color = "k"
methods.append(td0)

alpha = .5
mu = 8.
lam = .2
tdc = td.TDCLambda(alpha=alpha, lam=lam, mu=mu, phi=phi, gamma=gamma)
tdc.name = r"TDC({}) $\alpha$={} $\mu$={}".format(lam, alpha, mu)
tdc.color = "b"
methods.append(tdc)

alpha, mu = (.5, 8.)
tdc = td.TDC(alpha=alpha, beta=alpha * mu, phi=phi, gamma=gamma)
tdc.name = r"TDC $\alpha$={} $\mu$={}".format(alpha, mu)
tdc.color = "b"
methods.append(tdc)

lstd = td.RecursiveLSTDLambda(lam=0, eps=100, phi=phi, gamma=gamma)
lstd.name = r"LSTD({})".format(0)
lstd.color = "g"
methods.append(lstd)

tau = 1e-2
lam = 0.
lstd = regtd.LSTDl1(lam=lam, tau=tau, phi=phi)
lstd.name = r"LSTD-$\ell_1$({}) $\tau$={}".format(lam, tau)
lstd.color = "g"
methods.append(lstd)

tau = 1e-2
lam = 0.
lstd = regtd.LarsTD(lam=lam, tau=tau, phi=phi)
lstd.name = r"LarsTD({}) $\tau$={}".format(lam, tau)
lstd.color = "g"
methods.append(lstd)

alpha = .5
rg = td.ResidualGradient(alpha=alpha, phi=phi, gamma=gamma)
rg.name = r"RG $\alpha$={}".format(alpha)
rg.color = "brown"
#methods.append(rg)

reward_noise = 1e-1
ktd = td.KTD(phi=phi, gamma=gamma, theta_noise=None, eta=1e-5, P_init=1.,
             reward_noise=reward_noise)
ktd.name = r"KTD $r_n={}$".format(reward_noise)
#methods.append(ktd)

sigma = 0.2
gptdp = td.GPTDP(phi=phi, sigma=sigma)
gptdp.name = r"GPTDP $\sigma$={}".format(sigma)
#methods.append(gptdp)

l = 200
n_eps = 10
error_every = 200
name = "swingupmanyfeat_gauss_onpolicy"
title = "Cartpole Swingup Onpolicy"
n_indep = 1
criterion = "RMSPBE"
criteria = ["RMSPBE", "RMSBE"]
if __name__ == "__main__":
    from experiments import *
    mean, std, raw = run_experiment(n_jobs=1, **globals())
    #save_results(**globals())
    plot_errorbar(**globals())
