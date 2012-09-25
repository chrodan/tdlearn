import td
import examples
import numpy as np
#import matplotlib.pyplot as plt
import features
import policies
from task import LinearContinuousValuePredictionTask

gamma = 0.9
dt = 0.1


def make_slice(l, u, n):
    return slice(l, u + float(u - l) / (n - 1) / 2., float(u - l) / (n - 1))

mdp = examples.PendulumSwingUpCartPole(
    dt=dt, Sigma=np.zeros(4))  # np.array([0., 0.005, 0.005, 0.]))
n_slices = [3, 5, 7, 10]
bounds = [[0, 20], [-3, 4], [-12, 12], [-3, 3]]
s = [make_slice(b[0], b[1], n) for b, n in zip(bounds, n_slices)]
bounds = np.array(bounds, dtype="float")
means = np.mgrid[s[0], s[1], s[2], s[3]].reshape(4, -1).T

sigmas = np.ones_like(means) * (
    (bounds[:, 1] - bounds[:, 0]) / 2. / (np.array(n_slices) - 1)).flatten()
phi = features.gaussians(means, sigmas)


n_feat = len(phi(np.zeros(mdp.dim_S)))
print "Number of features:", n_feat
theta_p = np.array([-0.1, 0., 0., 0.])

policy = policies.MarcsPolicy(noise=np.array([0.1]))
policy_t = policies.MarcsPolicy(noise=np.array([0.01]))
theta0 = 0. * np.ones(n_feat)

task = LinearContinuousValuePredictionTask(
    mdp, gamma, phi, theta0, policy=policy, target_policy=policy_t,
    normalize_phi=False,
    mu_subsample=1, mu_iter=200,
    mu_restarts=30, mu_next=500)


methods = []

alpha = .5
mu = 0.01  # optimal
gtd = td.GTD(alpha=alpha, beta=mu * alpha, phi=phi)
gtd.name = r"GTD $\alpha$={} $\mu$={}".format(alpha, mu)
gtd.color = "r"
#methods.append(gtd)


alpha, mu = .5, 8.  # optimal
gtd = td.GTD2(alpha=alpha, beta=mu * alpha, phi=phi)
gtd.name = r"GTD2 $\alpha$={} $\mu$={}".format(alpha, mu)
gtd.color = "orange"
methods.append(gtd)

alpha = .5
td0 = td.LinearTD0(alpha=alpha, phi=phi, gamma=gamma)
td0.name = r"TD(0) $\alpha$={}".format(alpha)
td0.color = "k"
#methods.append(td0)

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

lstd = td.RecursiveLSTDLambdaJP(lam=0, eps=100, phi=phi, gamma=gamma)
lstd.name = r"LSTD-JP({})".format(0)
lstd.color = "g"
methods.append(lstd)

alpha = 1.
lam = 0.
lstd = td.RecursiveLSPELambda(lam=lam, alpha=alpha, phi=phi, gamma=gamma)
lstd.name = r"LSPE({}) $\alpha$={}".format(lam, alpha)
lstd.color = "g"
#methods.append(lstd)

alpha = 1.
lam = 0.
lstd = td.FPKF(lam=lam, alpha=alpha, phi=phi, gamma=gamma)
lstd.name = r"FPKF({}) $\alpha$={}".format(lam, alpha)
lstd.color = "g"
#methods.append(lstd)

alpha = .5
rg = td.ResidualGradient(alpha=alpha, phi=phi, gamma=gamma)
rg.name = r"RG $\alpha$={}".format(alpha)
rg.color = "brown"
methods.append(rg)


l = 200
n_eps = 1000
error_every = 4000
name = "swingup_" + str(n_slices[0]) + "-" + \
    str(n_slices[1]) + "-" + str(n_slices[2]) + "-" + str(n_slices[3]
                                                          ) + "_gauss_offpolicy"
title = "Cartpole Swingup Offpolicy"
n_indep = 10
criterion = "RMSPBE"

if __name__ == "__main__":
    from experiments import *
    mean, std, raw = run_experiment(n_jobs=-1, **globals())
    save_results(**globals())
    #plot_errorbar(**globals())
