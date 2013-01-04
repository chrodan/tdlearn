import td
import examples
import numpy as np
import matplotlib.pyplot as plt
import dynamic_prog as dp
import util
import features
import policies
from task import LinearLQRValuePredictionTask

gamma = 0.95
sigma = np.zeros(4)
sigma[-1] = 0.01
dt = 0.1
mdp = examples.PoleBalancingMDP(sigma=sigma, dt=dt)
phi = features.squared_diag(4)


n_feat = len(phi(np.zeros(mdp.dim_S)))
theta_p, _, _ = dp.solve_LQR(mdp, gamma=gamma)
print theta_p
theta_p = np.array(theta_p).flatten()
theta_o = theta_p.copy()
beh_policy = policies.LinearContinuous(theta=theta_o, noise=np.ones(1) * 0.01)
target_policy = policies.LinearContinuous(
    theta=theta_p, noise=np.ones(1) * 0.001)
theta0 = 0. * np.ones(n_feat)

task = LinearLQRValuePredictionTask(mdp, gamma, phi, theta0,
                                    policy=beh_policy, target_policy=target_policy,
                                    normalize_phi=True, mu_next=1000)



methods = []
alpha = 0.006
mu = .1
gtd = td.GTD(alpha=alpha, beta=mu * alpha, phi=phi)
gtd.name = r"GTD $\alpha$={} $\mu$={}".format(alpha, mu)
gtd.color = "r"
methods.append(gtd)

alpha, mu = 0.01, 0.1
gtd = td.GTD2(alpha=alpha, beta=mu * alpha, phi=phi)
gtd.name = r"GTD2 $\alpha$={} $\mu$={}".format(alpha, mu)
gtd.color = "orange"
methods.append(gtd)

alpha = td.RMalpha(0.5, 0.5)
lam = .0
td0 = td.LinearTDLambda(alpha=alpha, lam=lam, phi=phi, gamma=gamma)
td0.name = r"TD({}) $\alpha$={}".format(lam, alpha)
td0.color = "k"
methods.append(td0)

alpha = .002
lam = .2
td0 = td.LinearTDLambda(alpha=alpha, lam=lam, phi=phi, gamma=gamma)
td0.name = r"TD({}) $\alpha$={}".format(lam, alpha)
td0.color = "k"
methods.append(td0)

lam = 0.2
alpha = 0.002
mu = 0.0001
tdc = td.TDCLambda(alpha=alpha, mu = mu, lam=lam, phi=phi, gamma=gamma)
tdc.name = r"TDC({}) $\alpha$={} $\mu$={}".format(lam, alpha, mu)
tdc.color = "b"
methods.append(tdc)

lam = 0.2
alpha = 0.002
mu = 0.0001
tdc = td.GeriTDCLambda(alpha=alpha, mu = mu, lam=lam, phi=phi, gamma=gamma)
tdc.name = r"TDC({}) $\alpha$={} $\mu$={}".format(lam, alpha, mu)
tdc.color = "b"
methods.append(tdc)

alpha = .1
lam = 0.0
lstd = td.RecursiveLSPELambda(lam=lam, alpha=alpha, phi=phi, gamma=gamma)
lstd.name = r"LSPE({}) $\alpha$={}".format(lam, alpha)
lstd.color = "g"
methods.append(lstd)

alpha = 1.
lam = 0.0
lstd = td.RecursiveLSPELambdaCO(lam=lam, alpha=alpha, phi=phi, gamma=gamma)
lstd.name = r"LSPE({})-CO $\alpha$={}".format(lam, alpha)
lstd.color = "g"
methods.append(lstd)

lam = 0.
eps = 100
lstd = td.RecursiveLSTDLambdaJP(lam=lam, eps=eps, phi=phi, gamma=gamma)
lstd.name = r"LSTD-CO({}) $\epsilon$={}".format(lam, eps)
lstd.color = "g"
lstd.ls = "-."
methods.append(lstd)

lam = 0.
eps = 100
lstd = td.RecursiveLSTDLambda(lam=lam, eps=eps, phi=phi, gamma=gamma)
lstd.name = r"LSTD({}) $\epsilon$={}".format(lam, eps)
lstd.color = "g"
lstd.ls = "-."
methods.append(lstd)
#
alpha = 0.1
beta = 10.
mins = 0
lam = .4
lstd = td.FPKF(lam=lam, alpha = alpha, beta=beta, mins=mins, phi=phi, gamma=gamma)
lstd.name = r"FPKF({}) $\alpha$={}".format(lam, alpha)
lstd.color = "g"
lstd.ls = "-."
methods.append(lstd)

alpha = .005
rg = td.ResidualGradientDS(alpha=alpha, phi=phi, gamma=gamma)
rg.name = r"RG DS $\alpha$={}".format(alpha)
rg.color = "brown"
rg.ls = "--"
methods.append(rg)

alpha = .004
rg = td.ResidualGradient(alpha=alpha, phi=phi, gamma=gamma)
rg.name = r"RG $\alpha$={}".format(alpha)
rg.color = "brown"
methods.append(rg)


brm = td.RecursiveBRMDS(phi=phi)
brm.name = "BRMDS"
brm.color = "b"
brm.ls = "--"
methods.append(brm)

brm = td.RecursiveBRM(phi=phi)
brm.name = "BRM"
brm.color = "b"
methods.append(brm)

l = 30000
error_every = 300
n_indep = 50
n_eps = 1
episodic=False
criteria = ["RMSPBE", "RMSBE", "RMSE"]
criterion = "RMSPBE"
name = "lqr_imp_offpolicy"
title = "4-dim. State Pole Balancing Offpolicy Diagonal Features"


if __name__ == "__main__":
    from experiments import *
    mean, std, raw = run_experiment(n_jobs=-1, **globals())
    save_results(**globals())
#    plot_errorbar(**globals())
