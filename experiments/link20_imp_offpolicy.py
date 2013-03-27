import td
import examples
import numpy as np
import matplotlib.pyplot as plt
import dynamic_prog as dp
import util
import features
import policies
from joblib import Parallel, delayed
from task import LinearLQRValuePredictionTask
import itertools

gamma=0.95
dt = 0.1
dim = 20
sigma = np.ones(2*dim)*0.01
mdp = examples.NLinkPendulumMDP(np.ones(dim)*.5, np.ones(dim)*.6, sigma=sigma, dt=dt)
phi = features.squared_diag(2*dim)


n_feat = len(phi(np.zeros(mdp.dim_S)))
theta_p,_,_ = dp.solve_LQR(mdp, gamma=gamma)
theta_p = np.array(theta_p)
policy = policies.LinearContinuous(theta=theta_p, noise=np.ones(dim)*0.01)
target_policy = policies.LinearContinuous(theta=theta_p, noise=np.ones(dim)*0.005)
theta0 =  0.*np.ones(n_feat)

task = LinearLQRValuePredictionTask(mdp, gamma, phi, theta0, policy=policy, mu_next=1000,
                                    target_policy=target_policy, normalize_phi=True)

methods = []
alpha = 0.003
mu = 16.
gtd = td.GTD(alpha=alpha, beta=mu * alpha, phi=phi)
gtd.name = r"GTD $\alpha$={} $\mu$={}".format(alpha, mu)
gtd.color = "r"
methods.append(gtd)

alpha, mu = 0.5, .01
gtd = td.GTD2(alpha=alpha, beta=mu * alpha, phi=phi)
gtd.name = r"GTD2 $\alpha$={} $\mu$={}".format(alpha, mu)
gtd.color = "orange"
methods.append(gtd)

alpha = td.RMalpha(0.7, 0.25)
lam = .0
td0 = td.LinearTDLambda(alpha=alpha, lam=lam, phi=phi, gamma=gamma)
td0.name = r"TD({}) $\alpha$={}".format(lam, alpha)
td0.color = "k"
methods.append(td0)

alpha = .05
lam = .0
td0 = td.LinearTDLambda(alpha=alpha, lam=lam, phi=phi, gamma=gamma)
td0.name = r"TD({}) $\alpha$={}".format(lam, alpha)
td0.color = "k"
methods.append(td0)

lam = 0.0
alpha = 0.05
mu = .01
tdc = td.TDCLambda(alpha=alpha, mu = mu, lam=lam, phi=phi, gamma=gamma)
tdc.name = r"TDC({}) $\alpha$={} $\mu$={}".format(lam, alpha, mu)
tdc.color = "b"
methods.append(tdc)

lam = 0.0
alpha = 0.06
mu = .05
tdc = td.GeriTDCLambda(alpha=alpha, mu = mu, lam=lam, phi=phi, gamma=gamma)
tdc.name = r"GeriTDC({}) $\alpha$={} $\mu$={}".format(lam, alpha, mu)
tdc.color = "b"
methods.append(tdc)

alpha = .01
lam = .0
lstd = td.RecursiveLSPELambda(lam=lam, alpha=alpha, phi=phi, gamma=gamma)
lstd.name = r"LSPE({}) $\alpha$={}".format(lam, alpha)
lstd.color = "g"
methods.append(lstd)

alpha = .5
lam = .0
lstd = td.RecursiveLSPELambdaCO(lam=lam, alpha=alpha, phi=phi, gamma=gamma)
lstd.name = r"LSPE({})-CO $\alpha$={}".format(lam, alpha)
lstd.color = "g"
methods.append(lstd)

lam = 0.
eps = 100
lstd = td.RecursiveLSTDLambda(lam=lam, eps=eps, phi=phi, gamma=gamma)
lstd.name = r"LSTD({}) $\epsilon$={}".format(lam, eps)
lstd.color = "g"
lstd.ls = "-."
methods.append(lstd)

lam = 0.
eps = 100
lstd = td.RecursiveLSTDLambdaJP(lam=lam, eps=eps, phi=phi, gamma=gamma)
lstd.name = r"LSTD({})-CO $\epsilon$={}".format(lam, eps)
lstd.color = "g"
lstd.ls = "-."
methods.append(lstd)
#
alpha = 0.3
beta=10.
mins=0.
lam = .0
lstd = td.FPKF(lam=lam, alpha = alpha, beta=beta, mins=min,phi=phi, gamma=gamma)
lstd.name = r"FPKF({}) $\alpha$={} $\beta$={} m={}".format(lam, alpha, beta, mins)
lstd.color = "g"
lstd.ls = "-."
methods.append(lstd)

alpha = .05
rg = td.ResidualGradientDS(alpha=alpha, phi=phi, gamma=gamma)
rg.name = r"RG DS $\alpha$={}".format(alpha)
rg.color = "brown"
rg.ls = "--"
methods.append(rg)

alpha = .04
rg = td.ResidualGradient(alpha=alpha, phi=phi, gamma=gamma)
rg.name = r"RG $\alpha$={}".format(alpha)
rg.color = "brown"
methods.append(rg)

brm = td.RecursiveBRMDS(phi=phi,eps=100)
brm.name = "BRMDS"
brm.color = "b"
brm.ls = "--"
methods.append(brm)

brm = td.RecursiveBRM(phi=phi, eps=1e5)
brm.name = "BRM"
brm.color = "b"
methods.append(brm)

l=50000
error_every=500
n_indep=50
n_eps = 1
episodic=False
criteria = ["RMSPBE", "RMSBE", "RMSE"]
criterion="RMSPBE"
title = "12. 20-link Lin. Pole Balancing Off-pol."
name="link20_imp_offpolicy"
if __name__ =="__main__":
    from experiments import *
    mean, std, raw = run_experiment(n_jobs=-1, **globals())
    save_results(**globals())
    #plot_errorbar(**globals())
