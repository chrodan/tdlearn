import td
import examples
import numpy as np
import matplotlib.pyplot as plt
import dynamic_prog as dp
import features
import policies
from task import LinearLQRValuePredictionTask
import pickle

gamma = 0.9
sigma = np.array([0.] * 3 + [0.01])
#sigma = 0.
dt = 0.1
#mdp = examples.MiniLQMDP(dt=dt)
mdp = examples.PoleBalancingMDP(sigma=sigma, dt=dt)

phi = features.squared_tri(11)


n_feat = len(phi(np.zeros(mdp.dim_S)))
theta_p, _, _ = dp.solve_LQR(mdp, gamma=gamma)
print theta_p
theta_p = np.array(theta_p).flatten()

policy = policies.LinearContinuous(theta=theta_p, noise=np.zeros((1)))
#theta0 =  10*np.ones(n_feat)
theta0 = 0. * np.ones(n_feat)

task = LinearLQRValuePredictionTask(
    mdp, gamma, phi, theta0, policy=policy, normalize_phi=True, mu_next=1000)
#phi = task.phi
print "V_true", task.V_true
print "theta_true"
theta_true = phi.param_forward(*task.V_true)
print theta_true
#task.theta0 = theta_true
methods = []

#for alpha in [0.01, 0.005]:
#    for mu in [0.05, 0.1, 0.2, 0.01]:
#alpha = 0.1
alpha = 0.0005
mu = 0.01
gtd = td.GTD(alpha=alpha, beta=mu * alpha, phi=phi)
gtd.name = r"GTD $\alpha$={} $\mu$={}".format(alpha, mu)
gtd.color = "r"
methods.append(gtd)

alpha, mu = 0.004, 0.5
gtd = td.GTD2(alpha=alpha, beta=mu * alpha, phi=phi)
gtd.name = r"GTD2 $\alpha$={} $\mu$={}".format(alpha, mu)
gtd.color = "orange"
methods.append(gtd)

alpha = .01
lam = .2
td0 = td.LinearTDLambda(alpha=alpha, lam=lam, phi=phi, gamma=gamma)
td0.name = r"TD({}) $\alpha$={}".format(lam, alpha)
td0.color = "k"
methods.append(td0)

lam = 0.2
alpha = 0.003
mu = 0.5
tdc = td.TDCLambda(alpha=alpha, mu = mu, lam=lam, phi=phi, gamma=gamma)
tdc.name = r"TDC({}) $\alpha$={} $\mu$={}".format(lam, alpha, mu)
tdc.color = "b"
methods.append(tdc)

alpha = 1.
lam = 0.
lstd = td.RecursiveLSPELambda(lam=lam, alpha=alpha, phi=phi, gamma=gamma)
lstd.name = r"LSPE({}) $\alpha$={}".format(lam, alpha)
lstd.color = "g"
methods.append(lstd)

eps = 100
lstd = td.RecursiveLSTDLambda(lam=0, eps=eps, phi=phi, gamma=gamma)
lstd.name = r"LSTD({}) $\epsilon$={}".format(0, eps)
lstd.color = "g"
methods.append(lstd)
#
alpha = 0.1
lam = 0.
lstd = td.FPKF(lam=lam, alpha = alpha, phi=phi, gamma=gamma)
lstd.name = r"FPKF({}) $\alpha$={}".format(lam, alpha)
lstd.color = "g"
methods.append(lstd)[]

alpha = .02
rg = td.ResidualGradient(alpha=alpha, phi=phi, gamma=gamma)
rg.name = r"RG $\alpha$={}".format(alpha)
rg.color = "brown"
methods.append(rg)

reward_noise = 1e-1
ktd = td.KTD(phi=phi, gamma=gamma, theta_noise=None, eta=1e-5, P_init=1.,
             reward_noise=reward_noise)
ktd.name = r"KTD $r_n={}$".format(reward_noise)
#methods.append(ktd)


sigma = 1e-5
gptdp = td.GPTDP(phi=phi, sigma=sigma)
gptdp.name = r"GPTDP $\sigma$={}".format(sigma)
methods.append(gptdp)

l = 50000
error_every = 2000
n_indep = 50
n_eps = 1
name = "lqr_full_onpolicy"
title = "4-dim. State Pole Balancing Onpolicy"

criterion = "RMSPBE"

if __name__ == "__main__":
    from experiments import *
    mean, std, raw = run_experiment(n_jobs=2, **globals())
    #save_results(**globals())
    plot_errorbar(**globals())
