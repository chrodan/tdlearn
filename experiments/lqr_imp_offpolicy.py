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
#mdp = examples.MiniLQMDP(dt=dt)
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

#theta0 =  10*np.ones(n_feat)
theta0 = 0. * np.ones(n_feat)

task = LinearLQRValuePredictionTask(mdp, gamma, phi, theta0, policy=beh_policy, target_policy=target_policy, normalize_phi=True)

#phi = task.phi
print "V_true", task.V_true
print "theta_true"
theta_true = phi.param_forward(*task.V_true)
print theta_true

methods = []


alpha = .007
mu = .1  # optimal
gtd = td.GTD(alpha=alpha, beta=mu * alpha, phi=phi)
gtd.name = r"GTD $\alpha$={} $\mu$={}".format(alpha, mu)
gtd.color = "r"
methods.append(gtd)


alpha, mu = .02, .1  # optimal
gtd = td.GTD2(alpha=alpha, beta=mu * alpha, phi=phi)
gtd.name = r"GTD2 $\alpha$={} $\mu$={}".format(alpha, mu)
gtd.color = "orange"
methods.append(gtd)

alpha = .003
td0 = td.LinearTD0(alpha=alpha, phi=phi, gamma=gamma)
td0.name = r"TD(0) $\alpha$={}".format(alpha)
td0.color = "k"
methods.append(td0)

c = 0.04
mu = 0.25
alpha = td.RMalpha(c=c, mu=mu)
td0 = td.LinearTD0(alpha=alpha, phi=phi, gamma=gamma)
td0.name = r"TD(0) $\alpha={}\exp(-{}t)$".format(c, mu)
td0.color = "k"
methods.append(td0)

alpha = .003
mu = 0.0001
tdc = td.GeriTDC(alpha=alpha, mu=mu, phi=phi, gamma=gamma)
tdc.name = r"GeriTDC $\alpha$={} $\mu$={}".format(alpha, mu)
tdc.color = "b"
methods.append(tdc)

alpha = .003
mu = 0.0001
lam = .0
tdc = td.TDCLambda(alpha=alpha, lam=lam, mu=mu, phi=phi, gamma=gamma)
tdc.name = r"TDC({}) $\alpha$={} $\mu$={}".format(lam, alpha, mu)
tdc.color = "b"
methods.append(tdc)

lam = 0.
lstd = td.RecursiveLSTDLambda(lam=lam, eps=100, phi=phi, gamma=gamma)
lstd.name = r"LSTD({})".format(lam)
lstd.color = "g"
methods.append(lstd)

lstd = td.RecursiveLSTDLambdaJP(lam=0, eps=100, phi=phi, gamma=gamma)
lstd.name = r"LSTD-JP({})".format(0)
lstd.color = "g"
methods.append(lstd)

alpha = .1
lam = 0.
lstd = td.RecursiveLSPELambda(lam=lam, alpha=alpha, phi=phi, gamma=gamma)
lstd.name = r"LSPE({}) $\alpha$={}".format(lam, alpha)
lstd.color = "g"
methods.append(lstd)

alpha = 1.
lam = 0.8
lstd = td.FPKF(lam=lam, alpha=alpha, phi=phi, gamma=gamma)
lstd.name = r"FPKF({}) $\alpha$={}".format(lam, alpha)
lstd.color = "g"
methods.append(lstd)

alpha = .04
rg = td.ResidualGradient(alpha=alpha, phi=phi, gamma=gamma)
rg.name = r"RG $\alpha$={}".format(alpha)
rg.color = "brown"
methods.append(rg)

reward_noise = 1e-1
ktd = td.KTD(phi=phi, gamma=gamma, theta_noise=None, eta=1e-5, P_init=1.,
             reward_noise=reward_noise)
ktd.name = r"KTD $r_n={}$".format(reward_noise)
#methods.append(ktd)

sigma = 0.1
gptdp = td.GPTDP(phi=phi, sigma=sigma)
gptdp.name = r"GPTDP $\sigma$={}".format(sigma)
#methods.append(gptdp)

l = 20000
error_every = 100
name = "lqr_imp_offpolicy"
title = "4-dim. State Pole Balancing Offpolicy Diagonal Features"
criterion = "RMSPBE"
n_indep = 200
n_eps = 1

if __name__ == "__main__":
    from experiments import *
    mean, std, raw = run_experiment(n_jobs=-1, **globals())
    save_results(**globals())
    #plot_errorbar(**globals())
