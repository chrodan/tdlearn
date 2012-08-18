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
import task as t
gamma=0.9
sigma = np.array([0.]*3 + [0.01])

dt = 0.1
#mdp = examples.MiniLQMDP(dt=dt)
mdp = examples.PoleBalancingMDP(sigma=sigma, dt=dt)

phi = features.squared_diag()


n_feat = len(phi(np.zeros(mdp.dim_S)))
theta_p,_,_ = dp.solve_LQR(mdp,  gamma=gamma)
print theta_p
theta_p = np.array(theta_p).flatten()

policy = policies.LinearContinuous(theta=theta_p, noise=np.zeros((1,1)))
#theta0 =  10*np.ones(n_feat)
theta0 =  0.*np.ones(n_feat)

task = LinearLQRValuePredictionTask(mdp, gamma, phi, theta0, policy=policy, normalize_phi=True)
task.seed=0
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
alpha = 0.01
mu = 0.1 #optimal
gtd = td.GTD(alpha=alpha, beta=mu*alpha, phi=phi)
gtd.name = r"GTD $\alpha$={} $\mu$={}".format(alpha, mu)
gtd.color = "r"
methods.append(gtd)

#for alpha in [.005,0.01,0.02]:
#    for mu in [0.01, 0.1]:
alpha, mu = 0.01, 0.5 #optimal
gtd = td.GTD2(alpha=alpha, beta=mu*alpha, phi=phi)
gtd.name = r"GTD2 $\alpha$={} $\mu$={}".format(alpha, mu)
gtd.color = "orange"
methods.append(gtd)


alpha = .005
td0 = td.LinearTD0(alpha=alpha, phi=phi, gamma=gamma)
td0.name = r"TD(0) $\alpha$={}".format(alpha)
td0.color = "k"
methods.append(td0)

#for alpha in [0.005, 0.01, 0.02]:
#    for mu in [0.01, 0.1]:
for alpha, mu in [(.005,0.001)]: #optimal
    tdc = td.TDC(alpha=alpha, beta=alpha*mu, phi=phi, gamma=gamma)
    tdc.name = r"TDC $\alpha$={} $\mu$={}".format(alpha, mu)
    tdc.color = "b"
    methods.append(tdc)

#methods = []
#for eps in np.power(10,np.arange(-1,4)):
eps=100
lstd = td.RecursiveLSTDLambda(lam=0, eps=eps, phi=phi, gamma=gamma)
lstd.name = r"LSTD({}) $\epsilon$={}".format(0, eps)
lstd.color = "g"
methods.append(lstd)
#
#methods = []
#for alpha in [0.01, 0.02, 0.03]:
#alpha = .2
alpha=.04
rg = td.ResidualGradient(alpha=alpha, phi=phi, gamma=gamma)
rg.name = r"RG $\alpha$={}".format(alpha)
rg.color = "brown"
methods.append(rg)

eta = 0.001
reward_noise=0.01
P_init=1.
ktd = td.KTD(phi=phi, gamma=1., P_init=P_init,theta_noise=None, eta=eta, reward_noise=reward_noise)
ktd.name = r"KTD $\eta$={}, $\sigma^2$={}".format(eta, reward_noise)
methods.append(ktd)

eta = 0.0001
reward_noise=0.001
ktd = td.KTD(phi=phi, gamma=1., P_init=P_init,theta_noise=None, eta=eta, reward_noise=reward_noise)
ktd.name = r"KTD $\eta$={}, $\sigma^2$={}".format(eta, reward_noise)
methods.append(ktd)

l=16000
error_every=200
n_indep=50
name="lqr_imp_onpolicy"
title="4-dim. State Pole Balancing Onpolicy Diagonal Features"
criterion="RMSPBE"

if __name__ =="__main__":
    from experiments import *
    mean, std, raw = run_experiment(n_jobs=2, **globals())
    save_results(**globals())
    plot_errorbar(**globals())