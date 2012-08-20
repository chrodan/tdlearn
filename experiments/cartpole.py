import td
import examples
import numpy as np
import matplotlib.pyplot as plt
import features
import policies
from task import LinearContinuousValuePredictionTask

gamma=0.9
dt = 0.1

mdp = examples.PendulumSwingUpCartPole(dt = dt, Sigma=0.01)
s = slice(0., 1000., 1000.)
s2 = slice(-4., 4, 8./5)
s3 = slice(-10., 11., 20./5)
s4 = slice(-2.5, 3., 1.)
means = np.mgrid[s,s,s2,s2].reshape(4,-1).T
#means = np.zeros((5**4, 4), dtype="float")
sigmas = np.ones_like(means) * np.array([1000., 8./5, 4., 1.])
phi = features.gaussians(means,sigmas)


n_feat = len(phi(np.zeros(mdp.dim_S)))
print n_feat
theta_p = np.array([-0.1, 0., 0., 0.])

policy = policies.MarcsPolicy()
#theta0 =  10*np.ones(n_feat)
theta0 =  0.*np.ones(n_feat)

task = LinearContinuousValuePredictionTask(mdp, gamma, phi, theta0, policy=policy, 
                                           normalize_phi=True, 
                                           mu_subsample=200, mu_iter=200000,
                                           mu_restarts=3)
#print task.mu_phi
#task.seed=0
#phi = task.phi
#print "V_true", task.V_true
#print "theta_true"
#theta_true = phi.param_forward(*task.V_true)
#print theta_true
#task.theta0 = theta_true

methods = []

alpha = 0.0005
mu = 2 #optimal
gtd = td.GTD(alpha=alpha, beta=mu*alpha, phi=phi)
gtd.name = r"GTD $\alpha$={} $\mu$={}".format(alpha, mu)
gtd.color = "r"
#methods.append(gtd)


alpha, mu = 0.0005, 2 #optimal
gtd = td.GTD2(alpha=alpha, beta=mu*alpha, phi=phi)
gtd.name = r"GTD2 $\alpha$={} $\mu$={}".format(alpha, mu)
gtd.color = "orange"
#methods.append(gtd)

alpha = .0005
td0 = td.LinearTD0(alpha=alpha, phi=phi, gamma=gamma)
td0.name = r"TD(0) $\alpha$={}".format(alpha)
td0.color = "k"
#methods.append(td0)

alpha, mu = (.001,0.5)
tdc = td.TDC(alpha=alpha, beta=alpha*mu, phi=phi, gamma=gamma)
tdc.name = r"TDC $\alpha$={} $\mu$={}".format(alpha, mu)
tdc.color = "b"
#methods.append(tdc)

lstd = td.RecursiveLSTDLambda(lam=0, eps=1000, phi=phi, gamma=gamma)
lstd.name = r"LSTD({})".format(0)
lstd.color = "g"
methods.append(lstd)

alpha=.008
rg = td.ResidualGradient(alpha=alpha, phi=phi, gamma=gamma)
rg.name = r"RG $\alpha$={}".format(alpha)
rg.color = "brown"
#methods.append(rg)

l=200000
error_every=200
name="swingup_255gauss_onpolicy"
title="Cartpole Swingup Onpolicy"
n_indep=1
criterion="RMSPBE"

if __name__ =="__main__":
    from experiments import *
    mean, std, raw = run_experiment(n_jobs=1, **globals())
    #save_results(**globals())
    plot_errorbar(**globals())

