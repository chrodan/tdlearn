import td
import examples
import numpy as np
import matplotlib.pyplot as plt
import features
import policies
from task import LinearContinuousValuePredictionTask

gamma=0.9
dt = 0.1

def make_slice(l,u,n):
    return slice(l,u+1, float(u-l)/n)

mdp = examples.PendulumSwingUpCartPole(dt = dt, Sigma=np.array([0., 0.005, 0.005, 0.]))
n_slices=[2,3,5,10]
bounds = [[0,20], [-3, 4], [-12, 12], [-3, 3]]
s = [make_slice(b[0], b[1], n) for b,n in zip(bounds, n_slices)]
bounds = np.array(bounds, dtype="float")
means = np.mgrid[s[0], s[1], s[2], s[3]].reshape(4,-1).T

sigmas = np.ones_like(means) * ((bounds[:,1]-bounds[:,0])/2./np.array(n_slices)).flatten()
phi = features.gaussians(means,sigmas)


n_feat = len(phi(np.zeros(mdp.dim_S)))
print n_feat
theta_p = np.array([-0.1, 0., 0., 0.])

policy = policies.MarcsPolicy()#noise=np.array([[0.1]]))
theta0 =  0.*np.ones(n_feat)

task = LinearContinuousValuePredictionTask(mdp, gamma, phi, theta0, policy=policy, 
                                           normalize_phi=True, 
                                           mu_subsample=1, mu_iter=200,
                                           mu_restarts=30)


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

alpha = 50.
td0 = td.LinearTD0(alpha=alpha, phi=phi, gamma=gamma)
td0.name = r"TD(0) $\alpha$={}".format(alpha)
td0.color = "k"
methods.append(td0)

alpha, mu = (.001,0.5)
tdc = td.TDC(alpha=alpha, beta=alpha*mu, phi=phi, gamma=gamma)
tdc.name = r"TDC $\alpha$={} $\mu$={}".format(alpha, mu)
tdc.color = "b"
#methods.append(tdc)

lstd = td.RecursiveLSTDLambda(lam=0, eps=100, phi=phi, gamma=gamma)
lstd.name = r"LSTD({})".format(0)
lstd.color = "g"
methods.append(lstd)

alpha=.008
rg = td.ResidualGradient(alpha=alpha, phi=phi, gamma=gamma)
rg.name = r"RG $\alpha$={}".format(alpha)
rg.color = "brown"
#methods.append(rg)

l=200
n_eps=4000
error_every=8000
name="swingup_"+str(n_slices[0])+"-"+ \
    str(n_slices[1])+"-"+str(n_slices[2])+"-"+str(n_slices[3])+"_gauss_onpolicy"
title="Cartpole Swingup Onpolicy"
n_indep=1
criterion="RMSPBE"

if __name__ =="__main__":
    from experiments import *
    mean, std, raw = run_experiment(n_jobs=1, **globals())
    save_results(**globals())
    plot_errorbar(**globals())

