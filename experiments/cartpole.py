import td
import examples
import numpy as np
import matplotlib.pyplot as plt
import dynamic_prog as dp
import util
import features
import policies
from joblib import Parallel, delayed
from task import LinearContinuousValuePredictionTask

gamma=0.9
dt = 0.1

mdp = examples.PendulumSwingUpCartPole(dt = dt)
s = slice(-4., 5., 4.)
s2 = slice(-1., 1.1, 0.5)
means = np.mgrid[s,s,s2,s2].reshape(4,-1).T
#means = np.zeros((5**4, 4), dtype="float")
sigmas = np.ones(means.shape[0])
phi = features.gaussians(means,sigmas)


n_feat = len(phi(np.zeros(mdp.dim_S)))
print n_feat
theta_p = np.array([-0.1, 0., 0., 0.])

policy = policies.MarcsPolicy()
#theta0 =  10*np.ones(n_feat)
theta0 =  0.*np.ones(n_feat)

task = LinearContinuousValuePredictionTask(mdp, gamma, phi, theta0, policy=policy, normalize_phi=True)
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

l=100
error_every=1
name="swingup"

if __name__ =="__main__":
    err, rewards, states = task.ergodic_error_traces(methods,
        n_samples=l, error_every=error_every,
        criterion="RMSPBE",with_trace=True)

    plt.figure(figsize=(15,10))
    plt.subplot(311)
    plt.ylabel(r"$\sqrt{MSPBE}$")
    plt.xlabel("Timesteps")
    plt.title("Impoverished Swingup Onpolicy")
    for i, m in enumerate(methods):
        plt.plot(range(0,l,error_every), err[:,i], label=m.name)
        #plt.errorbar(range(0,l,error_every), mean[i,:], yerr=std[i,:], label=m.name)
    plt.legend()
    plt.subplot(312)
    plt.plot(range(0,l,error_every), rewards)
    plt.subplot(313)
    for i in range(4):
        plt.plot(range(0,l,error_every), states[:,i])
    plt.legend()
    plt.show()
"""
if __name__ =="__main__":
    mean, std, raw = task.avg_error_traces(methods, n_indep=1,
        n_samples=l, error_every=error_every,
        criterion="RMSPBE",
        verbose=True)

    plt.figure(figsize=(15,10))
    plt.ylabel(r"$\sqrt{MSPBE}$")
    plt.xlabel("Timesteps")
    plt.title("Impoverished Swingup Onpolicy")
    for i, m in enumerate(methods):
        plt.errorbar(range(0,l,error_every), mean[i,:], yerr=std[i,:], errorevery=l/error_every/8, label=m.name)
        #plt.errorbar(range(0,l,error_every), mean[i,:], yerr=std[i,:], label=m.name)
    plt.legend()
    plt.show()"""

