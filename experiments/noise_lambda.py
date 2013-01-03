import td
import examples
import numpy as np
import matplotlib.pyplot as plt
import dynamic_prog as dp
import features
import policies
from task import LinearDiscreteValuePredictionTask
import pickle

n = 100
n_a = 100
n_feat = 30
mdp = examples.AdvancedRandomMDP(n, n_a, determinism=1., reward_level=5., seed=10)
phi = features.lin_random(n_feat, n, constant=True)
#phi = features.eye(n)
gamma = .95
beh_pol = policies.Discrete(np.random.rand(n, n_a))
#tar_pol = policies.Discrete(np.random.rand(n, n_a))
task = LinearDiscreteValuePredictionTask(mdp, gamma, phi, np.zeros(phi.dim),
                                         policy=beh_pol)


methods = []
lambdas =  np.linspace(0.,1.,11)
noises = np.linspace(0., .1, 6)
#noises = [0.0001, 0.001, 0.01, 0.1]
for lam in lambdas:
    eps = 10000.
    lstd = td.RecursiveLSTDLambda(lam=lam, eps=eps, phi=phi, gamma=gamma)
    lstd.name = r"LSTD({}) $\epsilon$={}".format(lam, eps)
    lstd.color = "g"
    #lstd.ls = "-."
    methods.append(lstd)

l = 501
error_every = 10
n_indep = 4
n_eps = 1
episodic=False
criteria = ["RMSPBE", "RMSBE", "RMSE"]
criterion = "RMSE"
name = "noise_lambda"
title = "4-dim. State Pole Balancing Onpolicy Full Features"


if __name__ == "__main__":
    from experiments import *
    import matplotlib.pyplot as plt
    mserrors = np.zeros((len(lambdas), len(noises)))
    variances = np.zeros(len(noises))
    plt.ion()
    a = np.random.rand(n, n_a)
    a = np.ones((n,n_a)) / n_a
    b = np.zeros((n, n_a))
    b[:,int(n/2)] = 1.
    for i, noise in enumerate(noises):
        #phi = features.lin_random(n_feat, n, constant=True)
        c = noise*a + (1-noise) * b
        #b = a / a.max(axis=1)[:,None]
        #b = np.power(b, noise)
        c /= c.sum(axis=1)[:,None]
        beh_pol = policies.Discrete(c)
        task = LinearDiscreteValuePredictionTask(mdp, gamma, phi, np.zeros(phi.dim),
                                         policy=beh_pol)

        mean, std, raw = run_experiment(n_jobs=-1, **globals())
        val = mean[:, -1,:]
        val[mean[:,-1,:] > mean[0,-1,0]] = mean[0, -1, 0]
        val = val.mean(axis=1)
        mserrors[:,i] = val - np.mean(val)
        #plot_errorbar(**globals())
        #break
        variances[i] = task.estimate_variance(n_samples=100).mean()
        print noise, variances[i], lambdas[np.argmin(val)]
    for i, noise in enumerate(noises):
        plt.plot(lambdas, mserrors[:,i], label=str(noise)+" "+str(variances[i]))
    #plt.figure()
    #plot_errorbar(**globals())
    #plt.imshow(mserrors, interpolation="nearest")
    #p2 = noises
    #p1 = lambdas
    #plt.xticks(range(len(p1)), p1)
    #plt.xticks(range(len(p2)), p2, rotation=45, ha="right")
    #plt.xlabel(r"$\lambda$")
    #plt.ylabel("Noise on the cart position")
    #plt.colorbar()
    #plt.legend()

