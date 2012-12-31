import td
import examples
import numpy as np
import matplotlib.pyplot as plt
import dynamic_prog as dp
import features
import policies
from task import LinearDiscreteValuePredictionTask
import pickle

n = 200
n_a = 10
n_feat = 200
mdp = examples.RandomMDP(n, n_a)
phi = features.lin_random(n_feat, n, constant=True)
gamma = .95
beh_pol = policies.Discrete(np.random.rand(n, n_a))
tar_pol = policies.Discrete(np.random.rand(n, n_a))
task = LinearDiscreteValuePredictionTask(mdp, gamma, phi, np.zeros(phi.dim),
                                         policy=beh_pol, target_policy=tar_pol)


methods = []
lambdas =  np.linspace(0.,1.,7)
noises = [10,]
for lam in lambdas:
    eps = 1
    lstd = td.RecursiveLSTDLambda(lam=lam, eps=eps, phi=phi, gamma=gamma)
    lstd.name = r"LSTD({}) $\epsilon$={}".format(lam, eps)
    lstd.color = "g"
    #lstd.ls = "-."
    methods.append(lstd)

l = 501
error_every = 10
n_indep = 16
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
    plt.ion()
    for i, noise in enumerate(noises):
        n = noise
        mdp = examples.RandomMDP(n, n_a)
        phi = features.lin_random(n_feat, n, constant=True)
        beh_pol = policies.Discrete(np.random.rand(n, n_a))
        tar_pol = policies.Discrete(np.random.rand(n, n_a))
        task = LinearDiscreteValuePredictionTask(mdp, gamma, phi, np.zeros(phi.dim),
                                         policy=beh_pol)

        mean, std, raw = run_experiment(n_jobs=2, **globals())
        val = mean[:,-1,-1] #:].sum(axis=1)
        mserrors[:,i] = val - np.mean(val) + i
        plot_errorbar(**globals())
        #plt.plot(lambdas, mserrors[:,i], label=str(noise))
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

