import td
import examples
import numpy as np
import matplotlib.pyplot as plt
import features
import policies
from task import LinearDiscreteValuePredictionTask
from experiments import *
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.cm as cm
n = 40
mdp = examples.ActionMDP(n, reward_level=5.)
gamma = .95
phi = features.eye(n)


methods = []
lambdas = np.linspace(0., 1., 41)
noises = [0.001, 0.1, 0.2, 0.3, 0.4, 0.6, 0.7, 1.0]
for lam in lambdas:
    eps = 10000.
    lstd = td.RecursiveLSTDLambda(lam=lam, eps=eps, phi=phi)
    lstd.name = r"LSTD({}) $\epsilon$={}".format(lam, eps)
    lstd.color = "g"
    #lstd.ls = "-."
    methods.append(lstd)

l = 501
error_every = 10
n_indep = 16
n_eps = 1
episodic = False
criteria = ["RMSPBE", "RMSBE", "RMSE"]
criterion = "RMSE"
name = "noise_lambda"
title = "4-dim. State Pole Balancing Onpolicy Full Features"

def lambda_errors(phi, lambdas, noises):
    mserrors = np.zeros((len(lambdas), len(noises)))
    variances = np.zeros(len(noises))
    a = np.ones((n, n)) / n
    b = np.zeros((n, n))
    for i in range(n - 1):
        b[i, i + 1] = 1.
    b[-1, 0] = 1.
    for i, noise in enumerate(noises):
        c = noise * a + (1 - noise) * b
        c /= c.sum(axis=1)[:, None]
        beh_pol = policies.Discrete(c)
        task = LinearDiscreteValuePredictionTask(
            mdp, gamma, phi, np.zeros(phi.dim),
            policy=beh_pol)
        d = globals().copy()
        d["phi"] = phi
        d["task"] = task
        mean, std, raw = run_experiment(n_jobs=-1, **d)
        val = mean[:, -1, n:]
        val[mean[:, -1, n:] > mean[0, -1, 0]] = mean[0, -1, 0]
        val = val.mean(axis=1)
        mserrors[:, i] = val - np.mean(val)
        print noise, lambdas[np.argmin(val)]
    #mserrors -= mserrors.min(axis=1)[:,None]
    #mserrors /= mserrors.max(axis=1)[:,None]
    return mserrors

if __name__ == "__main__":
    plt.ion()
    n_feat = 20
    phi = features.lin_random(n_feat, n, constant=True)
    mserrors2 = lambda_errors(phi, lambdas, noises)
    mserrors2 -= mserrors.min(axis=0)
    mserror2 /= mserrors.max(axis=0)
    phi = features.eye(n)
    mserrors1 = lambda_errors(phi, lambdas, noises)

    plt.figure()
    mymap = mpl.colors.LinearSegmentedColormap.from_list('mycolors',['blue','red'])
    Z = [[0,0],[0,0]]
    CS3 = plt.contourf(Z, noises, cmap=mymap)
    plt.clf()

    for i, noise in enumerate(noises):
        col = mymap(i/float(len(noises)),1)
        plt.plot(lambdas, mserrors2[:, i], label=str(noise), color=col, linewidth=2)
    #plt.ylim(-.1,1.1)
    plt.colorbar(CS3)

    plt.figure()
    for i, noise in enumerate(noises):
        col = mymap(i/float(len(noises)),1)
        plt.plot(lambdas, mserrors2[:, i], label=str(noise), color=col, linewidth=2)
    plt.ylim(-.1,1.1)
    plt.colorbar(CS3)

