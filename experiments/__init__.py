__author__ = 'dann'
import pickle
import numpy as np
import os
import matplotlib.pyplot as plt
plt.ion()

def run_experiment(task, methods, n_indep, l, error_every, name, n_eps,
                   mdp, phi, title, verbose=True, n_jobs=1, criteria=None,
                   episodic=False, **kwargs):
    a, b, c = task.avg_error_traces(methods, n_indep=n_indep, n_eps=n_eps,
                                    n_samples=l, error_every=error_every,
                                    criteria=criteria,
                                    verbose=verbose, n_jobs=n_jobs, episodic=episodic)
    return a, b, c

def plot_path(path, method_id, methods, criterion, title):
    plt.figure(figsize=(15,10))
    plt.ylabel(criterion)
    plt.xlabel("Regularization Parameter")
    plt.title(title + " " + methods[method_id].name)

    par, theta, err = zip(*path[criterion][method_id])
    plt.plot(par, err)
    plt.show()

def save_results(name, l, criteria, error_every, n_indep, n_eps, methods,
                 mdp, phi, title, mean, std, raw, gamma, episodic=False, **kwargs):
    if not os.path.exists("data/{name}".format(name=name)):
        os.makedirs("data/{name}".format(name=name))

    with open("data/{name}/setting.pck".format(name=name), "w") as f:
        pickle.dump(dict(l=l, criteria=criteria, gamma=gamma,
                         error_every=error_every,
                         n_indep=n_indep,
                         episodic=episodic,
                         n_eps=n_eps,
                         methods=methods,
                         mdp=mdp, phi=phi, title=title, name=name), f)

    np.savez_compressed("data/{name}/results.npz".format(
        name=name), mean=mean, std=std, raw=raw)


def load_results(name):
    with open("data/{name}/setting.pck".format(name=name), "r") as f:
        d = pickle.load(f)

    d2 = np.load("data/{name}/results.npz".format(name=name))
    d.update(d2)
    return d


def plot_errorbar(title, methods, mean, std, l, error_every, criterion,
                  criteria, n_eps, episodic=False, ncol=1, **kwargs):
    plt.figure(figsize=(15, 10))
    plt.ylabel(criterion)
    plt.xlabel("Timesteps")
    plt.title(title)

    k = criteria.index(criterion)
    x = range(0, l * n_eps, error_every) if not episodic else range(n_eps)
    if episodic:
        ee = int(n_eps / 8.)
    else:
        ee = int(l * n_eps / error_every / 8.)
    if ee < 1:
        ee = 1
    lss = ["-","--",".-"]
    for i, m in enumerate(methods):
        if hasattr(m, "hide") and m.hide:
            continue
        ls = lss[int(i/7)]
        plt.errorbar(x, mean[i, k, :], yerr=std[i, k, :],
                     errorevery=ee, label=m.name, ls=ls)
    plt.legend(ncol=ncol)
    plt.show()
