__author__ = 'dann'
import pickle
import numpy as np
import os
from matplotlib.colors import LogNorm
import matplotlib.pyplot as plt
plt.ion()

exp_list = ["disc_random", "boyan", "lqr_imp_onpolicy", "lqr_imp_offpolicy",
            "lqr_full_onpolicy", "lqr_full_offpolicy", "swingup_gauss_onpolicy",
            "swingup_gauss_offpolicy", "baird", "link20_imp_onpolicy",
            "link20_imp_offpolicy"]


def load_result_file(fn, maxerr=5):
    with open(fn) as f:
        d = pickle.load(f)
    for i in range(d["res"].shape[-1]):
        print d["criteria"][i], np.nanmin(d["res"][..., i])
        best = d["params"][np.nanargmin(d["res"][..., i])]
        for n, v in zip(d["param_names"], best):
            print n, v
    return d

def plot_experiment(experiment, criterion):
    d = load_results(experiment)
    plot_errorbar(ncol=3, criterion=criterion, **d)

def plot_2d_error_grid_experiments(experiments, method, criterion, **kwargs):
    for e in experiments:
        fn = "data/{e}/{m}.pck".format(e=e, m=method)
        plot_2d_error_grid_file(
            fn, criterion, title="{m} {e}".format(m=method, e=e),
            **kwargs)


def plot_2d_error_grid_file(fn, criterion, **kwargs):
    with open(fn) as f:
        d = pickle.load(f)
    d.update(kwargs)
    plot_2d_error_grid(criterion=criterion, **d)


def plot_2d_error_grid(criterion, res, param_names, params, criteria, maxerr=5,
                       title="", pn1=None, pn2=None, settings={}, **kwargs):
    if pn1 is None and pn2 is None:
        pn1 = param_names[0]
        pn2 = param_names[1]
    erri = criteria.index(criterion)
    ferr = res[..., erri].copy()
    ferr[ferr > maxerr] = np.nan
    i = [slice(
        None) if (i == pn1 or i == pn2) else settings[i] for i in param_names]
    ferr = ferr[i]
    if param_names.index(pn1) < param_names.index(pn2):
        ferr = ferr.T
    plt.figure(figsize=(12, 10))
    plt.imshow(ferr, interpolation="nearest", cmap="hot", norm=LogNorm(
        vmin=np.nanmin(ferr), vmax=np.nanmax(ferr)))
    p1 = kwargs[pn1]
    p2 = kwargs[pn2]
    plt.yticks(range(len(p2)), p2)
    plt.xticks(range(len(p1)), p1, rotation=45, ha="right")
    plt.xlabel(pn1)
    plt.ylabel(pn2)
    plt.title(title)
    plt.colorbar()


def run_experiment(task, methods, n_indep, l, error_every, name, n_eps,
                   mdp, phi, title, verbose=1, n_jobs=1, criteria=None,
                   episodic=False, eval_on_traces=False, n_samples_eval=None, **kwargs):
    a, b, c = task.avg_error_traces(methods, n_indep=n_indep, n_eps=n_eps,
                                    n_samples=l, error_every=error_every,
                                    criteria=criteria, eval_on_traces=eval_on_traces,
                                    n_samples_eval=n_samples_eval,
                                    verbose=verbose, n_jobs=n_jobs, episodic=episodic)
    return a, b, c


def plot_path(path, method_id, methods, criterion, title):
    plt.figure(figsize=(15, 10))
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
                  criteria, n_eps, episodic=False, ncol=1, figsize=(15,10), **kwargs):
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
    lss = ["-", "--", "-."]
    for i, m in enumerate(methods):
        if hasattr(m, "hide") and m.hide:
            continue
        ls = lss[int(i / 7)]
        plt.errorbar(x, mean[i, k, :], yerr=std[i, k, :],
                     errorevery=ee, label=m.name, ls=ls)
    plt.legend(ncol=ncol)
    plt.show()
