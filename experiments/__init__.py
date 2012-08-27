__author__ = 'dann'
import pickle
import numpy as np
import os 
import matplotlib.pyplot as plt
def run_experiment(task, methods, n_indep, l, error_every, name, n_eps,
                   mdp, phi, title, criterion="RMSPBE", verbose=True, n_jobs=1, **kwargs):
    return task.avg_error_traces(methods, n_indep=n_indep, n_eps=n_eps,
        n_samples=l, error_every=error_every,
        criterion=criterion,
        verbose=verbose, n_jobs=n_jobs)

def save_results(name, l, criterion, error_every, n_indep, n_eps, methods,
                 mdp, phi, title, mean, std, raw, **kwargs):
    if not os.path.exists("data/{name}".format(name=name)):
        os.makedirs("data/{name}".format(name=name))

    with open("data/{name}/setting.pck".format(name=name), "w") as f:
        pickle.dump(dict(l=l, criterion=criterion,
                         error_every=error_every, 
                         n_indep=n_indep,
                         n_eps=n_eps,
                         methods=methods, 
                         mdp=mdp, phi=phi, title=title, name=name),f)

    np.savez_compressed("data/{name}/results.npz".format(name=name), mean=mean, std=std, raw=raw)
    
    
def load_results(name):
    with open("data/{name}/setting.pck".format(name=name), "r") as f:
        d = pickle.load(f)
        
    d2 = np.load("data/{name}/results.npz".format(name=name))
    d.update(d2)
    return d
    
def plot_errorbar(title, methods, mean, std, l, error_every, criterion, n_eps, **kwargs):
    plt.figure(figsize=(15,10))
    plt.ylabel(criterion)
    plt.xlabel("Timesteps")
    plt.title(title)
    for i, m in enumerate(methods):
        plt.errorbar(range(0,l*n_eps,error_every), mean[i,:], yerr=std[i,:], errorevery=l*n_eps/error_every/8, label=m.name)
        #plt.errorbar(range(0,l,error_every), mean[i,:], yerr=std[i,:], label=m.name)
    plt.legend()
    plt.show()