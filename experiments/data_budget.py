import td
from joblib import Parallel
import examples
import numpy as np
import regtd
#import matplotlib.pyplot as plt
import features
import policies
from task import LinearContinuousValuePredictionTask
import util
gamma = 0.95
dt = 0.1


mdp = examples.PendulumSwingUpCartPole(
    dt=dt, Sigma=np.zeros(4), start_amp=2.)  # np.array([0., 0.005, 0.005, 0.]))
policy = policies.MarcsPolicy(noise=np.array([.05]))


states,_,_,_,_ = mdp.samples_cached(n_iter=200, n_restarts=30,
                                policy=policy,seed=8000)

n_slices = [3, 5, 7,10]
n_slices2 = [5, 5, 14,20]
bounds = [[0, 35], [-3, 4], [-12, 12], [-3, 3]]
means, sigmas = features.make_grid(n_slices, bounds)
means2, sigmas2 = features.make_grid(n_slices2, bounds)
means = np.vstack([means,means2])
sigmas = np.vstack([sigmas, sigmas2])

phi = features.gaussians(means, sigmas, constant=False)
A = util.apply_rowise(arr=states, f=phi)
a = np.nonzero(np.sum(A > 0.05, axis=0) > 5)[0]
phi = features.gaussians(means[a], sigmas[a], constant=True)
print phi.dim, "features are used"
theta0 = 0. * np.ones(phi.dim)

task = LinearContinuousValuePredictionTask(
    mdp, gamma, phi, theta0, policy=policy,
    normalize_phi=False, mu_seed=1100,
    mu_subsample=1, mu_iter=200,
    mu_restarts=150, mu_next=300)

lam = 0.0
alpha = 0.3
mu = .1
tdc = td.TDCLambda(alpha=alpha, mu=mu, lam=lam, phi=phi)
tdc.name = r"TDC({}) $\alpha$={} $\mu$={}".format(lam, alpha, mu)

lam = 0.0
alpha = td.RMalpha(5., 0.3)
beta = td.RMalpha(.5, 0.3)
tdcrm = td.TDCLambda(alpha=alpha, beta=beta, lam=lam, phi=phi)
tdcrm.name = r"TDC({}) $\alpha$={} $\mu$={}".format(lam, alpha, mu)


lam = 0.
eps = 10
lstd = td.RecursiveLSTDLambda(lam=lam, eps=eps, phi=phi)
lstd.name = r"LSTD({}) $\epsilon$={}".format(lam, eps)

l = 200
n_eps = 300  # 1000
error_every = 600  # 4000
name = "swingup_data_budget"
title = "Cartpole Swingup Onpolicy"
n_indep = 10
episodic = False
criterion = "RMSPBE"
criteria = ["RMSPBE"]
eval_on_traces=False
n_samples_eval=10000
verbose=1
gs_ignore_first_n = 10000
gs_max_weight = 3.
max_t=400.
min_diff=1.
t = np.arange(min_diff, max_t, min_diff)
e = np.ones((len(t),3)) * np.nan

def  run(s):
    e = np.ones((len(t),3)) * np.nan
    lstd.time=0.
    tdc.time = 0.
    tdcrm.time = 0.
    e_,t_ = task.error_traces_cpu_time(lstd, max_passes=1, max_t=max_t, min_diff=min_diff,
                                        n_samples=l, n_eps=n_eps, verbose=0, seed=s,
                                        criteria=criteria)
    e[:len(e_),0] = e_
    e_,t_ = task.error_traces_cpu_time(tdc, max_passes=None, max_t=max_t, min_diff=min_diff,
                                        n_samples=l, n_eps=n_eps, verbose=0, seed=s,
                                        criteria=criteria)
    e[:len(e_),1] = e_
    e_,t_ = task.error_traces_cpu_time(tdcrm, max_passes=None, max_t=max_t, min_diff=min_diff,
                                        n_samples=l, n_eps=n_eps, verbose=0, seed=s,
                                        criteria=criteria)
    e[:len(e_),2] = e_
    return e

if __name__ == "__main__":
    from experiments import *
    import matplotlib.pyplot as plt
    #task.fill_trajectory_cache(seeds=range(n_indep), n_eps=n_eps, n_samples=l)
    task.mu
    fn = "data/data_budget.npz"
    if os.path.exists(fn):
        d = np.load(fn)
        globals().update(d)
    else:
        n_jobs = 10
        jobs = []
        for s in range(n_indep):
            jobs.append((run, [s], {}))
        res = Parallel(n_jobs=n_jobs, verbose=verbose)(jobs)
        res = np.array(res)
        np.savez(fn, res=res, title="Cartpole Swingup with {} features".format(phi.dim))
    plt.figure()
    for i in range(n_indep):
        o = res[i,0,0]
        for k in range(len(t)):
            if np.isnan(res[i,k,0]):
                res[i,k,0] = o
            o = res[i,k,0]
    r = np.nansum(res,axis=0) / np.sum(np.isfinite(res), axis=0)
    u = (res - r)**2
    std = np.sqrt(np.nansum(u,axis=0) / np.sum(np.isfinite(u), axis=0))
    plt.errorbar(t, r[:,0], yerr=std[:,0], errorevery=20, color="red", label="LSTD")
    plt.errorbar(t, r[:,1], yerr=std[:,1], errorevery=20, color="blue", label="TDC const.")
    plt.errorbar(t, r[:,2], yerr=std[:,2], errorevery=20, color="green", label="TDC decr.")
    plt.ylabel("RMSPBE")
    plt.xlabel("Runtime in s")
    plt.title(title)
    plt.legend()
    #    plt.plot(t["tdc"][s],np.vstack(e["tdc"][s])[:,0], "o", color="blue")
    #    plt.plot(t["tdcrm"][s],np.vstack(e["tdcrm"][s])[:,0], ".", color="green")

