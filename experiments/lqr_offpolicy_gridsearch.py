import td
import examples
import numpy as np
import matplotlib.pyplot as plt
import dynamic_prog as dp
import util
import features
import policies
import itertools
from task import LinearLQRValuePredictionTask
from joblib import Parallel, delayed

gamma=0.9
sigma = np.zeros((4,4))
sigma[-1,-1] = 0.01

dt = 0.1
#mdp = examples.MiniLQMDP(dt=dt)
mdp = examples.PoleBalancingMDP(sigma=sigma, dt=dt)

phi = features.squared_diag()


n_feat = len(phi(np.zeros(mdp.dim_S)))
theta_p,_,_ = dp.solve_LQR(mdp, gamma=gamma)
print theta_p
theta_p = np.array(theta_p).flatten()
theta_o = theta_p.copy()
beh_policy = policies.LinearContinuous(theta=theta_o, noise=np.eye(1)*0.01)
target_policy = policies.LinearContinuous(theta=theta_p, noise=np.eye(1)*0.001)

#theta0 =  10*np.ones(n_feat)
theta0 =  0.*np.ones(n_feat)

task = LinearLQRValuePredictionTask(mdp, gamma, phi, theta0, policy=beh_policy, target_policy=target_policy,  normalize_phi=True)

#phi = task.phi
print "V_true", task.V_true
print "theta_true"
theta_true = phi.param_forward(*task.V_true)
print theta_true

l=20000
error_every=200


def run(alpha, mu):
    np.seterr(all="ignore")
    m = td.TDC(alpha=alpha, beta=mu*alpha, phi=task.phi, gamma=gamma)
    mean, std, raw = task.avg_error_traces([m], n_indep=3, n_samples=l, error_every=error_every, criterion="RMSPBE", verbose=False)
    val = np.mean(mean)
    return val

alphas = [0.0002, 0.0005] + list(np.arange(0.001, .01, 0.001)) + list(np.arange(0.01, 0.1, 0.01)) + [0.1, 0.2]
mus = [0.0001, 0.001, 0.01,0.01, 0.1, 0.5,1,2,4,8,16]
params = list(itertools.product(alphas, mus))
#params = [(0.001, 0.5)]
k = (delayed(run)(*p) for p in params)
res = Parallel(n_jobs=-1, verbose=11)(k)
import pickle
res = np.array(res).reshape(len(alphas), -1)
with open("data/offpolicy_TDC_gs.pck", "w") as f:
    pickle.dump(dict(params=params, alphas=alphas, mus=mus, res=res), f)
print zip(params, res)
