__author__ = 'Christoph Dann <cdann@cdann.de>'

import td
import examples
import numpy as np
import matplotlib.pyplot as plt
import dynamic_prog as dp
import util
import features
import policies
from task import LinearLQRValuePredictionTask

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


theta0 =  0.*np.ones(n_feat)
lstd = td.LSTDLambda(lam=0, phi=phi, gamma=gamma)
lstd.name = r"LSTD({})".format(0)
lstd.color = "g"
lstdjp = td.LSTDLambdaJP(lam=0, phi=phi, gamma=gamma)
lstdjp.name = r"LSTD-JP({})".format(0)
lstdjp.color = "k"
l=20001
error_every= 2000

res = []

for f in [1., 0.9, 0.5, 0.3, 0.1, 0.01, 0.001, 0.0001, 0.00001, 1e-7, 1e-8]:
    target_policy = policies.LinearContinuous(theta=theta_p, noise=np.eye(1)*0.01*f)
    task = LinearLQRValuePredictionTask(mdp, gamma, phi, theta0, policy=beh_policy, target_policy=target_policy,  normalize_phi=True)
    print task.kl_policy()
    mean, std, raw = task.avg_error_traces([lstd, lstdjp], n_indep=5,
        n_samples=l, error_every=error_every,
        criterion="RMSPBE",
        verbose=10, n_jobs=1)
    res.append((task.kl_policy(), mean, std))

plt.title("Cart-Pole Off-Policy")
plt.ylabel(r"$\sqrt{MSPBE}$")
plt.xlabel(r"$KL(\pi_T\|\pi_B)$")
k,me,st = zip(*res)

plt.errorbar(k, [m[0,-1] for m in me], yerr=[s[0,-1] for s in st], label="LSTD(0)")
plt.errorbar(k, [m[0,-2] for m in me], yerr=[s[0,-2] for s in st], label="LSTD(0) pre")
plt.errorbar(k, [m[-1,-1] for m in me], yerr=[s[-1,-1] for s in st], label="LSTD-JP(0)")
plt.errorbar(k, [m[-1,-2] for m in me], yerr=[s[-1,-2] for s in st], label="LSTD-JP(0) pre")
plt.legend()
#plt.yscale("log")
plt.show()

