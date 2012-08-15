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
sigma = np.zeros(4)
sigma[-1] = 0.01

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
#task.theta0 = theta_true
methods = []

#for alpha in [0.01, 0.005]:
#    for mu in [0.05, 0.1, 0.2, 0.01]:
#alpha = 0.1
alpha = 0.01
mu = 0.5 #optimal
gtd = td.GTD(alpha=alpha, beta=mu*alpha, phi=phi)
gtd.name = r"GTD $\alpha$={} $\mu$={}".format(alpha, mu)
gtd.color = "r"
methods.append(gtd)

#for alpha in [.005,0.01,0.02]:
#    for mu in [0.01, 0.1]:
alpha, mu = 0.01, 0.5 #optimal
gtd = td.GTD2(alpha=alpha, beta=mu*alpha, phi=phi)
gtd.name = r"GTD2 $\alpha$={} $\mu$={}".format(alpha, mu)
gtd.color = "orange"
methods.append(gtd)


alpha = .004
td0 = td.LinearTD0(alpha=alpha, phi=phi, gamma=gamma)
td0.name = r"TD(0) $\alpha$={}".format(alpha)
td0.color = "m"
methods.append(td0)

#for alpha in [0.005, 0.01, 0.02]:
#    for mu in [0.01, 0.1]:
for alpha, mu in [(.008,0.001)]: #optimal
    tdc = td.TDC(alpha=alpha, beta=alpha*mu, phi=phi, gamma=gamma)
    tdc.name = r"TDC $\alpha$={} $\mu$={}".format(alpha, mu)
    tdc.color = "b"
    methods.append(tdc)

for alpha, mu in [(.01,0.001)]: #optimal
    tdc = td.GeriTDC(alpha=alpha, beta=alpha*mu, phi=phi, gamma=gamma)
    tdc.name = r"GeriTDC $\alpha$={} $\mu$={}".format(alpha, mu)
    tdc.color = "c"
    methods.append(tdc)

lstd = td.LSTDLambda(lam=0, phi=phi, gamma=gamma)
lstd.name = r"LSTD({})".format(0)
lstd.color = "g"
methods.append(lstd)
#
#methods = []
#for alpha in [0.01, 0.02, 0.03]:
#alpha = .2
alpha=.04
rg = td.ResidualGradient(alpha=alpha, phi=phi, gamma=gamma)
rg.name = r"RG $\alpha$={}".format(alpha)
rg.color = "brown"
methods.append(rg)


lstd = td.LSTDLambdaJP(lam=0, phi=phi, gamma=gamma)
lstd.name = r"LSTD-JP({})".format(0)
lstd.color = "k"
methods.append(lstd)

l=20000
error_every=100
name="cartpole_off"

if __name__ == "__main__":
    mean, std, raw = task.avg_error_traces(methods, n_indep=4,
        n_samples=l, error_every=error_every,
        criterion="RMSPBE",
        verbose=20, n_jobs=1)

    #plt.figure(figsize=(18,12))
    plt.title("Cart-Pole Off-Policy")
    plt.ylabel(r"$\sqrt{MSPBE}$")
    plt.xlabel("Timesteps")

    for i, m in enumerate(methods):
        #plt.errorbar(range(0,l,error_every), mean[i,:], yerr=std[i,:], errorevery=10000/error_every, label=m.name)
        plt.errorbar(range(0,l,error_every), mean[i,:], yerr=std[i,:], errorevery=l/error_every/10, color=m.color, label=m.name)
        #plt.errorbar(range(0,l,error_every), mean[i,:], yerr=std[i,:], color=m.color, label=m.name)
    
    plt.legend()
    #plt.yscale("log")
    plt.show()

