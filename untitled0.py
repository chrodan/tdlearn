import td
import examples
import numpy as np
import matplotlib.pyplot as plt
import dynamic_prog as dp
import util
from task import LinearLQRValuePredictionTask

gamma=0.9
sigma = np.zeros((4,4))
sigma[-1,-1] = 0.01
    
dt = 0.1    
#mdp = examples.MiniLQMDP(dt=dt)
mdp = examples.PoleBalancingMDP(sigma=sigma, dt=dt)

phi = mdp.full_tri_phi


n_feat = len(phi(np.zeros(mdp.dim_S)))
theta_p,_ = dp.solve_LQR(mdp, gamma=gamma)
print theta_p
theta_p = np.array(theta_p).flatten()
policy = mdp.linear_policy(theta_p)

theta0 =  10*np.ones(n_feat)
#theta0 =  0.*np.ones(n_feat)

task = LinearLQRValuePredictionTask(mdp, gamma, phi, theta0, policy=policy, normalize_phi=True)   
task.seed=0
#phi = task.phi
print "V_true", task.V_true
print "theta_true"
theta_true = task.V_true
theta_true = theta_true[np.triu_indices_from(theta_true)] * np.std(util.apply_rowise(phi,task.mu), axis=0)

methods = []

task.theta0 = theta_true

#for alpha in [0.01, 0.005]:
#    for mu in [0.05, 0.1, 0.2, 0.01]:
#alpha = 0.1
alpha = 0.005
mu = 0.1
gtd = td.GTD(alpha=alpha, beta=mu*alpha, phi=phi)
gtd.name = r"GTD $\alpha$={} $\mu$={}".format(alpha, mu)
gtd.color = "r"
methods.append(gtd)

#for alpha in [.005,0.01,0.02]:
#    for mu in [0.01, 0.1]:
alpha, mu = 0.01, 0.1
gtd = td.GTD2(alpha=alpha, beta=mu*alpha, phi=phi)
gtd.name = r"GTD2 $\alpha$={} $\mu$={}".format(alpha, mu)
gtd.color = "orange"
methods.append(gtd)


#for alpha in [0.005, 0.01, 0.02, 0.03, 0.04]:
alpha = .01
td0 = td.LinearTD0(alpha=alpha, phi=phi, gamma=gamma)
td0.name = r"TD(0) $\alpha$={}".format(alpha)    
td0.color = "k"
methods.append(td0)

#for alpha in [0.005, 0.01, 0.02]:
#    for mu in [0.01, 0.1]:
for alpha, mu in [(.01,0.1)]:        
    tdc = td.TDC(alpha=alpha, beta=alpha*mu, phi=phi, gamma=gamma)
    tdc.name = r"TDC $\alpha$={} $\mu$={}".format(alpha, mu)   
    tdc.color = "b"        
    methods.append(tdc)

#methods = []
#for eps in np.power(10,np.arange(-1,4)):
eps=100
lstd = td.LSTDLambda(lam=0, eps=eps, phi=phi, gamma=gamma)
lstd.name = r"LSTD({}) $\epsilon$={}".format(0, eps)    
lstd.color = "g"        
methods.append(lstd)
#
#methods = []    
#for alpha in [0.01, 0.02, 0.03]:
#alpha = .2
alpha=.02   
rg = td.ResidualGradient(alpha=alpha, phi=phi, gamma=gamma)
rg.name = r"RG $\alpha$={}".format(alpha)    
rg.color = "brown"
methods.append(rg)    
    
l=10000
error_every=100

mean, std, raw = task.ergodic_error_traces(methods, n_samples=l, error_every=error_every,
                                       criterion="RMSBE")

plt.figure(figsize=(18,12))
plt.ylabel(r"$\sqrt{MSPBE}$")
plt.xlabel("Timesteps")  

for i, m in enumerate(methods):
    plt.errorbar(range(0,l,error_every), mean[i,:], yerr=std[i,:], errorevery=10000/error_every, label=m.name)
plt.legend()
plt.ylim(0.,1.)