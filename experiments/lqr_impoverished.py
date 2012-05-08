'''
Experiment that compares TD approaches to estimate
the Value function of a pole balancing problem. 
Created on 25.04.2012

@author: Christoph Dann <cdann@cdann.de>
'''
import td
import examples
import numpy as np
import matplotlib.pyplot as plt
import dynamic_prog as dp
from task import LinearLQRValuePredictionTask

gamma=0.9
sigma = np.zeros((4,4))
sigma[-1,-1] = 0.01
    
dt = 0.1    
#mdp = examples.MiniLQMDP(dt=dt)
mdp = examples.PoleBalancingMDP(sigma=sigma, dt=dt)

phi = mdp.impoverished_phi


n_feat = len(phi(np.zeros(mdp.dim_S)))
theta_p,_ = dp.solve_LQR(mdp, gamma=gamma)
print theta_p
theta_p = np.array(theta_p).flatten()
policy = mdp.linear_policy(theta_p)

theta0 =  1000*np.ones(n_feat)
#theta0 =  0.*np.ones(n_feat)

task = LinearLQRValuePredictionTask(mdp, gamma, phi, theta0, policy=policy)    

print "V_true", task.V_true

methods = []

for alpha in [1.0]:
    for mu in [0.05]:
#alpha = 0.1
#alpha = 0.7
#mu = 0.005
        gtd = td.GTD(alpha=alpha, beta=mu*alpha, phi=phi)
        gtd.name = r"GTD $\alpha$={} $\mu$={}".format(alpha, mu)
        gtd.color = "#6E086D"
        methods.append(gtd)

#for alpha in [0.5,0.9, 1]:
#    for mu in [0.5, 0.3]:
alpha, mu = 0.9, 0.05
gtd = td.GTD2(alpha=alpha, beta=mu*alpha, phi=phi)
gtd.name = r"GTD2 $\alpha$={} $\mu$={}".format(alpha, mu)
gtd.color = "#6E086D"
methods.append(gtd)


#for alpha in [0.7, 0.9, 1.0]:
alpha = 1.0
td0 = td.LinearTD0(alpha=alpha, phi=phi, gamma=gamma)
td0.name = r"TD(0) $\alpha$={}".format(alpha)    
td0.color = "k"
methods.append(td0)

#for alpha in [0.7, 0.9]:
#    for mu in [0.01, 0.1, 0.05]:
for alpha, mu in[( 1.0, 0.1)]:        
    tdc = td.TDC(alpha=alpha, beta=alpha*mu, phi=phi, gamma=gamma)
    tdc.name = r"TDC $\alpha$={} $\mu$={}".format(alpha, mu)   
    tdc.color = "r"        
    methods.append(tdc)

#methods = []
#for eps in np.power(10,np.arange(-1,4)):
eps=100
lstd = td.LSTDLambda(lam=0, eps=eps, phi=phi, gamma=gamma)
lstd.name = r"LSTD({}) $\epsilon$={}".format(0, eps)    
lstd.color = "b"        
methods.append(lstd)
#
#methods = []    
#for alpha in [0.7, 0.9, 1.0]:
#alpha = .2
alpha=1.    
rg = td.ResidualGradient(alpha=alpha, phi=phi, gamma=gamma)
rg.name = r"RG $\alpha$={}".format(alpha)    
rg.color = "brown"
methods.append(rg)    
    

l=60000
error_every=500
mean, std, raw = task.avg_error_traces(methods, n_indep=1, 
                                       n_samples=l, error_every=error_every,
                                       criterion="RMSPBE",
                                       verbose=True)

plt.figure()
plt.ylabel(r"$\sqrt{MSPBE}$")
plt.xlabel("Timesteps")  

for i, m in enumerate(methods):
    plt.errorbar(range(0,l,error_every), mean[i,:], yerr=std[i,:], errorevery=500, label=m.name)
plt.legend()

plt.yscale("log")
plt.show()