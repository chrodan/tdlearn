from experiments.boyan import *
import matplotlib.pyplot as plt
from plots import *
plt.ioff()
x = np.linspace(0,13,131)
y = np.vstack([phi(xt) for xt in x])
plt.figure(figsize=(3.5,2.5))
for i in range(4):
    plt.plot(x+1,y[:,i])
#for i in range(4):
#    plt.plot(x[::10]+1, y[::10, i], "o")
plt.vlines(x[::10]+1, [0], np.max(y[::10], axis=1))
plt.xlim(1,14)
plt.ylim(0,1.5)
plt.xlabel("State")
plt.ylabel("Feature Activation")
save_figure("boyan_phi")
