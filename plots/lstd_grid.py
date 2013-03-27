from experiments import *
from plots import *
import td
from matplotlib.ticker import FuncFormatter, NullFormatter, MultipleLocator
__builtins__["exp_name"] = "lqr_full_offpolicy"
import experiments.gridsearch as gs
lambdas = np.linspace(0,1,21)
eps = np.power(10, np.linspace(-3,6,21))
njobs = -1
gs.gridsearch(td.RecursiveLSTDLambdaJP, gs_name="fine",eps=eps, lam=lambdas, batchsize=10, njobs=njobs)

"""
__builtins__["exp_name"] = "disc_random_on"
import experiments.gridsearch as gs
lambdas = np.linspace(0,1,21)
eps = np.power(10, np.linspace(-3,6,21))
njobs = -1
gs.gridsearch(td.RecursiveLSTDLambda, gs_name="fine",eps=eps, lam=lambdas, batchsize=10, njobs=njobs)
"""

"""
#plt.ioff()
f = plot_2d_error_grid_experiment("lqr_full_onpolicy", "LinearTDLambda", "RMSE", pn1="alpha", pn2="lam",
                                   settings={"lam": 0}, maxerr=50, figsize=(8,4))
#plt.title("")
#save_figure("tdc_mu_swingup", fig=f)
"""
plt.ioff()
nf = NullFormatter()
lf = FuncFormatter(lambda x,p: str(x)+" "+str(p))
cmap = "hot"
def tr(x):
    a = np.log(x)
    return a - np.nanmin(a)+ .1
f, p1, p2 = plot_2d_error_grid_file("data/lqr_full_offpolicy/RecursiveLSTDLambdaJP_fine.pck", "RMSE", pn1="lam",
                            pn2="eps", cmap=cmap, ticks=False, transform=tr,
                                   settings={"lam": 0}, maxerr=5, figsize=(4,4))
plt.xlabel(r"$\lambda$")
plt.xticks(range(len(p1))[::4], p1[::4])
plt.gca().yaxis.set_visible(False)
save_figure("lstd_grid_lqr_full_off", fig=f)

f,p1, p2 = plot_2d_error_grid_file("data/disc_random_on/RecursiveLSTDLambda_fine.pck", "RMSE",
                                  pn1="lam", pn2="eps", cmap=cmap, ticks=False, transform=tr,
                                  settings={"lam": 0}, maxerr=5, figsize=(4,4))
plt.xlabel(r"$\lambda$")
plt.xticks(range(len(p1))[::4], p1[::4])
plt.gca().yaxis.set_visible(False)
save_figure("lstd_grid_disc_random_on", fig=f)

f, p1, p2 = plot_2d_error_grid_file("data/lqr_imp_offpolicy/RecursiveLSTDLambdaJP_fine.pck",
                                  "RMSE", pn1="lam", pn2="eps", cmap=cmap, ticks=False,
                                    transform=tr,
                                   settings={"lam": 0}, maxerr=5, figsize=(4,4))
#plt.title("")
plt.xlabel(r"$\lambda$")
plt.ylabel(r"$\log(\epsilon)$")
plt.xticks(range(len(p1))[::4], p1[::4])
plt.yticks(range(len(p2))[::3], [str(np.log10(a)) for a in p2][::3])
save_figure("lstd_grid_lqr_imp_off", fig=f)
