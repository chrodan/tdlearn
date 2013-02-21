from experiments import *
from plots import *
import td
"""
__builtins__["exp_name"] = "lqr_full_onpolicy"
import experiments.gridsearch as gs
lambdas = np.linspace(0,1,21)
eps = np.power(10, np.linspace(-3,6,21))
njobs = -1
gs.gridsearch(td.RecursiveLSTDLambdaJP, gs_name="fine",eps=eps, lam=lambdas, batchsize=10, njobs=njobs)
"""
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
f = plot_2d_error_grid_file("data/lqr_full_onpolicy/RecursiveLSTDLambdaJP_fine.pck", "RMSE", pn1="lam",
                            pn2="eps", cmap="spectral",
                                   settings={"lam": 0}, maxerr=5, figsize=(8,8))
#plt.title("")
save_figure("lstd_grid_lqr_full_on", fig=f)
f = plot_2d_error_grid_file("data/disc_random_on/RecursiveLSTDLambda_fine.pck", "RMSE",
                                  pn1="lam", pn2="eps", cmap="spectral",
                                  settings={"lam": 0}, maxerr=5, figsize=(8,8))
#plt.title("")
save_figure("lstd_grid_disc_random_on", fig=f)
f = plot_2d_error_grid_file("data/lqr_imp_offpolicy/RecursiveLSTDLambdaJP_fine.pck",
                                  "RMSE", pn1="lam", pn2="eps", cmap="spectral",
                                   settings={"lam": 0}, maxerr=5, figsize=(8,8))
#plt.title("")
save_figure("lstd_grid_lqr_imp_off", fig=f)
