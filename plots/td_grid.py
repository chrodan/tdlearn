from experiments import *
from plots import *
import td
__builtins__["exp_name"] = "lqr_imp_offpolicy"

import experiments.gridsearch as gs
lambdas = np.linspace(0,1,21)
alphas = list(np.arange(0.0001, 0.001, 0.0001)) + list(np.arange(0.001, .01, 0.001)) +[0.01]
njobs = -1
gs.gridsearch(td.LinearTDLambda, gs_name="fine",alpha=alphas, lam=lambdas, batchsize=10, njobs=njobs)
#plt.ioff()
f = plot_2d_error_grid_experiment("lqr_full_onpolicy", "LinearTDLambda", "RMSE", pn1="alpha", pn2="lam",
                                   settings={"lam": 0}, maxerr=50, figsize=(8,4))
#plt.title("")
#save_figure("tdc_mu_swingup", fig=f)
f = plot_2d_error_grid_experiment("boyan", "LinearTDLambda", "RMSE", pn1="alpha", pn2="lam",
                                   settings={"lam": 0}, maxerr=50, figsize=(8,4))
#plt.title("")
#save_figure("tdc_mu_link20", fig=f)
f = plot_2d_error_grid_experiment("disc_random_on", "LinearTDLambda", "RMSE", pn1="alpha", pn2="lam",
                                   settings={"lam": 0}, maxerr=5, figsize=(8,4))
#plt.title("")
#save_figure("tdc_mu_lqr", fig=f)
f = plot_2d_error_grid_experiment("lqr_imp_offpolicy", "LinearTDLambda", "RMSE", pn1="alpha", pn2="lam",
                                   settings={"lam": 0}, maxerr=5, figsize=(8,4))
#plt.title("")
#save_figure("tdc_mu_baird", fig=f)
