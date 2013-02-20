from experiments import *
from plots import *
import td
__builtins__["exp_name"] = "lqr_imp_offpolicy"
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
__builtins__["exp_name"] = "lqr_full_onpolicy"
import experiments.gridsearch as gs
lambdas = np.linspace(0,1,21)
alphas = list(np.arange(0.0001, 0.001, 0.0001)) + \
        list(np.arange(0.001, .015, 0.001))
njobs = -1
gs.gridsearch(td.LinearTDLambda, gs_name="fine",alpha=alphas, lam=lambdas, batchsize=20, njobs=njobs)
"""
"""
__builtins__["exp_name"] = "boyan"
import experiments.gridsearch as gs
lambdas = np.linspace(0,1,21)
alphas = list(np.arange(0.001, 0.01, 0.001)) + \
        list(np.arange(0.01, .1, 0.01)) + \
        list(np.arange(0.1, 1., 0.1))
njobs = -1
gs.gridsearch(td.LinearTDLambda, gs_name="fine",alpha=alphas, lam=lambdas, batchsize=20, njobs=njobs)
"""
"""
#plt.ioff()
f = plot_2d_error_grid_experiment("lqr_full_onpolicy", "LinearTDLambda", "RMSE", pn1="alpha", pn2="lam",
                                   settings={"lam": 0}, maxerr=50, figsize=(8,4))
#plt.title("")
#save_figure("tdc_mu_swingup", fig=f)
f = plot_2d_error_grid_experiment("boyan", "LinearTDLambda", "RMSE", pn1="alpha", pn2="lam",
                                   settings={"lam": 0}, maxerr=50, figsize=(8,4))
#plt.title("")
#save_figure("tdc_mu_link20", fig=f)
"""
f = plot_2d_error_grid_experiment("data/disc_random_on/RecursiveLSTDLambda_fine.pck", "RMSE",
                                  pn1="lam", pn2="eps", cmap="hot",
                                  settings={"lam": 0}, maxerr=5, figsize=(8,8))
#plt.title("")
#save_figure("tdc_mu_lqr", fig=f)
f = plot_2d_error_grid_experiment("data/lqr_imp_offpolicy/RecursiveLSTDLambdaJP_fine.pck",
                                  "RMSE", pn1="lam", pn2="eps", cmap="hot",
                                   settings={"lam": 0}, maxerr=5, figsize=(8,8))
#plt.title("")
#save_figure("tdc_mu_baird", fig=f)
"""
