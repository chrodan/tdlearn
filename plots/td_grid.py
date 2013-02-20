from experiments import *
from plots import *
import td

"""
__builtins__["exp_name"] = "lqr_imp_offpolicy"
import experiments.gridsearch as gs
lambdas = np.linspace(0,1,21)
alphas = list(np.arange(0.0001, 0.001, 0.0001)) + list(np.arange(0.001, .01, 0.001)) +[0.01]
njobs = -1
gs.gridsearch(td.LinearTDLambda, gs_name="fine",alpha=alphas, lam=lambdas, batchsize=10, njobs=njobs)
"""
"""
__builtins__["exp_name"] = "disc_random_on"
import experiments.gridsearch as gs
lambdas = np.linspace(0,1,21)
alphas = list(np.arange(0.0001, 0.001, 0.0001)) + \
        list(np.arange(0.001, .01, 0.001)) +list(np.arange(0.01, 0.1, 0.01))
njobs = -1
gs.gridsearch(td.LinearTDLambda, gs_name="fine",alpha=alphas, lam=lambdas, batchsize=10, njobs=njobs)
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
plt.ioff()
f = plot_2d_error_grid_file("data/lqr_imp_offpolicy/LinearTDLambda_fine.pck",
                            "RMSE", pn1="lam", pn2="alpha", cmap="spectral",
                                   settings={"lam": 0}, maxerr=5, figsize=(8,8))

#plt.title("")
save_figure("td_grid_lqr_imp_offpolicy", fig=f)
f = plot_2d_error_grid_file("data/boyan/LinearTDLambda_fine.pck", "RMSE", pn1="lam",
                            pn2="alpha", cmap="spectral",
                            settings={"lam": 0}, maxerr=10, figsize=(8,8))
#plt.title("")
save_figure("td_grid_boyan", fig=f)
f = plot_2d_error_grid_file("data/disc_random_on/LinearTDLambda_fine.pck",
                            "RMSE", pn1="lam", pn2="alpha", cmap="spectral",
                                   settings={"lam": 0}, maxerr=5, figsize=(8,8))
#plt.title("")
save_figure("td_grid_disc_random_on", fig=f)
f = plot_2d_error_grid_file("data/lqr_full_onpolicy/LinearTDLambda_fine.pck", "RMSE",
                            pn1="lam", pn2="alpha",
                                   settings={"lam": 0}, cmap="spectral", maxerr=5, figsize=(8,8))
#plt.title("")
save_figure("td_grid_lqr_full_onpolicy", fig=f)
