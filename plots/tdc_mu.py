from experiments import *
from plots import *

plt.ioff()
f = plot_2d_error_grid_experiment("swingup_gauss_onpolicy", "TDCLambda", "RMSPBE", pn1="alpha", pn2="mu",
                                   settings={"lam": 0}, maxerr=50, figsize=(8,4))
plt.xlabel(r"Stepsize $\alpha$")
plt.ylabel(r"$\mu = \beta / \alpha$")
plt.title("")
save_figure("tdc_mu_swingup", fig=f)
f = plot_2d_error_grid_experiment("link20_imp_offpolicy", "TDCLambda", "RMSPBE", pn1="alpha", pn2="mu",
                                   settings={"lam": 0}, maxerr=50, figsize=(8,4))
plt.title("")
plt.xlabel(r"Stepsize $\alpha$")
plt.ylabel(r"$\mu = \beta / \alpha$")
save_figure("tdc_mu_link20", fig=f)
f = plot_2d_error_grid_experiment("lqr_imp_offpolicy", "TDCLambda", "RMSPBE", pn1="alpha", pn2="mu",
                                   settings={"lam": 0}, maxerr=5, figsize=(8,4))
plt.title("")
plt.xlabel(r"Stepsize $\alpha$")
plt.ylabel(r"$\mu = \beta / \alpha$")
save_figure("tdc_mu_lqr", fig=f)
f = plot_2d_error_grid_experiment("baird", "TDCLambda", "RMSPBE", pn1="alpha", pn2="mu",
                                   settings={"lam": 0}, maxerr=5, figsize=(8,4))
plt.title("")
plt.xlabel(r"Stepsize $\alpha$")
plt.ylabel(r"$\mu = \beta / \alpha$")
save_figure("tdc_mu_baird", fig=f)
