import matplotlib.pyplot as plt
import os.path
import subprocess
import sys

from matplotlib.ticker import MultipleLocator, FuncFormatter
folder="/home/christoph/Dropbox/ChristophDann/tdpaper/img/"
criteria_label = {"RMSBE": r"$\sqrt{\operatorname{MSBE}}$",
                  "RMSE": r"$\sqrt{\operatorname{MSE}}$",
                  "RMSPBE": r"$\sqrt{\operatorname{MSPBE}}$"}
def plot_errorbar(title, methods, mean, std, l, error_every, criterion,
                  criteria, n_eps, episodic=False, ncol=1, figsize=(8,6),
                  order=None, kformatter=False, **kwargs):
    f = plt.figure(figsize=figsize)
    if criterion in criteria_label:
        crit_label = criteria_label[criterion]
    else:
        crit_label = criterion
    plt.ylabel(crit_label)
    plt.xlabel("Timesteps")
    if title is not None:
        plt.title(title)

    k = criteria.index(criterion)
    x = range(0, l * n_eps, error_every) if not episodic else range(n_eps)
    if episodic:
        ee = int(n_eps / 8.)
    else:
        ee = int(l * n_eps / error_every / 8.)
    if ee < 1:
        ee = 1
    lss = ["-", "--", "-."]
    ind_list = order if order is not None else range(len(methods))
    for i in ind_list:
        m = methods[i]
        if hasattr(m, "hide") and m.hide:
            continue
        ls = m.ls if hasattr(m,"ls") else "-"
        marker = getattr(m, "marker", ".")

        if hasattr(m, "nobar") and m.nobar:
            plt.plot(x, mean[i,k,:], label=m.name, marker=marker, ls=ls,
                     markevery=ee)
        else:
            plt.errorbar(x, mean[i, k, :], yerr=std[i, k, :],
                         errorevery=ee, markevery=ee, label=m.name,
                         ls=ls, marker=marker)
    if kformatter:
        major_formatter = FuncFormatter(lambda x,pos:  str(int(x/1000))+"k" if x >= 1000 else str(int(x)))
        plt.gca().xaxis.set_major_formatter(major_formatter)
    plt.legend(ncol=ncol)
    return f

def save_figure(fname, fig=None, folder=folder, crop=True, dpi=None):
    if fig is None:
        fig = plt.gcf()
    epsfn = os.path.join(folder,fname+".eps")
    pdffn = os.path.join(folder,fname+".pdf")
    fig.savefig(epsfn, format="eps", dpi=dpi, bbox_inches="tight")
    subprocess.call(["epstopdf --outfile={} {}".format(pdffn, epsfn)], shell=True, cwd=folder)

    if crop:
        a = ["pdfcrop {}".format(pdffn)]
        subprocess.call(a, shell=True, cwd=folder)
