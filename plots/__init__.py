import matplotlib.pyplot as plt
import os.path
import subprocess
import sys

folder="/home/christoph/Dropbox/ChristophDann/tdpaper/img/"

def plot_errorbar(title, methods, mean, std, l, error_every, criterion,
                  criteria, n_eps, episodic=False, ncol=1, figsize=(8,6), **kwargs):
    f = plt.figure(figsize=figsize)
    plt.ylabel(criterion)
    plt.xlabel("Timesteps")
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
    for i, m in enumerate(methods):
        if hasattr(m, "hide") and m.hide:
            continue
        ls = m.ls if hasattr(m,"ls") else "-"
        if hasattr(m, "nobar") and m.nobar:
            plt.plot(x, mean[i,k,:], label=m.name, ls=ls)
        else:
            plt.errorbar(x, mean[i, k, :], yerr=std[i, k, :],
                     errorevery=ee, label=m.name, ls=ls)
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
