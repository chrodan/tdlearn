import matplotlib.pyplot as plt
import os.path
import subprocess
import sys

folder="/home/christoph/Dropbox/ChristophDann/tdpaper/img/"


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
