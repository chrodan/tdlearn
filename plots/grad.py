from plots import save_figure, plt
from experiments import *

d = load_results("lqr_full_onpolicy")
l = [0,1,2,3,4,7, 8,9]
for i,m in enumerate(d["methods"]):
    if i not in l:
        m.hide=True
    if i == 3:
        m.ls = "--"
plot_errorbar(criterion="RMSPBE", **d)
