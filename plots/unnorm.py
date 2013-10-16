from plots import *
from experiments import load_results

d = load_results("lqr_imp_onpolicy")
l = [1,4,6,9,11]
marker = ["o", "v", "D", "*", ","]
names = [r"GTD2", "TDC", "LSTD", "RG", "BRM"]
for i,m in enumerate(d["methods"]):
    if i not in l:
        m.hide=True
    else:
        m.ls = "-"
        m.name = names[0]
        names = names[1:]
        m.marker = marker[0]
        marker = marker[1:]
plt.ioff()
d["title"] = None
f = plot_errorbar(criterion="RMSPBE", ncol=3, kformatter=True, figsize=(0.8*7,0.8*4.5), **d)
plt.ylim(0,.5)
plt.xlim(0,15e3)
save_figure("norm", fig=f)

d = load_results("lqr_imp_onpolicy_unnorm")
names = [r"GTD2", "TDC", "LSTD", "RG", "BRM"]
marker = ["o", "v", "D", "*", ","]
for i,m in enumerate(d["methods"]):
    if i not in l:
        m.hide=True
    else:
        m.ls = "-"
        m.name = names[0]
        names = names[1:]
        m.marker = marker[0]
        marker = marker[1:]
plt.ioff()
d["title"] = None
f = plot_errorbar(criterion="RMSPBE", ncol=3, kformatter=True, figsize=(0.8*7,0.8*4.5), **d)
plt.xlim(0,15e3)
plt.ylim(0,.5)
save_figure("unnorm", fig=f)
