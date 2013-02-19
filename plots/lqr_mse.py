from plots import *
from experiments import load_results

d = load_results("lqr_imp_offpolicy")
l = [1,2,4,8]
names = [r"GTD2", r"TD $\searrow$", "TDC", "LSTD"]
for i,m in enumerate(d["methods"]):
    if i not in l:
        m.hide=True
    else:
        m.ls = "-"
        m.name = names[0]
        names = names[1:]

plt.ioff()
d["title"] = None
f = plot_errorbar(criterion="RMSE", ncol=2, figsize=(7,4.5), **d)
plt.ylim(2,4)
#plt.xlim(0,15e3)
save_figure("lqr_mse", fig=f)
