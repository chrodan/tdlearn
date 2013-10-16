from plots import *
from experiments import load_results

d = load_results("lqr_imp_offpolicy")
l = [1,2,4,8]
marker = ["o", "v", "D", "*"]
names = [r"GTD2", r"TD $\searrow$", "TDC", "LSTD"]
for i,m in enumerate(d["methods"]):
    if i not in l:
        m.hide=True
    else:
        m.marker = marker[0]
        marker = marker[1:]
        m.ls = "-"
        m.name = names[0]
        names = names[1:]

plt.ioff()
d["title"] = None
a=0.75
f = plot_errorbar(criterion="RMSE", ncol=2, figsize=(a*7,a*4.5),
                  kformatter=True, **d)
plt.ylim(2,4)
plt.xlim(0, 30000)
#plt.xlim(0,15e3)
save_figure("lqr_mse", fig=f)
