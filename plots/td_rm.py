from plots import *
from experiments import load_results

d = load_results("lqr_imp_offpolicy")
l = [2,3,4]
names = [r"TD $\searrow$", "TD", "TDC"]
for i,m in enumerate(d["methods"]):
    if i not in l:
        m.hide=True
    else:
        m.ls = "-"
        m.name = names[0]
        names = names[1:]
        if m.name == "TDC":
            m.ls = "--"

plt.ioff()
d["title"] = None
f = plot_errorbar(criterion="RMSPBE", ncol=3, figsize=(6,4), **d)
plt.ylim(0,.5)
save_figure("td_rm", fig=f)

d = load_results("link20_imp_onpolicy")
l = [2,3,4]
names = [r"TD $\searrow$", "TD", "TDC"]
for i,m in enumerate(d["methods"]):
    if i not in l:
        m.hide=True
    else:
        m.ls = "-"
        m.name = names[0]
        names = names[1:]
        if m.name == "TDC":
            m.ls = "--"

plt.ioff()
d["title"] = None
f = plot_errorbar(criterion="RMSPBE", ncol=3, figsize=(6,4), **d)
plt.ylim(0,.5)
save_figure("td_rm2", fig=f)
