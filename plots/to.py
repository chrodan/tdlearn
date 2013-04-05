from plots import *
from experiments import load_results

d = load_results("swingup_gauss_offpolicy")
l = [6,7,8,9]
names = ["LSPE-TO", "LSPE", "LSTD-TO", "LSTD"]
for i,m in enumerate(d["methods"]):
    if i not in l:
        m.hide=True
    else:
        m.ls = "-"
        m.name = names[0]
        names = names[1:]
        m.nobar = True
        if m.name == "LSTD":
            m.nobar=True
        elif m.name == "LSTD-TO":
            m.ls = "--"
plt.ioff()
d["title"] = None
f = plot_errorbar(criterion="RMSPBE", ncol=2, figsize=(6,4), **d)
plt.gca().set_yscale('log')
#plt.ylim(0,3)
save_figure("to", fig=f)

d = load_results("lqr_imp_offpolicy")
l = [6,7,8,9]
names = ["LSPE", "LSPE-TO", "LSTD-TO", "LSTD"]
for i,m in enumerate(d["methods"]):
    if i not in l:
        m.hide=True
    else:
        m.ls = "-"
        m.name = names[0]
        names = names[1:]
        if m.name == "LSTD":
            m.nobar=True
        elif m.name == "LSTD-TO":
            m.ls = "--"

plt.ioff()
d["title"] = None
f = plot_errorbar(criterion="RMSPBE", ncol=2, figsize=(6,4), **d)
plt.ylim(0,1)
save_figure("to_lqr", fig=f)
