from plots import *
from experiments import load_results

marker = ["o", "v", "D", "*", "1"]
d = load_results("swingup_gauss_offpolicy")
l = [7,6,8,9]
names = ["LSPE", "LSPE-TO", "LSTD-TO", "LSTD"]
for i in l:
    m = d["methods"][i]
    m.ls = "-"
    m.name = names[0]
    names = names[1:]
    m.nobar = True
    m.marker = marker[0]
    marker = marker[1:]
    if m.name == "LSTD":
        m.nobar=True
    elif m.name == "LSTD-TO":
        m.ls = "--"
plt.ioff()
d["title"] = None
f = plot_errorbar(criterion="RMSPBE", ncol=2, figsize=(6,4), order=l, **d)
plt.gca().set_yscale('log')
#plt.ylim(0,3)
save_figure("to", fig=f)

marker = ["o", "v", "D", "*", "1"]
d = load_results("lqr_imp_offpolicy")
l = [6,7,8,9]
names = ["LSPE", "LSPE-TO", "LSTD-TO", "LSTD"]
for i in l:
    m = d["methods"][i]
    m.ls = "-"
    m.name = names[0]
    names = names[1:]
    m.marker = marker[0]
    marker = marker[1:]
    if m.name == "LSTD":
        m.nobar=True
    elif m.name == "LSTD-TO":
        m.ls = "--"
d["title"] = None
f = plot_errorbar(criterion="RMSPBE", ncol=2, figsize=(6,4), kformatter=True, order=l, **d)
plt.ylim(0,1)
save_figure("to_lqr", fig=f)
