from plots import *
from experiments import load_results

marker = ["o", "v", "D", "*", "1"]
d = load_results("disc_random_on")
l = [0,3,4,6,9]
names = ["GTD", "TD", "TDC", "LSTD", "RG"]
for i,m in enumerate(d["methods"]):
    if i not in l:
        m.hide=True
    else:
        m.ls = "-"
        m.name = names[0]
        names = names[1:]
        m.marker = marker[0]
        marker = marker[1:]

d["title"] = "400 State Random MDP On-Policy"
plt.ioff()
f = plot_errorbar(criterion="RMSPBE", figsize=(6,4), ncol=3, kformatter=True, **d)
#plt.ylim(0,2)
save_figure("error_diff_MSPBE", fig=f)
f = plot_errorbar(criterion="RMSE", figsize=(6,4),kformatter=True,ncol=3, **d)
plt.ylim(0,14)
save_figure("error_diff_MSE", fig=f)
