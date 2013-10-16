from plots import *
from experiments import load_results

d = load_results("disc_random_off")
l = [11,12,13,14]
for i,m in enumerate(d["methods"]):
    if i not in l:
        m.hide=True
    else:
        m.ls = "-"

    if i == 13:
        m.name="BRM DS"
        m.nobar=True
        m.marker = "o"
    elif i == 11:
        m.name="RG DS"
        m.marker = "v"
    elif i == 12:
        m.name="RG"
        m.marker = "D"
    elif i == 14:
        m.name="BRM"
        m.marker = "*"
d["title"] = None
a = 0.9
plt.ioff()
f = plot_errorbar(criterion="RMSBE", ncol=2,figsize=(6 * a,4 * a), kformatter=True, **d)
plt.title("")
plt.ylim(0,1)
save_figure("ds_off", fig=f)
