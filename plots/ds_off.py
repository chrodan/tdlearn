from plots import *
from experiments import load_results

d = load_results("disc_random")
l = [11,12,13,14]
for i,m in enumerate(d["methods"]):
    if i not in l:
        m.hide=True
    else:
        m.ls = "-"

    if i == 13:
        m.name="BRM DS"
        m.nobar=True
    elif i == 11:
        m.name="RG DS"
    elif i == 12:
        m.name="RG"
    elif i == 14:
        m.name="BRM"

plt.ioff()
f = plot_errorbar(criterion="RMSBE", figsize=(6,4), **d)
plt.ylim(0,2)
save_figure("ds_off", fig=f)
