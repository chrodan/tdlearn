from plots import *
from experiments import load_results
from matplotlib.ticker import MultipleLocator, FuncFormatter

d = load_results("lqr_imp_onpolicy")
l = [2,3,4]
marker = ["o", "v", "D", "*"]
names = [r"TD $\searrow$", r"TD $\rightarrow$", r"TDC $\rightarrow$"]
for i,m in enumerate(d["methods"]):
    if i not in l:
        m.hide=True
    else:
        m.ls = "-"
        m.name = names[0]
        m.marker = marker[0]
        marker = marker[1:]
        names = names[1:]
        if i == 4:
            m.ls = "--"

plt.ioff()
d["title"] = None
f = plot_errorbar(criterion="RMSPBE", ncol=2, figsize=(6,4), **d)
major_formatter = FuncFormatter(lambda x,pos:  str(int(x/1000))+"k" if x >= 1000 else str(x))
plt.gca().xaxis.set_major_formatter(major_formatter)
plt.ylim(0,.4)
plt.xlim(0,15000)
save_figure("td_rm", fig=f)

marker = ["o", "v", "D", "*"]
names = [r"TD $\searrow$", r"TD $\rightarrow$", r"TDC $\rightarrow$"]
d = load_results("disc_random_on")
l = [2,3,4]
for i,m in enumerate(d["methods"]):
    if i not in l:
        m.hide=True
    else:
        m.ls = "-"
        m.name = names[0]
        m.marker = marker[0]
        marker = marker[1:]
        names = names[1:]
        if i == 4:
            m.ls = "--"

plt.ioff()
d["title"] = None
f = plot_errorbar(criterion="RMSPBE", ncol=2, figsize=(6,4), **d)
major_formatter = FuncFormatter(lambda x,pos:  str(int(x/1000))+"k" if x >= 1000 else str(x))
plt.gca().xaxis.set_major_formatter(major_formatter)
plt.ylim(0,.6)
save_figure("td_rm2", fig=f)
