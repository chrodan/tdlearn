from experiments import *
import td
exp_list = filter(lambda x: x != "baird", exp_list)
tab = np.zeros((len(exp_list), 3))
names = []
print r"& GTD & GTD2 & TDC \\"
for j,exp in enumerate(exp_list):
    d = load_results(exp)
    for i,m in enumerate(d["methods"]):
        if isinstance(m, td.GTD):
            a = i
        elif isinstance(m, td.GTD2):
            b = i
        elif isinstance(m, td.TDCLambda):
            c = i

    k = d["criteria"].index("RMSPBE")
    print d["title"], "&", "&".join(["{:.2f}".format(np.sum(d["mean"][i,k,:])) for i in [a,b,c]]), r"\\"
