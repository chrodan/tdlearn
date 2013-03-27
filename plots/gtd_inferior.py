from experiments import *
import td
#exp_list = filter(lambda x: x != "baird", exp_list)
tab = np.zeros((len(exp_list), 3))
names = []
print r"& GTD & GTD2 & TDC & TD\\"
for j,exp in enumerate(exp_list):
    e = load_results(exp, update_title=False)
    for i,m in enumerate(e["methods"]):
        if isinstance(m, td.GTD):
            a = i
        elif isinstance(m, td.GTD2):
            b = i
        elif isinstance(m, td.TDCLambda):
            c = i
        elif isinstance(m, td.LinearTDLambda) and not isinstance(m.alpha, td.RMalpha):
            d = i
    indices = [a,b,c,d]
    k = e["criteria"].index("RMSPBE")
    f = [np.sum(e["mean"][i, k, :]) for i in indices]
    if np.all(np.array(f) < .1):
        l = ["{:.2g}".format(a) for a in f]
    else:
        l = ["{:.2f}".format(a) for a in f]
    i = np.argmin(np.array(f))
    l[i] = r"\bf{"+l[i]+"}"
    print e["title"], "&", " & ".join(l), r"\\"
    #print e["title"], "&", "&".join(["{:.2f}".format(np.sum(e["mean"][i,k,:])) for i in [a,b,c,d]]), r"\\"
