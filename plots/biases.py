from experiments import *
import td
exp_list = filter(lambda x: x != "baird" and x != "disc_random_on" and x != "disc_random_off", exp_list)
exp_list += ["disc_random_on_conv", "disc_random_off_conv"]
tab = np.zeros((len(exp_list), 3))
names = []
print r"& MSTDE & MSBE & MSPBE \\"
for j,exp in enumerate(exp_list):
    d = load_results(exp)
    c = -10000
    a = -10000
    b = -10000
    for i,m in enumerate(d["methods"]):
        if type(m) is td.RecursiveBRM or type(m) is td.BRM:
            a = i
        if type(m) is td.RecursiveBRMDS or type(m) is td.BRMDS:
            b = i
        elif type(m) is td.RecursiveLSTDLambdaJP or (c < 0 and
                                    type(m) is td.RecursiveLSTDLambda):
            c = i

    k = d["criteria"].index("RMSE")
    l = [d["mean"][i,k,-1] for i in [a,b,c]]
    i = np.argmin(np.array(l))
    l = ["{:.2f}".format(a) for a in l]
    l[i] = r"\bf{"+l[i]+"}"
    print d["title"], "&", "&".join(l), r"\\"
