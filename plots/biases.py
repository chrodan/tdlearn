from experiments import *
import td
exp_list = filter(lambda x: x != "baird", exp_list)
tab = np.zeros((len(exp_list), 3))
names = []
print r"& MSTDE & MSBE & MSPBE \\"
for j,exp in enumerate(exp_list):
    d = load_results(exp)
    c = -1
    for i,m in enumerate(d["methods"]):
        if isinstance(m, td.RecursiveBRM) or isinstance(m, td.BRM):
            a = i
        elif isinstance(m, td.RecursiveBRMDS) or isinstance(m, td.BRMDS):
            b = i
        elif isinstance(m, td.RecursiveLSTDLambdaJP) or (c == -1 and
                                    isinstance(m, td.RecursiveLSTDLambda)):
            c = i

    k = d["criteria"].index("RMSE")
    print d["title"], "&", "&".join(["{:.2f}".format(d["mean"][i,k,-1]) for i in [a,b,c]]), r"\\"
