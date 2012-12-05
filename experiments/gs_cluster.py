#from experiments.gridsearch import *
import argparse
import sys
import os.path
sys.path[0] = os.path.realpath(os.path.join(os.path.dirname(os.path.realpath(__file__)), os.path.pardir))
import td
import time
import sqlite3


basepath="/tmp/"

def retry(ExceptionToCheck, tries=4, delay=3, backoff=2, logger=None):
    """Retry calling the decorated function using an exponential backoff.

    http://www.saltycrane.com/blog/2009/11/trying-out-retry-decorator-python/
    original from: http://wiki.python.org/moin/PythonDecoratorLibrary#Retry

    :param ExceptionToCheck: the exception to check. may be a tuple of
        excpetions to check
    :type ExceptionToCheck: Exception or tuple
    :param tries: number of times to try (not retry) before giving up
    :type tries: int
    :param delay: initial delay between retries in seconds
    :type delay: int
    :param backoff: backoff multiplier e.g. value of 2 will double the delay
        each retry
    :type backoff: int
    :param logger: logger to use. If None, print
    :type logger: logging.Logger instance
    """
    def deco_retry(f):
        def f_retry(*args, **kwargs):
            mtries, mdelay = tries, delay
            try_one_last_time = True
            while mtries > 1:
                try:
                    return f(*args, **kwargs)
                    try_one_last_time = False
                    break
                except ExceptionToCheck, e:
                    msg = "%s, Retrying in %d seconds..." % (str(e), mdelay)
                    if logger:
                        logger.warning(msg)
                    else:
                        print msg
                    time.sleep(mdelay)
                    mtries -= 1
                    mdelay *= backoff
            if try_one_last_time:
                return f(*args, **kwargs)
            return
        return f_retry  # true decorator
    return deco_retry

def method_type(string):
    if "." in string:
        l = string.split(".")
        if l[0] == "regtd":
            return regtd.__dict__[l[1]]
        else:
            return td.__dict__[l[1]]
    return td.__dict__[string]


parser = argparse.ArgumentParser(description='Do heady grid search.')
parser.add_argument('-e', '--experiment',
                   help='which experiment to test')
parser.add_argument('--alpha', type=float, nargs="+")
parser.add_argument('--lam', type=float, nargs="+")
parser.add_argument('--mu', type=float, nargs="+")
parser.add_argument('method', type=method_type)
parser.add_argument('--id', type=int, nargs="+")
parser.add_argument('-n','--name')


args = parser.parse_args()


exec "from experiments."+args.experiment+" import *"

gs_name = args.experiment+"_"+args.method.__name__
if args.name is not None:
    gs_name+="_"+args.name

error_every = int(l * n_eps / 20)
n_indep = 3
try:
    a = episodic
except Error:
    episodic = False

def run(cls, param):
    np.seterr(all="ignore")
    m = [cls(phi=task.phi, gamma=gamma, **p) for p in param]
    mean, std, raw = task.avg_error_traces(
        [m], n_indep=n_indep, n_samples=l, n_eps=n_eps,
        error_every=error_every, episodic=episodic, verbose=False)
    weights = np.linspace(1., 2., mean.shape[1])
    val = (mean * weights).sum() / weights.sum()
    return val

@retry(Exception, tries=20)
def store_result(fn,d):
    with open(fn, "a") as f:
        f.write(repr(d)+"\n")
    print repr(d)

param = []
con = sqlite3.connect('test.db')
with:
    cur = con.cursor()
    cur.execute("""create table if not exists {tabname}
(method text, alpha real, mu real, lam real, val real, id integer)
""".format(tabname=tabname)
    con.commit()

    for i in range(len(args.id)):
        d = {}
        if args.alpha is not None:
            d["alpha"] = args.alpha[i]
        if args.mu is not None:
            d["mu"] = args.mu[i]
        if args.lam is not None:
            d["lam"] = args.lam[i]
        d["id"] = args.id[i]
        d["method"] = args.method.__name__
        c.execute("SELECT val FROM {tabname} WHERE method=? AND id=?".format(tabname=tabname),
                  d["method"], d["id"])
        a = c.fetchone()
        if a == None:
            param.append(d)
        else:
            print a

np.seterr(all="ignore")
m = args.method(phi=task.phi, gamma=gamma, **p) for p in param]
mean, std, raw = task.avg_error_traces(
    [m], n_indep=n_indep, n_samples=l, n_eps=n_eps,
    error_every=error_every, episodic=episodic, verbose=False)
weights = np.linspace(1., 2., mean.shape[2])
val = (mean * weights).sum() / weights.sum()


for i in range(len(param)):
    d["val"] = val
    store_result(os.path.join(basepath, gs_name + ".txt"), d)
