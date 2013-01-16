import td
import util
try:
    import cvxopt as co
    import cvxopt.solvers as solv
    import gurobipy as grb
except Exception,e:
    print e
    co = None
    solv = None
    grb = None
import numpy as np
import copy
import sklearn.linear_model as lm
from numpy.random import RandomState


class DLSTD(td.LSTDLambdaJP):
    """
        Dantzig-LSTD
        regularized LSTD approach based on Dantzig Selectors.

        A Danzuig Selector Approach to Temporal Difference Learning
        Geist M., Scherrer B., ... (ICML 2012)
    """

    def __init__(self, tau, nonreg_ids, **kwargs):
        td.LSTDLambdaJP.__init__(self, **kwargs)
        self.tau = tau
        self.nonreg_ids = nonreg_ids
        #solv.options['featol'] = 1e-4

    def reset(self):
        td.LSTDLambdaJP.reset(self)
        n = len(self.init_vals["theta"])

        self.G = co.matrix(np.zeros((4 * n, 2 * n)))
        self.G[:n, :n] = -np.eye(n)
        self.G[:n, n:] = -np.eye(n)
        self.G[n:2 * n, :n] = -np.eye(n)
        self.G[n:2 * n, n:] = np.eye(n)

        self.c = co.matrix(0., (2 * n, 1))
        self.c[:n] += 1.

        self.h = co.matrix(0., (4 * n, 1))

    @property
    def theta(self):
        self._tic()
        A = -(self.C1 + self.C2)
        n = A.shape[0]
        m = grb.Model("DLSTD")
        u = []
        thetas = []
        for i in range(n):
            u.append(m.addVar(obj=1., name="u_{}".format(i), lb=0.))
            thetas.append(m.addVar(obj=0., name="theta_{}".format(i), lb=-grb.GRB.INFINITY))
        m.update()
        for i in range(n):
            if i not in self.nonreg_ids:
                m.addConstr(thetas[i]  <= u[i], "c_pos_{}".format(i))
                m.addConstr(thetas[i]  >= -u[i], "c_neg_{}".format(i))
            e = grb.quicksum([thetas[j]*A[i,j] for j in range(n)])
            m.addConstr(e - self.b[i] <= self.tau, "tau_pos{}".format(i))
            m.addConstr(e - self.b[i] >= -self.tau, "tau_neg{}".format(i))
        m.update()
        m.setParam( 'OutputFlag', False )
        m.optimize()
        #print m.status
        res = np.array([v.x for v in thetas])



        #self.G[2 * n:3 * n, n:] = +A
        #self.G[3 * n:, n:] = -A
        #self.h[2 * n:3 * n] = -self.b + self.tau
        #self.h[3 * n:] = self.b + self.tau

        #import ipdb
        #ipdb.set_trace()
        ##print "G Rank", np.array(self.G).shape, np.linalg.matrix_rank(np.array(self.G))
        #res = solv.lp(self.c, self.G, self.h)  # , solver="glpk")
        self._toc()
        #if res['status'] != "optimal":
            #import ipdb
            #ipdb.set_trace()
        #    pass
        #return np.array(res["x"][n:]).flatten()
        return res

    def regularization_path(self):
        taus = np.logspace(-10, 0, num=20)
        A = self.C1 + self.C2
        n = A.shape[0]
        self.G[2 * n:3 * n, n:] = +A
        self.G[3 * n:, n:] = -A
        res = []
        curres = None
        for tau in taus:
            self.h[2 * n:3 * n] = -self.b + tau
            self.h[3 * n:] = self.b + tau

            if curres is not None and curres["status"] == "optimal":
                curres = solv.lp(self.c, self.G, self.h, primalstart=curres)
            else:
                curres = solv.lp(self.c, self.G, self.h)
            res.append((tau, np.array(curres["x"][n:]).flatten()))
        return res


class LSTDl21(td.LinearValueFunctionPredictor, td.OffPolicyValueFunctionPredictor):
    """
        LSTD-l21
        regularized LSTD approach adding an l2-penalty on the bellman problem
        and an l1 penalty on the fixpoint problem.

        Regularized Least Squares Temporal Difference learning with nested
        l2 and l1 penalization
        Hoffman M., et al.
    """

    def __init__(self, beta, tau=1e-6, lars=False, **kwargs):
        self.lars = lars
        self.beta = beta
        td.LinearValueFunctionPredictor.__init__(self, **kwargs)
        td.OffPolicyValueFunctionPredictor.__init__(self, **kwargs)
        self.tau = tau
        self.reset()

    def reset(self):
        self.reset_trace()
        n = len(self.init_vals["theta"]) - 1
        for k, v in self.init_vals.items():
            if k == "theta":
                continue
            self.__setattr__(k, copy.copy(v))
        self.t = 0
        interc = hasattr(self.phi, "intercept")
        if not self.lars:
            self.lasso = lm.Lasso(
                alpha=self.tau, warm_start=True, fit_intercept=interc,
                normalize=False, max_iter=3000)
        else:
            self.lasso = lm.LassoLars(alpha=self.tau, fit_intercept=interc,
                                      normalize=False)
        self.Phi = util.GrowingMat(capacity=(n, n), shape=(0, n))
        self.Phit = util.GrowingMat(capacity=(n, n), shape=(0, n))
        self.R = util.GrowingVector(capacity=n, size=0)

    def update_V(self, s0, s1, r, f0=None, f1=None, theta=None, rho=1, **kwargs):
        """
            rho: weight for this sample in case of off-policy learning
        """
        if f0 is None or f1 is None:
            f0 = self.phi(s0)
            f1 = self.phi(s1)
        if hasattr(self.phi, "intercept"):
            f0 = f0[1:]
            f1 = f1[1:]

        self._tic()
        self.t += 1
        self.Phi.expand(rows=rho * f0[None, :])
        self.Phit.expand(rows=rho * f1[None, :])
        self.R.expand(rows=rho * r)
        self.oldrho = rho
        self._toc()

    def regularization_path(self):

        n = len(self.C1)
        Sigma = np.linalg.pinv(- self.C1 + self.beta * np.eye(n))
        A = np.dot(self.Phi.view, np.eye(n) + np.dot(Sigma, -self.C2))
        b = np.dot(self.Phi.view, np.dot(Sigma, self.b))
        alphas, _, coefs = lm.lars_path(A, b, eps=1e-6)
        #lst = [(model.alpha, model.coef_) for model in models]
        return zip(alphas, coefs.T)

    @property
    def theta(self):
        self._tic()
        #n = len(self.Phi.view)
        # normalization
        phi_m = self.Phi.view.mean(axis=0)
        phi_stddev = self.Phi.view.std(axis=0)
        phi_stddev[phi_stddev==0] = 1.
        omega = 1. / phi_stddev
        Phi_norm = (self.Phi.view - phi_m) * omega
        n = Phi_norm.shape[1]
        Phit_cent = (self.Phit.view - phi_m)
        R_m = self.R.view.mean()
        R_cent = self.R.view - R_m
        # Sigma
        P = np.dot(Phi_norm.T, Phi_norm) + self.beta * np.eye(n)
        P[np.isnan(P)] = 0.
        P[np.isposinf(P)] = 1000.
        P[np.isneginf(P)] = -1000.
        try:
            Sigma = np.linalg.pinv(P)
        except np.linalg.LinAlgError, e:
            print e
            return np.ones(n) * 1000
        Sigma = np.dot(np.dot(Phi_norm, Sigma), Phi_norm.T)

        A = self.Phi.view - self.gamma * np.dot(Sigma, Phit_cent) \
            - self.gamma * phi_m
        b = np.dot(Sigma, R_cent) + R_m
        self.lasso.fit(A, b)
        theta = np.zeros(n + 1)
        theta[0] = self.lasso.intercept_ / (1 - self.gamma)
        theta[1:] = self.lasso.coef_.flatten()

        self._toc()
        return theta


class LSTDl1(td.LSTDLambdaJP):
    """
        LSTD-l1
        regularized LSTD approach adding an l1 penalty on the A * theta - b residuals.

        A Danzuig Selector Approach to Temporal Difference Learning
        Geist M., Scherrer B., ... (ICML 2012)
    """

    def __init__(self, lars=False, **kwargs):
        self.lars = lars
        td.LSTDLambdaJP.__init__(self, **kwargs)

    def reset(self):

        td.LSTDLambdaJP.reset(self)
        interc = hasattr(self.phi, "intercept")
        if not self.lars:
            self.lasso = lm.Lasso(
                alpha=self.tau, warm_start=True, fit_intercept=interc,
                normalize=False, max_iter=50000)
        else:
            self.lasso = lm.LassoLars(alpha=self.tau, fit_intercept=interc,
                                      normalize=False)

    def regularization_path(self):
        A = -(self.C1 + self.C2)
        b = self.b
        alphas, _, coefs = lm.lars_path(A, b, eps=1e-6)
        #lst = [(model.alpha, model.coef_) for model in models]
        return zip(alphas, coefs.T)
        #models = lm.lars_path(A, b, fit_intercept=False, eps=1e-7, normalize=False)
        #lst = [(model.alpha, model.coef_) for model in models]
        #return lst

    @property
    def theta(self):
        self._tic()

        interc = hasattr(self.phi, "constant") and self.phi.constant == True
        A = -(self.C1 + self.C2)
        b = self.b
        if interc:
            A = A[:-1,:-1]
            b = b[:-1]
            s = 1
        else:
            s = 0
        self.lasso.fit(A,b)
        theta = np.zeros_like(self.b)
        if interc:
            theta[-1] = self.lasso.intercept_
            theta[:-1] = self.lasso.coef_.flatten()
        else:
            theta = self.lasso.coef_.flatten()
        self._toc()
        return theta


def _min_plus(vals):
    vals[vals <= 0] = np.inf
    i = np.nanargmin(vals)
    assert(vals[i] >= 0.)
    return vals[i], i


class LarsTD(td.LSTDLambdaJP):
    """
        l1-regularized LSTD. The theoretical formulation is known
        as Lasso-TD
        LarsTD is an algorithm to find the Lasso-TD fix-point.

        Implementation based on Figure 1 in
            Regularized Least-Squares Temporal Difference Learning
            Kolter Z., Ng A.

    """

    def __init__(self, tau, **kwargs):
        td.LSTDLambdaJP.__init__(self, **kwargs)
        self.tau = tau

    @property
    def theta(self):
        self._tic()
        res = self.lars_path(self.tau)
        theta = res[-1][1]
        self._toc()
        return theta

    def regularization_path(self):
        return self.lars_path()

    def lars_path(self, tau=0.):
        res = []

        A = -(self.C1 + self.C2)
        n = A.shape[0]
        theta = np.zeros(n)
        c = self.b.copy()
        i = np.argmax(np.abs(c))
        I = set([i])

        beta = c[i]
        while beta > tau + 1e-15:
            Il = list(I)
            Il.sort()
            Ilc = list(set(xrange(n)) - I)
            # Find direction
            try:
                dw = np.dot(np.linalg.pinv(A[Il][:, Il]), np.sign(c[Il]))
            except np.linalg.LinAlgError, e:
                print e
                return res
            # Find step size for adding an element

            d = np.dot(A[:, Il], dw).flatten()
            if len(Ilc) == 0:
                alpha1 = np.inf
                i1 = -1
            else:
                t = (c[Ilc] - beta) / (d[Ilc] - 1)
                alpha1, i1 = _min_plus(t)
                i1 = Ilc[i1]

                t = (c[Ilc] + beta) / (d[Ilc] + 1)
                alpha2, i2 = _min_plus(t)
                if alpha2 < alpha1:
                    alpha1 = alpha2
                    i1 = Ilc[i2]

            # Find step size fo reach zero coefficient
            t = - (theta[Il] / dw)
            alpha2, i2 = _min_plus(t)
            i2 = Il[i2]

            # Upates variables
            alpha = np.nanmin([alpha1, alpha2, beta - tau])
            assert(alpha > 0)

            theta[Il] += alpha * dw
            beta -= alpha
            c -= alpha * d

            # Update active set
            if alpha1 < alpha2:
                I.add(i1)
            else:
                I.remove(i2)

            # Sanity check (debug purpose)
            #diff = self.b - np.dot(A, theta)
            #assert(np.all(np.abs(diff) <= beta + 1e-2))
            res.append((beta, theta.copy()))
        res.append((beta, theta.copy()))
        return res


class LSTDRP(td.LSTDLambdaJP):
    """
        LSTD-l1
        regularized LSTD approach adding an l1 penalty on the A * theta - b residuals.

        A Danzuig Selector Approach to Temporal Difference Learning
        Geist M., Scherrer B., ... (ICML 2012)
    """

    def __init__(self, dim_lower, seed=None, **kwargs):
        self.dim_lower = dim_lower
        self.seed = seed
        td.LSTDLambdaJP.__init__(self, **kwargs)

    def reset(self):
        td.LSTDLambdaJP.reset(self)
        if self.seed is not None:
            self.prng = RandomState(self.seed)
        else:
            self.prng = np.random

    @property
    def theta(self):
        try:
            self._tic()
            D = self.C1.shape[0]
            n = self.t
            if self.dim_lower < 1:
                dim_lower = np.sqrt(n)
                dim_lower = np.maximum(1, dim_lower)
                dim_lower = int(dim_lower) + 1
            else:
                dim_lower = self.dim_lower
            proj = self.prng.normal(scale=1. / np.sqrt(dim_lower),
                                    size=(dim_lower, D))
            # for debugging, sanity check!
            #dim_lower = 6
            #proj = np.eye(D)[:dim_lower,:]
            A = np.dot(proj, np.dot(-self.C1 - self.C2, proj.T))
            b = np.dot(proj, self.b)
            #Phi = np.dot(self.Phi.finalized, proj)
            #Psi = np.dot(self.Psi.finalized, proj)
            #b = np.dot(Phi.T, self.R.finalized.flatten()).flatten()
            #A = np.dot(Phi.T, Psi)

            theta_t = np.dot(np.linalg.pinv(A), b)
            return np.dot(theta_t, proj).flatten()
        except np.linalg.LinAlgError, e:
            print e
            return np.zeros_like(self.b)
        finally:
            self._toc()

    def regularization_path(self):
        dim_lower = np.linspace(1, self.C1.shape[0], 30)

        res = []
        old_dim = self.dim_lower
        for n in dim_lower:
            self.dim_lower = n

            res.append((n, self.theta))
        self.dim_lower = old_dim
        return res
