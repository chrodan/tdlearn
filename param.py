import numpy as np

def twiddle(evaluation, p_start, dp=None, domain=None, tol = 0.2, ):
    if dp is None:
        dp = [1.] * len(p_start)
    if domain is None:
        domain = [lambda x: x] * len(p_start)
    p = p_start
    def p_d(p):
        return [domain[i](e) for i,e in enumerate(p)]
    best_err = evaluation(*p_d(p))
    while (np.sum(dp) > tol):
        for i in range(len(p)):
            k = p[i]
            p[i] = p[i] + dp[i]
            err = evaluation(*p_d(p))
            if err < best_err:
                best_err = err
                dp[i] *= 1.1
            else:
                p[i] = k - dp[i]
                err = evaluation(*p_d(p))
                if err < best_err:
                    best_err = err
                    dp[i] *= 1.1
                else:
                    p[i] = k
                    dp[i] *= 0.9
    
    return p

#def coordinate_descent(task, method, start, criterion="RMSPBE", step=None, domain=None):
    
