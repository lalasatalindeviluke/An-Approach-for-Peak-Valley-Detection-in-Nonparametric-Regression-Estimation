import numpy as np
import statsmodels.api as sm
import math
from scipy import stats
from patsy import bs

def rate_fun(x1, x, y, d0, n0=20, deg=1):
    '''
    x1:測試點，看要不要在此放knot
    d0:x1附近的小範圍的x
    '''
    index1 = np.where((x>(x1-d0)) & (x<x1))[0]
    if (index1.size < n0):
        return None
    
    index2 = np.where((x<(x1+d0)) & (x>x1))[0]
    if (index2.size < n0):
        return None
    
    yy = y[index1]
    xx = x[index1]
    if (deg > 1):
        for i in range(2, deg+1):
            xx = np.hstack((xx, (x[index1])**i))

    y1_lm = sm.OLS(yy, sm.add_constant(xx)).fit()
    ans1 = y1_lm.params

    yy = y[index2]
    xx = x[index2]
    if (deg > 1):
        for i in range(2, deg+1):
            xx = np.hstack((xx, (x[index2])**i))

    y2_lm = sm.OLS(yy, sm.add_constant(xx)).fit()
    ans2 = y2_lm.params

    sigma = y1_lm.cov_params() + y2_lm.cov_params()
    triangle = ans2 - ans1
    return triangle @ np.linalg.inv(sigma) @ triangle

def cl_fun(xx, score, d):
    '''
    xx想放節點的位置；score: xx對應的檢定統計量
    剔除範圍內已有節點的其他節點
    '''
    sol = xx[np.argmax(score)]
    ind = np.where(np.absolute(xx-sol)>d)[0]
    xnew = xx
    snew = score
    while (ind.size>0):
        xnew = xnew[ind]
        snew = snew[ind]
        sol1 = xnew[np.argmax(snew)]
        sol = np.hstack((sol, sol1))
        ind = np.where(abs(xnew-sol1)>d)[0]
    return sol

def rss0(xi, x, y, deg=3):
    if xi.size == 1:
        bx = bs(x, knots=[xi], degree=deg, include_intercept=True,
                lower_bound=0, upper_bound=1)
    else:
        bx = bs(x, knots=xi.tolist(), degree=deg, include_intercept=True,
                lower_bound=0, upper_bound=1)
    y_lm = sm.OLS(y, bx).fit()
    return (y_lm.resid**2).sum()

def rss1(xi, x, y, deg=3):
    if xi.size == 1:
        bx = bs(x, knots=[xi], degree=deg, include_intercept=True,
                lower_bound=0, upper_bound=1)
    bx = bs(x, knots=xi.tolist(), degree=deg, include_intercept=True,
            lower_bound=0, upper_bound=1)
    y_lm = sm.OLS(y, bx).fit()
    return y_lm.predict()

def get_knot(x, y):
    deg0 = 3
    deg1 = 1
    J = 5
    n = y.size
    evenid = np.arange(1, n, 2)
    sig = np.median(np.absolute(y[evenid] - y[evenid-1])) / (math.sqrt(2)*stats.norm.ppf(0.75))
    sol_list = []
    bic_val = []
    for j in range(J):
        m0 = 2**j
        newx = np.zeros((n, 2))
        for i in range(n):
            newx[i] = np.array([x[i], rate_fun(x[i], x, y, 1/m0, deg=deg1)])
        ind1 = (newx[:,1]>stats.chi2.ppf(0.95, df=deg1+1)*sig**2)
        sol_list.append(cl_fun(x[ind1], newx[ind1, 1], 1/m0))
        bic_val.append(n*np.log(rss0(sol_list[j], x, y, deg=deg0)/n) + sol_list[j].size*np.log(n))
    sol = sol_list[bic_val.index(min(bic_val))]
    return sol
