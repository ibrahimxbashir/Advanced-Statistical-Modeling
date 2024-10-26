import numpy as np
import scipy as sp
from scipy.special import kolmogi


def asymptotic_conf(X, xi, alpha=0.05):
    '''
    Computes asymptotic confidence band based on Kolmogorov-Smornov statistic.
    X: A numpy array with shape (n,); samples
    xi: A numpy array with shape (r,);  evaluation points
    alpha: Scalar; miscoverage probability alpha
    return: Tuple of 2 numpy arrays, each of shape (r,); the lower, upper envelopes of confidence band at each evaluation point
    '''
    Xn, n = np.sort(X), len(X)
    Fn, crit = np.arange(1, n + 1, dtype=float) / n, kolmogi(alpha) / np.sqrt(n)
    lower = np.maximum(Fn - crit, np.array([-crit for i in range(n)]))
    upper = np.minimum(Fn + crit, np.array([1 + crit for i in range(n)]))
    stepL = sp.interpolate.interp1d(Xn, lower, kind='previous', bounds_error=False, fill_value=(-crit, 1 - crit))
    stepU = sp.interpolate.interp1d(Xn, upper, kind='previous', bounds_error=False, fill_value=(crit, 1 + crit))
    L, U = stepL(xi), stepU(xi)
    return L, U
    pass


def exact_conf(X, xi, alpha=0.05):
    '''
    Computes the exact confidence band.
    X: A numpy array with shape (n,); samples
    xi: A numpy array with shape (r,);  evaluation points
    alpha: Scalar; miscoverage probability alpha
    return: Tuple of 2 numpy arrays, each of shape (r,); the lower, upper envelopes of confidence band at each evaluation point
    '''
    Xn, n = np.sort(X), len(X)
    Fn, crit = np.arange(1, n + 1, dtype=float) / n, np.sqrt(np.log(2 / alpha) / 2) / np.sqrt(n)
    lower = np.maximum(Fn - crit, np.array([-crit for i in range(n)]))
    upper = np.minimum(Fn + crit, np.array([1 + crit for i in range(n)]))
    stepL = sp.interpolate.interp1d(Xn, lower, kind='previous', bounds_error=False, fill_value=(-crit, 1 - crit))
    stepU = sp.interpolate.interp1d(Xn, upper, kind='previous', bounds_error=False, fill_value=(crit, 1 + crit))
    L, U = stepL(xi), stepU(xi)
    return L, U
    pass
    

def coverage(X, cdf, band, alpha=0.05):
    '''
    A function that computes band coverage.
    X: A numpy array with shape (n,); samples
    cdf: A numpy array with shape (n,) with values ordered corresponding to X: cdf[i] is the true CDF value of X[i]
    band: A string that can get 2 values: 'asymptotic' or 'exact'
    alpha: Scalar; miscoverage probability alpha
    return: A scalar that equals to 0 or 1 if successful
    '''
    n = len(X)
    lower, upper = exact_conf(X, X, alpha=alpha) if band == "exact" else asymptotic_conf(X, X, alpha=alpha)
    inside = np.sum((lower <= cdf) & (cdf <= upper))
    return 1 if (inside / n) >= 1 - alpha else 0
    pass

