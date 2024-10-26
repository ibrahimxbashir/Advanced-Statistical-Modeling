import numpy as np


def leg_kernel(u, d):
    '''
    Legendre kernel of order d estimated at the values u.
    u: A numpy array with shape (n,)
    d: an integer that equals 1 or 2
    return: A numpy array with shape (n,)
    '''
    k = np.full(u.shape, 1/2) if d==1 else (3 / 8) * (3 - 5*u**2) # This is the reduced/simplified versions of K1(u) and K2(u)
    return np.where(np.abs(u) <= 1, k, 0) # The sum of phi's (which is the Kernel of order d) * Indicator_{|u|<=1}
    pass


def KDE(x, X, h, d):
    '''
    the kernel density estimator at point x for the sample X, bandwidth h, and the Legendre kernel of order d.
    x: float
    X: A numpy array with shape (n,)
    h: float
    d: an integer that equals 1 or 2
    return: float
    '''
    return np.mean(leg_kernel((x - X)/h, d))/h  # The mean of K_d(u)/h (we don't divide n since that is done automatically with the mean function)
    pass


def CV(h, X, d):
    '''
    Cross-Validation criteria at bandwidth h for the sample X and the Legendre kernel of order d
    h: float
    X: A numpy array with shape (n,)
    d: an integer that equals 1 or 2
    return: float
    '''
    def q(xi, xj, hi, di):  # Function for p_n(x)^2
        u = np.abs(xi - xj)/hi # Define |u| = |X_i - X_j|/h
        return (2 - u)/4 if di==1 else (9/64)*((-5/6)*u**5 + 10*u** 3 - (40/3)*u**2 - 4*u + 8) # q(u)

    n = len(X) # Using broadcasting, (X[:,None] - X gives n by n matrix which subtracts each pair of elements of X) to implement function
    p2 = np.sum(np.where(np.abs(X[:, None] - X) <= 2*h, q(X[:,None], X, h, d), 0))/(h*n**2) # Estimate p^2_i = sum(q(u))/(h*n**2) where |Xi - Xj| <= 2h
    Tn = np.sum(np.where((np.eye(n)==0), leg_kernel((X[:,None] - X)/h, d), 0))/(n*(n - 1)*h) # Compute p_n\i = sum(K((Xj - x)/h), since Kernel already computes for |u|<=1, we don't need to include that requirement
    # We use np.eye(n)==0 for condition, because np.eye(n) creates an identity matrix of size n, and all values that are 0 are the non-diagonal (distinct pairs of X, i.e. i=/=j)
    return p2 - 2*Tn
    pass

