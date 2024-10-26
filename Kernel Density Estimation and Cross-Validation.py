import numpy as np


def kernel(u, d):
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
    return np.mean(kernel((x - X)/h, d))/h  # The mean of K_d(u)/h (we don't divide n since that is done automatically with the mean function)
    pass


def CV_criterion(h, X, d):
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
    Tn = np.sum(np.where((np.eye(n)==0), kernel((X[:,None] - X)/h, d), 0))/(n*(n - 1)*h) # Compute p_n\i = sum(K((Xj - x)/h), since Kernel already computes for |u|<=1, we don't need to include that requirement
    # We use np.eye(n)==0 for condition, because np.eye(n) creates an identity matrix of size n, and all values that are 0 are the non-diagonal (distinct pairs of X, i.e. i=/=j)
    return p2 - 2*Tn
    pass


import matplotlib.pyplot as plt
from scipy.stats import norm

if __name__ == '__main__':

    # TODO: YOUR CODE HERE
    X1, X2 = norm.rvs(size=100), norm.rvs(size=1000)  # Generate samples for n=100 and n=1000
    h = np.linspace(0.01, 3, 50)  # Generate 50 evenly spaced points for h to optimize through
    x = np.linspace(min(min(X1), min(X2)), max(max(X1), max(X2)),
                    100)  # Generate 100 evenly spaced points along x-axis within ranges of X1,X2
    pdf = norm.pdf(x, 0, 1)  # The pdf of N(0,1) along points in x


    def plotter(axes, X, d, cols):
        for i in range(len(cols)):  # Calculate KDE at each point xi given sample X, order d, and plot
            Xi, di = X[i // 2], d[i % 2]  # Define variable for sample and order of iteration
            min_index, min_value = min(((i, CV_criterion(hi, Xi, di)) for i, hi in enumerate(h)), key=lambda x: x[
                1])  # Get the index of the min value of CV(h) where hi in H, and then index H[i] to get optimal bandwidths
            kerns = np.array(
                [KDE(xi, Xi, h[min_index], di) for xi in x])  # Get KDE of Xi at x, of order di with optimal bandwidth
            axes.plot(x, kerns, lw=2, label=f'X{i // 2 + 1}, K{di}',
                      color=cols[i])  # Plot KDE with appropriate label and color


    fig, ax = plt.subplots(figsize=(10, 7))  # Plot data
    ax.hist(X1, bins='auto', density=True, label='X1, n=100', color='yellow', alpha=0.3)  # Histogram of samples
    ax.hist(X2, bins='auto', density=True, label='X2, n=1000', color='orchid', alpha=0.3)
    ax.plot(x, pdf, lw=2, label='true pdf', color='blue')  # True pdf of N(0,1)
    plotter(ax, [X1, X2], [1, 2],
            ['gold', 'goldenrod', 'plum', 'mediumorchid'])  # Plot each of the KDE's with their optimal bandwidth
    plt.title("KDE of N(0,1) using Legendre polynomials of order {1,2}")
    plt.legend()
    plt.show()
    pass
