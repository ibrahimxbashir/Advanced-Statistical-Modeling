import numpy as np
import scipy as sp
from scipy.special import kolmogi


def asymptotic_confidence(X, xi, alpha=0.05):
    '''
    A function that computes an asymptotic confidence band based on Kolmogorov-Smornov statistic.
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


def exact_confidence(X, xi, alpha=0.05):
    '''
    A function that computes the exact confidence band.
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
    
    # TODO: YOUR CODE HERE
    n = len(X)
    lower, upper = exact_confidence(X, X, alpha=alpha) if band == "exact" else asymptotic_confidence(X, X, alpha=alpha)
    inside = np.sum((lower <= cdf) & (cdf <= upper))
    return 1 if (inside / n) >= 1 - alpha else 0
    pass


import matplotlib.pyplot as plt
if __name__ == '__main__':

    # TODO: YOUR CODE HERE
    def monte_carlo(band, alpha=0.05, n=100, M=10000): # Monte Carlo Function
        freq = 0
        for i in range(M):
            points = np.random.chisquare(3, n) # The sample points
            true_cdf = sp.stats.chi2(3).cdf(points) # The points at the true cdf
            freq += coverage(points, true_cdf, band, alpha=alpha) # The coverage
        return freq

    # Everything below is the code made for the plotting
    def returner(x_values, y_values, x_list): # Function that interpolates values of a curve given some x-values (x-list), whose curve is defined on (x_values, y_values).
        y_list = []
        for x in x_list:
            if x > x_values[-1]:  # Check if x is greater than the largest value in x_values
                y_list.append(None)  # Appends None
                continue
            for i in range(len(x_values)):
                if x <= x_values[i]:
                    y_list.append(y_values[i - 1])
                    break
        return y_list

    def plotter(X, xr, alpha=0.05, dist=sp.stats.chi2(3)):
        # ECDF and True points
        Xn, n = np.sort(X), len(X)
        Fn = np.arange(1, n + 1, dtype=float) / n
        steps = sp.interpolate.interp1d(Xn, Fn, kind='previous', bounds_error=False, fill_value=(0, 1))
        step_ecdf, x_line = steps(Xn), np.linspace(Xn[0], Xn[-1], 1000)
        true_cdf = dist.cdf(x_line)

        # Bands of ECDF
        step_asym_l, step_asym_u = asymptotic_confidence(Xn, Xn, alpha)
        step_exact_l, step_exact_u = exact_confidence(Xn, Xn, alpha)

        # Bands at Xs
        step_asym_lxs, step_asym_uxs = asymptotic_confidence(Xn, xr, alpha) # make sure xr ar sorted if plotting line
        step_exact_lxs, step_exact_uxs = exact_confidence(Xn, xr, alpha)

        # Plot
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.step(Xn, step_ecdf, color='royalblue', label='ECDF')  # Step-function of ECDF
        plt.plot(x_line, true_cdf, color='seagreen', label='True CDF')  # Whole line of true CDF
        plt.fill_between(Xn, step_asym_l, step_asym_u, color='indianred', step='post', alpha=.5,
                         label='Asymptotic Band')  # Asymptotic band of ECDF
        plt.fill_between(Xn, step_exact_l, step_exact_u, color='cornflowerblue', step='post', alpha=.5,
                         label='Exact Band')  # Exact band of ECDF
        xr_true = returner(x_line, true_cdf, xr)
        plt.scatter(xr, xr_true, color='green', s=9, label='x1,..,xr at True CDF')  # Xs points on True CDF
        plt.scatter(xr+xr, np.concatenate((step_asym_lxs, step_asym_uxs), axis=0), color='indianred', s=15, label='x1,..,xr at Asym. Band')  # Xs points on Asym. Band
        plt.scatter(xr+xr, np.concatenate((step_exact_lxs, step_exact_uxs), axis=0), color='cornflowerblue', s=4, label='x1,..,xr at Exact Band')  # Xs points on Exact Band
        plt.title('Asymptotic and Exact Confidence Bands of Chi-square R.V. with 3 df,\nα = '+str(alpha)+', n = ' +str(n)+' (with x1,..,xr = '+str(xr)+')')
        plt.ylabel('Fn(x)')
        plt.legend()
        plt.show()

    def KA_compare(a=0.001, b=0.999, n=1000): # Function for plotting the values of Kolmogrov's K(1-α) and c(α)
        alphas = np.linspace(a, b, n)
        kolmogs = [kolmogi(alpha) for alpha in alphas]
        DWKs = [np.sqrt(np.log(2 / alpha) / 2) for alpha in alphas]

        plt.plot(alphas, kolmogs, color='indianred', label='K(1-α)')
        plt.plot(alphas, DWKs, color='cornflowerblue', label='c(α)')
        plt.xlabel('α')
        plt.title("Values of Kolmogrov's K(1-α) and c(α) as function of α")
        plt.legend()
        plt.show()

    print('Empirical Frequency of Monte Carlo Experiments (alpha=0.05):\nAsymptotic Coverage, n=10, M=10k: '+str(monte_carlo("asymptotic",n=10))+'\nAsymptotic Coverage, n=100, M=10k: '+str(monte_carlo("asymptotic"))+'\nExact Coverage, n=10, M=10k: '+str(monte_carlo("exact",n=10))+'\nExact Coverage, n=100, M=10k: '+str(monte_carlo("exact")))
    # print(monte_carlo("asymptotic",n=10))
    # print(monte_carlo("exact",n=10))
    # print(monte_carlo("asymptotic"))
    # print(monte_carlo("exact"))

    plotter(np.random.chisquare(3, 100), [1, 2.3, 3.6, 5, 6.4, 8.8], 0.05)
    #plotter(np.random.chisquare(3, 10), [1, 2.3, 3.6, 5], 0.05)
    plotter(np.random.chisquare(3, 100), [1, 2.3, 3.6, 5, 6.4, 8.8], 0.99)
    KA_compare()

    pass