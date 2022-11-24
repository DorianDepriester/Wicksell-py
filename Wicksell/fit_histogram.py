import numpy as np
from compute_transform import from_histogram
import scipy.stats as stats
from scipy.optimize import minimize, LinearConstraint


def _complete_histogram(bin_edges, freq):
    """Retrieve the right-most frequency value, so that the integral over all the bins is 1."""
    remain = 1 - np.sum(freq * np.diff(bin_edges[:-1]))
    last_freq = remain / np.diff(bin_edges)[-1]
    freq = np.append(freq, last_freq)
    return freq


def _histogram_error(sample, bin_edges, freq):
    """Perform KS test from the transformed distribution, computed from a given histogram, against a given sample."""
    freq = _complete_histogram(bin_edges, freq)
    hist = (freq, bin_edges)
    wh = from_histogram(hist)
    h=stats.kstest(sample, wh.cdf)[0]
    return h


def fit_histogram(sample, rmin=0.0, rmax=None, bins=10):
    """
    Unfold a sample to find the underlying histogram resulting in the best goodness-of-fit KS test.

    Parameters
    ----------
    sample : iterable
        The sample one wants to unfold. It consists in a list of value
    rmin : float
        Left-most location of bin edges. Default is 0.0.
    rmax : float
        Right-most location of bin edges. Default is max(sample).
    bins : int or list or tuple
        Number of bins to use. If int, the function will be run with only this number of bins. If bins is a list, like
        (n_min, n_max), this function will run for each value ranging between n_min and n_max. The returned value will
        be that minimizing the KS test.

    Returns
    -------
    hist : tuple
       The unfolded histogram which minimizes the KS test.

    res : scipy.optimize._optimize.OptimizeResult
        Information about minimizing procedure. In particular, the KS test value is provided by res.fun.

    """
    if rmax is None:
        rmax = max(sample)
    if isinstance(bins, tuple) or isinstance(bins, list):
        ks_min = 1.0
        hist = ()
        res = ()
        for n in range(bins[0], bins[1]+1):
            hist_n, res_n = fit_histogram(sample, rmin, rmax, bins=n)
            if res_n.fun < ks_min:
                hist = hist_n
                res = res_n
                ks_min = res.fun
        return hist, res
    elif isinstance(bins, int):
        bin_edges = np.linspace(rmin, rmax, bins + 1)
        n_freq = bins - 1
        lb = np.zeros(n_freq)
        ub = np.ones(n_freq)
        bounds = np.vstack((lb, ub)).T
        x_0 = np.ones(n_freq)/(rmax - rmin)

        def fun(x):
            return _histogram_error(sample, bin_edges, x)
        cons = LinearConstraint(np.diff(bin_edges[:-1]), 0, 1)  # Ensure that the last frequency is not negative
        res = minimize(fun, x_0, bounds=bounds, constraints=cons)
        freq = res.x
        freq = _complete_histogram(bin_edges, freq)
        hist = (freq, bin_edges)
        return hist, res




