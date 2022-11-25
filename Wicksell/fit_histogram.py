import numpy as np
from compute_transform import from_histogram
from scipy.optimize import minimize, LinearConstraint


def _complete_histogram(bin_edges, freq):
    """Retrieve the left-most frequency value, so that the integral over all the bins is 1."""
    remain = 1 - np.sum(freq * np.diff(bin_edges[1:]))
    left_freq = remain / np.diff(bin_edges)[0]
    freq = np.append(left_freq, freq)
    return freq


def _histogram_error_reduced(sample, bin_edges, freq):
    """Perform KS test from the transformed distribution, computed from a given histogram, against a given sample.
    The left-most frequency is inferred from all the others.
    """
    freq = _complete_histogram(bin_edges, freq)
    hist = (freq, bin_edges)
    wh = from_histogram(hist)
    return wh.nnlf((0, 1), sample)


def fit_histogram(sample, rmin=0.0, rmax=None, bins=10, spacing='linear'):
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
    spacing : str
        Spacing used for bins. Can be 'Linear' or 'geometric'. Default is 'linear' (i.e. evenly spaced bin edges in
        linear scale between rmin and rmax). 'geometric' uses an evenly space in log scale, so that the right-most bin
        is about twice smaller than the left-most one.

    Returns
    -------
    hist : tuple
       The unfolded histogram which minimizes the KS test.

    res : scipy.optimize._optimize.OptimizeResult
        Information about minimizing procedure. In particular, the KS test value is provided by res.fun.

    """
    if rmax is None:
        rmax = max(sample)*4/np.pi
    if isinstance(bins, tuple) or isinstance(bins, list):
        ks_min = 1.0
        hist = ()
        res = ()
        for n in range(bins[0], bins[1]+1):
            hist_n, res_n = fit_histogram(sample, rmin, rmax, bins=n, spacing=spacing)
            if res_n.fun < ks_min:
                hist, res = hist_n, res_n
                ks_min = res.fun
        return hist, res
    elif isinstance(bins, int):
        if spacing == 'linear':
            bin_edges = np.linspace(rmin, rmax, bins + 1)
        else:
            q = 2
            bin_edges = (-np.geomspace(q, 1, bins + 1) + q) * (rmax - rmin) / (q - 1) + rmin
        n_freq = bins - 1
        lb = np.zeros(n_freq)
        ub = np.ones(n_freq)
        bounds = np.vstack((lb, ub)).T
        x_0 = np.ones(n_freq)/(rmax - rmin)

        def fun(x):
            return _histogram_error_reduced(sample, bin_edges, x)
        cons = LinearConstraint(np.diff(bin_edges[:-1]), 0, 1)  # Ensure that the last frequency is not negative
        res = minimize(fun, x_0, bounds=bounds, constraints=cons)
        freq = res.x
        freq = _complete_histogram(bin_edges, freq)
        hist = (freq, bin_edges)
        return hist, res


def plot_histogram(ax, hist, *args, **kwargs):
    """
    Simple function to help plotting the resulting histogram.

    Parameters
    ----------
    ax : matplotlib.axes._subplots.AxesSubplot
        Handle to axis or figure where the histogram will go.

    hist : tuple
        Histogram details, so that hist == (frequencies, bin_edges)

    *args : tuple
        Extra value arguments passed to the ax.bar() function

    **kwargs : dict
        Extra keyword arguments passed to the ax.bar() function

    Returns
    -------
     l : list
        A list of lines representing the plotted data.

    Reference
    ---------
    Inspired from https://stackoverflow.com/a/33372888/12056867
    """
    freq, bin_edges = hist
    return ax.bar(bin_edges[:-1], freq, width=np.diff(bin_edges), align='edge', *args, **kwargs)




