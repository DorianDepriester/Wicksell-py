import numpy as np
from compute_transform import from_histogram
from scipy.optimize import minimize
from wickselluniform import cdf_uni
from scipy import stats
from collections.abc import Iterable


def _histogram_error(sample, bin_edges, freq):
    """
    Computes the negative loglikelihood function from the transformed PDF given by an histogram.
    """
    hist = (freq, bin_edges)
    wh = from_histogram(hist)
    return stats.kstest(sample, wh.cdf)[0]


def _wicksell(r, lb, ub):
    """
    When a sphere of radius r is cut at random latitude, the probability of finding a disk with radius comprised
    between lb and ub is:

    .. math::
        \frac{1r}\left( \sqrt{r^2-lb^2} - \sqrt{r^2-ub^2} \right)

    References
    ----------
    Sahagian, D. and Proussevitch, A. (1998). doi: 10.1016/S0377-0273(98)00043-2

    """
    return (np.sqrt(r ** 2 - lb ** 2) - np.sqrt(r ** 2 - ub ** 2)) / r


def _wicksell_uniform(a, b, lb, ub):
    return cdf_uni(b, lb, ub) - cdf_uni(a, lb, ub)


def Saltykov(sample, bins=10):
    """
    Compute the unfolded histogram from a folded sample, using the Saltykov method. This implementation of the Saltykov
    method uses a modified estimation of probabilities, taking advantage of cdf_uni function.

    Parameters
    ----------
    sample : Iterable
        Random values of apparent radii
    bins : int or Iterable
        If bins is int, the Saltykov method will use the specified numer of bins
        If bins is a series of increasing values, they will be used as bin edges.
    """
    freq, bin_edges = np.histogram(sample, bins=bins)
    bin_sizes = np.diff(bin_edges)
    freq = freq / np.sum(freq * bin_sizes)
    n_bins = len(freq)
    for i in reversed(range(0, n_bins)):
        p_i = _wicksell(bin_edges[i + 1], bin_edges[i], bin_edges[i + 1])
        for j in range(0, i):
            p_j = _wicksell_uniform(bin_edges[j], bin_edges[j + 1], bin_edges[i], bin_edges[i + 1])
            p_norm = p_j * freq[i] / p_i
            freq[j] = np.abs(freq[j] - p_norm)
    freq = freq / np.sum(freq * bin_sizes)
    return freq, bin_edges


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
    bins : Iterable
        Number of bins to use. If int, the function will be run with only this number of bins. If bins is a list, like
        (n_min, n_max), this function will run for each value ranging between n_min and n_max. The returned value will
        be that maximizing the likelihood.

    Returns
    -------
    hist : tuple
       The unfolded histogram which minimizes the KS test.

    res : scipy.optimize._optimize.OptimizeResult
        Information about minimizing procedure. In particular, the KS test value is provided by res.fun.

    """
    if rmax is None:
        rmax = max(sample) * 4 / np.pi
    if isinstance(bins, int):
        # e.g. bins=10
        bin_edges = np.linspace(rmin, rmax, bins + 1)
        return fit_histogram(sample, bins=bin_edges)
    elif isinstance(bins, Iterable):
        if len(bins) == 2 and isinstance(bins[0], int) and isinstance(bins[1], int):
            # e.g. bins=(10,20)
            cost = np.inf
            hist = ()
            res = ()
            for n in range(bins[0], bins[1] + 1):
                bin_edges = np.linspace(rmin, rmax, bins=n+1)
                hist_n, res_n = fit_histogram(sample, bins=bin_edges)
                if res_n.fun < cost:
                    hist, res = hist_n, res_n
                    cost = res_n.fun
            return hist, res
        else:
            # e.g. bins=(0., 1., 2., 3.)
            spacing = np.diff(bins)
            n_bins = len(bins) - 1

            def fun(x):
                return _histogram_error(sample, bins, x)

            def confun(x):
                return np.sum(x * spacing) - 1

            lb = np.zeros(n_bins)
            ub = np.ones(n_bins) * np.inf
            bounds = np.vstack((lb, ub)).T
            x_0, _ = Saltykov(sample, bins=bins)
            res = minimize(fun, x_0, bounds=bounds, constraints={'type': 'eq', 'fun': confun})
            freq = res.x
            hist = (freq, bins)
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

    **kwargs
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
