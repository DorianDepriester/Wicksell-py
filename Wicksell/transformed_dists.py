# -*- coding: utf-8 -*-
"""
Created on Tue Jun 23 16:42:21 2020

@author: Dorian
"""
import numpy as np
import scipy.stats as stats
from rv_continuous_transformed import rv_continuous_wicksell_transformed
from rv_histogram_transformed import rv_histogram_wicksell_transformed


def from_continuous(basedist, nbins=1000, rmin=0.0, **kwargs):
    """
    Creates a new distribution, defined as the Wicksell transform of an underlying continuous distribution [1].

    Parameters
    ----------
    basedist : scipy.stats.rv_continuous or scipy.stats._distn_infrastructure.rv_continuous_frozen
        The distribution to be transformed (base-distribution). This distribution can be frozen.

    rmin : float
        The value at which the transformed distribution is left-truncated (default is 0, i.e. no truncation)

    nbins : int
        The number of bins to use for constant-quantile histogram decomposition of the base-distribution (default is
         1000). See ref. [1] for details.

    Returns
    -------
    rv_histogram_transformed.rv_continuous_wicksell_transformed
        The transformed distribution. If the base-distribution is frozen, the returned distribution is frozen as well.

    References
    ----------
     .. [1] Wicksell S. (1925), doi:10.1093/biomet/17.1-2.84
     .. [2] Depriester D. and Kubler R. (2021), doi:10.1016/j.jsg.2021.104418
    """

    if isinstance(basedist, stats.rv_continuous):
        # If the base-distribution is rv_continuous, just return the transformed one.
        return rv_continuous_wicksell_transformed(basedist, nbins, rmin=rmin, **kwargs)
    elif isinstance(basedist, stats._distn_infrastructure.rv_continuous_frozen):
        # If the base-distribution is frozen, instance a transformed one, then freeze it.
        transformed_dist = rv_continuous_wicksell_transformed(basedist.dist, nbins=nbins, rmin=rmin, **kwargs)
        return transformed_dist.freeze(*basedist.args, **basedist.kwds)


def from_histogram(hist, rmin=0.0):
    """
    Creates a continuous distribution, defined as the Wicksell transform of a finite histogram. The transformed
    distribution is analytically computed with respect to the formula given in ref. [1].

    Parameters
    ----------
    hist : tuple or list
        histogram of data. hist[0] must be the list of frequencies and hist[1] must be the list of bin edges. Usually,
        hist is provided by numpy.histogram().

    rmin : float
        The value at which the transformed distribution is left-truncated (default is 0., i.e. no truncation)

    Returns
    -------
    rv_histogram_transformed.rv_histogram_transformed
        Continuous distribution, resulting from the Wicksell transform of the histogram.

    See also
    --------
    from_continuous : Computes the Wicksell transform of a continuous distribution
    from_sample : Computes the Wicksell transform from a finite sample of random values

    References
    ----------
     .. [1] Depriester D. and Kubler R. (2019), doi:10.5566/ias.2133
    """
    return rv_histogram_wicksell_transformed(hist, rmin=rmin)


def from_sample(sample, rmin=0.0, **kwargs):
    """Computes the histogram from a given sample, then apply the Wicksell transform on that histogram.

    Parameters
    ----------
    Sample : numpy.ndarray
        List of random values

    rmin : float
        The value at which the transformed distribution is left-truncated (default is 0., i.e. no truncation)

    kwargs : dict
        Extra arguments passed to numpy.histogram()

    Returns
    -------
    rv_histogram_transformed.rv_histogram_transformed
        Continuous distribution related to the Wicksell transform of the binned data, computed from the sample.

    See also
    --------
    from_continuous : Computes the Wicksell transform of a continuous distribution.
    from_histogram : Computes the Wicksell transform of a finite histogram.
    numpy.histogram : Computes the histogram of a dataset.
    """
    hist = np.histogram(sample, **kwargs)
    return from_histogram(hist, rmin=rmin)
