# -*- coding: utf-8 -*-
"""
Created on Tue Jun 23 16:42:21 2020

@author: Dorian
"""
import scipy.stats as stats
from scipy.stats._distn_infrastructure import rv_frozen
import scipy.optimize as opti
import numpy as np
from numpy import sqrt, log
import scipy.integrate as integrate
from numbers import Number as num


def pdf_uni(x, rmin, rmax):
    if x <= 0.0:
        return 0.0
    elif x <= rmin:
        return 2 * x / (rmax ** 2 - rmin ** 2) * log(
                             (rmax + sqrt(rmax ** 2 - x ** 2)) / (rmin + sqrt(rmin ** 2 - x ** 2)))
    elif (rmin < x) & (x <= rmax):
        return 2 * x / (rmax ** 2 - rmin ** 2) * log((rmax + sqrt(rmax ** 2 - x ** 2)) / x)
    else:
        return 0.0


def cdf_uni(x, rmin, rmax):
    gamma = lambda r: rmax * sqrt(rmax ** 2 - r ** 2) - r ** 2 * log(rmax + sqrt(rmax ** 2 - r ** 2))
    if x <= 0.0:
        return 0.0
    elif x <= rmin:
        return 1 - (gamma(x) + x ** 2 * log(rmin + sqrt(rmin ** 2 - x ** 2)) - rmin * sqrt(rmin ** 2 - x ** 2))\
               / (rmax ** 2 - rmin ** 2)
    elif (rmin < x) & (x <= rmax):
        return 1 - (gamma(x) + x ** 2 * log(x)) / (rmax ** 2 - rmin ** 2)
    else:
        return 1.0


def rv_cont2hist(frozen_dist, nbins):
    eps = 1 / (1000*nbins)
    q = np.linspace(0, 1-eps, nbins+1)
    lb = frozen_dist.ppf(q)
    ub = lb[1:]
    lb = lb[:-1]
    mid_points = (lb + ub) / 2
    freq = frozen_dist.cdf(ub) - frozen_dist.cdf(lb)
    freq = freq / np.sum(freq)
    return lb, mid_points, ub, freq


class wickselled_trans(stats.rv_continuous):
    """
    Wicksell's transform of a given distribution.

    Reference:
     - Wicksell S. (1925), doi:10.1093/biomet/17.1-2.84
     - Depriester D and Kubler R (2019), doi:10.5566/ias.2133
    """

    def __init__(self, basedist, nbins=1000, eps=1e-3, **kwargs):
        self.basedist = basedist
        self.nbins = nbins
        new_name = 'Wicksell''s transform of {}'.format(basedist.name)
        if basedist.shapes is None:
            shapes = 'baseloc, basescale'
        else:
            shapes = basedist.shapes + ', baseloc, basescale'
        super().__init__(shapes=shapes, a=max(0.0, basedist.a), b=basedist.b, name=new_name, **kwargs)
        self._pdf_vec = np.vectorize(self._pdf_single, otypes='d')
        self._cdf_vec = np.vectorize(self._cdf_single, otypes='d')


    def _argcheck(self, *args):
        """
        Check that all the following conditions are valid:
        - the argument passed to the base distribution are correct
        - basescale is positive
        - the support of base distribution is a subset of [0, +inf)
        """
        return self.basedist._argcheck(*args[:-2]) and (self.basedist.support(*args)[0] >= 0.0) and (args[-1] > 0.0)

    def wicksell(self, x, *args, **kwargs):
        *args, baseloc, basescale = args
        frozen_dist=self.basedist(*args, loc=baseloc, scale=basescale)
        E = frozen_dist.mean()
        if 0.0 < x:
            integrand = lambda R: frozen_dist.pdf(R) * (R ** 2 - x ** 2) ** (-0.5)
            return integrate.quad(integrand, x, np.inf)[0] * x / E
        else:
            return 0.0

    def _pdf_single(self, x, *args):
        *args, baseloc, basescale = args
        frozen_dist=self.basedist(*args, loc=baseloc, scale=basescale)
        lb, mid_points, ub, freq = rv_cont2hist(frozen_dist, self.nbins)
        ft = 0.0
        for i in range(0, len(freq)):
            ft += freq[i] * mid_points[i] * pdf_uni(x, lb[i], ub[i])
        return 1 / frozen_dist.mean() * ft

    def _cdf_single(self, x, *args):
        *args, baseloc, basescale = args
        frozen_dist=self.basedist(*args, loc=baseloc, scale=basescale)
        lb, mid_points, ub, freq = rv_cont2hist(frozen_dist, self.nbins)
        Ft = 0.0
        for i in range(0, len(freq)):
            Ft += freq[i] * mid_points[i] * cdf_uni(x, lb[i], ub[i])
        return 1 / frozen_dist.mean() * Ft

    def _pdf(self, x, *args):
        return self._pdf_vec(x, *args)

    def _cdf(self, x, *args):
        return self._cdf_vec(x, *args)

    def _stats(self, *args):
        data = self.rvs(*args[:-2], baseloc=args[-2], basescale=args[-1], size=10000)
        return np.mean(data), np.var(data), stats.skew(data), stats.kurtosis(data)

    def expect(self, *args, baseloc=0.0, basescale=1.0, **kwargs):
        integrand = lambda x: self._wicksellvec(x, *args, loc=baseloc, scale=basescale, **kwargs) * x
        return integrate.quad(integrand, 0, np.inf)[0]

    def _ppf(self, p, *args, baseloc=0.0, basescale=1.0, **kwargs):
        ppf_0 = self.basedist.ppf(p, *args, loc=baseloc, scale=basescale, **kwargs)
        return opti.newton_krylov(lambda x: self.cdf(x, *args, loc=baseloc, scale=basescale, **kwargs) - p, ppf_0)

    def isf(self, p, *args, baseloc=0.0, basescale=1.0, **kwargs):
        isf_0 = self.basedist.isf(p, *args, loc=baseloc, scale=basescale, **kwargs)
        return opti.newton_krylov(lambda x: self.cdf(x, *args, loc=baseloc, scale=basescale, **kwargs) + p - 1, isf_0)

    def rvs(self, *args, size=None, **kwargs):
        if size is None:
            n_req = 1
            init_size = 1000
        else:
            n_req = np.prod(size)
            if n_req < 1000:
                init_size = 1000
            else:
                init_size = n_req
        r = self.basedist.rvs(*args, size=init_size, **kwargs)
        x_ref = np.cumsum(2 * r) - r
        x_pick = np.random.rand(init_size) * np.sum(2 * r)
        i = [np.argmin((x_pick_i - x_ref) ** 2 - r ** 2) for x_pick_i in x_pick]
        r2 = r[i] ** 2 - (x_pick - x_ref[i]) ** 2
        if n_req == 1:
            return np.sqrt(r2[0])
        elif n_req < 1000:
            return np.sqrt(r2[:n_req]).reshape(size)
        else:
            return np.sqrt(r2).reshape(size)

    def _moment_distance(self, moments, *args):
        statistics = self.stats(*args, moments='mvsk')
        d = 0.0
        for i, moment in enumerate(moments):
            di = (statistics[i] - moments[i]) ** 2
            d += 1.0 / (1.0 + i) * di
        return d

    def _fitstart(self, data, args=None):
        """
        Here, we use the _fitstats method from the base distribution. Note that using this as an initial guess is a very
        poor idea. Still, it ensures that each value in the initial guess are valid, i.e.:
            self._argcheck(theta_0)==True
        """
        theta_0 = self.basedist._fitstart(data, args=args)
        return theta_0 + (0.0, 1.0)

    def fit(self, data, *args, floc=0.0, fscale=1.0, **kwds):
        return super().fit(data, *args, floc=floc, fscale=fscale, **kwds)

    def fit_moments(self, data, *args):
        moments = (np.mean(data), np.var(data), stats.skew(data), stats.kurtosis(data))

        def func(theta):
            return self._moment_distance(moments, *theta)

        return opti.fmin(func, self._fitstart(data, args=args), args=(np.ravel(data),), disp=False)
