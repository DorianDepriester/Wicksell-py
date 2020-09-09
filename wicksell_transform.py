# -*- coding: utf-8 -*-
"""
Created on Tue Jun 23 16:42:21 2020

@author: Dorian
"""
import scipy.stats as stats
import scipy.optimize as opti
import numpy as np
from numpy import sqrt, log
import scipy.integrate as integrate


def pdf_uni(x, rmin, rmax):
    x_m, rmin_m = np.meshgrid(x, rmin)
    _ , rmax_m = np.meshgrid(x, rmax)
    pdf = np.zeros(shape=x_m.shape)
    left = (0 < x_m) & (x_m <= rmin_m)
    x_l = x_m[left]
    pdf[left] = 2 * x_l / (rmax_m[left] ** 2 - rmin_m[left] ** 2) * log(
                             (rmax_m[left] + sqrt(rmax_m[left] ** 2 - x_l ** 2)) /
                             (rmin_m[left] + sqrt(rmin_m[left] ** 2 - x_l ** 2)))
    center = (rmin_m < x_m) & (x_m <= rmax_m)
    x_c = x_m[center]
    pdf[center] = 2 * x_c / (rmax_m[center] ** 2 - rmin_m[center] ** 2) * log((rmax_m[center] + sqrt(rmax_m[center] ** 2 - x_c ** 2)) / x_c)
    return pdf

def cdf_uni(x, rmin, rmax):
    x_m, rmin_m = np.meshgrid(x, rmin)
    _ , rmax_m = np.meshgrid(x, rmax)
    cdf = np.zeros(shape=x_m.shape)
    left = (0 < x_m) & (x_m <= rmin_m)
    x_l = x_m[left]
    gamma = rmax_m[left] * sqrt(rmax_m[left] ** 2 - x_l ** 2) - x_l ** 2 * log(rmax_m[left] + sqrt(rmax_m[left] ** 2 - x_l ** 2))
    cdf[left] = 1 - (gamma + x_l ** 2 * log(rmin_m[left] + sqrt(rmin_m[left] ** 2 - x_l ** 2)) - rmin_m[left] * sqrt(rmin_m[left] ** 2 - x_l ** 2))\
               / (rmax_m[left] ** 2 - rmin_m[left] ** 2)
    center = (rmin_m < x_m) & (x_m <= rmax_m)
    xc = x_m[center]
    gamma = rmax_m[center] * sqrt(rmax_m[center] ** 2 - xc ** 2) - xc ** 2 * log(rmax_m[center] + sqrt(rmax_m[center] ** 2 - xc ** 2))
    cdf[center] = 1 - (gamma + xc ** 2 * log(xc)) / (rmax_m[center] ** 2 - rmin_m[center] ** 2)
    cdf[x_m > rmax_m] = 1.0
    return cdf


class wicksell_trans(stats.rv_continuous):
    """
    Wicksell's transform of a given distribution.

    Reference:
     - Wicksell S. (1925), doi:10.1093/biomet/17.1-2.84
     - Depriester D and Kubler R (2019), doi:10.5566/ias.2133
    """

    def __init__(self, basedist, nbins=1000, rmin=0.0, **kwargs):
        self.basedist = basedist
        self.nbins = nbins
        new_name = 'Wicksell''s transform of {}'.format(basedist.name)
        if basedist.shapes is None:
            shapes = 'baseloc, basescale'
        else:
            shapes = basedist.shapes + ', baseloc, basescale'
        super().__init__(shapes=shapes, a=max(0.0, basedist.a), b=basedist.b, name=new_name, **kwargs)
        self._pdf_untruncated_vec = np.vectorize(self._pdf_untruncated_single, otypes='d')
        self._cdf_untruncated_vec = np.vectorize(self._cdf_untruncated_single, otypes='d')
        self.Rmax = -1.0
        self.rmin = rmin


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
        frozen_dist = self.basedist(*args, loc=baseloc, scale=basescale)
        E = frozen_dist.mean()
        if 0.0 < x:
            integrand = lambda R: frozen_dist.pdf(R) * (R ** 2 - x ** 2) ** (-0.5)
            return integrate.quad(integrand, x, np.inf)[0] * x / E
        else:
            return 0.0

    def _rv_cont2hist(self, *args, **kwargs):
        frozen_dist = self.basedist(*args, **kwargs)
        if frozen_dist.support()[1] == np.inf:
            q_max = self.nbins / (self.nbins + 1)
        else:
            q_max = 1.0
        q = np.linspace(0, q_max, self.nbins + 1)
        lb = frozen_dist.ppf(q)
        if self.Rmax > lb[-1] and q_max != 1.0:
            lb = np.append(lb, 1.01*self.Rmax)
        ub = lb[1:]
        lb = lb[:-1]
        mid_points = (lb + ub) / 2
        freq = frozen_dist.cdf(ub) - frozen_dist.cdf(lb)
        freq = freq / np.sum(freq)
        return lb, mid_points, ub, freq

    def _pdf_untruncated_single(self, x, *args):
        *args, baseloc, basescale = args
        if x < self.rmin:
            return 0.0
        else:
            lb, mid_points, ub, freq = self._rv_cont2hist(*args, loc=baseloc, scale=basescale)
            MF = freq * mid_points
            P = pdf_uni(x, lb, ub)
            return np.dot(P.T, MF) / np.sum(MF)

    def _cdf_untruncated_single(self, x, *args):
        *args, baseloc, basescale = args
        lb, mid_points, ub, freq = self._rv_cont2hist(*args, loc=baseloc, scale=basescale)
        MF = freq * mid_points
        C = cdf_uni(x, lb, ub)
        return np.dot(C.T, MF) / np.sum(MF)

    def _pdf(self, x, *args):
        if isinstance(x, int):
            self.Rmax = float(x)
        elif isinstance(x, float):
            self.Rmax = x
        else:
            self.Rmax = max(x)
        if self.rmin <= 0.0:
            return self._pdf_untruncated_vec(x, *args)
        else:
            return self._pdf_untruncated_vec(x, *args) / (1 - self._cdf_untruncated_vec(self.rmin, *args))

    def _cdf(self, x, *args):
        if isinstance(x, int):
            self.Rmax = float(x)
        elif isinstance(x, float):
            self.Rmax = x
        else:
            self.Rmax = max(x)
        if self.rmin <= 0.0:
            return self._cdf_untruncated_vec(x, *args)
        else:
            trunc = self._cdf_untruncated_vec(self.rmin, *args)
            return (self._cdf_untruncated_vec(x, *args) - trunc) / (1 - trunc)

    def _stats(self, *args):
        data = self.rvs(*args, size=10000)
        return np.mean(data), np.var(data), stats.skew(data), stats.kurtosis(data)

    def expect(self, *args, baseloc=0.0, basescale=1.0, **kwargs):
        integrand = lambda x: self.wicksell(x, *args, loc=baseloc, scale=basescale, **kwargs) * x
        return integrate.quad(integrand, 0, np.inf)[0]

    def _ppf(self, p, *args, **kwargs):
        ppf_0 = self.basedist.ppf(p, *args, **kwargs)
        return opti.newton_krylov(lambda x: self.cdf(x, *args, **kwargs) - p, ppf_0)

    def _isf(self, p, *args, **kwargs):
        isf_0 = self.basedist.isf(p, *args, **kwargs)
        return opti.newton_krylov(lambda x: self.cdf(x, *args, **kwargs) + p - 1, isf_0)

    def _rvs(self, *args, **kwargs):
        if self._size is None:
            n_req = 1
            init_size = 10000
        else:
            n_req = np.prod(self._size)
            if n_req < 10000:
                init_size = 10000
            else:
                init_size = n_req
        r = self.basedist.rvs(*args, size=init_size, **kwargs)
        x_ref = np.cumsum(2 * r) - r    # centers
        x_pick = np.random.rand(init_size) * np.sum(2 * r)
        i = [np.argmin((x_pick_i - x_ref) ** 2 - r ** 2) for x_pick_i in x_pick]
        r2 = r[i] ** 2 - (x_pick - x_ref[i]) ** 2
        if n_req == 1:
            return np.sqrt(r2[0])
        elif n_req < 10000:
            return np.sqrt(r2[:n_req]).reshape(self._size)
        else:
            return np.sqrt(r2).reshape(self._size)

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
