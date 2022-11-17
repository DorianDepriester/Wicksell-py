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
    _, rmax_m = np.meshgrid(x, rmax)
    pdf = np.zeros(shape=x_m.shape)
    left = (0 < x_m) & (x_m <= rmin_m)
    x_l = x_m[left]
    pdf[left] = 2 * x_l / (rmax_m[left] ** 2 - rmin_m[left] ** 2) * log(
        (rmax_m[left] + sqrt(rmax_m[left] ** 2 - x_l ** 2)) /
        (rmin_m[left] + sqrt(rmin_m[left] ** 2 - x_l ** 2)))
    center = (rmin_m < x_m) & (x_m <= rmax_m)
    x_c = x_m[center]
    pdf[center] = 2 * x_c / (rmax_m[center] ** 2 - rmin_m[center] ** 2) * log(
        (rmax_m[center] + sqrt(rmax_m[center] ** 2 - x_c ** 2)) / x_c)
    return pdf


def cdf_uni(x, rmin, rmax):
    x_m, rmin_m = np.meshgrid(x, rmin)
    _, rmax_m = np.meshgrid(x, rmax)
    cdf = np.zeros(shape=x_m.shape)
    left = (0 < x_m) & (x_m <= rmin_m)
    x_l = x_m[left]
    gamma = rmax_m[left] * sqrt(rmax_m[left] ** 2 - x_l ** 2) - x_l ** 2 * log(
        rmax_m[left] + sqrt(rmax_m[left] ** 2 - x_l ** 2))
    cdf[left] = 1 - (gamma + x_l ** 2 * log(rmin_m[left] + sqrt(rmin_m[left] ** 2 - x_l ** 2)) - rmin_m[left] * sqrt(
        rmin_m[left] ** 2 - x_l ** 2)) \
                / (rmax_m[left] ** 2 - rmin_m[left] ** 2)
    center = (rmin_m < x_m) & (x_m <= rmax_m)
    xc = x_m[center]
    gamma = rmax_m[center] * sqrt(rmax_m[center] ** 2 - xc ** 2) - xc ** 2 * log(
        rmax_m[center] + sqrt(rmax_m[center] ** 2 - xc ** 2))
    cdf[center] = 1 - (gamma + xc ** 2 * log(xc)) / (rmax_m[center] ** 2 - rmin_m[center] ** 2)
    cdf[x_m > rmax_m] = 1.0
    return cdf


class WicksellTransform(stats.rv_continuous):
    """
    Wicksell transform of a given distribution.
    """

    def __init__(self, basedist, nbins=1000, rmin=0.0, **kwargs):
        """
        Creates a new distribution, defined as the Wicksell transform of an underlying continuous distribution [1].

        Parameters
        ----------
        basedist : scipy.stats.rv_continuous
            The distribution to be transformed (base-distribution)

        rmin : float, optional
            The value at which the transformed distribution is left-truncated (default is 0, i.e. no truncation)

        nbins : int, optional
            The number of bins to use for constant-quantile histogram decomposition of the base-distribution (default is
             1000). See ref. [1] for details.

        References
        ----------
         .. [1] Wicksell S. (1925), doi:10.1093/biomet/17.1-2.84
         .. [2] Depriester D. and Kubler R. (2019), doi:10.5566/ias.2133
        """
        self.basedist = basedist
        new_name = 'Wicksell transform of {}'.format(basedist.name)
        super().__init__(shapes=basedist.shapes, a=max(0.0, basedist.a), b=np.inf, name=new_name, **kwargs)
        self._pdf_untruncated_vec = np.vectorize(self._pdf_untruncated_single, otypes='d')
        self._cdf_untruncated_vec = np.vectorize(self._cdf_untruncated_single, otypes='d')

        # Overwrites the default argument parsers
        self._parse_args = self._parse_args_modified
        self._parse_args_rvs = self._parse_args_rvs_modified
        self._parse_args_stats = self._parse_args_stats_modified

        # Integration options
        self.nbins = nbins
        self.Rmax = -1.0
        self.rmin = rmin

    def _argcheck(self, *args):
        """
        Check that:
        - the argument passed to the base distribution are correct
        - the support of base distribution is a subset of [0, +inf)
        """
        args, _, _ = self.basedist._parse_args(*args)
        return self.basedist._argcheck(*args) and (self.basedist.support(*args)[0] >= 0.0)

    def _parse_args_modified(self, *args, **kwargs):
        """
        Fool the argument parser so that loc and scale parameters are considered as related to the base-distribution
        (not the transformed one).
        """
        args, loc, scale = self.basedist._parse_args(*args, **kwargs)
        return args + (loc, scale), 0, 1

    def _parse_args_rvs_modified(self, *args, **kwargs):
        """
        Fool the argument parser so that loc and scale parameters are considered as related to the base-distribution
        (not the transformed one).
        """
        args, loc, scale, size = self.basedist._parse_args_rvs(*args, **kwargs)
        args.extend([loc, scale])
        return args, 0, 1, size

    def _parse_args_stats_modified(self, *args, **kwargs):
        """
        Fool the argument parser so that loc and scale parameters are considered as related to the base-distribution
        (not the transformed one).
        """
        args, loc, scale, moments = self.basedist._parse_args_stats(*args, **kwargs)
        args + (loc, scale)
        return args, 0, 1, moments

    def _get_support(self, *args, **kwargs):
        """
        The support is actually given by the left truncation (if any, 0 otherwise) and the maximum value given by the
        base-distribution.
        """
        return self.rmin, self.basedist.support(*args, **kwargs)[1]

    def wicksell(self, x, *args, **kwargs):
        """
        Performs full intregration of the Wicksell equation.
        """
        args, _, _ = self._parse_args(*args, **kwargs)
        frozen_dist = self.basedist(*args)
        E = frozen_dist.mean()
        if 0.0 < x:
            integrand = lambda R: frozen_dist.pdf(R) * (R ** 2 - x ** 2) ** (-0.5)
            return integrate.quad(integrand, x, np.inf)[0] * x / E
        else:
            return 0.0

    def _rv_cont2hist(self, *args, **kwargs):
        """
        Converts the continuous distribution into a finite histogram. Each class of this histogram has constant
        frequency (constant-quantile decomposition).
        """
        args, loc, scale = self.basedist._parse_args(*args, **kwargs)
        if self.basedist == stats.uniform:
            lb = loc
            ub = lb + scale
            mid_points = (lb + ub) / 2
            freq = 1 / scale
        else:
            frozen_dist = self.basedist(*args, **kwargs)
            if frozen_dist.support()[1] == np.inf:
                q_max = self.nbins / (self.nbins + 1)
            else:
                q_max = 1.0
            q = np.linspace(0, q_max, self.nbins + 1)
            lb = frozen_dist.ppf(q)
            if self.Rmax > lb[-1] and q_max != 1.0:
                lb = np.append(lb, 1.001 * self.Rmax)
            ub = lb[1:]
            lb = lb[:-1]
            mid_points = (lb + ub) / 2
            freq = frozen_dist.cdf(ub) - frozen_dist.cdf(lb)
            freq = freq / np.sum(freq)
        return lb, mid_points, ub, freq

    def _pdf_untruncated_single(self, x, *args):
        """
        Numerically compute the (untruncated) PDF of the Wicksell transform using the histogram decomposition.
        """
        args, baseloc, basescale = self.basedist._parse_args(*args)
        if x < self.rmin:
            return 0.0
        else:
            lb, mid_points, ub, freq = self._rv_cont2hist(*args, loc=baseloc, scale=basescale)
            MF = freq * mid_points
            P = pdf_uni(x, lb, ub)
            return np.dot(P.T, MF) / np.sum(MF)

    def _cdf_untruncated_single(self, x, *args):
        """
        Numerically compute the (untruncated) CDF of the Wicksell transform using the histogram decomposition.
        """
        args, baseloc, basescale = self.basedist._parse_args(*args)
        lb, mid_points, ub, freq = self._rv_cont2hist(*args, loc=baseloc, scale=basescale)
        MF = freq * mid_points
        C = cdf_uni(x, lb, ub)
        return np.dot(C.T, MF) / np.sum(MF)

    def _pdf(self, x, *args):
        """
        Numerically compute the (possibly truncated) PDF of the Wicksell transform using the histogram decomposition.
        """
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
        """
        Numerically compute the (possibly truncated) CDF of the Wicksell transform using the histogram decomposition.
        """
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
        """
        For the sake a efficiency, the stats are computed from RVs.
        """
        data = self.rvs(*args, size=10000)
        return np.mean(data), np.var(data), stats.skew(data), stats.kurtosis(data)

    def expect(self, *args, **kwargs):
        """
        The expectation is estimated from full integration of the full integration of Wicksell transform.
        """
        integrand = lambda x: self.wicksell(x, *args, **kwargs) * x
        return integrate.quad(integrand, 0, np.inf)[0]

    def _ppf(self, p, *args, **kwargs):
        """
        Quantile function
        """
        ppf_0 = self.basedist.ppf(p, *args, **kwargs)
        return opti.newton_krylov(lambda x: self.cdf(x, *args, **kwargs) - p, ppf_0)

    def _isf(self, p, *args, **kwargs):
        """
        Inverse survival function
        """
        isf_0 = self.basedist.isf(p, *args, **kwargs)
        return opti.newton_krylov(lambda x: self.cdf(x, *args, **kwargs) + p - 1, isf_0)

    def _rvs(self, *args, size=None, random_state=None):
        """
        Random variate generator.
        """
        if size is None:
            n_req = 1
        else:
            n_req = np.prod(size)
        nbr_spheres = max(10000, int(10 * n_req))  # Number of spheres to choose
        r = self.basedist.rvs(*args, size=nbr_spheres, random_state=random_state)
        centers = np.cumsum(2 * r) - r  # centers
        x_pick = stats.uniform.rvs(size=n_req, scale=np.sum(2 * r), random_state=random_state)
        i = [np.argmin((x_pick_i - centers) ** 2 - r ** 2) for x_pick_i in x_pick]
        r2 = r[i] ** 2 - (x_pick - centers[i]) ** 2
        if size is None:
            return np.sqrt(r2[0])
        else:
            return np.sqrt(r2).reshape(size)

    def _unpack_loc_scale(self, theta):
        """
        Fool the fit() method so that it cannot try to reduce the function to minimize. This is because loc and scale
        parameters must be passed to the base-distribution (and not treated as-is).
        """
        loc, scale, args = self.basedist._unpack_loc_scale(theta)
        return 0, 1, args + (loc, scale)

    def _fitstart(self, data, args=None):
        """
        Here, we use the _fitstarts method from the base distribution. Note that using this as an initial guess is a very
        poor idea. Still, it ensures that each value in the initial guess are valid, i.e.:
            self._argcheck(theta_0)==True
        """
        if self.basedist == stats.uniform:
            theta_0 = (max(data) / 2, max(data))
        else:
            theta_0 = self.basedist._fitstart(data)
        return theta_0

    def _moment_error(self, theta, x, data_moments):
        """When the Method of Moments (MM) is used for fit(), _unpack_theta() is incompatible with the way the
        moment_error is programed.
        """
        # The hack: use the default version of _unpack_loc_scale
        loc, scale, args = super()._unpack_loc_scale(theta)

        # Everything below is the same as in _distn_infrastructure.py
        if not self._argcheck(*args) or scale <= 0:
            return np.inf

        dist_moments = np.array([self.moment(i + 1, *args, loc=loc, scale=scale)
                                 for i in range(len(data_moments))])
        if np.any(np.isnan(dist_moments)):
            raise ValueError("Method of moments encountered a non-finite "
                             "distribution moment and cannot continue. "
                             "Consider trying method='MLE'.")

        return (((data_moments - dist_moments) /
                 np.maximum(np.abs(data_moments), 1e-8)) ** 2).sum()
