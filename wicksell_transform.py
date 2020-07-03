# -*- coding: utf-8 -*-
"""
Created on Tue Jun 23 16:42:21 2020

@author: Dorian
"""
import scipy.stats as stats
from scipy.stats._distn_infrastructure import rv_frozen
import scipy.optimize as opti
import numpy as np
import scipy.integrate as integrate
from numbers import Number as num

class wickselled_trans(stats.rv_continuous):
    """
    Wicksell's transform of a given distribution.

    Reference:
     - Wicksell S. (1925), doi:10.1093/biomet/17.1-2.84
     - Depriester D and Kubler R (2019), doi:10.5566/ias.2133
    """
    
    def __init__(self, basedist, nint=500, eps=1e-3, **kwargs):
        self.basedist = basedist
        self.nint = nint
        self.eps = eps
        new_name = 'Wicksell''s transform of {}'.format(basedist.name)
        if basedist.shapes is None:
            shapes = 'baseloc, basescale'
        else:
            shapes = basedist.shapes + ', baseloc, basescale'
        super().__init__(shapes=shapes, a=max(0.0, basedist.a), b=basedist.b, name=new_name, **kwargs)
        self._wicksellvec = np.vectorize(self._wicksell_single, otypes='d')

    def _argcheck(self, *args):
        """
        Check that all the following conditions are valid:
        - the argument passed to the base distribution are correct
        - basescale is positive
        - the support of base distribution is a subset of [0, +inf]
        """
        return self.basedist._argcheck(*args[:-2]) and (self.basedist.support(*args)[0] >= 0.0) and (args[-1] > 0.0)

    def _parse_args(self, * args, **kwargs):
        return self.basedist._parse_args(*args)

    def _parse_args_stats(self, * args, **kwargs):
        return self.basedist._parse_args_stats(*args)

    def _parse_args_rvs(self, * args, **kwargs):
        return self.basedist._parse_args_rvs(*args)
        
    def _wicksell_single(self, x, *args, **kwargs):
        E = self.basedist.mean(*args, **kwargs)
        if 0.0 < x:
            integrand=lambda R: self.basedist.pdf(R, *args, **kwargs)*(R**2-x**2)**(-0.5)
            return integrate.quad(integrand, x, np.inf)[0]*x/E
        else:
            return 0.0
        
    def _pdf(self, x, *args):
        print(args)
        *args, baseloc, basescale = args
        if isinstance(x, num) or x.size == 1:
            return self._wicksellvec(x, *args, loc=baseloc, scale=basescale)
        else:
            if x.size < self.nint:
                return [self._wicksellvec(xi, *args, loc=baseloc, scale=basescale) for xi in x]
            else:
                if not isinstance(baseloc, num):
                    baseloc = baseloc[0]
                    basescale = basescale[0]
                    args = [args_i[0] for args_i in args]
                x1 = self.basedist.isf(self.eps, *args, loc=baseloc, scale=basescale)
                x0 = 0.999 * x1
                y0 = self._wicksellvec(x0, *args, loc=baseloc, scale=basescale)
                y1 = self._wicksellvec(x1, *args, loc=baseloc, scale=basescale)
                fp = (y1 - y0) / (x1 - x0)
                a = fp / y1            
                xp = np.linspace(np.min(x), np.max(x), self.nint)
                yp = [self._wicksellvec(xi, *args, loc=baseloc, scale=basescale) for xi in xp]
                y = np.zeros(x.shape)
                y[x <= x1] = np.interp(x[x <= x1], xp, yp)
                y[x > x1] = y1 * np.exp(a * (x[x > x1] - x1))
                return y
    
    def _cdf(self, x, *args):
        *args, baseloc, basescale = args
        integrand = lambda r: self._wicksellvec(r, *args, loc=baseloc, scale=basescale)
        if isinstance(x, num) or x.size == 1:
            return integrate.quad(integrand, 0, x)[0]
        else:
            if x.size < 0.0:
                return [integrate.quad(integrand, 0, xi)[0] for xi in x]
            else:
                loc0 = baseloc[0]
                scale0 = basescale[0]
                args = [args_i[0] for args_i in args]
                xint = np.linspace(0, np.max(x), self.nint)
                pdfi = [self._wicksellvec(xi, *args, loc=loc0, scale=scale0) for xi in xint]
                yint = integrate.cumtrapz(pdfi, x=xint, initial=0.0)
                return np.interp(x, xint, yint, left=0.0, right=1.0)

    def _stats(self, *args):
        data = self.rvs(*args[:-2], baseloc=args[-2], basescale=args[-1], size=10000)
        return np.mean(data), np.var(data), stats.skew(data), stats.kurtosis(data)
    
    def expect(self, *args, baseloc=0.0, basescale=1.0, **kwargs):
        integrand = lambda x: self._wicksellvec(x, *args, loc=baseloc, scale=basescale, **kwargs) * x
        return integrate.quad(integrand, 0, np.inf)[0]
        
    def _ppf(self, p, *args, baseloc=0.0, basescale=1.0, **kwargs):
        ppf_0 = self.basedist.ppf(p, *args, loc=baseloc, scale=basescale, **kwargs)
        return opti.newton_krylov(lambda x: self.cdf(x, *args, loc=baseloc, scale=basescale, **kwargs)-p, ppf_0)

    def isf(self, p, *args, baseloc=0.0, basescale=1.0, **kwargs):
        isf_0 = self.basedist.isf(p, *args, loc=baseloc, scale=basescale, **kwargs)
        return opti.newton_krylov(lambda x: self.cdf(x, *args, loc=baseloc, scale=basescale, **kwargs)+p-1, isf_0)
    
    def rvs(self, *args, baseloc=0.0, basescale=1.0, size=None, **kwargs):
        if size is None:
            n_req = 1
            init_size = 1000
        else:
            n_req = np.prod(size)
            if n_req < 1000:
                init_size = 1000
            else:
                init_size = n_req
        r = self.basedist.rvs(*args, size=init_size, loc=baseloc, scale=basescale, **kwargs)
        x_ref = np.cumsum(2*r) - r
        x_pick = np.random.rand(init_size) * np.sum(2*r)
        i = [np.argmin((x_pick_i-x_ref)**2 - r**2) for x_pick_i in x_pick]
        r2 = r[i]**2 - (x_pick-x_ref[i])**2
        if n_req == 1:
            return np.sqrt(r2[0])
        elif n_req < 1000:
            return np.sqrt(r2[:n_req]).reshape(size)
        else:
            return np.sqrt(r2).reshape(size)
    
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
