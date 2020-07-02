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
    "Wicksell's transform of a mixture of two normal distributions"
    
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
        return self.basedist._argcheck(*args[:-2])

    def _penalized_nnlf(self, theta, x):
        return self.basedist._penalized_nnlf(theta[:-2], x)

    def _parse_args(self, * args, **kwargs):
        return self.basedist._parse_args(*args)

    def _parse_args(self, * args, **kwargs):
        return self.basedist._parse_args_rvs(*args)

    def _reduce_func(self, args, kwds):
        args += (0.0, 1.0)
        return self.basedist._reduce_func(args, kwds)
        
    def _wicksell_single(self, x, *args, **kwargs):
        """
        Analytical computation of the Probability Density Function
        """
        E = self.basedist.mean(*args, **kwargs)
        if 0.0 < x:
            integrand=lambda R: self.basedist.pdf(R, *args, **kwargs)*(R**2-x**2)**(-0.5)
            return integrate.quad(integrand, x, np.inf)[0]*x/E
        else:
            return 0.0
        
    def _pdf(self, x, *args):
        *args, baseloc, basescale = args
        if isinstance(x, num) or x.size == 1:
            return self._wicksellvec(x, *args, loc=baseloc, scale=basescale)
        else:
            if x.size < self.nint:
                return [self._wicksellvec(xi, *args, loc=baseloc, scale=basescale) for xi in x]
            else:
                loc0 = baseloc[0]
                scale0 = basescale[0]
                args = [args_i[0] for args_i in args]
                x1 = self.basedist.isf(self.eps, *args, loc=loc0, scale=scale0)
                x0 = 0.999 * x1
                y0 = self._wicksellvec(x0, *args, loc=loc0, scale=scale0)
                y1 = self._wicksellvec(x1, *args, loc=loc0, scale=scale0)
                fp = (y1 - y0) / (x1 - x0)
                a = fp / y1            
                xp = np.linspace(np.min(x), np.max(x), self.nint)
                yp = [self._wicksellvec(xi, *args, loc=loc0, scale=scale0) for xi in xp]
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
    
    def expect(self, *args, baseloc=0.0, basescale=1.0, **kwargs):
        integrand = lambda x: self._wicksellvec(x, *args, loc=baseloc, scale=basescale, **kwargs) * x
        return integrate.quad(integrand, 0, np.inf)[0]
    
    def mean(self, *args, baseloc=0.0, basescale=1.0, size=10000, **kwargs):
        data = self.rvs(size=size, *args, **kwargs)
        return np.mean(data)
        
    def _ppf(self, p, *args, baseloc=0.0, basescale=1.0, **kwargs):
        ppf_0 = self.basedist.ppf(p, *args, loc=baseloc, scale=basescale, **kwargs)
        return opti.newton_krylov(lambda x: self._cdf(x, *args, loc=baseloc, scale=basescale, **kwargs)-p, ppf_0)

    def isf(self, p, *args, baseloc=0.0, basescale=1.0, **kwargs):
        isf_0 = self.basedist.isf(p, *args, loc=baseloc, scale=basescale, **kwargs)
        return opti.newton_krylov(lambda x: self._cdf(x, *args, loc=baseloc, scale=basescale, **kwargs)+p-1, isf_0)
    
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

    # def pdf(self, x, *args, **kwargs):
    #     baseloc=kwargs.get('baseloc', 0.0)
    #     basescale=kwargs.get('basescale', 1.0)
    #     return self._pdf(x, *args, baseloc, basescale)

    # def cdf(self, x, *args, **kwargs):
    #     baseloc=kwargs.get('baseloc', 0.0)
    #     basescale=kwargs.get('basescale', 1.0)
    #     return self._cdf(x, *args, baseloc, basescale)
    
    def _fitstart(self, data, args=None):
        theta_0 = self.basedist._fitstart(data, args=args)
        return theta_0

    def freeze(self, *args, **kwds):
        return rv_wickselled_frozen(self, *args, **kwds)

    def __call__(self, *args, **kwds):
        return self.freeze(*args, **kwds)