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
    
    def __init__(self, basedist, *args, nint=500, eps=1e-3, **kwargs):
        self.basedist = basedist
        self.nint = nint
        self.eps = eps
        new_name = 'Wicksell''s transform of {}'.format(basedist.name)
        super().__init__(*args, shapes=basedist.shapes, a=max(0.0, basedist.a), b=basedist.b, name=new_name, **kwargs)

    def _argcheck(self, *args):
        return self.basedist._argcheck(*args)
        
    def _wicksell(self, x, *args, **kwargs):
        """
        Analytical computation of the Probability Density Function
        """
        E = self.basedist.mean(*args, **kwargs)
        if 0.0 < x:
            integrand=lambda R: self.basedist.pdf(R, *args, **kwargs)*(R**2-x**2)**(-0.5)
            return integrate.quad(integrand, x, np.inf)[0]*x/E
        else:
            return 0.0
        
    def _pdf(self, x, *args, **kwargs):
        if isinstance(x, num) or x.size == 1:
            return self._wicksell(x, *args, **kwargs)
        else:
            if x.size < self.nint:
                return [self._wicksell(xi, *args, **kwargs) for xi in x]
            else:
                x1 = self.basedist.isf(self.eps, *args, **kwargs)
                x0 = 0.999 * x1
                y0 = self._wicksell(x0, *args, **kwargs)
                y1 = self._wicksell(x1, *args, **kwargs)
                fp = (y1 - y0) / (x1 - x0)
                a = fp / y1            
                xp = np.linspace(np.min(x), np.max(x), self.nint)
                yp = [self._wicksell(xi, *args, **kwargs) for xi in xp]
                y = np.zeros(x.shape)
                y[x <= x1] = np.interp(x[x <= x1], xp, yp)
                y[x > x1] = y1 * np.exp(a * (x[x > x1] - x1))
                return y
    
    def _cdf(self, x, *args, **kwargs):
        integrand = lambda r: self._wicksell(r, *args, **kwargs)
        if isinstance(x, num) or x.size == 1:
            return integrate.quad(integrand, 0, x)[0]
        else:
            if x.size < 0.0:
                return [integrate.quad(integrand, 0, xi)[0] for xi in x]
            else:
                xint = np.linspace(0, np.max(x), self.nint)
                pdfi = [self._wicksell(xi, *args, **kwargs) for xi in xint]
                yint = integrate.cumtrapz(pdfi, x=xint, initial=0.0)
                return np.interp(x, xint, yint, left=0.0, right=1.0)
    
    def expect(self, *args, **kwargs):
        integrand = lambda x: self._wicksell(x, *args, **kwargs) * x
        return integrate.quad(integrand, 0, np.inf)[0]
    
    def mean(self, *args, size=10000, **kwargs):
        data = self.rvs(size=size, *args, **kwargs)
        return np.mean(data)
        
    def _ppf(self, p, *args, **kwargs):
        ppf_0 = self.basedist.ppf(p, *args, **kwargs)
        return opti.newton_krylov(lambda x: self._cdf(x, *args, **kwargs)-p, ppf_0)  

    def isf(self, p, *args, **kwargs):
        isf_0 = self.basedist.isf(p, *args, **kwargs)
        return opti.newton_krylov(lambda x: self._cdf(x, *args, **kwargs)+p-1, isf_0)
    
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

    def pdf(self, x, *args, **kwargs):
        return self._pdf(x, *args, **kwargs)

    def cdf(self, x, *args, **kwargs):
        return self._cdf(x, *args, **kwargs)
    
    def _fitstart(self, data, args=None):
        return self.basedist._fitstart(data, args=args)

    def freeze(self, *args, **kwds):
        return rv_wickselled_frozen(self, *args, **kwds)

    def __call__(self, *args, **kwds):
        return self.freeze(*args, **kwds)