import numpy as np
from scipy import stats as stats
import scipy.integrate as integrate
from wickselluniform import pdf_uni, cdf_uni


def _fun_from_hist(x, freq, edges, fun):
    lb = edges[:-1]
    ub = edges[1:]
    mid_points = (lb + ub) / 2
    a = freq * mid_points
    if fun == 'pdf':
        b = pdf_uni(x, lb, ub)
    else:
        b = cdf_uni(x, lb, ub)
    return np.dot(b.T, a) / np.sum(a)


class rv_histogram_wicksell_transformed(stats.rv_continuous):
    def __init__(self, hist, rmin=0.0, **kwargs):
        freq = hist[0]
        bin_edges = hist[1]
        if min(bin_edges) < 0:
            raise ValueError('The bin edges must all be positives.')
        self.bin_edges = bin_edges
        self.freq = freq / np.sum(freq * np.diff(hist[1]))
        super().__init__(**kwargs)
        self.rmin = rmin
        self.a = rmin
        self.b = bin_edges[-1]

    def _pdf(self, x, *args):
        pdf = _fun_from_hist(x, self.freq, self.bin_edges, 'pdf')
        if self.rmin == 0.:
            return pdf
        else:
            trunc = _fun_from_hist(self.rmin, self.freq, self.bin_edges, 'cdf')
            return pdf / (1 - trunc)

    def _cdf(self, x, *args):
        cdf = _fun_from_hist(x, self.freq, self.bin_edges, 'cdf')
        if self.rmin == 0.:
            return cdf
        else:
            trunc = _fun_from_hist(self.rmin, self.freq, self.bin_edges, 'cdf')
            return (cdf - trunc) / (1 - trunc)

    def _munp(self, n):
        r = 0
        bounds = self.bin_edges
        n_bounds = len(bounds)-1
        for i in range(0, n_bounds):
            r += integrate.quad(lambda x: x**n*self.pdf(x), bounds[i], bounds[i+1])[0]
        return r
