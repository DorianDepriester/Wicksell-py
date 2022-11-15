import scipy.stats as stats
import numpy as np


class posnorm_gen(stats.rv_continuous):
    """Positive normal distribution"""
    def _pdf(self, x, mu, s):
        a = -mu / s
        return stats.truncnorm.pdf(x, a, np.inf, scale=s, loc=mu)

    def _cdf(self, x, mu, s):
        a = -mu / s
        return stats.truncnorm.cdf(x, a, np.inf, scale=s, loc=mu)

    def _fitstart(self, data, args=None):
        return np.mean(data), np.std(data), 0.0, 1.0

    def _get_support(self, *args, **kwargs):
        return 0.0, np.inf

    def _argcheck(self, *args):
        return args[1] > 0

    def _ppf(self, q, mu, s):
        a = -mu / s
        A = (1 - stats.norm.cdf(a)) * q + stats.norm.cdf(a)
        return s * stats.norm.ppf(A) + mu

