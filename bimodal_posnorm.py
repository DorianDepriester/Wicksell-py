import scipy.stats as stats
import numpy as np

class bimodalposnorm_gen(stats.rv_continuous):
    """Mixture of two positive normal distributions"""
    def _pdf(self, x, *args):
        mu1, s1, mu2, s2, f1 = args
        a1 = -mu1 / s1
        pdf1 = stats.truncnorm.pdf(x, a1, np.inf, scale=s1, loc=mu1)
        a2 = -mu2 / s2
        pdf2 = stats.truncnorm.pdf(x, a2, np.inf, scale=s2, loc=mu2)
        return f1*pdf1 + (1-f1)*pdf2

    def _get_support(self, *args, **kwargs):
        return 0., np.inf

    def _rvs(self, *args, size=None, random_state=None):
        mu1, s1, mu2, s2, f1 = args
        if size is None:
            n_req = 1
        else:
            n_req = np.prod(size)
        rand_dist = stats.uniform.rvs(size=n_req, random_state=random_state)
        n1 = np.count_nonzero(rand_dist < f1)
        n2 = n_req - n1
        a1 = -mu1 / s1
        rvs1 = stats.truncnorm.rvs(a1, np.inf, scale=s1, loc=mu1, size=n1)
        a2 = -mu2 / s2
        rvs2 = stats.truncnorm.rvs(a2, np.inf, scale=s2, loc=mu2, size=n2)
        rvs = np.concatenate((rvs1, rvs2))
        if size is None:
            return rvs[0]
        else:
            return rvs.reshape(size)
