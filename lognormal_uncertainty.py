import numpy as np


def uncertainty(n, sigma, scale=1):
    """
    Computes the uncertainty on the estimator given by Maximum Likelihood Estimation (MLE) on the Wicksell transform of
    lognormal distribution.

    Parameters
    ----------
    n : int
        Sample size
    sigma : float
        Shape parameter of the lognormal distribution, given by MLE.
    scale : float, optional
        Scale option of the lognormal distribution. The default is 1.

    Returns
    -------
    Dict
        A dictionary with confidence intervals for the shape parameter (sigma), the expectation (E) and the log-scale
        factor (mu).

    Reference
    ---------
    Depriester D. and Kubler R. (2019), doi:10.5566/ias.2133

    """

    mu = np.log(scale)
    E = np.exp(mu + sigma**2/2)
    sigma_min = sigma / (1 + 2.402/np.sqrt(n))
    sigma_max = sigma / (1 - 2.402/np.sqrt(n))
    E_min = E / (1 + 3.003 * sigma_max / np.sqrt(n))
    E_max = E / (1 - 3.003 * sigma_max / np.sqrt(n))
    mu_min = np.log(E_min) - sigma_max**2 / 2
    mu_max = np.log(E_max) - sigma_min**2 / 2
    return {'sigma': (sigma_min, sigma_max),
            'E': (E_min, E_max),
            'mu': (mu_min, mu_max)}