import numpy as np
import warnings


def expect(sigma, mu):
    return np.exp(mu + sigma ** 2 / 2)


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

    References
    ----------
    Depriester D. and Kubler R. (2021), doi:10.1016/j.jsg.2021.104418

    """

    mu = np.log(scale)
    E = expect(sigma, mu)
    sigma_min = sigma / (1 + 2.402 / np.sqrt(n))
    sigma_max = sigma / (1 - 2.402 / np.sqrt(n))
    E_min = E / (1 + 3.003 * sigma_max / np.sqrt(n))
    E_max = E / (1 - 3.003 * sigma_max / np.sqrt(n))
    mu_min = np.log(E_min) - sigma_max ** 2 / 2
    mu_max = np.log(E_max) - sigma_min ** 2 / 2
    return {'sigma': (sigma_min, sigma_max),
            'E': (E_min, E_max),
            'mu': (mu_min, mu_max)}


def polyhedron_bias(sigma, **kwargs):
    """
    Correct the bias introduced by the spherical assumption, when one tries to unfold a lognormal size distribution
    on polyhedrons. For convenience, the scale parameter can be defined different ways (see below).

    Parameters
    ----------
    sigma : float
        Shape parameter of the lognormal distribution, given by fitting a transformed lognormal distribution on a
        sample.

    kwargs : float
        scale parameter. It can be:
            - scale, which correspond to conventional scale parametrization in scipy.stats;
            - E, which sets the scale parameter through the expectation;
            - mu, which sets the scale parameter through the usual mu parameter.
        Default is scale=1

    Returns
    -------
    tuple
        The 1st output is the corrected value of sigma.
        The 2nd output is the scale parameter, using the same convention as in keyword arguments (can be scale, E or
        mu). Default is scale.

    References
    ----------
    Depriester D. and Kubler R. (2021), doi:10.1016/j.jsg.2021.104418
    """
    if kwargs == {}:
        scale = 1
        E = expect(sigma, np.log(scale))
        out = 'scale'
    elif len(kwargs) == 1:
        if 'E' in kwargs:
            E = kwargs['E']
            out = 'E'
        elif 'scale' in kwargs:
            E = expect(sigma, np.log(kwargs['scale']))
            out = 'scale'
        elif 'mu' in kwargs:
            E = expect(sigma, kwargs['mu'])
            out = 'mu'
        else:
            raise TypeError('The keyword argument should be either E, scale or mu.')
    else:
        raise TypeError('Only one keyword argument is allowed (E, scale or mu).')

    # Polynom coefficients in eqs. (43) in ref.
    p_E = np.array((0.402, -0.589, 1.019))
    p_sigma = np.array((-0.430, 1.361, 0.014))

    # 1st, check the validity domain
    lb = 0.1
    ub = 0.9
    sigma_min, sigma_max = np.polyval(p_sigma, [lb, ub])
    if (sigma < sigma_min) or (sigma > sigma_max):
        mess = 'The bias function was initially set up for sigma in [{:.3f},{:.3f}].'.format(sigma_min, sigma_max)
        warnings.warn(mess)

    # Solve  0 = p[0]*sigma^2 + p[1]*sigma + (p[1] - sigma)
    p_roots = p_sigma
    p_roots[-1] += -sigma
    roots = np.roots(p_roots)
    sigma_real = roots[(roots >= 0) & (roots <= 1)]
    if not np.any(sigma_real):
        err_mess = 'Sigma is out of range (too far from the [{:.3f},{:.3f}] range)'.format(sigma_min, sigma_max)
        raise ValueError(err_mess)

    sigma_real = sigma_real[0]
    E_real = E / np.polyval(p_E, sigma_real)
    scale_real = E_real * np.exp(-sigma_real**2 / 2)
    if out == 'scale':
        return sigma_real, scale_real
    elif out == 'E':
        return sigma_real, E_real
    else:
        return sigma_real, np.log(scale_real)
