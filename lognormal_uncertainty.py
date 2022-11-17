import numpy as np

valid_params_msg = 'E, mu or scale.'


def uncertainty(sigma, n, **kwargs):
    if len(kwargs) == 0:
        E = 1
    elif len(kwargs) > 1:
        raise TypeError('Only one optional argument is valid. It can be either ' + valid_params_msg)
    else:
        keyword, value = kwargs.popitem()
        if keyword == 'E':
            E = value
        elif keyword == 'mu':
            mu = value
            E = np.exp(mu + sigma**2 / 2)
        elif keyword == 'scale':
            return uncertainty(sigma, n, mu=np.log(value))
        else:
            raise TypeError('Unknown optional argument. It can be ' + valid_params_msg)
    sigma_min = sigma / (1 + 2.402/np.sqrt(n))
    sigma_max = sigma / (1 - 2.402/np.sqrt(n))
    E_min = E / (1 + 3.003 * sigma_max / np.sqrt(n))
    E_max = E / (1 - 3.003 * sigma_max / np.sqrt(n))
    mu_min = np.log(E_min) - sigma_max**2 / 2
    mu_max = np.log(E_max) - sigma_min**2 / 2
    return {'sigma':(sigma_min, sigma_max),
            'E':(E_min, E_max),
            'mu':(mu_min, mu_max)}