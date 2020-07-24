# Wicksell-py
A Python class for computing Wicksell's transforms of continuous distributions.

## Purpose
Consider a medium consisting in a large number of spheres whose radii follow a Probability Density Function (PDF) *f*. 
If sections of the medium are made at random lattitudes, the radius apparents disks (denoted *r* below) would follow the PDF:

![a](https://latex.codecogs.com/gif.latex?\tilde{f}(r)=\frac{r}{E}\int_{r}^{\infty}\frac{f(R)}{\sqrt{R^2-r^2}}\mathrm{d}R)

where *E* is the mean value of *f*. The previous formula is refered as the Wicksell's equation. 
The histogram approximation is used to compute the Wicksell's equation of the continuous distribution *f*.


The aim of this project is to provide a robust and convinient way to compute the statistics of apparents disks (related to values of *r*).

## Usage
Just import the ``wicksell_trans`` class

    from wicksell_transform import wicksell_trans
    
and create an instance of that class, passing the underlying distribution (that used for computing the Wicksell transform).

    wt = wicksell_trans(distro)
    
In the example above, ``distro`` must be continuous distribution, as defined in the [scipy's stats](https://docs.scipy.org/doc/scipy/reference/stats.html) module. Finally, use this instance as a usual scipy's distribution. All the parameters related to the underlying distribution are inferred to the transformed one. The ``loc`` and ``scale`` parameters of the underlying distribution are renamed ``baseloc`` and ``basescale``, respectivelly.

## Example
In the following, the [lognormal](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.lognorm.html) distribion is considered.

    import scipy.stats as stats
    from wicksell_transform import wicksell_trans
    
    wlognorm = wicksell_trans(stats.lognorm)
    s = 0.1                 # Shape parameter for lognormal
    mu = 0.5
    baseloc = 0
    basescale = np.exp(mu)  # loc parameter of underlying distribution
    
### Compute the transformed PDF/CDF

    import numpy as np
    
    x = np.linspace(0, 4, 1000)
    pdf = wlognorm.pdf(x, s, baseloc, basescale)
    cdf = wlognorm.cdf(x, s, baseloc, basescale)

### Generate random variables

    data = wlognorm.rvs(s, baseloc, basescale, size=1000)
    
### Plot results

    from matplotlib import pyplot as plt
    
    fig, ax1 = plt.subplots()
    ax1.hist(data, bins=20, density=True, label='RVs')
    ax1.set_ylim(bottom=0.0)
    ax1.plot(x, pdf, 'r', label='PDF')
    ax1.plot(x, cdf, 'g', label='CDF')
    ax1.set_ylim(bottom=0.0)
    ax1.legend()
    ax1.set_xlabel('r')
    ax1.set_ylabel('Frequency')
    plt.show()
    
### Fit the empirical data

Empirical data can be used to fit the distribution in odrer to get the optimal distribution parameters:

    theta = wlognorm.fit(data, fbaseloc=0.0)
    
Here, the fit is made assuming that the location parameter is 0 (as a reminder, this parameter has been renamed ``baseloc``). The ``fit`` method is a build-in method provided in all rv_continuous distributions. See the [related documentation](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.rv_continuous.fit.html#scipy.stats.rv_continuous.fit) for details.

The example below roughly leads to:

    (0.10232127760439322, 0.0, 1.6544771155069116, 0.0, 1.0)
    
It appears that the first parameter is close to ``s`` (0.1) whereas the ``basescale`` (3rd one) corresponds to µ=ln(1.654)=0.503 (instead of 0.5). Note that the 2 last arguments relate to the location and scale parameters of __the transformed__ distribution. Thus, they are not relevent at all.

### Perform a goodness of fit test

The transformed CDF can be used to perform the [Kolmogov-Smirnov test](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.kstest.html). For instance, the parameters evaluated by fitting lead to:

    stats.kstest(data, wlognorm.cdf, theta)

    KstestResult(statistic=0.013768823282845344, pvalue=0.9914096826147232)

## References
- Wicksell, S. D. (1925). The corpuscle problem: A mathematical study of a biometric
problem. *Biometrika*, 17(1/2):84–99, DOI: [10.2307/2332027](https://www.doi.org/10.2307/2332027)
- Depriester, D. and Kubler, R. (2019). Resolution of the Wicksell's equation by minimum
distance estimation. *Image Analysis & Stereology*, 38(3):213–226, DOI: [10.5566/ias.2133](https://www.doi.org/10.5566/ias.2133)

