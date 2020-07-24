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
    
In the example above, ``distro`` must be continuous distribution, as defined in the [scipy's stats](https://docs.scipy.org/doc/scipy/reference/stats.html) module. Finally, use this instance as a usual scipy's distribution.

## References
- Wicksell, S. D. (1925). The corpuscle problem: A mathematical study of a biometric
problem. *Biometrika*, 17(1/2):84–99, DOI: [10.2307/2332027](https://www.doi.org/10.2307/2332027)
- Depriester, D. and Kubler, R. (2019). Resolution of the Wicksell's equation by minimum
distance estimation. *Image Analysis & Stereology*, 38(3):213–226, DOI: [10.5566/ias.2133](https://www.doi.org/10.5566/ias.2133)

