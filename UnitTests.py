import scipy.stats as stats
import numpy as np
from Wicksell.transform import WicksellTransform
from matplotlib import pyplot as plt
from posnorm import posnorm_gen

if __name__ == "__main__":
    distros = {
        "uniform":
            {"distro": stats.uniform,
             "param": [],
             "baseloc": 1,
             "basescale": 1},
        "positiveNormal":
            {"distro": posnorm_gen(),
             "param": [0.2, 0.5],
             "baseloc": 0,
             "basescale": 1},
        "lognorm":
            {"distro": stats.lognorm,
             "param": [0.2],
             "baseloc": 0,
             "basescale": 2}
    }

    x = np.linspace(0, 3.5, 1000)
    fig, axs = plt.subplots(3, 1)

    for i, dist in enumerate(distros):
        basedist = distros[dist]['distro']
        trans_dist = WicksellTransform(basedist)
        baseloc = distros[dist]['baseloc']
        basescale = distros[dist]['basescale']
        param = distros[dist]['param']
        pdf = basedist.pdf(x, *param, loc=baseloc, scale=basescale)
        tpdf = trans_dist.pdf(x, *param, loc=baseloc, scale=basescale)
        sample = trans_dist.rvs(*param, loc=baseloc, scale=basescale, size=500)

        if dist == 'uniform':
            theta = trans_dist.fit(sample)
        elif dist == 'positiveNormal':
            theta = trans_dist.fit(sample, floc=0.0, fscale=1)
        else:
            theta = trans_dist.fit(sample, floc=0.0)
        print("Distribution: " + dist)
        print("Fit: {}".format(theta))
        ks = stats.kstest(sample, trans_dist.cdf, theta)
        print('KS test: {}'.format(ks))
        print("---")

        axs[i].set_ylim(bottom=0.0, top=1.1 * max(pdf))
        axs[i].plot(x, pdf, 'r', label='PDF')
        axs[i].plot(x, tpdf, 'b', label='transf. PDF')
        axs[i].plot(x, trans_dist.pdf(x, *theta), 'b', linestyle='dotted', label='Fit')
        axs[i].hist(sample, bins=25, density=True, label='Rand. samp.')
        axs[i].set_xlabel('R')
        axs[i].set_ylabel('Frequency')
        axs[i].legend()
        axs[i].set_title(dist)
    plt.show()
