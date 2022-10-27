import scipy.stats as stats
import numpy as np
from wicksell_transform import wicksell_trans
from matplotlib import pyplot as plt
from posnorm import posnorm_gen

posnorm=posnorm_gen()

if __name__ == "__main__":
    distros = {
        "uniform":
            {"distro": stats.uniform,
             "param": [],
             "baseloc": 1,
             "basescale": 2},
         "positiveNormal":
             {"distro": posnorm,
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
        trans_dist = wicksell_trans(basedist)
        baseloc = distros[dist]['baseloc']
        basescale = distros[dist]['basescale']
        param = distros[dist]['param']
        pdf = basedist.pdf(x, *param, loc=baseloc, scale=basescale)
        tpdf = trans_dist.pdf(x, *param, baseloc=baseloc, basescale=basescale)
        sample = trans_dist.rvs(*param, baseloc=baseloc, basescale=basescale, size=500)

        axs[i].set_ylim(bottom=0.0, top=1.1*max(pdf))
        axs[i].plot(x, pdf, 'r', label='PDF')
        axs[i].plot(x, tpdf, 'b', label='transf. PDF')
        axs[i].legend()
        axs[i].set_xlabel('R')
        axs[i].set_ylabel('Frequency')
        axs[i].hist(sample, bins=25, density=True, label='Random samp.')
        axs[i].set_title(dist)

        if dist == 'uniform':
            theta = trans_dist.fit(sample)
        elif dist == 'positiveNormal':
            theta = trans_dist.fit(sample, fbaseloc=0.0, fbasescale=1)
        else:
            theta = trans_dist.fit(sample, fbaseloc=0.0)
        ks = stats.kstest(sample, trans_dist.cdf, theta)
        print("Distribution: " + dist)
        print("Fit: {}".format(theta))
        print('KS test: {}'.format(ks))
        print("---")
    plt.show()
