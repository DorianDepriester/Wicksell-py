import scipy.stats as stats
import numpy as np
from Wicksell.transform import WicksellTransform
from matplotlib import pyplot as plt
from posnorm import posnorm_gen

if __name__ == "__main__":
    """
    The script will run 3 tests, investigating different base distributions (namely: uniform, positive normal and 
    logNormal). For each distribution, this script will:
        1. compute the transformed Probability Density Function (PDF),
        2. Generate random data from the considered distribution,
        3. Try to retrieve the distribution parameters by fitting the random data (estimator),
        4. Perform the Kolmogorov-Smirnov (KS) goodness-of-fit test from the estimator,
        5. Plot:
            a. the PDF of the base-distribution,
            b. the transformed PDF,
            c. the transformed PDF, computed from the estimator
            d. the values of the random data as histogram.
    """

    # List of investigated distributions, and their related parameters
    distros = {
        "uniform":
            {"distro": stats.uniform,
             "param": [],
             "baseloc": 1,
             "basescale": 1.5},
        "positiveNormal":
            {"distro": posnorm_gen(name='Positive Normal'),
             "param": [1, 0.5],
             "baseloc": 0,
             "basescale": 1},
        "lognorm":
            {"distro": stats.lognorm,
             "param": [0.2],
             "baseloc": 0,
             "basescale": 2}
    }

    n_sample = 1000     # Size of the random data

    # Plotting options
    x = np.linspace(0, 3.5, 1000)   # Used for plotting the PDFs
    fig, axs = plt.subplots(len(distros), 1)
    fig.tight_layout(h_pad=3)

    for i, dist in enumerate(distros):
        print("Distribution: " + dist)

        # Compute PDF and transformed PDF
        basedist = distros[dist]['distro']
        trans_dist = WicksellTransform(basedist)
        baseloc = distros[dist]['baseloc']
        basescale = distros[dist]['basescale']
        param = distros[dist]['param']
        pdf = basedist.pdf(x, *param, loc=baseloc, scale=basescale)
        tpdf = trans_dist.pdf(x, *param, loc=baseloc, scale=basescale)

        # Generate random data
        sample = trans_dist.rvs(*param, loc=baseloc, scale=basescale, size=n_sample)

        # Estimate the distribution parameter from the random data
        if dist == 'uniform':
            theta = trans_dist.fit(sample)
        elif dist == 'positiveNormal':
            theta = trans_dist.fit(sample, floc=0.0, fscale=1)
        else:
            theta = trans_dist.fit(sample, floc=0.0)

        # Print results and perform KS test
        print("Fit: {}".format(theta))
        ks = stats.kstest(sample, trans_dist.cdf, theta)
        print('KS test: {}'.format(ks))

        # Plot all of this
        axs[i].set_ylim(bottom=0.0, top=1.1 * max(pdf))
        axs[i].set_xlim(left=0.0, right=3.5)
        axs[i].plot(x, pdf, 'r', label='PDF')
        axs[i].plot(x, tpdf, 'b', label='transf. PDF')
        axs[i].plot(x, trans_dist.pdf(x, *theta), 'b', linestyle='dotted', label='Fit')
        axs[i].hist(sample, ec='yellow', fc='orange', bins=25, density=True, label='Rand. samp.')
        axs[i].set_xlabel('R')
        axs[i].set_ylabel('Frequency')
        axs[i].legend()
        axs[i].set_title(dist)

        print("---")

    plt.show()
