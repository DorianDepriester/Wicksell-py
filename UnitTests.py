import scipy.stats as stats
import numpy as np
from Wicksell.transform import WicksellTransform
from matplotlib import pyplot as plt
from posnorm import posnorm_gen

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
         "loc": 1,
         "scale": 1.5,
         "fit_option": {}},
    "positivenorm":
        {"distro": posnorm_gen(name='Positive Normal'),
         "param": [1, 0.5],
         "loc": 0,
         "scale": 1,
         "fit_option": {'floc': 0, 'fscale': 1}},
    "lognorm":
        {"distro": stats.lognorm,
         "param": [0.2],
         "loc": 0,
         "scale": 2,
         "fit_option": {'floc': 0}}
}


def run_test(distributions=None, fit=True, kstest=True, nsample=1000):
    if distributions is None:
        distributions = ['uniform', 'positivenorm', 'lognorm']

    # Plotting options
    x = np.linspace(0, 3.5, 1000)   # Used for plotting the PDFs
    fig, axs = plt.subplots(len(distributions), 1)
    if len(distributions) == 1:
        axs = [axs]   # If only one subplot is used, it should still be subscriptable.
    fig.tight_layout(h_pad=3)

    for i, dist in enumerate(distributions):

        # Compute PDF and transformed PDF
        basedist = distros[dist]['distro']
        trans_dist = WicksellTransform(basedist)
        loc = distros[dist]['loc']
        scale = distros[dist]['scale']
        param = distros[dist]['param']
        if len(param):
            formatted_params = "{}, loc={}, scale={}".format(', '.join(map(str, param)), loc, scale)
        else:
            formatted_params = "loc={}, scale={}".format(loc, scale)
        print("Distribution: {}({})".format(dist, formatted_params))

        pdf = basedist.pdf(x, *param, loc=loc, scale=scale)
        tpdf = trans_dist.pdf(x, *param, loc=loc, scale=scale)

        # Generate random data and plot them as histogram
        sample = trans_dist.rvs(*param, loc=loc, scale=scale, size=nsample)
        axs[i].hist(sample, ec='yellow', fc='orange', bins=25, density=True, label='Rand. samp.')

        # Plot PDFs
        axs[i].set_ylim(bottom=0.0, top=1.1 * max(pdf))
        axs[i].set_xlim(left=0.0, right=3.5)
        axs[i].plot(x, pdf, 'r', label='PDF')
        axs[i].plot(x, tpdf, 'b', label='transf. PDF')

        # Estimate the distribution parameter from the sample
        if fit:
            fit_option = distros[dist]['fit_option']
            theta = trans_dist.fit(sample, **fit_option)
            print("Fit: {}".format(theta))

            # Plot fitted curve
            axs[i].plot(x, trans_dist.pdf(x, *theta), 'b', linestyle='dotted', label='Fit')

            # Print results and perform KS test
            if kstest:
                ks = stats.kstest(sample, trans_dist.cdf, theta)
                print('KS test: {}'.format(ks))

        # Plot all of this
        axs[i].set_xlabel('R')
        axs[i].set_ylabel('Frequency')
        axs[i].legend()
        axs[i].set_title(dist)
        print("---")

    plt.show()


if __name__ == "__main__":
    run_test()


