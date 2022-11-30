import scipy.stats as stats
import numpy as np
from Wicksell import compute_transform as wt
from matplotlib import pyplot as plt
from posnorm import posnorm_gen
import histogram_tools as ht

"""
The script will run 4 tests, investigating different base distributions (namely: uniform, positive normal, 
logNormal and Weibull). For each distribution, this script will:
    1. compute the transformed Probability Density Function (PDF),
    2. Generate random data from the considered distribution,
    3. Try to retrieve the distribution parameters by fitting the random data (estimator),
    4. Perform the Kolmogorov-Smirnov (KS) goodness-of-fit test from the estimator,
    5. Plot:
        a. the PDF of the base-distribution,
        b. the transformed PDF,
        c. the transformed PDF, computed from the estimator,
        d. the values of the random data as histogram.
Using default options, this script will take about 10 minutes to complete on a modern computer.
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
         "fit_option": {'floc': 0}},
    "weibull":
        {"distro": stats.weibull_min,
         "param": [1.5],
         "loc": 0,
         "scale": 0.5,
         "fit_option": {'floc': 0}},
}

styles = {
    "basedist": {"color": 'black',                      "label": 'Real (target) distr.'},
    "two_step": {"color": 'red',                        "label": 'Two-step method'},
    "fit":      {'color': 'b', 'linestyle': 'dotted',   "label": 'Fit (from distribution)'},
    "fit_histogram": {"color": 'blue',                  "label": 'Fit (from histogram)'},
    "Saltykov": {"color": 'red',                        "label": 'Saltykov method'}
}


def run_test(distributions=None, two_step=True, fit_distribution=True, fit_histogram=True, kstest=True, nsample=1000):
    """
    Run tests on the distributions given above. By default, all the distributions will be considered, and the following
    test will be performed on each:
        1. Compute the transformed PDF
        2. Generate a random sample
        3. Fit the distribution on the underlying data
        4. Perform the KS goodness-of-fit test
        5. Plot the results

    Parameters
    ----------
    distributions : list, optional
        List of the names of the investigated parameters. They can be 'uniform', 'positivenorm' of 'lognorm' and
        'weibull'.
    fit_distribution : bool
        Turn on/off the fit step. Default is True
    kstest : bool
        Turn on/off the KS goodness-of-fit test. Requires fit=True. Default is True
    nsample : int
        Size of the random sample to generate. Default if 1000.
    """
    if distributions is None:
        distributions = distros.keys()

    # Plotting options
    x = np.linspace(0, 3.5, 1000)  # Used for plotting the PDFs

    for i, dist in enumerate(distributions):
        fig, (ax_basedist, ax_transformed) = plt.subplots(2, 1, constrained_layout=True)

        # Compute PDF and transformed PDF
        basedist = distros[dist]['distro']
        trans_dist = wt.from_continuous(basedist)
        loc = distros[dist]['loc']
        scale = distros[dist]['scale']
        param = distros[dist]['param']
        fit_option = distros[dist]['fit_option']
        if len(param):
            formatted_params = "{}, loc={}, scale={}".format(', '.join(map(str, param)), loc, scale)
        else:
            formatted_params = "loc={}, scale={}".format(loc, scale)
        print("Distribution: {}({})".format(dist, formatted_params))

        pdf = basedist.pdf(x, *param, loc=loc, scale=scale)
        tpdf = trans_dist.pdf(x, *param, loc=loc, scale=scale)

        # Generate random data
        sample = trans_dist.rvs(*param, loc=loc, scale=scale, size=nsample)

        # Set up the bins here so that they are consistent en each bar plot
        bins = np.linspace(0, 1.3 * max(sample), 21)

        # Plot underlying/unfolded distributions
        ax_basedist.set_ylim(bottom=0.0, top=1.1 * max(pdf))
        ax_basedist.set_xlim(left=0.0, right=3.5)
        style = {'color': 'black'}
        ax_basedist.plot(x, pdf, label='Underlying PDF', **style)
        ax_transformed.plot(x, tpdf, label='transf. PDF', **style)

        # Plot transformed distributions and apparent histogram
        ax_transformed.set_ylim(bottom=0.0, top=1.1 * max(tpdf))
        ax_transformed.set_xlim(left=0.0, right=3.5)
        ax_transformed.hist(sample, ec='yellow', fc='orange', bins=bins, density=True, label='Rand. samp.')

        if two_step:
            theta, hist_salt = ht.two_step_method(sample, basedist, bins=bins)
            color = 'red'
            label = 'Two-step method'
            ht.plot_histogram(ax_basedist, hist_salt, ec=color, fc=color, alpha=0.2, label='Saltykov method')
            ax_basedist.plot(x, basedist.pdf(x, *theta), color=color, label=label)
            trans_hist_salt = wt.from_histogram(hist_salt)
            ax_transformed.plot(x, trans_hist_salt.pdf(x), color=color, label=label, linestyle='dashdot')

        # Estimate the distribution parameter from the sample
        if fit_distribution:
            theta = trans_dist.fit(sample, **fit_option)
            print("Fit: {}".format(theta))

            # Plot fitted curve
            style = {'label': 'Fit (from distribution)', 'color': 'b', 'linestyle': 'dotted'}
            ax_transformed.plot(x, trans_dist.pdf(x, *theta), **style)
            ax_basedist.plot(x, basedist.pdf(x, *theta), **style)

            # Print results and perform KS test
            if kstest:
                ks = stats.kstest(sample, trans_dist.cdf, theta)
                print('KS test: {}'.format(ks))

        # Unfold the histogram
        if fit_histogram:
            # Unfold the distribution (w/o considering the continuous distribution)
            hist, res = ht.fit_histogram(sample, bins=bins)
            trans_hist = wt.from_histogram(hist)
            color = 'blue'
            label = 'Fit (from histogram)'
            ht.plot_histogram(ax_basedist, hist, ec=color, fc=color, alpha=0.2, label=label)
            ax_transformed.plot(x, trans_hist.pdf(x), color=color, label=label, linestyle='dashed')

        # Plot all of this
        fig.suptitle(dist, fontsize=16)
        ax_transformed.set_xlabel('r')
        ax_transformed.set_ylabel('Frequency')
        ax_transformed.legend()
        ax_transformed.set_title('Apparent/transformed distribution')

        ax_basedist.set_xlabel('R')
        ax_basedist.set_ylabel('Frequency')
        ax_basedist.legend()
        ax_basedist.set_title('Underlying/unfolded distributions')

        fig.show()
        print("---")

    plt.show()


if __name__ == "__main__":
    run_test(nsample=1000)
