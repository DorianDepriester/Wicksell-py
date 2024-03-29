import scipy.stats as stats
import numpy as np
from Wicksell import wicksell_transform as wt
from matplotlib import pyplot as plt
from posnorm import posnorm_gen
import histogram_tools as ht

"""
The script will run 4 tests, investigating different base distributions (namely: uniform, positive normal, 
logNormal and Weibull). For each distribution, this script will:
    1. compute the transformed PDF,
    2. generate a random sample,
    3. apply the two-step method (which includes the Saltykov method),
    3. fit a continuous distribution,
    4. Perform the KS goodness-of-fit test from fit
    5. Unfold the histogram, without considering a given distribution
At each step, the results will be illustrated as distribution plots. Using default options, this script will take 
about 15 minutes to complete on a modern computer.
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

labels = {"dist": 'Real PDF', "two_step": 'Two-step method', "salt": 'Saltykov', "fit": 'Fit (from distr.)',
         "fit_hist": 'Fit (from hist.)'}
colors = {"dist": 'k', "two_step": 'b', "salt": 'b', "fit": 'm',  "fit_hist": 'r'}
styles = {"dist": '-', "two_step": '--', "salt": '-', "fit": ':', "fit_hist": '-.'}


def run_test(distributions=None, two_step=True, fit_distribution=True, fit_histogram=True, kstest=True, nsample=1000):
    """
    Run tests on the distributions given above. By default, all the distributions will be considered, and the following
    test will be performed on each:
        1. compute the transformed PDF,
        2. generate a random sample,
        3. apply the two-step method (which includes the Saltykov method),
        3. fit a continuous distribution,
        4. Perform the KS goodness-of-fit test from fit
    At each step, the results will be illustrated as distribution plots.

    Parameters
    ----------
    distributions : str or list or tuple
        List of the names of the investigated parameters. They can be 'uniform', 'positivenorm' of 'lognorm' and
        'weibull'.
    two_step : bool
        Turn on/off the attempt to retrieve the distribution using the two-step method. Default is True.
    fit_distribution : bool
        Turn on/off the fit step. Default is True
    kstest : bool
        Turn on/off the KS goodness-of-fit test. Requires fit=True. Default is True
    nsample : int
        Size of the random sample to generate. Default if 1000.
    """
    if distributions is None:
        distributions = distros.keys()
    elif isinstance(distributions, str):
        distributions = [distributions]
    elif isinstance(distributions, tuple):
        distributions = list(distributions)

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

        # Generate random data and plot them as histogram
        sample = trans_dist.rvs(*param, loc=loc, scale=scale, size=nsample)
        bins = np.linspace(0, 1.3 * max(sample), 21)
        ax_transformed.hist(sample, ec='yellow', fc='orange', bins=bins, density=True, label='Rand. samp.')

        # Plot underlying/unfolded distributions
        x = np.linspace(0, 1.1*max(sample), 1000)  # Used for plotting the PDFs
        pdf = basedist.pdf(x, *param, loc=loc, scale=scale)
        tpdf = trans_dist.pdf(x, *param, loc=loc, scale=scale)
        ax_basedist.plot(x, pdf, label=labels['dist'], color=colors['dist'])
        ax_transformed.plot(x, tpdf, label=labels['dist'], color=colors['dist'])

        if two_step:
            theta, hist_salt = ht.two_step_method(sample, basedist, bins=bins)
            ht.plot_histogram(ax_basedist, hist_salt, ec=colors['salt'], fc=colors['salt'], alpha=0.3, label='Saltykov method')
            ax_basedist.plot(x, basedist.pdf(x, *theta),
                             color=colors['two_step'], linestyle=styles['two_step'], label=labels['two_step'])
            trans_hist_salt = wt.from_histogram(hist_salt)
            ax_transformed.plot(x, trans_hist_salt.pdf(x),
                                color=colors['two_step'], linestyle=styles['two_step'], label=labels['two_step'])

        # Estimate the distribution parameter from the sample
        if fit_distribution:
            theta = trans_dist.fit(sample, **fit_option)
            print("Fit: {}".format(theta))

            # Plot fitted curve
            ax_transformed.plot(x, trans_dist.pdf(x, *theta),
                                color=colors['fit'], linestyle=styles['fit'], label=labels['fit'])
            ax_basedist.plot(x, basedist.pdf(x, *theta),
                            color=colors['fit'], linestyle=styles['fit'], label=labels['fit'])

            # Print results and perform KS test
            if kstest:
                ks = stats.kstest(sample, trans_dist.cdf, theta)
                print('KS test: {}'.format(ks))

        # General plotting options
        fig.suptitle(dist, fontsize=16)

        ax_transformed.set_xlim(left=0.0)
        ax_transformed.set_ylim(bottom=0.0, top=1.1 * max(tpdf))
        ax_transformed.set_xlabel('r')
        ax_transformed.set_ylabel('Frequency')
        ax_transformed.legend()
        ax_transformed.set_title('Apparent/transformed distribution')

        ax_basedist.set_xlim(left=0.0)
        ax_basedist.set_ylim(bottom=0.0, top=1.1 * max(pdf))
        ax_basedist.set_xlabel('R')
        ax_basedist.set_ylabel('Frequency')
        ax_basedist.legend()
        ax_basedist.set_title('Underlying/unfolded distributions')

        fig.show()
        print("---")

    plt.show()


if __name__ == "__main__":
    run_test()
