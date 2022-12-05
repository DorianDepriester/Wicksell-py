import Wicksell.wicksell_transform as wt
import numpy as np
import matplotlib.pyplot as plt
from Wicksell.histogram_tools import fit_histogram, plot_histogram, Saltykov, two_step_method
from bimodal_posnorm import bimodalposnorm_gen

bimod = bimodalposnorm_gen(name='Bimodal positive Normal')
mu1 = 1
s1 = 0.5
mu2 = 3
s2 = 0.5
f1 = 0.3
dist = bimod(mu1, s1, mu2, s2, f1)
wdist = wt.from_continuous(dist)

size = 10000
bins = 30
real_radii = dist.rvs(size=size)
Rmax=max(real_radii)
r = np.linspace(0, Rmax, 1000)
pdf = dist.pdf(r)

apparent_radii = wdist.rvs(size=size)
bins = np.linspace(0, max(apparent_radii), bins + 1)
hist_salt = Saltykov(apparent_radii, bins=bins)
hist_fit, res = fit_histogram(apparent_radii, bins=bins)

fig, axs = plt.subplots(2, 1, constrained_layout=True)
axs[0].set_title('Apparent radii')
axs[0].set_xlim(left=0.0, right=Rmax)
axs[0].set_xlabel('r')
axs[0].hist(apparent_radii, density=True, bins=50, label='Random data')
axs[0].legend()

axs[1].set_title('Unfolded distribution')
axs[1].set_xlim(left=0.0, right=Rmax)
axs[1].set_xlabel('R')
plot_histogram(axs[1], hist_salt, label='Saltykov method', fc='orange', ec='red')
plot_histogram(axs[1], hist_fit, label='Histogram fit', alpha=0.5, fc='cyan', ec='blue')
axs[1].plot(r, pdf, label='Exact solution (PDF)')
axs[1].legend()
fig.show()



