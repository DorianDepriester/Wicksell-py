---
title: 'Wycksell-py: A Python integration for the corpuscle problem'
tags:
  - Python
  - stereology
  - spherical particles
  - grain size analysis
  - Wicksell equation
authors:
  - name: Dorian Depriester
    orcid: 0000-0002-2881-8942
    corresponding: true # (This is how to denote the corresponding author)    
    affiliation: 1
affiliations:
 - name: Arts et Métiers Institute of Technology, MSMP, HESAM Université, F-13617 Aix-en-Provence, France
   index: 1
date: 25 october 2022
bibliography: paper.bib
--- 

# Summary
`Wicksell-py` is a Python package for statistical analysis of sphere radii, when observed in 2D sections. The relashiontionhip between the distribution of the apparent
size (in 2D) and the actual ones (in 3D) is indeed given by the so-called Wicksell equation. The proposed implementation of the Wicksell equation is made as part of a 
subclass from the `scipy.stats` module.

# Statement of need
In materials sciences, such as petrology, geology or metallography, the microstructures of granular materials are usually characterized by 2D observations (e.g. optical
microscopy or Scanning Electron Microscopy). Thus, the real grain size distribution cannot be directly inferred from these measurements, and the relationship between the
2D apparent size distribution and the real 3D distribution is commonly refered to as the corpuscle problem. 

For equiaxed materials, grains are usually considered as perfect spheres. Indeed, in this case, the corpuscle problem can be solved through the so-called Wicksell
equation [@Wicksell:1925]. Let $f(R)$ be the Probability Density Function (PDF) of the radii $R$ of spheres randomly located in space. If these spheres are cut at random latitudes, 
the resulting disks will have radii $r$ following the PDF $\tilde{f}(r)$ so that:

\begin{equation}
\tilde{f}(r)=\frac{r}{E}\int_0^\infty \frac{f(R)}{\sqrt{R^2-r^2}}\mathrm{d}R
\label{eq:Wicksell}
\end{equation}

where $E$ is the expectation on $R$:

$$E=\int_0^\infty Rf(R)\mathrm{d}R$$

The most widely used technique to *unfold* a given distribution of apparent radii is the so-called Saltykov technique [@Slatikov:1967]. This method uses a finite histogram (bins) then uses \autoref{eq:Wicksell} to get the unfolded histogram. @Lopez-Sanchez:2018 has proposed a tool to automatically fit a continuous distribution (e.g. lognormal) on the unfolded histogram.

# Implementation
A Python subclass of the `scipy.stats.rv_continuous` module [@Virtanen:2020] have been developped so that the user can easily compute the Wicksell transform of any kind of continuous distribution. The numerical computation of the Wicksell transform \autoref{eq:Wicksell} works on the constant-quantile histogram decomposition [@Depriester:2021] and takes advantage mathematical developpments made in an earlier work [@Depriester:2019]. Hence, this allows to easily fit a continuous distribution on the unfolded distribution (through the `scipy.stats.rv_continuous.fit` inherited method).


