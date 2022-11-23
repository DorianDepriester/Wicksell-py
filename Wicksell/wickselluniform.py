from numpy import sqrt, log
import numpy as np


def pdf_uni(x, rmin, rmax):
    x_m, rmin_m = np.meshgrid(x, rmin)
    _, rmax_m = np.meshgrid(x, rmax)
    pdf = np.zeros(shape=x_m.shape)
    left = (0 < x_m) & (x_m <= rmin_m)
    x_l = x_m[left]
    pdf[left] = 2 * x_l / (rmax_m[left] ** 2 - rmin_m[left] ** 2) * log(
        (rmax_m[left] + sqrt(rmax_m[left] ** 2 - x_l ** 2)) /
        (rmin_m[left] + sqrt(rmin_m[left] ** 2 - x_l ** 2)))
    center = (rmin_m < x_m) & (x_m <= rmax_m)
    x_c = x_m[center]
    pdf[center] = 2 * x_c / (rmax_m[center] ** 2 - rmin_m[center] ** 2) * log(
        (rmax_m[center] + sqrt(rmax_m[center] ** 2 - x_c ** 2)) / x_c)
    return pdf


def cdf_uni(x, rmin, rmax):
    x_m, rmin_m = np.meshgrid(x, rmin)
    _, rmax_m = np.meshgrid(x, rmax)
    cdf = np.zeros(shape=x_m.shape)
    left = (0 < x_m) & (x_m <= rmin_m)
    x_l = x_m[left]
    gamma = rmax_m[left] * sqrt(rmax_m[left] ** 2 - x_l ** 2) - x_l ** 2 * log(
        rmax_m[left] + sqrt(rmax_m[left] ** 2 - x_l ** 2))
    cdf[left] = 1 - (gamma + x_l ** 2 * log(rmin_m[left] + sqrt(rmin_m[left] ** 2 - x_l ** 2)) - rmin_m[left] * sqrt(
        rmin_m[left] ** 2 - x_l ** 2)) \
                / (rmax_m[left] ** 2 - rmin_m[left] ** 2)
    center = (rmin_m < x_m) & (x_m <= rmax_m)
    xc = x_m[center]
    gamma = rmax_m[center] * sqrt(rmax_m[center] ** 2 - xc ** 2) - xc ** 2 * log(
        rmax_m[center] + sqrt(rmax_m[center] ** 2 - xc ** 2))
    cdf[center] = 1 - (gamma + xc ** 2 * log(xc)) / (rmax_m[center] ** 2 - rmin_m[center] ** 2)
    cdf[x_m > rmax_m] = 1.0
    return cdf